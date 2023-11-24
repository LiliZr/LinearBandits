import numpy as np

from LinearBandits import *

############################################################
#                                                          #
#                        LinUCB                            #
#                                                          #
############################################################


class LinUCB(LinearBandit):
    def update_V(self, a_t):
        """
            Update V with given action
                Compute : V_t = V_0 + ∑ action_t ⋅ action_t.T
        """
        self.V = self.V + np.outer(a_t, a_t)


    def update_ar(self, a_t, r_t):
        """
            Update ar cumulative sum
                Compute : ar_t = ∑ action_t ⋅ reward_t
        """
        self.ar = self.ar + (a_t * r_t)

    def init_run(self, n):
        super().init_run(n)
        ## Params used in regression
        # Init inverse of covariance matrix
        self.Vinv = (1/self.lam) * np.identity(self.d) if self.lam != 0 else np.zeros((self.V.shape))
        # Init product action x reward
        self.ar = np.zeros(self.d)
        # Estimator of the parameter theta
        self.theta_est = np.zeros(self.d)

    def run(self, n=100):
        # Initialize variables
        self.t0 = time.process_time()
        self.init_run(n)
        self.t = 0

        while self.t < n:
            # Action recommended 
            a_t = self.recommend()
    
            # Observed reward
            r_t = self.generate_reward(a_t)

            # Update parameters
            self.update_ar(a_t, r_t)

            # Linear regression
            inv_sherman_morrison(self.Vinv, a_t)
            self.theta_est = self.Vinv @ self.ar 

            self.time[self.t] =  time.process_time() - self.t0
            self.t += 1
            self.cumulative_reward[self.t] =  self.cumulative_reward[self.t - 1] + r_t
        self.cumulative_reward = self.cumulative_reward[1:]


    def recommend(self):
        """ 
            recommends over a finite set
                return: 
                    argmax_a ⟨ a, ̂θ ⟩ + β ||a||_V^-1 
                with β defined as in https://papers.nips.cc/paper_files/paper/2011/file/e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf
        """
        beta = ((np.sqrt(self.lam) * self.m2) + (self.sigma * np.sqrt((- 2 * np.log10(self.delta)) + 
                               (self.d * np.log10(1 + ((self.t * self.L**2)/(self.lam*self.d)))))) ) * self.scale

        # Compute UCB for each action
        ucb_max = float('-inf')
        a_max = self.action_set[0]
        self.selected_action_idx = 0
        for idx, a in enumerate(self.action_set):
            ucb = (a @ self.theta_est) + (beta * np.sqrt(a @ (self.Vinv @ a)))
            if ucb > ucb_max:
                ucb_max = ucb
                a_max = a
                self.selected_action_idx = idx  
        return a_max
    
def inv_sherman_morrison(B, u):
    """
        Efficient Inverse of 1-rank update 
            return : (B + uu.T)^-1
    """
    u = u.reshape(-1, 1)
    Bu = B @ u
    np.subtract(B, (Bu @ (u.T @ B)) / (1 + (u.T @ Bu)), out=B)
