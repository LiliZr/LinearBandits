#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time




############################################################
#                                                          #
#               Linear Bandit General Class                #
#                                                          #
############################################################

class LinearBandit():
    def __init__(self, theta=None, lam=0.01, sigma=0., delta=0.01, 
                 scale=0.001, action_set=None, seed=48):
        """
            theta: True parameter to estimate
            lam: Regularization parameter λ used in regression
            sigma: noise standard deviation σ used in reward computation
            delta: probability param
            scale: scaling factor for beta (ellipsoid param)
            action_set: given action set (implies finite set case)
        """
        self.rng = np.random.RandomState(seed)
        self.theta = theta
        self.sigma = sigma
        self.lam = lam
        self.action_set = action_set
        self.set_size = len(action_set) 
        self.d = self.action_set.shape[1]
        self.scale = scale
        self.delta = delta





    def generate_reward(self, a_t):
        """
            Generate observed noisy reward given an action/context vector
        """
        r =  (self.theta @ a_t) + (self.rng.randn() * self.sigma)
        return r


    def init_run(self, n):
        """
            Initialize different variables to keep track of reward, CPU time, ...etc
        """
        
        # Init cumulative reward to zero
        self.cumulative_reward = np.empty(n + 1, dtype=object)
        self.cumulative_reward[0], self.cumulative_reward[-1] = 0, 0
        # CPU Time
        self.time = np.zeros(n) 
        # Upper bound norm of theta
        self.m2 = 1 if self.theta is None else np.linalg.norm(self.theta, 2)
        # Upper bound norm actions
        self.L = max(np.linalg.norm(self.action_set, 2, axis=1))


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

            self.time[self.t] =  time.process_time() - self.t0
            self.t += 1
            self.cumulative_reward[self.t] =  self.cumulative_reward[self.t - 1] + r_t
        self.cumulative_reward = self.cumulative_reward[1:]

class Optimal(LinearBandit):
    """ 
        Optimal model
            Knows the true model parameter so it recommends the best action possible each time
    """
    def recommend(self):
        # Get the action with highest value
        self.selected_action_idx = np.argmax(self.action_set @ self.theta)
        return self.action_set[self.selected_action_idx]

class Random(LinearBandit):
    """ 
        Random model
            Recommends actions randomly
    """
    def recommend(self):
        # Recommending random action over finite set
        self.selected_action_idx = self.rng.randint(0, self.action_set.shape[0])
        return self.action_set[self.selected_action_idx]