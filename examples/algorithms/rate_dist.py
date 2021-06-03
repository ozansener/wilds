import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


class BlahutArimoto:
    def __init__(self, _beta):
        self.beta = _beta
        self.device = torch.device("cuda") 
        self.conditionals_pos = {}
        self.iter_count = None
        self.marginals = []

    def initialize_conditionals_only(self, grad_shapes):
        for env in self.envs:
            for i, p_shape in enumerate(grad_shapes):
                self.conditionals_pos[env][i].fill_(0.5)
                
    def initialize_marginals_only(self, grad_shapes):
        for i in range(len(self.marginals)):
            self.marginals[i].fill_(0.5)

    def initialize_uniform(self, grad_shapes):
        for env in self.envs:
            if env in self.conditionals_pos:
                for i, p_shape in enumerate(grad_shapes):
                    self.conditionals_pos[env][i].fill_(0.5)
            else:
                self.conditionals_pos[env] = []
                for p_shape in grad_shapes:
                    self.conditionals_pos[env].append(0.5*torch.ones(p_shape).to(self.device))
        if len(self.marginals)>0:
            for i in range(len(self.marginals)):
                self.marginals[i].fill_(0.5)
        else:
            for p_shape in grad_shapes:
                self.marginals.append(torch.zeros(p_shape).to(self.device))
    
    def compute_conditionals(self, dist_pos_all, dist_neg_all):
        for env in self.envs:
            for i,p in enumerate(self.marginals):
                pos_conditional = p*torch.exp(-1.0*self.beta*dist_pos_all[env][i]) + 1e-12
                neg_conditional = (1 - p)*torch.exp(-1.0*self.beta*dist_neg_all[env][i]) + 1e-12
                self.conditionals_pos[env][i] = pos_conditional / (pos_conditional+neg_conditional)

    def compute_marginals(self):
        for i,p in enumerate(self.conditionals_pos[self.envs[0]]):
            self.marginals[i] = p*(1.0/self.num_envs)
        for env in self.envs[1:]:
            for i, p in enumerate(self.conditionals_pos[env]):
                self.marginals[i] += p * (1.0/self.num_envs)
    
    def compute_cost(self):
        cost = 0 
        for i in range(len(self.conditionals_pos[self.envs[0]])):
            cost += torch.mean(torch.abs(self.conditionals_pos[self.envs[0]][i] - self.conditionals_pos[self.envs[1]][i]))  
        self.costs.append(cost.item())

    def run_algorithm(self, dist_pos_all, dist_neg_all, compute_cost=False, num_iterations=300):
        self.costs = []
        for _ in range(num_iterations):
            if compute_cost:
                self.compute_cost()
            self.compute_conditionals(dist_pos_all, dist_neg_all)
            self.compute_marginals()
        return self.marginals, self.costs

    def optimize_rd_from_scratch(self, dist_pos_all, dist_neg_all, sizes):
        self.envs = list(dist_pos_all.keys())
        self.num_envs = len(self.envs)
        
        self.initialize_uniform(sizes)
        self.compute_marginals()
        self.run_algorithm(dist_pos_all, dist_neg_all, compute_cost=False, num_iterations=100)
    
    def optimize_rd(self, dist_pos_all, dist_neg_all, sizes):
        self.envs = list(dist_pos_all.keys())
        self.num_envs = len(self.envs)

        if self.iter_count is None:
            _num_iterations = 200
            self.iter_count = 0
            self.initialize_uniform(sizes)
            self.compute_marginals()
        elif self.iter_count % 500 == 499:
            _num_iterations = 50
            self.iter_count += 1
            self.initialize_uniform(sizes)
            self.compute_marginals()
        else:
            _num_iterations = 50
            self.iter_count += 1
        return self.run_algorithm(dist_pos_all, dist_neg_all, compute_cost=False, num_iterations=_num_iterations)
