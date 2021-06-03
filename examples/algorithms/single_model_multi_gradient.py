import torch
import torch.nn as nn
import numpy as np
import time

from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from algorithms.rate_dist import BlahutArimoto
from torch.nn.utils import clip_grad_norm_

class MultiGradient(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out).to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self._is_satisficing = False
        self.and_mask = False
        """
        if config.rd_type == 0:
            self.beta_min = 0.1
            self.beta_max = 0.1
        elif config.rd_type == 1:
            self.beta_min = 0.01
            self.beta_max = 0.01
        elif config.rd_type == 2:
            self.beta_min = 1
            self.beta_max = 1
            #self._is_satisficing = True
        elif config.rd_type == 3:
            self.beta_min = 0.0001
            self.beta_max = 0.1
        """

        if config.rd_type == 0:
            self.beta_min = 0.0001
            self.beta_max = 0.0001
        elif config.rd_type == 1:
            self.beta_min = 0.001
            self.beta_max = 0.001
        elif config.rd_type == 2:
            self.beta_min = 0.01
            self.beta_max = 0.01
            #self._is_satisficing = True
        elif config.rd_type == 3:
            self.beta_min = 0.1
            self.beta_max = 0.1
        elif config.rd_type == 4:
            self.beta_min = 1
            self.beta_max = 1
        elif config.rd_type == 5:
            self.beta_min = 10
            self.beta_max = 10
        elif config.rd_type == 42:
            self.beta_min = 1
            self.beta_max = 1
            self.and_mask = True
        
        self.ba = BlahutArimoto(self.beta_min)
        self.warm_started = False
        self.control_only_direction = config.control_only_direction
        self.only_inconsistent = config.only_inconsistent
        self.without_sampling = config.without_sampling

        print("Settings: {} {} {} {}".format(self.control_only_direction, 
                                             self.only_inconsistent, 
                                             self.without_sampling,
                                             self.and_mask))
        
        # Compute size
        self.total_size = 0
        for p in self.model.parameters():
            self.total_size += np.prod(list(p.shape))
        print("Total Size", self.total_size)

        self.dist_pos = {}
        self.dist_neg = {}
        for env_name in range(config.n_groups_per_batch):
            self.dist_pos[env_name] = [torch.zeros(self.total_size).to(torch.device("cuda"))]
            self.dist_neg[env_name] = [torch.zeros(self.total_size).to(torch.device("cuda"))]
 
        self.pos_grad = torch.zeros(self.total_size).to(torch.device("cuda"))
        self.neg_grad = torch.zeros(self.total_size).to(torch.device("cuda"))
        self.total_grad = torch.zeros(self.total_size).to(torch.device("cuda"))
        self.all_positive = torch.zeros(self.total_size).to(torch.device("cuda"))
        self.all_negative = torch.zeros(self.total_size).to(torch.device("cuda"))
 
    def update_beta(self, epoch, num_epochs):
        lin_rate = float(epoch) / float(num_epochs)
        new_d = self.beta_min + ((self.beta_max - self.beta_min)*lin_rate)
        self.ba.beta = new_d
        print('Updated to {} with Sat {}'.format(new_d, self._is_satisficing))


    def objective(self, results):
        return self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)

    def partial_objective(self, pred, gtrue):
        return self.loss.compute(pred, gtrue, return_dict=False)


    def update_batch_multi_env(self, batch):
        """
        A helper function for update() and evaluate() that processes the batch
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor)
                - g (Tensor)
                - metadata (Tensor)
                - output (Tensor)
                - y_true
        """
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)

        # Perform per group gradient computation and RD
        environments = torch.unique(g).tolist()
        sizes = []
        pos_acc = [] 
        neg_acc = []
        total_grad = []

        self.pos_grad.fill_(0)
        self.neg_grad.fill_(0)
        self.total_grad.fill_(0)
        self.all_positive.fill_(1)
        self.all_negative.fill_(1)
        full_outputs = None
        for env_id, env_name in enumerate(environments):
            # get the ratio of size of env to batch
            env_ratio = torch.sum(g==env_name).item() / g.shape[0]
            self.model.zero_grad()

            if torch.is_tensor(x):
                outputs = self.model(x[g==env_name])
                partial_objective = self.partial_objective(outputs, y_true[g==env_name])
                if full_outputs is None:
                    partial_shape = list(outputs.shape)
                    partial_shape[0] = y_true.shape[0]
                    full_outputs = torch.zeros(*partial_shape, dtype=outputs.dtype).to(torch.device("cuda"))
                full_outputs[g==env_name] = outputs
            else:
                outputs = self.model(x)
                partial_objective = self.partial_objective(outputs[g==env_name], y_true[g==env_name])
                full_outputs = outputs
            if not partial_objective.requires_grad:
                print("Loss function returned 0 (possibly input is NaN")
            else:
                partial_objective.backward()
            # Save the gradients here
            cur_pos = 0
            for i, param in enumerate(self.model.parameters()):
                p = param.grad.clone().detach()
                next_pos = cur_pos + np.prod(list(p.shape))

                p_neg =  (p<0)*p
                p_pos =  (p>0)*p

                self.all_positive[cur_pos:next_pos] = torch.logical_and(p.view(-1)>0, self.all_positive[cur_pos:next_pos])
                self.all_negative[cur_pos:next_pos] = torch.logical_and(p.view(-1)<0, self.all_negative[cur_pos:next_pos])
                
                self.dist_pos[env_id][0][cur_pos:next_pos] = -1.0 * p_neg.view(-1)
                self.dist_neg[env_id][0][cur_pos:next_pos] = p_pos.view(-1)

                self.pos_grad[cur_pos:next_pos] += p_pos.view(-1) * env_ratio
                self.neg_grad[cur_pos:next_pos] += p_neg.view(-1) * env_ratio
                self.total_grad[cur_pos:next_pos] += p.view(-1) * env_ratio
                cur_pos = next_pos
            
        if self._is_satisficing:
            for env_id, env_name in enumerate(environments):
                self.dist_pos[env_id][0] = self.dist_pos[env_id][0] * torch.abs(self.total_grad)
                self.dist_neg[env_id][0] = self.dist_neg[env_id][0] * torch.abs(self.total_grad)

        if self.and_mask:
            consistent_ones = torch.logical_or(self.all_positive, self.all_negative)
            final_grads = self.total_grad * consistent_ones
        else: 
            marginals, _ = self.ba.optimize_rd(self.dist_pos, self.dist_neg, [self.dist_pos[0][0].shape])

            marginals[0][marginals[0]<0.0] = 1e-8
            marginals[0][marginals[0]>1.0] = 1.0 - 1e-8

            if self.without_sampling:
                pos_ones = marginals[0]
            else:
                # Sample the directions
                pos_ones = torch.bernoulli(marginals[0])

            if self.control_only_direction:
                magnitude = torch.abs(self.total_grad)
                direction = 2*pos_ones - 1
                final_rd_grads = magnitude * direction
            else:
                final_rd_grads = self.pos_grad*pos_ones + self.neg_grad*(1-pos_ones)


            if self.only_inconsistent:
                consistent_ones = torch.logical_or(self.all_positive, self.all_negative)
                final_grads = final_rd_grads * torch.logical_not(consistent_ones) + self.total_grad * consistent_ones
            else:
                final_grads = final_rd_grads

        self.model.zero_grad()
 
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': full_outputs,
            'metadata': metadata,
            }

        objective = self.objective(results)
        results['objective'] = objective.item()
        cur_pos = 0
        for i,p in enumerate(self.model.parameters()):
            next_pos = cur_pos + np.prod(list(p.shape))
            p.grad.data.copy_(final_grads[cur_pos:next_pos].data.view(p.shape))
            cur_pos = next_pos

        if self.max_grad_norm:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.step_schedulers(
            is_epoch=False,
            metrics=results,
            log_access=False)
        return results 
    

    def update_batch_erm(self, batch):
        """
        A helper function for update() and evaluate() that processes the batch
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor)
                - g (Tensor)
                - metadata (Tensor)
                - output (Tensor)
                - y_true
        """
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        outputs = self.model(x)
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
            }

        objective = self.objective(results)
        results['objective'] = objective.item()
        # update
        self.model.zero_grad()
        objective.backward()
        if self.max_grad_norm:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.step_schedulers(
            is_epoch=False,
            metrics=results,
            log_access=False)

        return results 
 
    def update(self, batch):
        """
        Process the batch, update the log, and update the model
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - outputs (Tensor)
                - y_pred (Tensor)
                - objective (float)
        """
        assert self.is_training
        # process batch
        if self.warm_started:
            results = self.update_batch_multi_env(batch)
        else:
            results = self.update_batch_erm(batch)
        # log results
        self.update_log(results)
        return self.sanitize_dict(results)

