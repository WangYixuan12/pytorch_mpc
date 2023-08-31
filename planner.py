import numpy as np

import torch
import torch.nn.functional as F

# Tips to tune MPC:
# - When sampling actions, noise_level should be large enough to have enough coverage, but not too large to cause instability
# - Larger n_sample should lead to better performance, but it will also increase the computation cost
# - Properly tune reward_weight, higher reward_weight encourages to 'exploit' the current best action sequence, while lower reward_weight encourages to 'explore' more action sequences
# - Plot reward vs. iteration to see the convergence of the planner

class Planner(object):

    def __init__(self, config):
        # config contains following keys:
        
        # REQUIRED
        # - action_dim: the dimension of the action
        # - state_dim: the dimension of the state
        # - model_rollout_fn:
        #   - description: the function to rollout the model
        #   - input:
        #     - state_cur (shape: [n_his, state_dim] torch tensor)
        #     - action_seqs (shape: [n_sample, n_look_ahead, action_dim] torch tensor)
        #   - output: a dict containing the following keys:
        #     - state_seqs: the sequence of the state, shape: [n_sample, n_look_ahead, state_dim] torch tensor
        #     - any other keys that you want to return
        # - evaluate_traj_fn:
        #   - description: the function to evaluate the trajectory
        #   - input:
        #     - state_seqs (shape: [n_sample, n_look_ahead, state_dim] torch tensor)
        #     - action_seqs (shape: [n_sample, n_look_ahead, action_dim] torch tensor)
        #   - output: a dict containing the following keys:
        #     - reward_seqs (shape: [n_sample] torch tensor)
        #     - any other keys that you want to return
        # - n_sample: the number of action trajectories to sample
        # - n_look_ahead: the number of steps to look ahead
        # - n_update_iter: the number of iterations to update the action sequence
        # - reward_weight: the weight of the reward to aggregate action sequences
        # - action_lower_lim:
        #   - description: the lower limit of the action
        #   - shape: [action_dim]
        #   - type: torch tensor
        # - action_upper_lim: the upper limit of the action
        #   - description: the upper limit of the action
        #   - shape: [action_dim]
        #   - type: torch tensor
        # - planner_type: the type of the planner (options: 'GD', 'MPPI', 'MPPI_GD')
        self.config = config
        self.action_dim = config['action_dim']
        # self.state_dim = config['state_dim']
        self.model_rollout = config['model_rollout_fn']
        self.evaluate_traj = config['evaluate_traj_fn']
        self.n_sample = config['n_sample']
        self.n_look_ahead = config['n_look_ahead']
        self.n_update_iter = config['n_update_iter']
        self.reward_weight = config['reward_weight']
        self.action_lower_lim = config['action_lower_lim']
        self.action_upper_lim = config['action_upper_lim']
        self.planner_type = config['planner_type']
        assert self.planner_type in ['GD', 'MPPI', 'MPPI_GD']
        assert self.action_lower_lim.shape == (self.action_dim,)
        assert self.action_upper_lim.shape == (self.action_dim,)
        assert type(self.action_lower_lim) == torch.Tensor
        assert type(self.action_upper_lim) == torch.Tensor
        
        # OPTIONAL
        # - device: 'cpu' or 'cuda', default: 'cuda'
        # - verbose: True or False, default: False
        # - sampling_action_seq_fn:
        #   - description: the function to sample the action sequence
        #   - input: init_act_seq (shape: [n_look_ahead, action_dim] torch tensor)
        #   - output: act_seqs (shape: [n_sample, n_look_ahead, action_dim] torch tensor)
        #   - default: sample action sequences from a normal distribution
        # - noise_type: the type of the noise (options: 'normal'), default: 'normal'
        # - noise_level: the level of the noise, default: 0.1
        # - n_his: the number of history states to use, default: 1
        # - rollout_best: whether rollout the best act_seq and get model prediction and reward. True or False, default: True
        self.device = config['device'] if 'device' in config else 'cuda'
        self.verbose = config['verbose'] if 'verbose' in config else False
        self.sample_action_sequences = config['sampling_action_seq_fn'] if 'sampling_action_seq_fn' in config else self.sample_action_sequences_default
        self.noise_type = config['noise_type'] if 'noise_type' in config else 'normal'
        assert self.noise_type in ['normal'] # TODO: support more noise_type
        self.noise_level = config['noise_level'] if 'noise_level' in config else 0.1
        self.n_his = config['n_his'] if 'n_his' in config else 1
        self.rollout_best = config['rollout_best'] if 'rollout_best' in config else True

    def sample_action_sequences_default(self, act_seq):
        # init_act_seq: shape: [n_look_ahead, action_dim] torch tensor
        # return: shape: [n_sample, n_look_ahead, action_dim] torch tensor
        assert act_seq.shape == (self.n_look_ahead, self.action_dim)
        assert type(act_seq) == torch.Tensor

        beta_filter = 0.7

        # [n_sample, n_look_ahead, action_dim]
        act_seqs = torch.stack([act_seq.clone()] * self.n_sample)

        # [n_sample, action_dim]
        act_residual = torch.zeros((self.n_sample, self.action_dim), dtype=act_seqs.dtype, device=self.device)

        # actions that go as input to the dynamics network
        for i in range(self.n_look_ahead):
            if self.noise_type == "normal":
                noise_sample = torch.normal(0, self.noise_level, (self.n_sample, self.action_dim), device=self.device)
            else:
                raise ValueError("unknown noise type: %s" %(self.noise_type))

            act_residual = beta_filter * noise_sample + act_residual * (1. - beta_filter)

            # add the perturbation to the action sequence
            act_seqs[:, i] += act_residual

            # clip to range
            act_seqs[:, i] = torch.clamp(act_seqs[:, i],
                                         self.action_lower_lim,
                                         self.action_upper_lim)

        assert act_seqs.shape == (self.n_sample, self.n_look_ahead, self.action_dim)
        assert type(act_seqs) == torch.Tensor
        return act_seqs

    def optimize_action(self, act_seqs, reward_seqs):
        # act_seqs: shape: [n_sample, n_look_ahead, action_dim] torch tensor
        # reward_seqs: shape: [n_sample] torch tensor
        assert act_seqs.shape == (self.n_sample, self.n_look_ahead, self.action_dim)
        assert reward_seqs.shape == (self.n_sample,)
        assert type(act_seqs) == torch.Tensor
        assert type(reward_seqs) == torch.Tensor

        if self.planner_type == 'MPPI':
            return self.optimize_action_mppi(act_seqs, reward_seqs)
        elif self.planner_type == 'GD' or self.planner_type == 'MPPI_GD':
            raise NotImplementedError
        else:
            raise ValueError("unknown planner type: %s" %(self.planner_type))

    def trajectory_optimization(self, state_cur, act_seq):
        # input:
        # - state_cur: current state, shape: [n_his, state_dim] torch tensor
        # - act_seq: initial action sequence, shape: [n_look_ahead, action_dim] torch tensor
        # output:
        # - a dictionary with the following keys:
        #   - 'act_seq': optimized action sequence, shape: [n_look_ahead, action_dim] torch tensor
        #   - 'model_outputs' if verbose is True, otherwise None, might be useful for debugging
        #   - 'eval_outputs' if verbose is True, otherwise None, might be useful for debugging
        #   - 'best_model_output' if rollout_best is True, otherwise None, might be useful for debugging
        #   - 'best_eval_output' if rollout_best is True, otherwise None, might be useful for debugging
        # assert state_cur.shape == (self.n_his, self.state_dim)
        assert type(state_cur) == torch.Tensor
        assert act_seq.shape == (self.n_look_ahead, self.action_dim)
        assert type(act_seq) == torch.Tensor
        if self.planner_type == 'MPPI':
            return self.trajectory_optimization_mppi(state_cur, act_seq)
        elif self.planner_type == 'GD' or self.planner_type == 'MPPI_GD':
            raise NotImplementedError
        else:
            raise ValueError("unknown planner type: %s" %(self.planner_type))
    
    def optimize_action_mppi(self, act_seqs, reward_seqs):
        return torch.sum(act_seqs * F.softmax(reward_seqs * self.reward_weight, dim=0).unsqueeze(-1).unsqueeze(-1), dim=0)
    
    def optimize_action_gd(self, act_seqs, reward_seqs):
        pass
    
    def optimize_action_mppi_gd(self, act_seqs, reward_seqs):
        pass
    
    def trajectory_optimization_mppi(self, state_cur, act_seq):
        if self.verbose:
            model_outputs = []
            eval_outputs = []
        for i in range(self.n_update_iter):
            with torch.no_grad():
                act_seqs = self.sample_action_sequences(act_seq)
                assert act_seqs.shape == (self.n_sample, self.n_look_ahead, self.action_dim)
                assert type(act_seqs) == torch.Tensor
                model_out = self.model_rollout(state_cur, act_seqs)
                state_seqs = model_out['state_seqs']
                # assert state_seqs.shape == (self.n_sample, self.n_look_ahead, self.state_dim)
                assert type(state_seqs) == torch.Tensor
                eval_out = self.evaluate_traj(state_seqs, act_seqs)
                reward_seqs = eval_out['reward_seqs']
                act_seq = self.optimize_action(act_seqs, reward_seqs)
                if self.verbose:
                    model_outputs.append(model_out)
                    eval_outputs.append(eval_out)

        if self.rollout_best:
            best_model_out = self.model_rollout(state_cur, act_seq.unsqueeze(0))
            best_eval_out = self.evaluate_traj(best_model_out['state_seqs'], act_seq.unsqueeze(0))
                
        return {'act_seq': act_seq,
                'model_outputs': model_outputs if self.verbose else None,
                'eval_outputs': eval_outputs if self.verbose else None,
                'best_model_output': best_model_out if self.rollout_best else None,
                'best_eval_output': best_eval_out if self.rollout_best else None}
    
    def trajectory_optimization_gd(self, state_cur, act_seq = None):
        pass
    
    def trajectory_optimization_mppi_gd(self, state_cur, act_seq = None):
        pass
