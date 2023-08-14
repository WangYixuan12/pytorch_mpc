# import gym
import gymnasium as gym
import numpy as np
import torch
# import logging
import math
# from gym import wrappers, logger as gym_log
from planner import Planner
from matplotlib import pyplot as plt

# gym_log.set_level(gym_log.INFO)
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG,
#                     format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
#                     datefmt='%m-%d %H:%M:%S')

def env_state_to_input(state, device):
    # :return (1, state_dim)
    return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

def vis_res(res):
    eval_outputs = res['eval_outputs']
    rew = [out['reward_seqs'].mean().item() for out in eval_outputs]
    plt.plot(rew)
    plt.show()

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v1"
    TIMESTEPS = 15  # T
    N_SAMPLES = 1000  # K
    device = 'cuda'
    ACTION_LOW = torch.tensor([-2.0], device=device)
    ACTION_HIGH = torch.tensor([2.0], device=device)

    d = "cpu"
    dtype = torch.double

    def dynamics(state, perturbed_action):
        # true dynamics from gym
        # :param state: (n_his, state_dim)
        # :param perturbed_action: (n_samples, n_look_ahead, action_dim)
        # :return a dict containing
        # - states_seqs: (n_samples, n_look_ahead, state_dim)
        # - any other optional info you want to pass
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = perturbed_action
        u = torch.clamp(u, -2, 2)

        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)

        state = torch.cat((newth, newthdot), dim=-1)
        return {
            'state_seqs': state,
        }


    def angle_normalize(x):
        return (((x + math.pi) % (2 * math.pi)) - math.pi)


    def running_cost(state, action):
        # :param state: (n_sample, n_look_ahead, state_dim)
        # :param action: (n_samples, n_look_ahead, action_dim)
        # :return a dict containing
        # - reward_seqs: (n_samples) - higher is better
        # - any other optional info you want to pass
        theta = state[:, :, 0]
        theta_dt = state[:, :, 1]
        action = action[:, :, 0]
        cost = angle_normalize(theta) ** 2 + 0.01 * theta_dt ** 2
        cost = cost.sum(dim=-1)
        return {
            'reward_seqs': -cost,
        }


    def train(new_data):
        pass


    downward_start = True
    env = gym.make(ENV_NAME, render_mode='human', g=10.0)
    env.reset()
    if downward_start:
        # env.state = [np.pi, 1]
        # env.set_state([np.pi, 1])
        env.__setattr__('state', np.array([np.pi, 1]))

    config = {
        'action_dim': 1,
        'state_dim': 2,
        'model_rollout_fn': dynamics,
        'evaluate_traj_fn': running_cost,
        'n_sample': N_SAMPLES,
        'n_look_ahead': TIMESTEPS,
        'n_update_iter': 100,
        'reward_weight': 100.0,
        'action_lower_lim': ACTION_LOW,
        'action_upper_lim': ACTION_HIGH,
        'planner_type': 'MPPI',
        'device': device,
        'verbose': True,
        'noise_level': 2.0,
        'rollout_best': True,
    }
    planner = Planner(config)
    
    act_seq = (torch.rand((TIMESTEPS, ACTION_HIGH.shape[0]), device=device) * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW)
    for i in range(100):
        res = planner.trajectory_optimization(env_state_to_input(env.env.state, device=device), act_seq)
        env.step(res['act_seq'][0].detach().cpu().numpy())
        act_seq = torch.cat((res['act_seq'][1:], torch.rand((1, ACTION_HIGH.shape[0]), device=device) * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW), dim=0)
