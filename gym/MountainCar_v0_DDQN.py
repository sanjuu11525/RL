import gym
import numpy as np
import random
import os, datetime, math
from collections import deque
import time

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
env = gym.make('MountainCar-v0')

# observation limitation [position, velocity]
print(f'Env upper bound in [position, velocity]: {env.observation_space.high}')
print(f'Env lower bound in [position, velocity]: {env.observation_space.low}')

# number of actions
print(f'Number of actions: {env.action_space.n}')

config = {'MEM_CAPACITY'    :10000,
          'DISCOUNT'        :0.95,
          'EPISODE'         :20000,
          'SHOW_EVERY'      :100,
          'EPS_START'       :1.0,
          'EPS_START_DECAY' :1,
          'EPS_END_DECAY'   :8000,
          'TARGET_UPDATE'   :10,
          'BATCH_SIZE'      :128,
          'NUM_HIDDEN_SH'   :64,
          'NUM_HIDDEN_MID'  :128,
          }

# episode statistic
ep_rewards = []
agg_ep_rewards = {'ep':[], 'avg':[], 'max':[], 'min':[]}

# tb initialization
DATETIME_FORMAT = '%Y-%m-%dT%H-%M-%S'
log_dir = os.path.join(os.getcwd(), "dqn_out", datetime.datetime.now().strftime(DATETIME_FORMAT))
tb_writer = SummaryWriter(log_dir)


class ReplayMemory():
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()
        layer_a = config['NUM_HIDDEN_SH']
        layer_b = config['NUM_HIDDEN_MID']

        self.model = nn.Sequential( nn.Linear(2      , layer_a), nn.LeakyReLU(negative_slope=1, inplace=True),
                                    nn.Linear(layer_a, layer_a), nn.LeakyReLU(negative_slope=1, inplace=True),
                                    nn.Linear(layer_a, layer_a), nn.LeakyReLU(negative_slope=1, inplace=True),
                                    nn.Linear(layer_a, 3      ))
    def forward(self, x):
        return self.model(x)


def select_action(net, state, epsilon):
    sample = np.random.random()

    if sample > epsilon:
        with torch.no_grad():
            net.eval()
            out = net(torch.FloatTensor(state).view(1, -1).to(device))
            action = out.max(1)[1].cpu().item()
    else:
        action = np.random.randint(0, env.action_space.n)

    return action


def optim_model(memory, policy_net, target_net, optimizer):
    if len(memory) < config['BATCH_SIZE']:
        return
    
    # sample the batch
    minibatch = memory.sample(config['BATCH_SIZE'])
    list_batch = list(zip(*minibatch))

    # prepare data for inference
    state_batch      = torch.FloatTensor(list_batch[0]).to(device)
    next_state_batch = torch.FloatTensor(list_batch[1]).to(device)
    reward_batch     = torch.FloatTensor(list_batch[2]).view(-1, 1).to(device)
    action_batch     = torch.LongTensor (list_batch[3]).view(-1, 1).to(device)
    done_batch       = torch.BoolTensor (list_batch[4]).view(-1, 1).to(device)
    mask_batch       = ~ done_batch

    # compute Q values
    curr_Q = policy_net(state_batch).gather(1, action_batch)
    next_Q = target_net(next_state_batch)
    max_next_Q = next_Q.max(1, keepdim=True)[0]
    expected_Q = reward_batch + config['DISCOUNT'] * max_next_Q * mask_batch

    # optimize weights
    loss = F.mse_loss(curr_Q, expected_Q.detach())
    optimizer.zero_grad()    
    loss.backward()
    optimizer.step()
 

def main():
    # epsilon 
    epsilon = config['EPS_START']

    # Double Q-learning in Hasselt et al., 2015
    policy_net = DQNNet().to(device) # only training
    target_net = DQNNet().to(device) # only inference
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(capacity=config['MEM_CAPACITY'])

    tb_writer.add_text('config', '   \n'.join([str(key) + ": " + str(val) for key, val in config.items()]))
    tb_writer.add_text('config', str(policy_net))

    for episode in range(config['EPISODE']):

        rewards_per_ep = 0
        state = env.reset()
        done = False

        while not done:
            action = select_action(policy_net, state, epsilon)
            next_state, reward, done, info = env.step(action)

            memory.push((state, next_state, reward, action, done and not info.get('TimeLimit.truncated', False)))

            policy_net.train()
            optim_model(memory, policy_net, target_net, optimizer)

            rewards_per_ep += reward
            state = next_state

        # update the traget model w.r.t DDQN
        if episode % config['TARGET_UPDATE'] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # update epsilon with the linear decay
        if config['EPS_END_DECAY'] >= episode >= config['EPS_START_DECAY']:
            decay_value = config['EPS_START']/(config['EPS_END_DECAY'] - config['EPS_START_DECAY'])
            epsilon -= decay_value

        ep_rewards.append(rewards_per_ep)

        SHOW_EVERY = config['SHOW_EVERY']
        if not episode % config['SHOW_EVERY']:
            avg_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
            min_v = min(ep_rewards[-SHOW_EVERY:])
            max_v = max(ep_rewards[-SHOW_EVERY:])
            agg_ep_rewards['ep'].append(episode)
            agg_ep_rewards['avg'].append(avg_reward)
            agg_ep_rewards['min'].append(min_v)
            agg_ep_rewards['max'].append(max_v)

            if tb_writer is not None:
                figure = plt.figure()
                plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['avg'], label='avg')
                plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['max'], label='max')
                plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['min'], label='min')
                plt.legend(loc='upper left')
                plt.ylim(-210, -70)
                tb_writer.add_figure("Rewards", figure, episode)
                tb_writer.add_scalar('Epsilon', epsilon, episode)
                tb_writer.flush()

            print(f"Episode: {episode} avg: {avg_reward} max: {max_v} min: {min_v} epsilon: {epsilon}")

    env.close()


if __name__ == "__main__":
    main()