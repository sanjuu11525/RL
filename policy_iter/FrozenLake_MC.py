import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

env = gym.make('FrozenLake-v0')
# the number of states
N_states = 16

# number of actions
print(f'Number of actions: {env.action_space.n}')

# actions
# 0 	Move Left
# 1 	Move Down
# 2 	Move Right
# 3 	Move Up

# parameters
DEMO_EVERY = 5000
EPISODE = 50000
SHOW_EVERY = 500
ALPHA = 0.001
GAMMA = 1
EPSILON = 0.9
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODE // 4

# epsilon decay
epsilon_decay_value = 0.9/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# q table and counters w.r.t [Left, Down, Right, Up]
table = np.zeros((N_states, env.action_space.n))
counter = np.zeros((N_states, env.action_space.n))

ep_rewards = []
def main():
    epsilon = EPSILON

    for episode in range(EPISODE):
        cur_state = env.reset()
        done = False

        queue_pair = deque([])
        queue_reward = deque([])
        rewards_per_ep = 0

        # policy evaluation
        while not done:
            # exploration versus exploitation
            if np.random.random() > epsilon:
                action = np.argmax(table[cur_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            next, reward, done, _ = env.step(action)

            # track action pairs and reward
            queue_pair.appendleft([cur_state, action])
            queue_reward.appendleft(reward)

            rewards_per_ep += reward
            cur_state = next
  
            if not episode % DEMO_EVERY:
                env.render()
                time.sleep(0.2)

        # policy improvement
        if END_EPSILON_DECAYING >= episode >=START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        ep_rewards.append(rewards_per_ep)

        for (state, action) in queue_pair:
            counter[state][action] += 1

        Gt = 0
        for reward, (state, action) in zip(queue_reward, queue_pair):
            Gt = reward + GAMMA * Gt
            table[state][action] += (1/counter[state][action]) * (Gt - table[state][action])        
            # table[state][action] += ALPHA * (Gt - table[state][action])

        if not episode % SHOW_EVERY:
            print(f'Accumulated reward: {sum(ep_rewards[-SHOW_EVERY:])}')


if __name__=='__main__':
    main()