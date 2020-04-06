import gym
import numpy as np
from typing import List

import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

# observation limitation [position, velocity]
print(f'Env upper bound in [position, velocity]: {env.observation_space.high}')
print(f'Env lower bound in [position, velocity]: {env.observation_space.low}')

# number of actions
print(f'Number of actions: {env.action_space.n}')

LEARNING_RATE = 0.5
DISCOUNT = 0.95
EPISODE = 12500
SHOW_EVERY = 100

epsilon = 0.75
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODE // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# episode statistic
ep_rewards = []
agg_ep_rewards = {'ep':[], 'avg':[], 'max':[], 'min':[]}


class Qtable:
    def __init__(self, row_size=20, col_size=20):
        
        discrete_os_size = [row_size] + [col_size]
        action_size = [env.action_space.n]
        table_size = discrete_os_size + action_size
        
        print(f'Create Q table with size: {table_size}')
        self.q_table = self.create_q_table(table_size)
        self.discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

    def create_q_table(self, table_size: List):

        return np.random.uniform(low=-2, high=0, size=table_size)

    def __call__(self, state: List):

        assignment = (state - env.observation_space.low) / self.discrete_os_win_size
        index = tuple(assignment.astype(np.int))

        return self.q_table[index]


def main():

    global epsilon
    q_table = Qtable(row_size=20, col_size=20)

    for episode in range(EPISODE):

        rewards_per_ep = 0
        cur_state = env.reset()
        done = False

        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(q_table(cur_state))
            else:
                action = np.random.randint(0, env.action_space.n)
            
            future_state, reward, done, _ = env.step(action)

            rewards_per_ep += reward

            # if episode % SHOW_EVERY == 0 and episode != 0:
            #     env.render()

            if not done:
                # compute new q value
                max_future_q = np.max(q_table(future_state))
                current_q = q_table(cur_state)[action]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                # update ocurrent state
                q_table(cur_state)[action] = new_q

            elif abs(future_state[0] - env.goal_position) < 0.01:
                # achieve the goal
                q_table(cur_state)[action] = 0

            cur_state = future_state

        if END_EPSILON_DECAYING >= episode >=START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        ep_rewards.append(rewards_per_ep)

        if not episode % SHOW_EVERY:
            avg_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
            min_v = min(ep_rewards[-SHOW_EVERY:])
            max_v = max(ep_rewards[-SHOW_EVERY:])
            agg_ep_rewards['ep'].append(episode)
            agg_ep_rewards['avg'].append(avg_reward)
            agg_ep_rewards['min'].append(min_v)
            agg_ep_rewards['max'].append(max_v)

            print(f"Episode: {episode} avg: {avg_reward} max: {max_v} min: {min_v} epsilon: {epsilon}")

    env.close()

    plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['avg'], label='avg')
    plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['max'], label='max')
    plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['min'], label='min')
    plt.legend(loc=4)
    plt.show()


if __name__ == "__main__":
    main()
