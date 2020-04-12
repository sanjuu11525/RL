import numpy as np
import matplotlib.pyplot as plt
from collections import deque

EPISODE = [0, 10, 100, 1000]
ALPHA = 0.1
GAMMA = 1.0
STATIC = True
COLOR = ['r', 'g', 'm', 'k']

class State():
    def __init__(self, name, value, done, reward, counter):
        self.state = name
        self.value = value
        self.done = done
        self.reward = reward
        self.counter = counter

# the main dataset we operate on
graph = np.array([State('Left' , 0.0, True , 0, 0), State('A', 0.5, False, 0, 0),
                  State('B'    , 0.5, False, 0, 0), State('C', 0.5, False, 0, 0),
                  State('D'    , 0.5, False, 0, 0), State('E', 0.5, False, 0, 0),
                  State('Right', 0.0, True , 1, 0)])


def MDP():
    '''
    Solve Bellman expectation eq.
    '''
    P = np.array([[0  , 1.0, 0  , 0  , 0  ],
                  [1.0, 0  , 1.0, 0  , 0  ],
                  [0  , 1.0, 0  , 1.0, 0  ],
                  [0  , 0  , 1.0, 0  , 1.0],
                  [0  , 0  , 0  , 1.0, 0  ]])

    R = np.array([0, 0, 0, 0, 1])
    I = np.diag(np.ones((5)))
    v = np.dot(np.linalg.inv(I - 0.5 * GAMMA * P), 0.5 * R)
    return v


def reset():
    '''
    Reset to initial condition
    '''
    for state in graph[1:-1]:
        state.value = 0.5
        state.counter = 0


if __name__ == "__main__":
    '''
    Monte Carlo offline policy evalution.
    '''

    for episode in EPISODE:

        epi_queue_idx = []
        epi_queue_reward = []

        # evaluate policy with the assigned episode
        for _ in range(episode):

            done = False

            # always start from middle if static
            index = 3 if STATIC else np.random.randint(1, 6)

            queue_idx = deque([])
            queue_reward = deque([])
            # track trajectories by random walk
            while not done:
                # track index
                queue_idx.appendleft(index)

                # the probability of transition is 50%
                action = 1 if np.random.rand() < 0.5 else -1
            
                next = index + action
                done, reward = graph[next].done, graph[next].reward
                index = next
                
                # track reward
                queue_reward.appendleft(reward)

            epi_queue_idx.append(queue_idx)
            epi_queue_reward.append(queue_reward)

        if episode != 0:
            # accumulate return for each step
            for queue_idx, queue_reward in zip(epi_queue_idx, epi_queue_reward):
                
                Gt = 0
                for idx, reward in zip(queue_idx, queue_reward):
                    graph[idx].counter += 1
                    Gt = reward + GAMMA * Gt
                    graph[idx].value += Gt

            # compute empirical mean
            for state in graph[1:-1]:
                state.value = state.value/state.counter
                
        # plot the result w.r.t episode
        plt_x = [node.state for node in graph[1:-1]]
        plt_y = [node.value for node in graph[1:-1]]
        plt.plot(plt_x, plt_y, color=COLOR.pop(), marker='x',linestyle='--', linewidth=0.75, markersize=3, label=f'episode:{episode}')

        reset()

    v = MDP()
    plt.plot(plt_x, v, color='b', marker='o',linestyle='-', linewidth=0.75, markersize=3, label=f'MDP')
    plt.legend(loc=2, title=f'GAMMA:{GAMMA}')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()



