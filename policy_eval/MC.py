import numpy as np
import matplotlib.pyplot as plt
from collections import deque

EPISODE = [0, 10, 100, 500, 5000]
ALPHA = 0.01
GAMMA = 1.0
STATIC = True
MARKER = ['x', 'x', 'x', 'x', '*']
COLOR = ['b', 'r', 'g', 'k', 'c']

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


def reset():
    '''
    Reset to initial condition
    '''
    for state in graph[1:-1]:
        state.value = 0.5
        state.counter = 0


if __name__ == "__main__":

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
                queue_idx.appendleft(index)

                # the probability of transition is 50%
                action = 1 if np.random.rand() < 0.5 else -1
            
                next = index + action
                done, reward = graph[next].done, graph[next].reward
                index = next
                queue_reward.appendleft(reward)

            epi_queue_idx.append(queue_idx)
            epi_queue_reward.append(queue_reward)

        # calculate values statistically
        if episode != 0:
            for queue_idx, queue_reward in zip(epi_queue_idx, epi_queue_reward):

                Gt = 0
                for idx, reward in zip(queue_idx, queue_reward):
                    graph[idx].counter += 1
                    Gt = reward + GAMMA * Gt
                    graph[idx].value += Gt

            for state in graph[1:-1]:
                state.value = state.value/state.counter
                
        # plot the result w.r.t episode
        plt_x = [node.state for node in graph[1:-1]]
        plt_y = [node.value for node in graph[1:-1]]
        plt.plot(plt_x, plt_y, color=COLOR.pop(), marker=MARKER.pop(), linewidth=0.75, markersize=3, label=f'episode:{episode}')

        reset()

    plt.legend(loc=2, title=f'GAMMA:{GAMMA}')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()



