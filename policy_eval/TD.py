import numpy as np
import matplotlib.pyplot as plt


EPISODE = [0, 10, 100, 1000]
ALPHA = 0.01
GAMMA = 1.0
STATIC = True
COLOR = ['r', 'g', 'm', 'k']

class State():
    def __init__(self, name, value, done, reward):
        self.state = name
        self.value = value
        self.done = done
        self.reward = reward

# the main dataset we operate on
graph = np.array([State('Left' , 0.0, True , 0), State('A', 0.5, False, 0),
                  State('B'    , 0.5, False, 0), State('C', 0.5, False, 0),
                  State('D'    , 0.5, False, 0), State('E', 0.5, False, 0),
                  State('Right', 0.0, True , 1)])


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


if __name__ == "__main__":
    '''
    TD(0) online policy evaluation.
    '''
    for episode in EPISODE:

        # evaluate policy with the assigned episode
        for _ in range(episode):

            done = False

            # always start from middle if static
            index = 3 if STATIC else np.random.randint(1, 6)

            # update by random walk
            while not done:
                value = graph[index].value

                # the probability of transition is 50%
                action = 1 if np.random.rand() < 0.5 else -1
            
                next = index + action
            
                # access info of the next state
                future_value, reward, done = graph[next].value, graph[next].reward, graph[next].done

                # update next state by TD(0)
                value = value + ALPHA * (reward + GAMMA * future_value - value)
                graph[index].value = value

                index = next

        # plot the result w.r.t episode
        plt_x = [node.state for node in graph[1:-1]]
        plt_y = [node.value for node in graph[1:-1]]
        plt.plot(plt_x, plt_y, color=COLOR.pop(), marker='x',linestyle='--', linewidth=0.75, markersize=3, label=f'episode:{episode}')

        reset()

    v = MDP()
    plt.plot(plt_x, v, color='b', marker='o',linestyle='-', linewidth=0.75, markersize=3, label=f'MDP')
    plt.legend(loc=2, title=f'GAMMA:{GAMMA} ALPHA:{ALPHA}')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()



