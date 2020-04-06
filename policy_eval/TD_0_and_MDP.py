import numpy as np
import matplotlib.pyplot as plt


POLICY_THRESHOLD = 0.5
EPISODE = [0, 1, 10, 100]
ALPHA = 0.1
GAMMA = 1
STATIC = True
MARKER = ['o', 'v', 'x', '*']
COLOR = ['c', 'r', 'g', 'k']

class State():
    def __init__(self, name, value, done, reward):
        self.state = name
        self.value = value
        self.done = done
        self.reward = reward

graph = np.array([State('Left' , 0.0, True , 0), State('A', 0.5, False, 0),
                  State('B'    , 0.5, False, 0), State('C', 0.5, False, 0),
                  State('D'    , 0.5, False, 0), State('E', 0.5, False, 0),
                  State('Right', 0.0, True , 1)])


def reset(graph):
    for state in graph[1:-1]:
        state.value = 0.5


def compute_MDP():
    '''
    Implement policy evaluation of Markov process.
    '''
    P = np.array([[0  , 0  , 0  , 0  ,0  ,0  ,0  ],
                  [0.5, 0  , 0.5, 0  ,0  ,0  ,0  ],
                  [0  , 0.5, 0  , 0.5,0  ,0  ,0  ],
                  [0  , 0  , 0.5, 0  ,0.5,0  ,0  ],
                  [0  , 0  , 0  , 0.5,0  ,0.5,0  ],
                  [0  , 0  , 0  , 0  ,0.5,0  , 0.5],
                  [0  , 0  , 0  , 0  ,0  ,0  , 0  ]])

    R = np.array([0, 0, 0, 0, 0, 0, 1])
    I = np.diag(np.ones((7)))
    v = np.dot(np.linalg.inv(I - GAMMA * P), R)
    
    return v[1:-1]


if __name__ == "__main__":

    for episode in EPISODE:
    
        for _ in range(episode):

            done = False

            index = 3 if STATIC else np.random.randint(1, 6)

            # evaluate policy by random walk
            while not done:
                value = graph[index].value

                # 
                action = 1 if np.random.rand() < 0.5 else -1
            
                next = index + action
            
                # access info of the next state
                future_value, reward, done = graph[next].value, graph[next].reward, graph[next].done

                # update next state by TD(0)
                graph[index].value = value + ALPHA * (reward + GAMMA * future_value - value)

                index = next

        plt_x = [node.state for node in graph[1:-1]]
        plt_y = [node.value for node in graph[1:-1]]
        plt.plot(plt_x, plt_y, color=COLOR.pop(), marker=MARKER.pop(), linewidth=0.75, markersize=3, label=f'episode:{episode}')

        reset(graph)

    v = compute_MDP()
    plt.plot(plt_x, v, color='BLUE', marker='o', linewidth=0.75, markersize=3, label='MDP')
    plt.legend(loc=4)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()



