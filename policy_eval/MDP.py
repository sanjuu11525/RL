import numpy as np

GAMMA = 1

if __name__ == "__main__":
    '''
    Implement policy evaluation of Markov process for Student diagram.
    '''
    P = np.array([[0  , 0.5, 0  , 0  , 0  , 0.5, 0  ],
                  [0  , 0  , 0.8, 0  , 0  , 0  , 0.2],
                  [0  , 0  , 0  , 0.6, 0.4, 0  , 0  ],
                  [0  , 0  , 0  , 0  , 0  , 0  , 1  ],
                  [0.2, 0.4, 0.4, 0  , 0  , 0  , 0  ],
                  [0.1, 0  , 0  , 0  , 0  , 0.9, 0  ],
                  [0  , 0  , 0  , 0  , 0  , 0  , 1  ]])

    R = np.array([-2, -2, -2, 10, 1, -1, 0])
    I = np.diag(np.ones((7)))
    v = np.dot(np.linalg.inv(I - GAMMA * P), R)
    print(v)