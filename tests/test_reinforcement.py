from nostalgia import *

import numpy as np

if __name__ == '__main__':
    env = reinforcement.GridWorld()
    agent = reinforcement.SampleAgent(env)

    print 'DP Policy'
    policy = reinforcement.DPPolicy(env)
    agent.set_policy(policy)
    while agent.next_step():
        pass

    print 'MC Policy'
    policy = reinforcement.MCPolicy(env)
    agent.set_policy(policy)
    while agent.next_step():
        pass