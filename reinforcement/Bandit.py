# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

import numpy as np
from basic import * 

class BanditAction(Action):
    pass

class BanditState(State):
    pass

class BanditReward(Reward):
    pass

class BanditEnv(Env):
    def __init__(self,n):
        state = State(0)
        actions = {}
        for i in range(n):
            action = Action(i)
            actions[action] = np.random.normal()
        self.states = { state : actions }
        self.currentState = state
        self.currentReward = 0.
    def getStates(self):
        return self.states.keys()
    def getCurrentState(self):
        return self.currentState
    def getCurrentReward(self):
        return self.currentReward
    def getAvailableActions(self,state):
        return self.states[state].keys()
    def takeAction(self,action):
        self.currentReward = self.states[self.currentState][action] + np.random.normal()
        return (self.currentState,self.currentReward)

class BanditPolicy(Policy):
    def __init__(self):
        self.valuefunc = BanditValueFunc()
        self.actionValueFunc = BanditActionValueFunc()
    def chooseAction(self,state):
        epsilon = np.random.uniform()
        actions = self.actionValueFunc.states[state].keys()
        action = actions[0]        
        s = 0
        t = 0.1
        d = 0
        for i in range(len(actions)):
            s += np.exp(self.actionValueFunc.states[state][actions[i]] / t)
        for i in range(len(actions)):
            d += np.exp(self.actionValueFunc.states[state][actions[i]] / t)
            if d/s > epsilon:
                action = actions[i]
                break
        return action
    def update(self,state,action,reward):
        self.valueFunc.update(state,action,reward)
        self.actionValueFunc.update(state,action,reward)

class BanditValueFunc():
    def __init__(self):
        self.states = {}
    def update(self,state,action,reward):
        pass

class BanditActionValueFunc():
    def __init__(self):
        self.states = {}
    def update(self,state,action,reward):
        if not self.states.has_key(state):
            self.states[state] = {}
        if not self.states[state].has_key(action):
            self.states[state][action] = 0.
        self.states[state][action] = self.states[state][action] + 0.1*(reward - self.states[state][action])
    def __str__(self):
        msg = ''
        states = self.states.keys()
        for i in range(len(states)):
            actions = self.states[states[i]].keys()
            for j in range(len(actions)):
                msg += str(states[i]) + ' ' 
                msg += str(actions[j]) + ' '
                msg += 'action_value = ' + str(self.states[states[i]][actions[j]]) + '\n'
        return msg
        
        
class BanditAgent(Agent):
    def __init__(self,env,epsilon):
        self.env = env
        self.epsilon = epsilon
        self.policy = BanditPolicy()
    def step(self):
        action = self.policy.chooseAction()
        self.env.takeAction(action)
        state = self.env.getCurrentState()
        reward = self.env.getCurrentReward()
        policy.update(action,state,reward)
    def __str__(self):
        msg  = '==== BanditAgent ====\n'
        msg += 'epsilon : ' + str(self.epsilon) + '\n'
        msg += self.policy.__str__()
        return  msg        

if __name__ == '__main__':
    env = BanditEnv(10)
    agent = BanditAgent(env,0.1)
    for i in range(100):
        agent.step()
