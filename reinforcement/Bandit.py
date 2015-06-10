# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

import numpy as np
from basic import * 

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
        self.currentState = self.currentState
    def __str__(self):
        msg = '==== BanditEnv ====\n'
        for state in self.states.keys():
            for action in self.states[state].keys():
                msg += str(state) + ' ' + str(action) + ' ' + str(self.states[state][action]) + '\n'
        return msg

class BanditPolicy(Policy):
    def __init__(self,env):
        self.temp = 1.0
        self.valueFunc = BanditValueFunc(env)
        self.actionValueFunc = BanditActionValueFunc(env)
    def chooseAction(self,state):
        epsilon = np.random.uniform()
        actions = self.actionValueFunc.states[state].keys()
        action = actions[0]        
        s = 0
        d = 0
        for i in range(len(actions)):
            s += np.exp(self.actionValueFunc.states[state][actions[i]] / self.temp)
        for i in range(len(actions)):
            d += np.exp(self.actionValueFunc.states[state][actions[i]] / self.temp)
            if d/s > epsilon:
                action = actions[i]
                break
        return action
    def update(self,state,action,reward):
        self.valueFunc.update(state,action,reward)
        self.actionValueFunc.update(state,action,reward)
    def __str__(self):
        return '==== Policy ====\n'
        
class BanditValueFunc():
    def __init__(self,env):
        self.states = {}
        for state in env.getStates():
            self.states[state] = {}
            for action in env.getAvailableActions(state):
                self.states[state][action] = 0.
    def update(self,state,action,reward):
        pass

class BanditActionValueFunc():
    def __init__(self,env):
        self.alpha = 0.3
        self.states = {}
        for state in env.getStates():
            self.states[state] = {}
            for action in env.getAvailableActions(state):
                self.states[state][action] = 0.
    def update(self,state,action,reward):
        if not self.states.has_key(state):
            self.states[state] = {}
        if not self.states[state].has_key(action):
            self.states[state][action] = 0.
        self.states[state][action] = self.states[state][action] + self.alpha*(reward - self.states[state][action])
    def __str__(self):
        msg = '==== ActionValueFunc ====\n'
        for state in self.states.keys():
            for action in self.states[state].keys():
                msg += str(state) + ' ' + str(action) + ' '
                msg += 'action_value = ' + str(self.states[state][action]) + '\n'
        return msg
        
        
class BanditAgent(Agent):
    def __init__(self,env,epsilon):
        self.env = env
        self.epsilon = epsilon
        self.policy = BanditPolicy(env)
        self.cnt = {}
    def step(self):
        state = self.env.getCurrentState()        
        action = self.policy.chooseAction(state)

        if not self.cnt.has_key(state):
            self.cnt[state] = {}
        if not self.cnt[state].has_key(action):
            self.cnt[state][action] = 0
        self.cnt[state][action] += 1
            
        self.env.takeAction(action)
        reward = self.env.getCurrentReward()
        self.policy.update(state,action,reward)

    def __str__(self):
        msg = '==== BanditAgent ====\n'
        for state in self.cnt.keys():
            for action in self.cnt[state].keys():
                msg += str(state) + ' ' + str(action) + ' ' + str(self.cnt[state][action]) + '\n'
        return msg        
                    
if __name__ == '__main__':
    epsilon = 0.1        
    env = BanditEnv(10)  
    agent = BanditAgent(env,epsilon)
           
    for i in range(2000):
        agent.step()
    
    print str(env) 
    print str(agent)
    print str(agent.policy.actionValueFunc)