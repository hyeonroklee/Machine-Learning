# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

import numpy as np
import matplotlib.pyplot as plt

class Action:
    def __init__(self,a):
        self.a = a
    def get(self):
        return self.a
    def set(self,a):
        self.a = a
    def __eq__(self,obj):
        return isinstance(obj,Action) and obj.a == self.a
    def __ne__(self,obj):
        return not self == obj
    def __str__(self):
        return 'Action : a = ' + str(self.a)
    def __hash__(self):
        return self.a
    
class State:
    def __init__(self,s):
        self.s = s
    def get(self):
        return self.s
    def set(self,s):
        self.s = s
    def __eq__(self,obj):
        return isinstance(obj,State) and obj.s == self.s
    def __ne__(self,obj):
        return not self == obj
    def __str__(self):
        return 'State : s = ' + str(self.s)
    def __hash__(self):
        return self.s      
        
class BanditReward():
    def __init__(self):
        self.mean = np.random.normal()
    def get(self):
        return self.mean + np.random.normal()

class BanditEnv():
    def __init__(self,n=10):
        state = State(0)
        actions = {}
        for i in range(n):
            action = Action(i)
            actions[action] = BanditReward()
        self.states = { state : actions }
        self.currentState = state
        self.currentReward = 0
    def getStates(self):
        return self.states.keys()
    def getCurrentState(self):
        return self.currentState
    def getCurrentReward(self):
        return self.currentReward
    def getAvailableActions(self,state):
        return self.states[state].keys()
    def takeAction(self,state,action):
        self.currentReward = self.states[state][action].get()
        return self.currentReward
    def __str__(self):
        return self.state.__str__() + " " + str(self.n)
        
class BanditValueFunc():
    def __init__(self):
        self.states = {}
    def addState(self,state):
        self.states[state] = 0
    def update(self,state,action,reward):
        pass

class BanditActionValueFunc():
    def __init__(self):
        self.states = {}
    def addStateActions(self,state,actions):
        self.states[state] = {}
        for i in range(len(actions)):
            self.states[state][actions[i]] = 0
    def update(self,state,action,reward):
        self.states[state][action] = self.states[state][action] + 0.1*(reward - self.states[state][action])
    def __str__(self):
        msg = ''
        states = self.states.keys()
        for i in range(len(states)):
            actions = self.states[states[i]].keys()
            for j in range(len(actions)):
                msg += str(states[i]) + ' ' 
                msg += str(actions[j]) + ' '
                msg += str(self.states[states[i]][actions[j]]) + '\n'
        return msg

class BanditPolicy():
    def __init__(self,valueFunc,actionValueFunc):
        self.valueFunc = valueFunc
        self.actionValueFunc = actionValueFunc
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
    
if __name__ == '__main__':    
    env = BanditEnv()
    valueFunc = BanditValueFunc()
    actionValueFunc = BanditActionValueFunc()
    policy = BanditPolicy(valueFunc,actionValueFunc)
    states = env.getStates()    
    for i in range(len(states)):
        actions = env.getAvailableActions(states[i])
        valueFunc.addState(states[i])
        actionValueFunc.addStateActions(states[i],actions)
    
    fa = 0
    for i in range(100):
        state = env.getCurrentState()
        action = policy.chooseAction(state)
        reward = env.takeAction(state,action)        
        policy.update(state,action,reward)
        fa = action
        
    print(actionValueFunc)
    print(fa)