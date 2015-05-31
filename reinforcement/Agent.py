'''
Created on 2014. 6. 21.

@author: Lee
'''

from reinforcement.StateAction import *
from reinforcement.Environment import *
import numpy as np

class ValueFunc:
    def __init__(self,env):
        pass
    pass

class ActionValueFunc:
    def __init__(self,env):
        self.states = {}
        if isinstance(env,Environment):
            states = env.getStates()
            for state in states:
                actions = env.getAvailableActions(state)
                self.states[state] = {}
                for action in actions:
                    self.states[state][action] = 0           
        pass
    def update(self,state,action,reward):
        self.states[state][action] = self.states[state][action] + 0.5 * (reward - self.states[state][action])
        pass

class Policy:
    def __init__(self,env):
        self.states = {}
        if isinstance(env,Environment):
            states = env.getStates()
            for state in states:
                actions = env.getAvailableActions(state)
                self.states[state] = {}
                for action in actions:
                    self.states[state][action] = 0        
    def selectActionByExploration(self,state):
        actions = self.states[state].keys()
        return actions[ np.random.randint(len(actions))]
    def selectActionByExploitation(self,state):
        return max(self.states[state])
    def update(self,state,action,reward):
        self.states[state][action] = self.states[state][action] + 0.5 * (reward - self.states[state][action])
        pass
    
class Agent:
    def __init__(self,env):
        self.env = env
        self.epsilon = 0.0
        self.valuefunc = ValueFunc(env)
        self.actionValueFunc = ActionValueFunc(env)
        self.policy = Policy(env)
    def selectAction(self,state):
        if np.random.uniform() > self.epsilon:
            self.selectedAction = self.policy.selectActionByExploitation(state)
        else:
            self.selectedAction =  self.policy.selectActionByExploration(state)
        return self.selectedAction
    def receive(self,state,reward):
        self.actionValueFunc.update(state,self.selectedAction,reward)
        self.policy.update(state,self.selectedAction,reward)
