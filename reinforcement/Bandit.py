# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 22:42:18 2015

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
            actions[Action(i)] = BanditReward(state,action)
            print(actions[Action(i)])
        self.states = { state : actions }
        self.currentState = state
    def getStates(self):
        return self.states.keys()
    def getCurrentState(self):
        return self.currentState
    def getAvailableActions(self,state):
        return self.states[state].keys()
    def takeAction(self,action):
        reward = self.states[self.currentState][action].get()
        previousState = self.currentState
        nextState = previousState
        self.currentState = nextState
        return (nextState,reward)

class BanditPolicy(Policy):
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
        action = max(self.states[state],key=self.states[state].get)
        return action
    def update(self,state,action,reward):
        self.states[state][action] = self.states[state][action] + 0.1 * (reward - self.states[state][action])
    def __str__(self):
        return str(self.states)

class BanditAgent(Agent):
    def __init__(self,env,epsilon):
        self.env = env
        self.epsilon = epsilon
        self.valuefunc = ValueFunc(env)
        self.actionValueFunc = ActionValueFunc(env)
        self.policy = BanditPolicy(env)
    def selectAction(self,state):
        if np.random.uniform() > self.epsilon:
            self.selectedAction = self.policy.selectActionByExploitation(state)
        else:
            self.selectedAction =  self.policy.selectActionByExploration(state)
        return self.selectedAction
    def receive(self,state,reward):
        self.actionValueFunc.update(state,self.selectedAction,reward)
        self.policy.update(state,self.selectedAction,reward)
    def __str__(self):
        msg  = '==== BanditAgent ====\n'
        msg += 'epsilon : ' + str(self.epsilon) + '\n'
        msg += self.policy.__str__()
        return  msg


if __name__ == '__main__':
    a1 = BanditAction(1)
    a2 = BanditAction(1)
    print a1 == a2
