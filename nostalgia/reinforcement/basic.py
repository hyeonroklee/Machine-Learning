# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

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
        return 'action = ' + str(self.a)
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
        return 'state = ' + str(self.s)
    def __hash__(self):
        return self.s      

class Env:
    def takeAction(self):
        pass
    def getStates(self):
        pass
    def getAvailableActions(self,state):
        pass
    def getCurrentState(self):
        pass
    def getCurrentReward(self):
        pass

class Policy:
    def chooseAction(self,state):
        pass
    def update(self,state,action,reward):
        pass
    

class Agent:
    def step(self):
        pass