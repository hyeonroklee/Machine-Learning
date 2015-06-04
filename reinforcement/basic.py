# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

class Env:
    def takeAction(self):
        pass
    def getStates(self):
        pass
    def getAvailableActions(self,state):
        pass
    def getCurrentState(self):
        pass

class Policy:
    def selectAction(self):
        pass
    def update(self,state,action,reward):
        pass
    
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

class Reward:
    def __init__(self,r):
        self.r = r
    def get(self):
        return self.r

class Agent:
    pass