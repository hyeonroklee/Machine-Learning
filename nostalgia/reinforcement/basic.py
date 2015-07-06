# -*- coding: utf-8 -*-

class Action(object):
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
    
class State(object):
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

class Env(object):
    def __init__(self):
        pass
    def take_action(self,action):
        pass
    def get_states(self):
        pass
    def get_available_actions(self,state):
        pass
    def set_initial_state(self,state):
        pass
    def get_current_state(self):
        pass
    def get_current_reward(self):
        pass
    def is_terminated(self):
        pass
    def is_terminal_state(self,state):
        pass
    def get_transition(self):
        pass
    def get_reward(self):
        pass

class Policy(object):
    def __init__(self):
        pass
    def choose_action(self,state):
        pass
    def update(self,state,action,reward):
        pass
    

class Agent(object):
    def __init__(self):
        pass
    def step(self):
        pass