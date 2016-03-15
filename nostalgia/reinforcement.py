# -*- coding: utf-8 -*-

from abc import ABCMeta,abstractmethod

class Action(object):
    def __init__(self,a):
        self._a = a
    def get(self):
        return self._a
    def set(self,a):
        self._a = a
    def __eq__(self,obj):
        return isinstance(obj,Action) and obj._a == self._a
    def __ne__(self,obj):
        return not self == obj
    def __str__(self):
        return 'action = ' + str(self._a)
    def __hash__(self):
        return self._a

class State(object):
    def __init__(self,s):
        self._s = s
    def get(self):
        return self._s
    def set(self,s):
        self._s = s
    def __eq__(self,obj):
        return isinstance(obj,State) and obj._s == self._s
    def __ne__(self,obj):
        return not self == obj
    def __str__(self):
        return 'state = ' + str(self._s)
    def __hash__(self):
        return self._s

class Environment(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def initialize(self,state=None):
        pass

    @abstractmethod
    def get_current_state(self):
        pass

    @abstractmethod
    def get_current_reward(self):
        pass

    @abstractmethod
    def take_action(self,action):
        pass

    @abstractmethod
    def get_states(self):
        pass

    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def get_available_action(self,state):
        pass

    @abstractmethod
    def get_transition_prob(self):
        pass

    @abstractmethod
    def get_expected_reward(self):
        pass

class Policy(object):
    __metaclass__ = ABCMeta

    def __init__(self,env):
        self._env = env

    @abstractmethod
    def choose_action(self,state):
        pass

class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self,env,policy):
        self._env = env
        self._policy = policy

    @abstractmethod
    def next_step(self):
        return False

'''
0   1  2  3
4   5  6  7
8   9 10 11
12 13 14 15
'''

class GridWorld(Environment):
    def __init__(self):
        super(GridWorld,self).__init__()

        self._num_of_states = 16
        self._num_of_actions = 4
        self._states = []
        self._actions = []
        self._expected_reward = {}
        self._transition_prob = {}
        self._current_state = None
        self._current_reward = 0.

        for i in range(self._num_of_states):
            self._states.append(State(i))
        for i in range(self._num_of_actions):
            self._actions.append(Action(i))

    def initialize(self,state=None):
        pass
    def get_current_state(self):
        pass
    def get_current_reward(self):
        pass
    def take_action(self,action):
        pass
    def get_states(self):
        pass
    def get_actions(self):
        pass
    def get_available_action(self,state):
        pass
    def get_transition_prob(self):
        pass
    def get_expected_reward(self):
        pass

class DPPolicy(Policy):
    def __init__(self):
        super(DPPolicy,self).__init__()

    def choose_action(self,state):
        pass

class DPAgent(Agent):
    def __init__(self):
        super(DPAgent,self).__init__()

    def next_step(self):
        return False

class MCPolicy(Policy):
    def __init__(self):
        super(DPPolicy,self).__init__()

    def choose_action(self,state):
        pass

class MCAgent(Agent):
    def __init__(self):
        super(DPAgent,self).__init__()

    def next_step(self):
        return False

if __name__ == '__main__':
    e = GridWorld()
