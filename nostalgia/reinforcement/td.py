# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""

import numpy as np
from basic import *

'''
0   1  2  3
4   5  6  7
8   9 10 11
12 13 14 15
'''

class TDEnv(Env):
    def __init__(self,start_state=State(5)):
        self.num_of_states = 16
        self.num_of_actions = 4
        self.states = []
        self.actions = []
        self.reward = {}
        self.transition = {}        
        self.current_state = start_state
        self.current_reward = 0.     
        
        for i in range(num_of_states):
            self.states.append(State(i))
        for i in range(num_of_actions):
            self.actions.append(Action(i))
        
        for from_state in self.states:
            for action in self.actions:
                for to_state in self.states:
                    if not self.reward.has_key(from_state):
                        self.reward[from_state] = {}
                    if not self.reward[from_state].has_key(action):
                        self.reward[from_state][action] = {}
                    if not self.transition.has_key(from_state):
                        self.transition[from_state] = {}
                    if not self.transition[from_state].has_key(action):
                        self.transition[from_state][action] = {}
                    self.reward[from_state][action][to_state] = -1
                    self.transition[from_state][action][to_state] = 0.
        
        for i in range(num_of_states):
            target_states = [ (State(i-4),Action(0)) ,(State(i+1),Action(1)),(State(i+4),Action(2)),(State(i-1),Action(3)) ]
            for target_state,target_action in target_states:
                if self.transition[State(i)][target_action].has_key(target_state) and \
                    ( i / 4 == target_state.get() / 4 or i % 4 == target_state.get() % 4 ) :
                    self.transition[State(i)][target_action][target_state] = 1.
                else:
                    self.transition[State(i)][target_action][State(i)] = 1.

                
    def get_states(self):
        return self.states
    
    def get_actions(self):
        return self.actions
    
    def get_current_state(self):
        return self.current_state
    
    def get_current_reward(self):
        return self.current_reward
       
    def reset_current_state(self,state):
        self.current_state = state
        self.current_reward = 0.
        
    def is_terminated(self):
        return (self.current_state == State(0) or self.current_state == State(15))
       
    def take_action(self,action):
        if self.current_state == State(0) or self.current_state == State(15):
            return False
        else:
            d = np.random.uniform()
            s = 0.
            for state in self.states:
                #print self.current_state,action,state
                s += self.transition[self.current_state][action][state]
                if d < s:
                    self.current_reward = self.reward[self.current_state][action][state]
                    self.current_state = state
                    break
        return True

class TDPolicy(Policy):
    def __init__(self):
        super(TDPolicy, self).__init__()

    def choose_action(self, state):
        super(TDPolicy, self).choose_action(state)

    def update(self, state, action, reward):
        super(TDPolicy, self).update(state, action, reward)

class TDAgent(Agent):
    def __init__(self,env,policy):
        super(TDAgent,self,env,policy).__init__()

    def step(self):
        super(TDAgent, self).step()

if __name__ == '__main__':
    e = env()
    states = e.get_states()
    actions = e.get_actions()
    policy = {}
    action_value_func = {}
    
    ''' initialize policy '''
    ''' p(a|s) = 1/4 '''
    for state in states:
        for action in actions:
            if not policy.has_key(state):
                policy[state] = {}
            policy[state][action] = 1./num_of_actions

    ''' initialize action value function '''
    for state in states:
        for action in actions:
            if not action_value_func.has_key(state):
                action_value_func[state] = {}
            action_value_func[state][action] = -1.            
    
    while not e.is_terminated():
        current_state = e.get_current_state()
        current_reward = e.get_current_reward()
        
        # take an action
        r = np.random.uniform()
        c = 0.
        for action in policy[current_state].keys():
            c += policy[current_state][action]
            if r < c:
                e.take_action(action)                
                break
    
        # update action value function
    
        # update policy