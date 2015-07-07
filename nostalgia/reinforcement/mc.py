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

class MCEnv(Env):
    def __init__(self):
        super(MCEnv, self).__init__()

        self._num_of_states = 16
        self._num_of_actions = 4
        self._states = []
        self._actions = []
        self._reward = {}
        self._transition = {}
        self._current_state = State(6)
        self._current_reward = 0.
        
        for i in range(self._num_of_states):
            self._states.append(State(i))
        for i in range(self._num_of_actions):
            self._actions.append(Action(i))
        
        for from_state in self._states:
            for action in self._actions:
                for to_state in self._states:
                    if not self._reward.has_key(from_state):
                        self._reward[from_state] = {}
                    if not self._reward[from_state].has_key(action):
                        self._reward[from_state][action] = {}
                    if not self._transition.has_key(from_state):
                        self._transition[from_state] = {}
                    if not self._transition[from_state].has_key(action):
                        self._transition[from_state][action] = {}
                    self._reward[from_state][action][to_state] = -1
                    self._transition[from_state][action][to_state] = 0.
        
        for i in range(self._num_of_states):
            target_states = [ (State(i-4),Action(0)) ,(State(i+1),Action(1)),(State(i+4),Action(2)),(State(i-1),Action(3)) ]
            for target_state,target_action in target_states:
                if self._transition[State(i)][target_action].has_key(target_state) and \
                    ( i / 4 == target_state.get() / 4 or i % 4 == target_state.get() % 4 ) :
                    self._transition[State(i)][target_action][target_state] = 1.
                else:
                    self._transition[State(i)][target_action][State(i)] = 1.

                
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

    def set_initial_state(self, state):
        super(MCEnv, self).set_initial_state(state)

    def is_terminal_state(self, state):
        super(MCEnv, self).is_terminal_state(state)

    def get_available_actions(self, state):
        super(MCEnv, self).get_available_actions(state)

    def get_transition(self):
        raise Exception('Not Supported')

    def get_reward(self):
        raise Exception('Not Supported')

class MCPolicy(Policy):
    def __init__(self):
        super(MCPolicy, self).__init__()

    def choose_action(self, state):
        super(MCPolicy, self).choose_action(state)

    def update(self, state, action, reward):
        super(MCPolicy, self).update(state, action, reward)


class MCAgent(Agent):
    def __init__(self,env,policy):
        super(MCAgent, self,env,policy).__init__()

    def step(self):
        super(MCAgent, self).step()


if __name__ == '__main__':
    
    states = []
    actions = []
    policy = {}
    action_value_func = {}
    e = env(State(7))
    states = e.get_states()
    actions = e.get_actions()
    
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
   
    all_occurence_return = {}
    for i in range(20):
        # generate an episode
        start_state = states[np.random.randint(1,len(states)-1)]
        taken_action = actions[np.random.randint(1,len(actions)-1)]
        e.reset_current_state(start_state)
        s = [start_state,taken_action]
        while e.take_action(taken_action):
            current_state = e.get_current_state()
            current_reward = e.get_current_reward()
            s.append(current_reward)
            s.append(current_state)
            if e.is_terminated():
                break
            r = np.random.uniform()
            c = 0.
            for action in policy[current_state].keys():
                c += policy[current_state][action]
                if r < c:
                    taken_action = action
                    s.append(action)
                    break
        
        first_occurence_return = {}
        for i in range(0,len(s)-1,3):
            if not first_occurence_return.has_key(s[i]):
               first_occurence_return[s[i]] = {}
            if not first_occurence_return[s[i]].has_key(s[i+1]):
               first_occurence_return[s[i]][s[i+1]] = []
               for j in range(i+2,len(s)-1,3):
                   first_occurence_return[s[i]][s[i+1]].append(s[j])
        
        '''
        for state in first_occurence_return.keys():
            for action in first_occurence_return[state].keys():
                print state,action,first_occurence_return[state][action]
        '''
        
        ''' update action value function '''
        for state in states:
            for action in actions:
                if first_occurence_return.has_key(state) and first_occurence_return[state].has_key(action):
                    action_value_func[state][action] = 0.5*(action_value_func[state][action] + np.sum( first_occurence_return[state][action] ))
        
    ''' update policy '''
    for state in states:
        v = action_value_func[state].values()
        m = max(v)
        p = 1. / float(v.count(m))
        for action in actions:
            print state,action,action_value_func[state][action],m,p
            if action_value_func[state][action] == m:
                policy[state][action] = p
            else:
                policy[state][action] = 0.
        
    print '######## action value function #######'
    for state in action_value_func.keys():
        for action in action_value_func[state].keys():
            if action_value_func[state][action] != 0:
                print state,action,action_value_func[state][action]
    print '######## Policy #######'
    for state in policy.keys():
        for action in policy[state].keys():
            print state,action,policy[state][action]
    