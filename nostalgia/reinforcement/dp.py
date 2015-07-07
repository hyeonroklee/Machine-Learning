# -*- coding: utf-8 -*-

import numpy as np
from basic import *

'''
0   1  2  3
4   5  6  7
8   9 10 11
12 13 14 15
'''

class DPEnv(Env):
    def __init__(self):
        super(DPEnv, self).__init__()

        self._num_of_states = 16
        self._num_of_actions = 4
        self._states = []
        self._actions = []
        self._current_state = State(5)
        self._current_reward = -1
        self._transition = {}
        self._reward = {}

        for i in range(self._num_of_states):
            self._states.append(State(i))
        for i in range(self._num_of_actions):
            self._actions.append(Action(i))

        ''' initialize transition probability p(s'|a,s) '''
        for from_state in self._states:
            for action in self._actions:
                for to_state in self._states:
                    if not self._transition.has_key(from_state):
                        self._transition[from_state] = {}
                    if not self._transition[from_state].has_key(action):
                        self._transition[from_state][action] = {}
                    self._transition[from_state][action][to_state] = 0.

        for i in range(self._num_of_states):
            target_states = [ (State(i-4),Action(0)) ,(State(i+1),Action(1)),(State(i+4),Action(2)),(State(i-1),Action(3)) ]
            for target_state,target_action in target_states:
                if self._transition[State(i)][target_action].has_key(target_state) and \
                    ( i / 4 == target_state.get() / 4 or i % 4 == target_state.get() % 4 ) :
                    self._transition[State(i)][target_action][target_state] = 1.
                else:
                    self._transition[State(i)][target_action][State(i)] = 1.


        ''' initialize reward e(r|s',a,s) '''
        for from_state in self._states:
            for action in self._actions:
                for to_state in self._states:
                    if not self._reward.has_key(from_state):
                        self._reward[from_state] = {}
                    if not self._reward[from_state].has_key(action):
                        self._reward[from_state][action] = {}
                    self._reward[from_state][action][to_state] = -1

    def get_current_state(self):
        return self._current_state

    def set_initial_state(self, state):
        self._current_state = state

    def take_action(self,action):
        if not self.is_terminated():
            p = np.random.uniform()
            s = 0.
            states = self._transition[self._current_state][action].keys()
            for to_state in states:
                s += self._transition[self._current_state][action][to_state]
                if p < s:
                    self._current_state = to_state
                    self._current_reward = self._reward[self._current_state][action][to_state]
                    break

    def is_terminated(self):
        return (self._current_state == State(0) or self._current_state == State(15))

    def get_states(self):
        return self._states

    def is_terminal_state(self,state):
        return (state == State(0) or state == State(15))

    def get_current_reward(self):
        return self._current_reward

    def get_actions(self):
        return self._actions

    def get_available_actions(self, state):
        return self._actions

    def get_transition(self):
        return self._transition

    def get_reward(self):
        return self._reward

class DPPolicy(Policy):
    def __init__(self,env):
        super(DPPolicy, self).__init__()
        self._policy = {}
        self._value_func = {}
        self._temp_value_func = {}
        self._action_value_func = {}
        self._temp_action_value_func = {}
        self._discount = 0.5
        self._states = env.get_states()
        self._actions = env.get_actions()
        self._transition = env.get_transition()
        self._reward = env.get_reward()
        self._env = env

        ''' initialize policy probability'''
        for state in self._states:
            for action in self._actions:
                if not self._policy.has_key(state):
                    self._policy[state] = {}
                self._policy[state][action] = 1./numOfActions
        ''' initialize value / action-value functions '''
        for state in self._states:
            self._value_func[state] = 0.
            self._temp_value_func[state] = 0.
            for action in self._actions:
                if not self._action_value_func.has_key(state):
                        self._action_value_func[state] = {}
                        self._temp_action_value_func[state] = {}
                self._action_value_func[state][action] = 0.
                self._temp_action_value_func[state][action] = 0.

    def choose_action(self, state):
        super(DPPolicy, self).choose_action(state)

    def update(self, state, action, reward):
        super(DPPolicy, self).update(state, action, reward)

    def update(self):

        # update value function
        for i in range(2000):
            for from_state in self._states:
                if not self.env.is_terminal_state(from_state):
                    temp = 0.
                    for action in self._actions:
                        for to_state in self._states:
                            temp += self._policy[from_state][action] * self._transition[from_state][action][to_state] * \
                                    (self._reward[from_state][action][to_state] + self._discount * self._value_func[to_state])
                    self._temp_value_func[from_state] = temp
            self._value_func = self._temp_value_func

        # update action value function
        for from_state in self._states:
            if not self.env.is_terminal_state(from_state):
                for action in self._actions:
                    temp = 0.
                    for to_state in self._states:
                        temp += self._transition[from_state][action][to_state] * (self._reward[from_state][action][to_state] + self._discount * self._value_func[to_state])
                    self._temp_action_value_func[from_state][action] = temp
        self._action_value_func = self._temp_action_value_func

        # improve policy
        for from_state in self._states:
            if not self.env.is_terminal_state(from_state):
                v = self._action_value_func[from_state].values()
                m = max(v)
                p = 1. / float(v.count(m))
                for action in self._actions:
                    #print from_state,action,action_value_func[from_state][action],m,p
                    if self._action_value_func[from_state][action] == m:
                        self._policy[from_state][action] = p
                    else:
                        self._policy[from_state][action] = 0.

class DPAgent(Agent):
    def __init__(self,env,policy):
        super(DPAgent, self,env,policy).__init__()

    def step(self):
        super(DPAgent, self).step()



if __name__ == '__main__':

    numOfStates = 16
    numOfActions = 4    

    states = []
    actions = []
    policy = {}
    transition = {}
    reward = {}
    
    value_func = {}  
    temp_value_func = {}
    
    action_value_func = {}
    temp_action_value_func = {}
    
    for i in range(numOfStates):
        states.append(State(i))
    for i in range(numOfActions):
        actions.append(Action(i))
    
    ''' initial policy probability'''
    for state in states:
        for action in actions:
            if not policy.has_key(state):
                policy[state] = {}
            policy[state][action] = 1./numOfActions
            
    ''' initial transition probability '''
    for from_state in states:
        for action in actions:
            for to_state in states:
                if not transition.has_key(from_state):
                    transition[from_state] = {}
                if not transition[from_state].has_key(action):
                    transition[from_state][action] = {}
                transition[from_state][action][to_state] = 0.
    
    for i in range(16):
        target_states = [ (State(i-4),Action(0)) ,(State(i+1),Action(1)),(State(i+4),Action(2)),(State(i-1),Action(3)) ]
        for target_state,target_action in target_states:
            if transition[State(i)][target_action].has_key(target_state) and \
                ( i / 4 == target_state.get() / 4 or i % 4 == target_state.get() % 4 ) :
                transition[State(i)][target_action][target_state] = 1.
            else:
                transition[State(i)][target_action][State(i)] = 1.
                
        
    ''' initial reward '''
    for from_state in states:
        for action in actions:
            for to_state in states:
                if not reward.has_key(from_state):
                    reward[from_state] = {}
                if not reward[from_state].has_key(action):
                    reward[from_state][action] = {}
                reward[from_state][action][to_state] = -1
                

    ''' interative policy evaluation '''
    for state in states:        
        value_func[state] = 0.
        temp_value_func[state] = 0.
        for action in actions:
            if not action_value_func.has_key(state):
                    action_value_func[state] = {}
                    temp_action_value_func[state] = {}
            action_value_func[state][action] = 0.
            temp_action_value_func[state][action] = 0.
        
    '''
    print '#### State ####'
    for state in states:
        print state
    print '#### Action ####'
    for action in actions:
        print action
    print '#### Policy ####'
    for state in policy.keys():
        for action in policy[state].keys():
            print state,action,policy[state][action]
    print '#### Transition ####'
    for from_state in transition.keys():
        for action in transition[from_state].keys():
            for to_state in transition[from_state][action].keys():
                if transition[from_state][action][to_state] == 1:
                    print from_state,action,to_state,transition[from_state][action][to_state]
    '''

    active_states = []
    for i in range(1,15):
        active_states.append(State(i))
        
    for i in range(2000):
        epsilon = 0.1
        delta = 0.0
        discount = 0.999
        
        # update value function
        for from_state in active_states:
            temp = 0.
            for action in actions:
                for to_state in states:
                    temp += policy[from_state][action] * transition[from_state][action][to_state] * (reward[from_state][action][to_state] + discount * value_func[to_state])
            temp_value_func[from_state] = temp
        value_func = temp_value_func
        
    # update action value function
    for from_state in active_states:
        for action in actions:
            temp = 0.
            for to_state in states:
                temp += transition[from_state][action][to_state] * (reward[from_state][action][to_state] + discount * value_func[to_state])
            temp_action_value_func[from_state][action] = temp
    action_value_func = temp_action_value_func
         
    # improve policy
    for from_state in active_states:
        v = action_value_func[from_state].values()
        m = max(v)
        p = 1. / float(v.count(m))
        for action in actions:
            #print from_state,action,action_value_func[from_state][action],m,p
            if action_value_func[from_state][action] == m:
                policy[from_state][action] = p
            else:
                policy[from_state][action] = 0.
                    
        
    for state in value_func.keys():
        print state,value_func[state]
           
    print '#### Action Value Function ####'    
    for state in action_value_func.keys():
        for action in action_value_func[state].keys():
            print state,action,action_value_func[state][action]  
    
    print '#### Policy ####'
    for state in policy.keys():
        for action in policy[state].keys():
            if policy[state][action] > 0.25:
                print state,action,policy[state][action]
    
    '''
    while True:
        for from_state in states:
            temp = 0.
            for action in actions:
                for to_state in states:
                    t = policy[from_state][action] * transition[from_state][action][to_state] * (reward[from_state][action][to_state] + discount * value_func[to_state])
                    if t != 0 and from_state == State(11):
                        print from_state,action,to_state
                        print policy[from_state][action],transition[from_state][action][to_state],reward[from_state][action][to_state],value_func[to_state],t
                    temp += t
            delta = max(delta,abs(temp-temp_value_func[from_state]))
            temp_value_func[from_state] = temp
        value_func = temp_value_func
        for state in value_func.keys():
            print state,value_func[state]
        if delta < epsilon:
            break
        
    print value_func
    for state in value_func.keys():
        print state
    '''
    
    '''

    '''
    ''' find optimal policy '''
    ''' policy improvement '''
                
    