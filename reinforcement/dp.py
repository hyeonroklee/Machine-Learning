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
                
    