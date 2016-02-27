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
        self._current_state = State(9)
        self._current_reward = -1
        self._transition = {}
        self._reward = {}

        for i in range(self._num_of_states):
            self._states.append(State(i))
        for i in range(self._num_of_actions):
            self._actions.append(Action(i))

        ''' initialize transition probability P(s'|a,s) '''
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
                (i / 4 == target_state.get() / 4 or i % 4 == target_state.get() % 4):
                    self._transition[State(i)][target_action][target_state] = 1.
                else:
                    self._transition[State(i)][target_action][State(i)] = 1.


        ''' initialize reward E(r|s',a,s) '''
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
                    return True
        else:
            return False

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

        ''' initialize policy probability P(a|s) '''
        for state in self._states:
            for action in self._actions:
                if not self._policy.has_key(state):
                    self._policy[state] = {}
                self._policy[state][action] = 1./len(self._actions)

        ''' initialize value / action-value functions E(r|s) , E(r|a,s)'''
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
        p = np.random.uniform()
        s = 0.
        for action in self._policy[state]:
            s += self._policy[state][action]
            if p < s:
                return action

    def update(self, state, action, reward):
        pass

    def get_policy(self):
        return self._policy

    def get_value_func(self):
        return self._value_func

    def get_action_value_func(self):
        return self._action_value_func

    def update(self):

        # update value function
        for i in range(2000):
            for from_state in self._states:
                if not self._env.is_terminal_state(from_state):
                    temp = 0.
                    for action in self._actions:
                        for to_state in self._states:
                            temp += self._policy[from_state][action] * self._transition[from_state][action][to_state] * \
                                    (self._reward[from_state][action][to_state] + self._discount * self._value_func[to_state])
                    self._temp_value_func[from_state] = temp
            self._value_func = self._temp_value_func

        # update action value function
        for from_state in self._states:
            if not self._env.is_terminal_state(from_state):
                for action in self._actions:
                    temp = 0.
                    for to_state in self._states:
                        temp += self._transition[from_state][action][to_state] * (self._reward[from_state][action][to_state] + self._discount * self._value_func[to_state])
                    self._temp_action_value_func[from_state][action] = temp
        self._action_value_func = self._temp_action_value_func

        # improve policy
        for from_state in self._states:
            if not self._env.is_terminal_state(from_state):
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
    def __init__(self,env):
        self._env = env
        self._policy = DPPolicy(env)
        self._policy.update()

    def next_step(self):
        if self._env.is_terminated():
            print 'Terminated ...'
            return False
        current_state = self._env.get_current_state()
        chosen_action = self._policy.choose_action(current_state)
        print 'current_state = ' + str(current_state) + ' chosen_action = ' + str(chosen_action)
        self._env.take_action(chosen_action)
        return True

if __name__ == '__main__':
    env = DPEnv()
    agent = DPAgent(env)
    for i in range(10):
        if not agent.next_step():
            break
