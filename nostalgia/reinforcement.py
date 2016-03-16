# -*- coding: utf-8 -*-

from abc import ABCMeta,abstractmethod

import numpy as np

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

    @abstractmethod
    def is_terminated(self):
        pass

    @abstractmethod
    def is_terminal_state(self,state):
        pass

class Policy(object):
    __metaclass__ = ABCMeta

    def __init__(self,env):
        self._env = env

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def choose_action(self,state):
        pass

    @abstractmethod
    def get_value_func(self):
        pass

    @abstractmethod
    def get_action_value_func(self):
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

        ''' initialize transition probability P(s'|a,s) '''
        for from_state in self._states:
            for action in self._actions:
                for to_state in self._states:
                    self._transition_prob[from_state] = self._transition_prob.get(from_state,{})
                    self._transition_prob[from_state][action] = self._transition_prob[from_state].get(action,{})
                    self._transition_prob[from_state][action][to_state] = 0.

        for i in range(self._num_of_states):
            target_states = [ (State(i-4),Action(0)) ,(State(i+1),Action(1)),(State(i+4),Action(2)),(State(i-1),Action(3)) ]
            for target_state,target_action in target_states:
                if self._transition_prob[State(i)][target_action].has_key(target_state) and \
                (i / 4 == target_state.get() / 4 or i % 4 == target_state.get() % 4):
                    self._transition_prob[State(i)][target_action][target_state] = 1.
                else:
                    self._transition_prob[State(i)][target_action][State(i)] = 1.

        ''' initialize reward E(r|s',a,s) '''
        for from_state in self._states:
            for action in self._actions:
                for to_state in self._states:
                    self._expected_reward[from_state] = self._expected_reward.get(from_state,{})
                    self._expected_reward[from_state][action] = self._expected_reward[from_state].get(action,{})
                    self._expected_reward[from_state][action][to_state] = -1.

        self.initialize()

    def initialize(self,state=None):
        if state is None:
            state = State(np.random.randint(1,self._num_of_states-1))
        self._current_state = state
        self._current_reward = 0.

    def get_current_state(self):
        return self._current_state

    def get_current_reward(self):
        return self._current_reward

    def take_action(self,action):
        if not self.is_terminated():
            p = np.random.uniform()
            s = 0.
            for to_state in self._transition_prob[self._current_state][action]:
                s += self._transition_prob[self._current_state][action][to_state]
                if p < s:
                    self._current_state = to_state
                    self._current_reward = self._expected_reward[self._current_state][action][to_state]
                    return self._current_state,self._current_reward
        else:
            return None

    def get_states(self):
        return self._states

    def get_actions(self):
        return self._actions

    def get_available_action(self,state):
        return self._actions

    def get_transition_prob(self):
        return self._transition_prob

    def get_expected_reward(self):
        return self._expected_reward

    def is_terminated(self):
        return (self._current_state == State(0) or self._current_state == State(15))

    def is_terminal_state(self,state):
        return (state == State(0) or state == State(15))

    def __str__(self):
        pass

class DPPolicy(Policy):
    def __init__(self,env):
        super(DPPolicy,self).__init__(env)
        self._policy = {}
        self._value_func = {}
        self._temp_value_func = {}
        self._action_value_func = {}
        self._temp_action_value_func = {}
        self._discount = 0.5
        self._states = env.get_states()
        self._actions = env.get_actions()
        self._transition_prob = env.get_transition_prob()
        self._expected_reward = env.get_expected_reward()
        self._env = env

        ''' initialize policy probability P(a|s) '''
        for state in self._states:
            self._policy[state] = self._policy.get(state,{})
            for action in self._actions:
                self._policy[state][action] = 1./len(self._actions)

        ''' initialize value / action-value functions E(r|s) , E(r|a,s)'''
        for state in self._states:
            self._value_func[state] = 0.
            self._temp_value_func[state] = 0.
            self._action_value_func[state] = self._action_value_func.get(state,{})
            self._temp_action_value_func[state] = self._temp_action_value_func.get(state,{})
            for action in self._actions:
                self._action_value_func[state][action] = 0
                self._temp_action_value_func[state][action] = 0

        self.initialize()
        self._env.initialize()

    def initialize(self):

        # evaluate policy
        # update value function
        for i in range(2000):
            for from_state in self._states:
                if not self._env.is_terminal_state(from_state):
                    temp = 0.
                    for action in self._actions:
                        for to_state in self._states:
                            temp += self._policy[from_state][action] * self._transition_prob[from_state][action][to_state] * \
                                    (self._expected_reward[from_state][action][to_state] + self._discount * self._value_func[to_state])
                    self._temp_value_func[from_state] = temp
            self._value_func = self._temp_value_func

        # update action value function
        for from_state in self._states:
            if not self._env.is_terminal_state(from_state):
                for action in self._actions:
                    temp = 0.
                    for to_state in self._states:
                        temp += self._transition_prob[from_state][action][to_state] * (self._expected_reward[from_state][action][to_state] + self._discount * self._value_func[to_state])
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

    def choose_action(self,state):
        p = np.random.uniform(0,np.sum(self._policy[state].values()))
        s = 0.
        for action in self._policy[state]:
            s += self._policy[state][action]
            if p < s:
                return action

    def get_value_func(self):
        return self._value_func

    def get_action_value_func(self):
        return self._action_value_func

    def __str__(self):
        s = ''
        for state in self._policy:
            for action in self._policy[state]:
                s += '%s %s %s\n' % (str(state),str(action),self._policy[state][action])
        return s


class MCPolicy(Policy):
    def __init__(self,env):
        super(MCPolicy,self).__init__(env)
        self._policy = {}
        self._value_func = {}
        self._temp_value_func = {}
        self._action_value_func = {}
        self._temp_action_value_func = {}
        self._discount = 0.5
        self._states = env.get_states()
        self._actions = env.get_actions()
        self._env = env

        ''' initialize policy probability P(a|s) '''
        for state in self._states:
            self._policy[state] = self._policy.get(state,{})
            for action in self._actions:
                self._policy[state][action] = 1./len(self._actions)

        ''' initialize value / action-value functions E(r|s) , E(r|a,s)'''
        for state in self._states:
            self._value_func[state] = 0.
            self._temp_value_func[state] = 0.
            self._action_value_func[state] = self._action_value_func.get(state,{})
            self._temp_action_value_func[state] = self._temp_action_value_func.get(state,{})
            for action in self._actions:
                self._action_value_func[state][action] = 0
                self._temp_action_value_func[state][action] = 0

        self.initialize()
        self._env.initialize()

    def initialize(self):
        all_occurence_return = {}

        for i in range(1000):
            # generate an episode
            start_state = self._states[np.random.randint(1,len(self._states)-1)]
            taken_action = self.choose_action(start_state)
            self._env.initialize(start_state)
            s = [start_state,taken_action]
            while self._env.take_action(taken_action):
                current_state = self._env.get_current_state()
                current_reward = self._env.get_current_reward()
                s.append(current_reward)
                s.append(current_state)
                if self._env.is_terminated():
                    break
                r = np.random.uniform(0,np.sum(self._policy[current_state].values()))
                c = 0.
                for action in self._policy[current_state]:
                    c += self._policy[current_state][action]
                    if r < c:
                        taken_action = action
                        s.append(action)
                        break

            # print [ str(i) for i in s]

            first_occurence_return = {}
            for i in range(0,len(s)-1,3):
                if not first_occurence_return.has_key(s[i]):
                    first_occurence_return[s[i]] = {}
                if not first_occurence_return[s[i]].has_key(s[i+1]):
                    first_occurence_return[s[i]][s[i+1]] = []
                    for j in range(i+2,len(s)-1,3):
                        first_occurence_return[s[i]][s[i+1]].append(s[j])
                    if not all_occurence_return.has_key(s[i]):
                        all_occurence_return[s[i]] = {}
                    if not all_occurence_return[s[i]].has_key(s[i+1]):
                        all_occurence_return[s[i]][s[i+1]] = []
                    all_occurence_return[s[i]][s[i+1]].append(np.sum(first_occurence_return[s[i]][s[i+1]]))

            ''' update action value function '''
            for state in self._states:
                for action in self._actions:
                    if first_occurence_return.has_key(state) and all_occurence_return[state].has_key(action):
                        self._action_value_func[state][action] = np.mean( all_occurence_return[state][action] )

            ''' update policy '''
            for state in self._states:
                v = self._action_value_func[state].values()
                m = max(v)
                c = v.count(m)
                for action in self._actions:
                    if self._action_value_func[state][action] == m:
                        self._policy[state][action] = 1 - 0.2 + 0.2/len(self._actions)
                    else:
                        self._policy[state][action] = 0.2/len(self._actions)

    def choose_action(self,state):
        p = np.random.uniform(0,np.sum(self._policy[state].values()))
        s = 0.
        for action in self._policy[state]:
            s += self._policy[state][action]
            if p < s:
                return action

    def get_value_func(self):
        return self._value_func

    def get_action_value_func(self):
        return self._action_value_func

class SampleAgent(Agent):
    def __init__(self,env=None,policy=None):
        super(SampleAgent,self).__init__(env,policy)
    def set_environment(self,env):
        self._env = env
    def set_policy(self,policy):
        self._policy = policy
    def next_step(self):
        state = self._env.get_current_state()
        action = self._policy.choose_action(state)
        if not self._env.is_terminated():
            next_state,reward = self._env.take_action(action)
            print state,action,reward
            return True
        else:
            return False

if __name__ == '__main__':
    env = GridWorld()
    agent = SampleAgent(env)

    print 'DP Policy'
    policy = DPPolicy(env)
    agent.set_policy(policy)
    while agent.next_step():
        pass

    print 'MC Policy'
    policy = MCPolicy(env)
    agent.set_policy(policy)
    while agent.next_step():
        pass
