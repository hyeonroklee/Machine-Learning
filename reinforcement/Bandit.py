'''
Created on 2014. 6. 22.

@author: Lee
'''

from reinforcement.StateAction import *
from reinforcement.Environment import *
from reinforcement.Agent import *

class BanditReward(Reward):
    def __init__(self,state,action):
        self.mean = np.random.normal()
        self.state = state
        self.action = action
    def getState(self):
        return self.state
    def getAction(self):
        return self.action
    def get(self):
        return self.mean + np.random.normal()
    def __str__(self):
        return 'BanditReward : mean = ' + str(self.mean)

class BanditEnv(Environment):
    def __init__(self,n):
        state = State(0)
        actions = {}
        for i in range(n):
            action = Action(i)
            actions[Action(i)] = BanditReward(state,action)
            print(actions[Action(i)])
        self.states = { state : actions }
        self.currentState = state
    def getStates(self):
        return self.states.keys()
    def getCurrentState(self):
        return self.currentState
    def getAvailableActions(self,state):
        return self.states[state].keys()
    def takeAction(self,action):
        reward = self.states[self.currentState][action].get()
        previousState = self.currentState
        nextState = previousState
        self.currentState = nextState
        return (nextState,reward)
    
class BanditValueFunc(ValueFunc):
    pass

class BanditActionValueFunc(ActionValueFunc):
    def __init__(self,env):
        self.states = {}
        if isinstance(env,Environment):
            states = env.getStates()
            for state in states:
                actions = env.getAvailableActions(state)
                self.states[state] = {}
                for action in actions:
                    self.states[state][action] = 0           
        pass
    def update(self,state,action,reward):
        self.states[state][action] = self.states[state][action] + 0.5 * (reward - self.states[state][action])
        pass
        
class BanditPolicy(Policy):
    def __init__(self,env):
        self.states = {}
        if isinstance(env,Environment):
            states = env.getStates()
            for state in states:
                actions = env.getAvailableActions(state)
                self.states[state] = {}
                for action in actions:
                    self.states[state][action] = 0        
    def selectActionByExploration(self,state):
        actions = self.states[state].keys()
        return actions[ np.random.randint(len(actions))]
    def selectActionByExploitation(self,state):
        action = max(self.states[state],key=self.states[state].get)
        return action
    def update(self,state,action,reward):
        self.states[state][action] = self.states[state][action] + 0.1 * (reward - self.states[state][action])
    def __str__(self):
        return str(self.states)

class BanditAgent(Agent):
    def __init__(self,env,epsilon):
        self.env = env
        self.epsilon = epsilon
        self.valuefunc = ValueFunc(env)
        self.actionValueFunc = ActionValueFunc(env)
        self.policy = BanditPolicy(env)
    def selectAction(self,state):
        if np.random.uniform() > self.epsilon:
            self.selectedAction = self.policy.selectActionByExploitation(state)
        else:
            self.selectedAction =  self.policy.selectActionByExploration(state)
        return self.selectedAction
    def receive(self,state,reward):
        self.actionValueFunc.update(state,self.selectedAction,reward)
        self.policy.update(state,self.selectedAction,reward)
    def __str__(self):
        msg  = '==== BanditAgent ====\n'
        msg += 'epsilon : ' + str(self.epsilon) + '\n'
        msg += self.policy.__str__()
        return  msg
 

if __name__ == '__main__':
    pass
    