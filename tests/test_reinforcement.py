from nostalgia.reinforcement import *

if __name__ == '__main__':
    count = 1000

    # Dynamic Programming method
    dp_env = DPEnv()
    dp_policy = DPPolicy()
    dp_agent = DPAgent(dp_env,dp_policy)
    for i in range(count):
        dp_agent.step()

    # monte carlo method
    mc_env = MCAgent()
    mc_policy = MCPolicy()
    mc_agent = MCAgent(mc_env,mc_policy)
    for i in range(count):
        mc_agent.step()

    # temporary decision method
    td_env = MCAgent()
    td_policy = MCPolicy()
    td_agent = MCAgent(td_env,td_policy)
    for i in range(count):
        td_agent.step()




