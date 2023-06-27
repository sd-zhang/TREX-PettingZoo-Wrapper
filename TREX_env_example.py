from trexenv import TrexEnv
import numpy as np

'''The goal of this piece of code is to show how to:
    - launch the TREX-core (our digityal twin)
    - launch the TREX-gym env - connect the env to the subprocess
    - do some basic interactions with the env
    '''
def random_heuristic(action_space):
    actions = action_space.sample()
    return list(actions)

def zero_heuristic(action_space):
    action_sample = action_space.sample()
    actions = [np.zeros_like(agent_action_space) for agent_action_space in action_sample]
    return actions

def run_sim_with_heuristic(heuristic, config_name, **kwargs):
    trex_env = TrexEnv(config_name=config_name, **kwargs)
    # getting some useful stuff from the environment

if __name__ == '__main__':
    heuristic = random_heuristic
    config_name = 'GymIntegration_test'
    kwargs = {'action_space_type': 'discrete',  # discrete or continuous
              'action_space_entries': 30}  # if discrete, we need to know how many quantitizations we want between the min and max defined in the config file

    trex_env = TrexEnv(config_name=config_name, **kwargs)

    # getting some useful stuff from the environment
    # ToDo: push observation keys
    # ToDo: add SoC to the observation space
    # bid ask spread: range between highest bid price and lowest ask price
    agent_names = trex_env.get_agent_names()  # this is a list of the names for each agent
    agents_action_keys = trex_env.get_action_keys()  # this is a list of the names for each agent's actions
    agents_action_spaces = trex_env.get_action_spaces()  # this is a dict of the action spaces for each agent
    agents_obs_keys = trex_env.get_obs_keys()  # this is a list of the names for each agent's observations
    agents_obs_spaces = trex_env.get_obs_spaces()  # this is a dict of the observation spaces for each agent
    episode_length = trex_env.episode_length # this is the length of the episode, also defined in the config
    n_agents = trex_env.n_agents  # because agents are defined in the config

    episodes = 3  # we can also get treex_env.episode_limit, which is the number of episodes defined in the config
    cumulative_reward = [[] for _ in range(n_agents)]
    for episode in range(episodes):
        obs = trex_env.reset()  # this should print out a warning. The reset only resets stuff internally in the gym env, it does not reset the connected TREX-core sim. Steven should be on this but it's not high priority atm
        steps = 0
        terminated = False
        episode_cumulative_reward = [0 for _ in range(n_agents)]
        while steps <= episode_length:
            # query the policy
            actions = heuristic(trex_env.action_space)  # this is a list of actions, one for each agent
            # test to make sure if we have matching bid and ask prices we settle
            # agent 1: actions = [10, 100, 10, 10, 0]
            # agent 2 actions = [10, 10, 10, 100, 0]
            # actions = [[10, 29, 10, 1, 0], [10, 1, 10, 29, 0]]
            # print('actions: ', actions, flush=True)
            # for agent, action in enumerate(actions):
            #    print('agent: ', agent, ' action: ', action, flush=True)
            obs, reward, terminated, truncated, info = trex_env.step(actions)
            for agent, agent_reward in enumerate(reward):
                if agent_reward == agent_reward: #testing for a nan
                    episode_cumulative_reward[agent] += agent_reward
            # Disclaimer: Rewards at the first 2 steps of an episode are nans, because the market settles for 1 step ahead.
            # t==0 we have nothing in the market and therefore no reward
            # t==1 the market is processing the settlements of the previous timestep, so we have no reward
            # t==2 the has processed the settlements and now we get reward
            # it might make learning faster to accomodate for this shift somehow, but it is not strictly necessary!

            # print('reward: ', reward, flush=True)
            # if terminated:
            #    print('done at step: ', steps, 'expected to be at', episode_length, flush=True)
            steps += 1
        for agent, agent_reward in enumerate(episode_cumulative_reward):
            cumulative_reward[agent].append(agent_reward)

    for agent_names, agent_reward in zip(agent_names, cumulative_reward):
        print('agent: ', agent, ' mean return: ', np.mean(agent_reward))

    # print('simulation done, closing env')
    trex_env.close()  # ATM it is necessary to do this as LAST step!


