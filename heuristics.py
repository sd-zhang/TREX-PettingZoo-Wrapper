import datetime
import os
import shutil

import numpy as np
import tensorflow as tf

from clients.pettingzoo.trex_env import TrexEnv

'''The goal of this piece of code is to show how to:
    - launch the TREX-core (our digityal twin)
    - launch the TREX-gym env - connect the env to the subprocess
    - do some basic interactions with the env
    '''
def random_heuristic(action_space, **kwargs):
    actions = action_space.sample()
    return list(actions)

#ToDo: important note, this does NOT check wether the action space can actually use 0s
def zero_heuristic(action_space, **kwargs):
    action_sample = action_space.sample()
    actions = [np.zeros_like(agent_action_space) for agent_action_space in action_sample]

    return actions
def constant_price_heuristic(action_space, price=0.11, **kwargs):
    assert 'agents_action_keys' in kwargs.keys(), 'this heuristic needs to know the names of the actions, please provide agents_action_keys'
    assert 'agents_obs_keys' in kwargs.keys(), 'this heuristic needs to know the names of the observations, please provide agents_obs_keys'
    assert 'obs' in kwargs.keys(), 'this heuristic needs to know the observations, please provide obs'

    obs = kwargs['obs']
    agent_obs_keys = kwargs['agents_obs_keys']
    agent_action_keys = kwargs['agents_action_keys']
    #get 0 actions
    zero_actions = zero_heuristic(action_space)
    actions = zero_actions.copy()
    for agent_idx, agent_name in enumerate(agent_obs_keys):
        agent_obs = obs[agent_idx]
        load_index = agent_obs_keys[agent_name].index('load_settle')
        load_demand = agent_obs[load_index]
        generation_index = agent_obs_keys[agent_name].index('generation_settle')
        generation = agent_obs[generation_index]
        battery_index = agent_action_keys[agent_name].index('storage')
        battery = agent_obs[battery_index]

        #calculate the net load at t_settle
        net_load = load_demand - generation + battery

        # if netload is positive, we sell:
        if net_load > 0:
            # set quantity to netload
            price_bid_index = agent_action_keys[agent_name].index('price_bid')
            quantity_bid_index = agent_action_keys[agent_name].index('quantity_bid')
            actions[agent_idx][price_bid_index] = price
            actions[agent_idx][quantity_bid_index] = net_load
        elif net_load < 0:
            # set quantity to negative netload
            price_ask_index = agent_action_keys[agent_name].index('price_ask')
            quantity_ask_index = agent_action_keys[agent_name].index('quantity_ask')
            actions[agent_idx][price_ask_index] = price
            actions[agent_idx][quantity_ask_index] = -net_load

        else:
            pass #we dont bid or ask anything

    # print('actions in heuristic: {}'.format(actions), flush=True)
    return actions

def greedy_battery_management_heuristic(action_space, **kwargs):
    assert 'agents_action_keys' in kwargs.keys(), 'this heuristic needs to know the names of the actions, please provide agents_action_keys'
    assert 'agents_obs_keys' in kwargs.keys(), 'this heuristic needs to know the names of the observations, please provide agents_obs_keys'
    assert 'obs' in kwargs.keys(), 'this heuristic needs to know the observations, please provide obs'

    obs = kwargs['obs']
    agent_obs_keys = kwargs['agents_obs_keys']
    agent_action_keys = kwargs['agents_action_keys']
    #get 0 actions
    actions = zero_heuristic(action_space)

    for agent_idx, agent_name in enumerate(agent_obs_keys):

        assert 'SoC' in agent_obs_keys[agent_name], 'this heuristic needs to know the SoC'
        assert 'generation_settle' in agent_obs_keys[agent_name], 'this heuristic needs to know the generation at t_settle'
        assert 'load_settle' in agent_obs_keys[agent_name], 'this heuristic needs to know the load at t_settle'
        agent_obs = obs[agent_idx]
        # get the net-load at t_settle and aim to make that the battery action
        load_index = agent_obs_keys[agent_name].index('load_settle')
        generation_index = agent_obs_keys[agent_name].index('generation_settle')
        battery_goal = agent_obs[load_index] - agent_obs[generation_index]

        # schedule the battery action, clip it first according to action space
        battery_index = agent_action_keys[agent_name].index('storage')
        # agent_action_space_low = action_space[agent_idx].low
        # agent_action_space_high = action_space[agent_idx].high
        battery_goal = np.clip(battery_goal, action_space[agent_idx].low[battery_index], action_space[agent_idx].high[battery_index])
        actions[agent_idx][battery_index] = battery_goal
        # print('agent {} battery goal: {}'.format(agent_name, battery_goal), flush=True)
    return actions


def add_to_obs_histograms(obs_histogram_info, obs_t, agents_obs_keys, agent_names):
    for agent_idx, agent_name in enumerate(agent_names):
        if agent_name not in obs_histogram_info.keys():
            obs_histogram_info[agent_name] = {}

        for obs_index, obs_key in enumerate(agents_obs_keys[agent_name]):
            if obs_key not in obs_histogram_info[agent_name].keys():
                obs_histogram_info[agent_name][obs_key] = []

            a_o_t = obs_t[agent_idx][obs_index]

            if a_o_t is not None:
                obs_histogram_info[agent_name][obs_key].append(a_o_t)

    # print(obs_t[agent_idx])

    return obs_histogram_info


def run_heuristic(heuristic, config_name='GymIntegration_test', action_space_type='continuous', **kwargs):
    if 'summary_writer' in kwargs:
        summary_writer = kwargs['summary_writer']

        logging = True

        if 'obs_histograms' in kwargs:
            obs_histogram_info = {}
            obs_histogram = True
        else:
            obs_histogram = False
    else:
        logging = False

    kwargs = {'action_space_type': action_space_type,  # discrete or continuous
              'action_space_entries': 30}  # if discrete, we need to know how many quantitizations we want between the min and max defined in the config file

    trex_env = TrexEnv(config_name=config_name, **kwargs)

    # getting some useful stuff from the environmentxham
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

    episodes = 5# we can also get treex_env.episode_limit, which is the number of episodes defined in the config
    cumulative_rewards = [[] for _ in range(n_agents)]
    episode_steps = []
    for episode in range(episodes):
        if obs_histogram:
            obs_histogram_info = {}
        obs, info = trex_env.reset()  # this should print out a warning. The reset only resets stuff internally in the gym env, it does not reset the connected TREX-core sim. Steven should be on this but it's not high priority atm
        steps = 0
        terminated = False
        episode_cumulative_reward = [0 for _ in range(n_agents)]
        while not terminated:
            # query the policy
            actions = heuristic(action_space=trex_env.action_space,
                                agents_action_keys=agents_action_keys,
                                obs=obs,
                                agents_obs_keys=agents_obs_keys)  # this is a list of actions, one for each agent
            # test to make sure if we have matching bid and ask prices we settle
            # agent 1: actions = [10, 100, 10, 10, 0]
            # agent 2 actions = [10, 10, 10, 100, 0]
            # actions = [[10, 29, 10, 1, 0], [10, 1, 10, 29, 0]]
            # print('actions: ', actions, flush=True)
            # for agent, action in enumerate(actions):
            #    print('agent: ', agent, ' action: ', action, flush=True)
            obs, reward, terminated, truncated, info = trex_env.step(actions)
            obs_histogram_info = add_to_obs_histograms(obs_histogram_info, obs, agents_obs_keys, agent_names)
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
            # if we're logging, log all the interesting info using tensorboard
            if logging:
                with summary_writer.as_default():
                    for agent_reward_index, agent in enumerate(agent_names):
                        a_r_t = reward[agent_reward_index]
                        tf.summary.scalar('agent_{}_reward'.format(agent), a_r_t, step=steps)

                        tf.summary.scalar('agent_{}_steps'.format(agent), steps, step=steps)

                    summary_writer.flush()

            steps += 1

        for agent, agent_reward in enumerate(episode_cumulative_reward):
            cumulative_rewards[agent].append(agent_reward)
            episode_steps.append(steps)

        if logging:

            if obs_histogram:
                with summary_writer.as_default():
                    agent_names = trex_env.get_agent_names()
                    for obs_key in agents_obs_keys[agent_names[0]]: #Important: this assumes every agent has the same obs keys

                        #plot the obs that the agents dont share for each agent
                        if obs_key in ["generation_now", "load_now",  "generation_settle", "load_settle", "SoC"]:
                            for agent_name in agent_names:
                                tf.summary.histogram('obs_{}_agent_{}'.format(obs_key, agent_name), obs_histogram_info[agent_name][obs_key], step=episode)
                        else: #these are obs that are shared
                            tf.summary.histogram('obs_{}'.format(obs_key), obs_histogram_info[agent_names[0]][obs_key], step=episode)
                    summary_writer.flush()

            summary_writer.flush()

    # print('simulation done, closing env')
    trex_env.close()  # ATM it is necessary to do this as LAST step!

    return cumulative_rewards, agent_names, episode_steps

if __name__ == '__main__':
    # # run the zero actions baseline
    # print('zero actions heuristic')
    # cumulative_reward, agent_names = run_heuristic(heuristic=zero_heuristic, config_name='GymIntegration_test')
    # for agent_name, agent_reward in zip(agent_names, cumulative_reward):
    #     print('agent: ', agent_name, ' mean bill: ', np.mean(agent_reward))

    # # run the battery heuristic
    # print('battery actions heuristic')
    # cumulative_reward, agent_names = run_heuristic(heuristic=greedy_battery_management_heuristic, config_name='GymIntegration_test')
    # for agent_name, agent_reward in zip(agent_names, cumulative_reward):
    #     print('agent: ', agent_name, ' mean bill: ', np.mean(agent_reward))
    #
    # # run the random baseline
    # print('random actions heuristic')
    # cumulative_reward, agent_names = run_heuristic(heuristic=random_heuristic, config_name='GymIntegration_test')
    # for agent_name, agent_reward in zip(agent_names, cumulative_reward):
    #     print('agent: ', agent_name, ' mean bill: ', np.mean(agent_reward))

    # run the constant price baseline
    print('constant price heuristic')
    cwd = os.getcwd()
    logdir = os.path.join(cwd, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # check if dir exists, if it exists delete it and remake
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    summary_writer = tf.summary.create_file_writer(logdir)

    cumulative_rewards, agent_names, episode_steps = run_heuristic(heuristic=constant_price_heuristic, config_name='GymIntegration_test', summary_writer=summary_writer, obs_histograms=True)

    median_episode_length = np.median(episode_steps)
    median_episode_lengh_percentage = (1-len(np.where(episode_steps != median_episode_length)[0])/len(episode_steps))*100
    print('median episode length: ', median_episode_length, ' for ', median_episode_lengh_percentage, ' of the time')
    non_median_episode_length_indices = np.where(episode_steps != median_episode_length)[0]

    if non_median_episode_length_indices.size == 0:
        non_median_episode_length_indices = np.array([0])
        non_median_episode_lengths = None
    else:
        non_median_episode_lengths = episode_steps[non_median_episode_length_indices]
        print('Discovered non_median_episode_lengths: ', non_median_episode_lengths, ' at positions: ', non_median_episode_length_indices)

    agents_returns = {}
    for agent_name, agent_returns in zip(agent_names, cumulative_rewards):
        #     #find outliers in agent returns
        agents_returns[agent_name] = {}
        agent_returns = np.array(agent_returns)
        median_agent_return = np.median(agent_returns)
        median_agent_return_indices = np.where(agent_returns == median_agent_return)[0]
        median_percentage = len(median_agent_return_indices)/len(agent_returns)

        non_median_agent_return_indices = np.where(agent_returns != median_agent_return)[0]
        non_median_agent_returns = agent_returns[non_median_agent_return_indices]

        print('agent: ', agent_name, ' median returns: ', median_agent_return, 'at ', median_percentage, ' of the time')
        print('non_median_agent_returns: ', non_median_agent_returns)
        print('non_median_agent_return_indices: ', non_median_agent_return_indices)
        print('------------------')







# battery heuristic
# agent:  Building_1  mean bill:  -102.14451931514002
# agent:  Building_2  mean bill:  -78.99161397492999
# agent:  Building_3  mean bill:  -91.31732216039995
# agent:  Building_4  mean bill:  -66.94351313042999

# #random baseline
# agent:  Building_1  mean bill:  -102.11894510222668
# agent:  Building_2  mean bill:  -85.12669829586336
# agent:  Building_3  mean bill:  -97.32864160179668
# agent:  Building_4  mean bill:  -75.81927020421999

# zero actions
# agent:  Building_1  mean bill:  -91.35787152011999
# agent:  Building_2  mean bill:  -71.00918475999008
# agent:  Building_3  mean bill:  -84.09825423030003
# agent:  Building_4  mean bill:  -58.89945161796002