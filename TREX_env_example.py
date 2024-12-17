import numpy as np
import pandas as pd

from trexenv import TrexEnv

'''The goal of this piece of code is to show how to:
    - launch the TREX-core (our digityal twin)
    - launch the TREX-gym env - connect the env to the subprocess
    - do some basic interactions with the env
    '''
# def random_heuristic(action_space, **kwargs):
#     actions = action_space.sample()
#     return list(actions)

def reward_recorder(rewards, records, step):
    assert isinstance(rewards, dict) #assuming PettingZoo format here
    assert isinstance(records, dict) #records are a dict, too
    for agent in rewards:
        if agent not in records:
            records[agent] = []

        num_steps_recorded = len(records[agent])
        if num_steps_recorded <= step: #if this is the first time we have this step
            records[agent].append([])
            records[agent][step] = [rewards[agent]]
        else:
            rewards_record = records[agent][step]
            rewards_record.append(rewards[agent])
            records[agent][step] = rewards_record

    return records

# obs are recorded in 24h histograms over all agents, for each observation key
def obs_recorder(obs, records, step, obs_names):
    assert isinstance(obs, dict) #assuming PettingZoo format here
    assert isinstance(records, dict) #records are a dict, too
    daytime_hour = step % 24
    for agent in obs:
        for obs_index, obs_key in enumerate(obs_names[agent]):
            if obs_key not in records:
                records[obs_key] = {}

            if daytime_hour not in records[obs_key]:
                records[obs_key][daytime_hour] = []

                records[obs_key][daytime_hour].append(obs[agent][obs_index])

    return records


#ToDo: important note, this does NOT check wether the action space can actually use 0s
def zero_heuristic(action_spaces, **kwargs):
    zero_actions = {}
    for agent in action_spaces:
        zero_actions[agent] = np.zeros_like(action_spaces[agent].sample())

    return zero_actions
def constant_price_heuristic(action_spaces, price=0.11, **kwargs):
    assert 'agents_action_keys' in kwargs.keys(), 'this heuristic needs to know the names of the actions, please provide agents_action_keys'
    assert 'agents_obs_keys' in kwargs.keys(), 'this heuristic needs to know the names of the observations, please provide agents_obs_keys'
    assert 'obs' in kwargs.keys(), 'this heuristic needs to know the observations, please provide obs'

    obs = kwargs['obs']
    agent_obs_keys = kwargs['agents_obs_keys']
    agent_action_keys = kwargs['agents_action_keys']
    #get 0 actions
    zero_actions = zero_heuristic(action_spaces)
    actions = zero_actions.copy()
    for agent_name in action_spaces:
        agent_obs = obs[agent_name]
        if 'netload_settle' in agent_obs_keys[agent_name]:
            netload_settle_index = agent_obs_keys[agent_name].index('netload_settle')
            #calculate the net load at t_settle
            netload_settle = agent_obs[netload_settle_index]

        if 'netload_now' in agent_obs_keys[agent_name]:
            netload_now_index = agent_obs_keys[agent_name].index('netload_now')
            #calculate the net load at t_settle
            netload_now = agent_obs[netload_now_index]

        if 'netload_deliver' in agent_obs_keys[agent_name]:
            netload_deliver_index = agent_obs_keys[agent_name].index('netload_deliver')
            #calculate the net load at t_settle
            netload_deliver = agent_obs[netload_deliver_index]

        if 'storage' in agent_action_keys[agent_name]:
            SoC_index = agent_obs_keys[agent_name].index('SoC_settle')
            SoC = agent_obs[SoC_index]
            # print('agent {} has SoC {}'.format(agent_name, SoC))
            battery_index = agent_action_keys[agent_name].index('storage')
            actions[agent_name][battery_index] = 0 #netload_settle/3000

            # netload deliver: 0.3
            # 0 action: 0
            # - netload settle: 8.129579796950543



        else:
            pass #we dont bid or ask anything


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

def run_heuristic(heuristic, config_name='GymIntegration_test', action_space_type='continuous', record_reward=False, record_obs=False, **kwargs):
    env_args = dict(config_name=config_name, action_space_type=action_space_type, action_space_entries=None, baseline_offset_rewards=True)
    trex_env = TrexEnv(**env_args)

    # getting some useful stuff from the environmentxham
    # ToDo: push observation keys
    # ToDo: add SoC to the observation space
    # bid ask spread: range between highest bid price and lowest ask price
    agent_names = trex_env.agents  # this is a list of the names for each agent
    agents_action_keys = trex_env.agent_action_array  # this is a list of the names for each agent's actions
    agents_action_spaces = trex_env.action_spaces  # this is a dict of the action spaces for each agent
    agents_obs_keys = trex_env.agents_obs_names  # this is a list of the names for each agent's observations
    agents_obs_spaces = trex_env.observation_spaces  # this is a dict of the observation spaces for each agent
    episode_length = trex_env.episode_length # this is the length of the episode, also defined in the config
    num_agents = trex_env.num_agents  # because agents are defined in the config

    episodes = 2# we can also get treex_env.episode_limit, which is the number of episodes defined in the config
    agents_episode_returns = {agent_name: [] for agent_name in agent_names}
    episode_steps = []
    max_sin = 0
    min_sin = 0

    if record_reward:
        reward_records = {}

    if record_obs:
        obs_records = {}


    for episode in range(episodes):
        obs, info = trex_env.reset()  # this should print out a warning. The reset only resets stuff internally in the gym env, it does not reset the connected TREX-core sim. Steven should be on this but it's not high priority atm

        if record_obs:
            obs_records = obs_recorder(obs, obs_records, 0, agents_obs_keys)

        steps = 0
        terminated = False
        episode_cumulative_reward = {agent_name: 0 for agent_name in agent_names}
        while not terminated:
            # query the policy
            actions = heuristic(action_spaces=trex_env.action_spaces,
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
            obs, reward, terminateds, truncated, info = trex_env.step(actions)

            if record_reward:
                reward_records = reward_recorder(reward, reward_records, steps)
            if record_obs:
                obs_records = obs_recorder(obs, obs_records, steps,agents_obs_keys)

            # print(reward)

            if 'yeartime_sin' in agents_obs_keys[agent_names[0]]:
                sin_index = agents_obs_keys[agent_names[0]].index('yeartime_sin')
                sin = obs[agent_names[0]][sin_index]
                # if sin > max_sin:
                #     print('max_sin: ', sin)
                #     max_sin = sin
                # elif sin < min_sin:
                #     print('min sin:', sin)
                #     min_sin = sin
            terminated = [terminateds[agent] for agent in terminateds.keys()]
            terminated = all(terminated)
            for agent in reward:
                agent_reward = reward[agent]
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

        if record_obs: #now we assemble the histogram and print it all out
                for obs_key in obs_records:
                    for hour in range(24):
                        obs_records[obs_key][hour] = np.mean(obs_records[obs_key][hour])
                    print('obs_records for', obs_key, ':', obs_records[obs_key])


        for agent in episode_cumulative_reward:
            agent_return = episode_cumulative_reward[agent]
            agents_episode_returns[agent].append(agent_return)
            episode_steps.append(steps)

    # print('simulation done, closing env')
    trex_env.close()  # ATM it is necessary to do this as LAST step!

    if record_reward:
            # make sure all the rewards are causal
        for building in reward_records:
            if reward_records[building][0][0] != np.mean(reward_records[building][0]):
                # print('correcting building', building, 'step 0 rewards. GO FIX THE BUG U LAZY ASS')
                reward_records[building][0] = [0]

            for step in range(len(reward_records[building])): #make sure all the rewards are causal
                step_rewards = reward_records[building][step]
                if len(set(step_rewards)) != 1:
                    print('WTFFF')

                reward_records[building][step] = reward_records[building][step][0]

        #save the first element of each run in a csv
        # row = timesep
        # column = house name
        print('Recording rewards!! Make sure this is only toggled when we want to record a new baseline!!!')
        dataframe = pd.DataFrame.from_dict(reward_records)
        csv_path = config_name + '.csv'
        dataframe.to_csv(csv_path)
    return agents_episode_returns, agent_names, episode_steps

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
    agents_episode_returns, agent_names, episode_steps = run_heuristic(heuristic=constant_price_heuristic,
                                                                       config_name='MultiHouseTest_Year_NewMarket',)

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

    mean_returns = []
    for agent_name in agents_episode_returns:

        #     #find outliers in agent returns
        agent_episode_returns = np.array(agents_episode_returns[agent_name])
        median_agent_return = np.median(agent_episode_returns)
        mean_returns.append(agent_episode_returns)
        median_agent_return_indices = np.where(agent_episode_returns == median_agent_return)[0]
        median_percentage = len(median_agent_return_indices)/len(agent_episode_returns)

        non_median_agent_return_indices = np.where(agent_episode_returns != median_agent_return)[0]
        non_median_agent_returns = agent_episode_returns[non_median_agent_return_indices]

        print('agent: ', agent_name, 'episode returns: ', agent_episode_returns)
        print('------------------')

    print('Mean Returns per episode: ', np.mean(mean_returns, axis=0))

    print('Overall Mean Returns: ', np.mean(mean_returns))
