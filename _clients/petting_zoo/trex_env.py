import numpy as np
import os
# import pandas as pd
import pettingzoo as pz
from gymnasium import spaces
import threading
import random

# #ToDo: check how we're fucking up the reset, we're overflowing 2h?
#
# def bin_array(num, m):
#     """Convert a positive integer num into an m-bit bit vector"""
#     return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)
# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     return np.eye(num_classes, dtype='uint8')[y]

class Env(pz.ParallelEnv): #
    """

    """
    metadata = {'name': 'TREXEnv'}

    def __init__(self,
                 client,
                 config,
                 run_name=hash(os.times()) % 100,  # ToDo: add a default here
                 action_space_type='continuous',  # continuous or discrete
                 action_space_entries=None,  # only applicable if we have discrete actions
                 one_hot_encode_agent_ids=True,
                 baseline_offset_rewards=True,
                 only_positive_rewards=False,
                 **kwargs
                 ):
        """
        This method initializes the environment and sets up the action and observation spaces
        :param kwargs:
        'TREX_config': name of the trex-core experiment config file
        'TREX_path': path of the trex env
        'env-id': if using the parallel runner
        'action_space_type': list of 'discrete' or 'continuous' where len(list) == n_actions,
        'seed': random seed, not necessarily fully enforced yet! #FixMe: enforce random seeds properly
        """

        # self.loop=loop
        # print(self.client)
        self.client = client
        self.config = config
        self.render_mode = False

        #set up agent names
        self.agents = [agent for agent in self.config['participants'] if self.config['participants'][agent]['trader'][
            'type'] == 'policy_client']
        self.possible_agents = self.agents #change if that ever becomes a thing
        self.one_hot_encode_agent_ids = one_hot_encode_agent_ids  # needed for CLDE
        self.agents_obs_names = {} #holds the observations of each agent
        self.agent_action_array = {} #holds the action types of each agent

        self.market_id = 'training'
        self.client_connected = threading.Event()
        self.get_actions_event = threading.Event()
        self.get_obs_event = threading.Event()
        self.end_step_event = threading.Event()
        self.end_episode_event = threading.Event()

        self.actions = dict()
        self.obs = dict()
        self.reward = dict()
        self.step_count = 0
        self.poke_count = 0

        # set up spaces
        self.action_space_type = action_space_type
        if self.action_space_type == 'discrete':
            # assert isinstance(action_space_entries, int), 'action_space_entries must be specified in the environment yaml for discrete action space'
            self.action_space_entries = action_space_entries

        self._setup_obs_spaces()
        self._setup_action_spaces()

        # self.agents_obs_names = {} #holds the observations of each agent

    def render(self, mode="human"):
        raise NotImplementedError
    # def async_sender(self, sender):
    #     async with sender:
    #         sender.send('event')

    def obs_check(self, payload):
        """ every time obs/rewards come in this function is called
        when the entire set of obs/rewards are in, set get_obs_event
        """
        self.obs = payload['obs']
        self.reward = payload['reward']
        self.get_obs_event.set()

    def decode_actions(self, actions):
        agent_actions_decoded = {}
        for i, agent in enumerate(self.agents):

            agent_action = actions[agent]

            # log maxes and mins
            if agent_action > self.max_storage:
                self.max_storage = agent_action
            if agent_action < self.min_storage:
                self.min_storage = agent_action

            if self.action_space_type == 'discrete':
                if isinstance(actions[i], np.ndarray):
                    agent_action = agent_action.tolist()
                agent_action = [agent_action] if len(self.agent_action_translation[
                                                         agent]) == 1 else agent_action  # reformat actions to match, might need to change for multidimensional action space
                agent_action = [action_list[action] for action_list, action in
                                zip(self.agent_action_translation[agent], agent_action)]

            # if continuous we don't need to do anything
            elif self.action_space_type == 'continuous':
                if isinstance(agent_action, np.ndarray):
                    agent_action = agent_action.astype(float)
                    agent_action = agent_action.tolist()

            agent_actions_decoded[agent] = agent_action
        return agent_actions_decoded

    def step(self, actions):
        #ToDo: change this to process an action dict of the form {agent_name: action}
        '''
        https://gymnasium.farama.org/api/env/#gymnasium.Env.step
        :return Obs, reward (list of floats), terminated (bool), truncated (bool), info (dict)
        [bid price, bid quantity, solar ask price, solar ask quantity, bess ask price, bess ask quantity]
        '''

        # obs = dict()
        rewards, terminations, truncations, infos = {}, {}, {}, {}

        # TODO: wait for agents(TBD) to request actions. keep count to check if all erp agents requested actions
        # TODO: clear the obs/rewards dict


        # self.get_obs_event.wait()
        # TODO: DECODE ACTIONS AND STORE IN DICT: see what daniel did
        # {'Building_1': array([0.8560092], dtype=float32), 'Building_2': array([-0.94756734], dtype=float32)}
        self.actions = self.decode_actions(actions)
        print('ready to send actions')
        # self.client.publish('/'.join([self.market_id, 'algorithm', 'policy_sever_ready']), '', qos=2)
        # pub.sendMessage('pz_event')

        # self.client.put_nowait(random.random())
        # self.client.join()
        # print(self.client)
        # self.client.set()
        # self.client.publish('/'.join([self.market_id, 'algorithm', 'policy_sever_ready']), '', qos=0)
        # self.client.emit('event')
        # from_thread.run(self.tg.start_soon,self.client.send, 'event')
        # from_thread.run(self.client.publish, '/'.join([self.market_id, 'algorithm', 'policy_sever_ready'], ), '')
        # self.client.publish('/'.join([self.market_id, 'algorithm', str(actor)]), actions)
        # self.has_message.set()
        # loop = asyncio.get_event_loop()
        # loop = asyncio.get_running_loop()
        # time.sleep(10)



        # asyncio.run_coroutine_threadsafe(self.client.publish('/'.join([self.market_id, 'algorithm', 'policy_sever_ready']), ''), self.loop)



        # TODO: wait for observations and rewards
        self.get_obs_event.wait()
        self.get_obs_event.clear()
        self.step_count += 1
        print('next', self.step_count)
        # self.get_actions_event.wait()

        # if not self.step_count % 1000:
        #     print('learning for 10 seconds')
        #     time.sleep(10)
        # observations are from the beginning of the next round
        # rewards are as a consequence of these actions which also gets sent out at the start of the next round
        # essentially PZ and TREX are out of phase by half a round

        # TODO: check if sim controller sent the terminated signal
        # self.end_episode_event.wait()
        # terminateds = {}
        # truncateds = {}
        # for agent in self.agents:
        #     terminateds[agent] =  True if self.t_env_steps >= self.episode_length else False
        #     truncateds[agent] = True if self.t_env_steps >= self.episode_length else False

        # self.t_env_steps += 1
        obs = self.obs

        return obs, rewards, terminations, truncations, infos



    def reset(self, seed=None, **kwargs):
        if not hasattr(self, 'max_storage'):
            self.max_storage = -np.inf
            self.min_storage = np.inf
        else:
            print(self.max_storage, self.min_storage)

        # TODO: make sure we can talk to the sim controller

        # TODO: send sim controller a signal that we're ready to start the episode

        # self.get_obs_event.wait()
        self.client_connected.wait()
        print('waiting for obs (reset)')
        self.get_obs_event.wait()
        obs = self.obs
        self.get_obs_event.clear()
        print('got obs (reset)', self.obs)

        infos = {}
        self.agents_max_actions = {}
        self.agents_min_actions = {}
        for agent in self.agents:
            infos[agent] = dict()

        return obs, infos

    def close(self):
        pass

    def state(self):
        '''
        returns the current state of the environment
        '''
        state = []
        for agent in self.agents:
            state.expand(self._obs[agent])

        return np.array(state)


    def observation_space(self, agent):
        """
        THIS METHOD IS REQUIRED FOR PettingZoo
        return the obs space for the agent
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """
        takes agent id, returns agent action space
        """

        return self.action_spaces[agent]

    def _setup_obs_spaces(self):
        '''
        This method sets up the action and observation spaces based on the values that are in the config
        For now, agents are assumed to be homogenous in the
        '''
        # FixMe: might nor support discrete actiion and observation spaces at the moment
        # ToDo: then change action and obs spaces to be able to be diverse (Rabbithole warning!)

        self.observation_spaces = {}
        if self.one_hot_encode_agent_ids:  # append the one-hot-encoding space here
            num_one_hot_bits = int(np.ceil(np.sqrt(self.num_agents)))
            self.num_one_hot_bits = num_one_hot_bits
        for agent in self.agents:
            try:
                agent_obs_names = self.config['participants'][agent]['trader']['observations']
                lows = [-np.inf]*len(agent_obs_names)
                highs = [np.inf]*len(agent_obs_names)
                if self.one_hot_encode_agent_ids:
                    for bit in range(num_one_hot_bits):
                        agent_obs_names.append('Agent_id_bit_' + str(bit))
                        lows.append(0.0)
                        highs.append(1.0)
                self.agents_obs_names[agent] = agent_obs_names

            except:
                print('There was a problem loading the config observations')
            num_agent_obs = len(self.agents_obs_names[agent])
            agent_obs_space = spaces.Box(low=np.array(lows), high=np.array(highs), shape=(num_agent_obs,))
            self.observation_spaces[agent] = agent_obs_space

    def _setup_action_spaces(self):
        '''
        This method sets up the action and observation spaces based on the values that are in the config
        For now, agents are assumed to be homogenous in the
        '''
        # FixMe: might nor support discrete actiion and observation spaces at the moment
        # ToDo: then change action and obs spaces to be able to be diverse (Rabbithole warning!)

        self.action_spaces = {}
        if self.action_space_type == 'discrete':
            self.agent_action_translation = {}

        for agent in self.agents:
            #try:
            actions = self.config['participants'][agent]['trader']['actions']
            self.agent_action_array[agent] = [action for action in actions if actions[action]['heuristic'] == 'learned']

            min_action = [action['min'] for action in actions.values() if action['heuristic'] == 'learned']
            max_action = [action['max'] for action in actions.values() if action['heuristic'] == 'learned']
            # assert len(min_action) == len(max_action) == len(self.agent_action_array[agent]), 'There was a problem with the action space'
            num_actions = len(self.agent_action_array[agent]) #FixMe: this is obviously in the wrong spot

            if self.action_space_type == 'discrete':
                if len(self.agent_action_array[agent]) == 1:
                    agent_action_space = spaces.Discrete(self.action_space_entries)
                else:
                    entries = [self.action_space_entries for _ in range(len(self.agent_action_array[agent]))]
                    agent_action_space = spaces.MultiDiscrete(entries)

                # this is needed to decode the discrete actions into their continuous counterparts
                agent_actions_array = [np.linspace(min_action, max_action, self.action_space_entries).tolist() for min_action, max_action in zip(min_action, max_action)]
                self.agent_action_translation[agent] = agent_actions_array

            elif self.action_space_type == 'continuous':
                agent_action_space = spaces.Box(low=np.array(min_action), high=np.array(max_action), shape=(num_actions,))

            else:
                print('Action space type not recognized:', self.action_space_type)
                raise NotImplementedError

            self.action_spaces[agent] = agent_action_space
            # except:
            #    print("there was a problem loading the actions")