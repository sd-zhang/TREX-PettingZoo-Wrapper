import multiprocessing.managers

from TREX_env._utils.sml_utils import read_flag_x_times
from TREX_env._utils.trex_utils import prep_trex, run_subprocess, add_envid_to_launchlist
import TREX_Core._utils.runner
from gymnasium import spaces
import numpy as np
import os
import multiprocessing as mp
import time
import tenacity
import pettingzoo as pz
from mathutils import RunningMeanStdMinMax

#ToDo: check how we're fucking up the reset, we're overflowing 2h?

class TrexEnv(pz.ParallelEnv): #ToDo: make this inherit from PettingZoo or sth else?
    """

    """
    metadata = {}

    def __init__(self,
                 config_name=None, #ToDo: add a default here
                 run_name=hash(os.times()) % 100, #ToDo: add a default here
                 normalize_obs=False,
                 normalize_rewards=False,
                 action_space_type='continuous', #continuous or discrete
                    action_space_entries=None, #only applicable if we have discrete actions
                 **kwargs):
        """
        This method initializes the environment and sets up the action and observation spaces
        :param kwargs:
        'TREX_config': name of the trex-core experiment config file
        'TREX_path': path of the trex env
        'env-id': if using the parallel runner
        'action_space_type': list of 'discrete' or 'continuous' where len(list) == n_actions, #ToDo: continuous NOT implemented ATM
        'seed': random seed, not necessarily fully enforced yet! #FixMe: enforce random seeds properly
        """
        TREX_path = TREX_Core.__path__[0]  ##ToDo: James - adjust this to whichever path is yours, TREX has to be set up as a package
        # changes where python is looking to open the right config file
        #ToDo: Daniel - Get steven to write a function that does this without having to launch the runner and then kill it
        cur_dir = os.getcwd()
        os.chdir(TREX_path)
        runner = TREX_Core._utils.runner.Runner(config_name, resume=False, purge=False, path=TREX_path)
        self.config = runner.configs
        del runner
        os.chdir(cur_dir)

        #set up agent names
        self.agents = [agent for agent in self.config['participants'] if self.config['participants'][agent]['trader']['type'] == 'gym_agent']
        # self.num_agents = len(self.agents) might be an attribute thats auto set
        assert self.num_agents > 0, 'There are no gym_agents in the config, please pick a config with at least 1 gym_agent'

        self.possible_agents = self.agents #change if that ever becomes a thing
        # self.max_num_agents = self.num_agents #might be an autoset attribute

        self.normalize_obs = normalize_obs
        self.normalize_rewards = normalize_rewards
        # set up general env variables
        self.episode_length = int(np.floor(self.config['study']['days'] * 24 * 60 * 60 / self.config['study']['time_step_size']) + 1) #Because the length of an episode is given by the config
        self.episode_limit = int(np.floor(self.config['study']['generations'])) #number of max episodes
        self.t_env_steps = 0
        self.episode_current = 0
        self._seed = kwargs['seed'] if 'seed' in kwargs else 0
        if 'seed' in kwargs:
            print('setting seed to', kwargs['seed'], 'BEWARE that this is not fully enforced yet!')

        self.agents_obs_names = {} #holds the observations of each agent
        self.agent_action_array = {} #holds the action types of each agent

        # set up spaces
        self.action_space_type = action_space_type
        if self.action_space_type == 'discrete':
            assert isinstance(action_space_entries, int), 'action_space_entries must be specified in the environment yaml for discrete action space'
            self.action_space_entries = action_space_entries
        self._setup_spaces()

        # set up env_id for the memory lists, etc
        self.env_ids = kwargs['env_id'] if 'env_id' in kwargs else [0]
        # print('initializing TREX env, env_id:', self.env_ids, flush=True)
        self.smm_hash ='000000'
        self.smm_address = ''
        self.smm_port = 6666
        self.run_name = run_name
        self._setup_interprocess_memory()

        # set up trex
        self.trex_pool = self.__startup_TREX_Core(config_name)

    def render(self, mode="human"):
        '''
        #TODO: August 16, 2022: make this compatible with the browser code that steven has finished.

        '''

        raise NotImplementedError
        return False

    def step(self, actions):
        #ToDo: change this to process an action dict of the form {agent_name: action}
        '''
        https://gymnasium.farama.org/api/env/#gymnasium.Env.step
        :return Obs, reward (list of floats), terminated (bool), truncated (bool), info (dict)
        [bid price, bid quantity, solar ask price, solar ask quantity, bess ask price, bess ask quantity]
        '''

        # SEND ACTIONS
        # we wait for the memlist to give us the goahead to be filled again with a new action
        # actions are provided, so this method needs to put the actions into the right action bufferes
        # actions are tensor (n_actions x n_agent)
        # Trex will have price, quantity,
        # print('In Trexenv Step')

        # first we decode the actions for each agent
        agent_actions_decoded = {}
        for i, agent in enumerate(self.agents):

            agent_action = actions[agent]

            #log maxes and mins
            if agent_action > self.max_storage:
                self.max_storage = agent_action
            if agent_action < self.min_storage:
                self.min_storage = agent_action

            if self.action_space_type == 'discrete':
                if isinstance(actions[i], np.ndarray):
                    agent_action = agent_action.tolist()
                agent_action = [agent_action] if len(self.agent_action_translation[agent])== 1 else agent_action #reformat actions to match, might need to change for multidimensional action space
                agent_action = [action_list[action] for action_list, action in zip(self.agent_action_translation[agent],agent_action)]

            #if continuous we don't need to do anything
            elif self.action_space_type == 'continuous':
                if isinstance(agent_action, np.ndarray):
                    agent_action = agent_action.astype(float)
                    agent_action = agent_action.tolist()

            agent_actions_decoded[agent] = agent_action

        # then we wait cycle over each agent's memorylist to tell us we can write
        # once we hit a 'can write' we write the action in
        # should_be_false_now = [self.agent_mem_lists[agent]['actions'][0] for agent in self.agents]
        self.write_to_action_smls(agent_actions_decoded)

        # terminated:
        # this will need to be able to get set on the end of each generation

        # Reward:
        # Rewards are going to have to be sent over from the gym trader, which will be able to
        # get information from the reward
        rewards = self._get_rewards()

        # info:
        # Imma keep it as a open dictionary for now:
        infos = {}
        for agent in self.agents:
            infos[agent] = dict()
            # infos[agent]['unscaled_rewards'] = self.unscaled_rewards[agent]
        # info['rewards'] = {agent: rewards[i] for i, agent in enumerate(self.agents)}  # include the rewards individually for each agent
        # info['terminated'] = {agent: terminated[i] for i, agent in enumerate(self.agents)} #include terminated for each agent

        obs = self._get_obs()

        terminateds = {}
        truncateds = {}
        for agent in self.agents:
            terminateds[agent] =  True if self.t_env_steps >= self.episode_length else False
            truncateds[agent] = True if self.t_env_steps >= self.episode_length else False

        self.t_env_steps += 1

        return obs, rewards, terminateds, truncateds,  infos

    def reset(self, seed=None, **kwargs):
        '''
        https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
        This method resets the trex environment.
        The reset would have to be able to kill all the TREX processes,
        then reboot them all and have the gym traders reconnect to the shared memory objects.
        TODO Peter: November 30, 2022; This is going to need to reset the TREX instance
        '''

        # print('reset trex_env', self.env_ids, flush=True)
        #resetting the memlists to be sure nothing gets fucked up here
        # reset the simulation to next gen



        if not hasattr(self, 'max_storage'):
            self.max_storage = -np.inf
            self.min_storage = np.inf
        else:
            print(self.max_storage, self.min_storage)

        for env_id in self.env_ids:
            assert self.controller_smls[env_id]['kill'][2] == 'reset', 'reset section not found in envcontroller sml'
            self.controller_smls[env_id]['kill'][3] = True #set the reset flag

        self._force_nonblocking_sml()

        # envs_reset = [self.controller_smls[env_id]['kill'][3] for env_id in self.controller_smls]
        self.wait_for_controller_smls()

        # print('done reset')
        self._reset_interprocess_memory()

        #now we're reset and g2g
        self.t_env_steps = 0
        self.episode_current += 1
        obs = self._get_obs()

        # if hasattr(self, 'agents_max_actions'):
        #    print('agent max actions:', self.agents_max_actions, flush=True)
        # if hasattr(self, 'agents_min_actions'):
        #    print('agent min actions:', self.agents_min_actions, flush=True)
        infos = {}
        self.agents_max_actions = {}
        self.agents_min_actions = {}
        for agent in self.agents:
            infos[agent] = dict()
            # infos[agent]['unscaled_rewards'] = 0

        return obs, infos

    def close(self):
        # gets rid of the smls
        # print('WARNING: this might be unreliable ATM, check that the processes are actually killed!')

        # here we send the kill command to the sim controller and wait for the confirmation flag
        # print('sending kill signal to trex processes')
        for env_id in self.env_ids:
            self.controller_smls[env_id]['kill'][1] = True #setting the command flag to kill
        self._force_nonblocking_sml()
        # print('waiting for trex processes to die')

        self.wait_for_kill_smls()

        self._close_agent_memlists()
        self._close_controller_smls()

        self.trex_pool.terminate()

    def state(self):
        '''
        returns the current state of the environment
        '''
        state = []
        for agent in self.agents:
            state.expand(self._obs[agent])

        return np.array(state)

    @tenacity.retry(wait=tenacity.wait_fixed(0.01)
                          + tenacity.wait_random(0, 0.01),
                    )
    def write_to_action_smls(self, agent_actions_decoded):
        #we want alll agent flages to be false
        # print('waiting to write action smls')
        if any([self.agent_mem_lists[agent]['actions'][0] for agent in self.agent_mem_lists]): #ToDo: this should be any, nno?
            raise tenacity.TryAgain
        else:
            # print('writing to action smls')
            # should_be_false_now = [self.agent_mem_lists[agent]['actions'][0] for agent in self.agent_mem_lists]
            assert not any([self.agent_mem_lists[agent]['actions'][0] for agent in self.agent_mem_lists]), 'actions were not read ready'
            # now that all past actions have been consumed, we can write the new actions
            for i, agent in enumerate(self.agent_mem_lists):
                agent_action = agent_actions_decoded[agent]
                for j, action in enumerate(agent_action):
                    self.agent_mem_lists[agent]['actions'][j+1] = action
                self.agent_mem_lists[agent]['actions'][0] = True #set the can be read to true

            assert all([self.agent_mem_lists[agent]['actions'][0] for agent in self.agent_mem_lists]), 'actions were not written correctly'
            # print('done writing to action smls')
            return True

    def _get_obs(self):

        """
        This gets the full observation state for that timestep.
        Only call explicitly for T==0 and for testing purposes!

        :return: a single list that contains all the agents individual observations as lists:
        [agent1_obs_list, agent2_obs_list, ..., agentn_obs_list

        """
        #FixMe: this is an epymarl leftover and will need to be removed at some point

        # in the lbf this simply returns self._obs
        # self._obs is populated in env.step, but the values are pulled before the next
        # steps
        self._read_obs_smls()
        return self._scale_obs(self._obs) if self.normalize_obs else self._obs
    @tenacity.retry(wait=tenacity.wait_fixed(0.01)
                          + tenacity.wait_random(0, 0.01),
                    )
    def _read_obs_smls(self):
        """
        This method cycles through the mem lists of the agents until they all have all read the information.
        """


        # lets make sure all agent obs are ready to be read
        # agent_status = [self.agent_mem_lists[agent]['obs'][0] for agent in self.agent_mem_lists]
        # #We expect these to be True by now, if some are false we retry
        # print('waiting to read obs smls')
        if not all([self.agent_mem_lists[agent]['obs'][0] for agent in self.agent_mem_lists]):
            raise tenacity.TryAgain
        else:
            self._obs = {}
            #after having made sure all are true, we can read the values
            # print('reading obs smls')
            for i, agent_name in enumerate(self.agent_mem_lists):
            # agent is a dictionary 'obs', 'actions', 'rewards'

                agent_obs = [self.agent_mem_lists[agent_name]['obs'][j] for j in range(1,len(self.agent_mem_lists[agent_name]['obs']))] #get the values, THIS SEEMS TO WORK WITH SHAREABLE LISTS SO THIS IS WHAT WE DO
                self._obs[agent_name] = np.expand_dims(agent_obs, axis=0)
                self.agent_mem_lists[agent_name]['obs'][0] = False #Set flag to false, obs were read and are ready to be written again

            assert all([self.agent_mem_lists[agent]['obs'][0] for agent in self.agent_mem_lists]) == False, 'all agent obs should be read by now and ready to be written'
            # print('self._obs after', self._obs)
            # print('read obs smls')
            return True
    def _scale_obs(self, obs):
        """
        This method scales obs using a running mean and std for each observation
        It assumes a shared obs space between all agents
        """

        if not hasattr(self, 'obs_rms'):
            #check if all obs are the same
            #agent_obs_names = self.agents_obs_names
            agent_0 = self.agents[0]
            agent_0_obs_names = self.agents_obs_names[self.agents[0]]
            assert all([agent_0_obs_names == self.agents_obs_names[agent] for agent in self.agents]), 'obs spaces are not the same for all agents'
            obs_shape = self.observation_spaces[agent_0].shape

            obs_names = self.agents_obs_names[agent_0]
            #for all the obs where we know the maxes and mins, we force them here:
            obs_forced_max = []
            obs_forced_min = []
            for obs_name in obs_names:
                if 'price' in obs_name:
                    obs_forced_max.append(0.1449) #ToDo: make sure this is right, and maybe autoupdate?
                    obs_forced_min.append(0.069)
                elif 'SoC' in obs_name:
                    obs_forced_max.append(1.0)
                    obs_forced_min.append(0.0)
                elif 'load' in obs_name:
                    obs_forced_max.append(None) #ToDo: log the value and put it in here
                    obs_forced_min.append(None)
                elif 'generation' in obs_name:
                    obs_forced_max.append(None) #ToDo: log the value and put it in here
                    obs_forced_min.append(None)
                elif 'netload' in obs_name:
                    obs_forced_max.append(None) #ToDo: log the value and put it in here
                    obs_forced_min.append(None)
                else:
                    obs_forced_max.append(None)
                    obs_forced_min.append(None)

            self.obs_rms = RunningMeanStdMinMax(shape=obs_shape, forced_maxes=np.array(obs_forced_max), forced_mins=np.array(obs_forced_min))
        #update the running mean and std

        for agent in self.agents:
            expaned_obs = np.expand_dims(np.array(obs[agent]), axis=0)
            self.obs_rms.update(expaned_obs)

        #clip obs for all agents
        for agent in self.agents:
            agent_obs = np.array(obs[agent])
            scaled_agent_obs = (agent_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            obs[agent] = scaled_agent_obs

        return obs
    def _scale_rewards(self, rewards):
        """
        This method scales rewards using a running mean and std
        """

        if not hasattr(self, 'rewards_rms'):
            self.rewards_rms = RunningMeanStdMinMax(shape=())

        #update the running mean and std
        for agent in self.agents:
            agent_rewards = np.expand_dims(np.array(rewards[agent]), axis=0)
            self.rewards_rms.update(agent_rewards)

        #clip rewards
        self.unscaled_rewards = rewards
        for agent in self.agents:
            agent_rewards = np.expand_dims(np.array(rewards[agent]), axis=0)
            scaled_agent_rewards = agent_rewards / np.sqrt(self.rewards_rms.var + 1e-8)
            scaled_agent_rewards = np.squeeze(scaled_agent_rewards, axis=0)
            rewards[agent] = scaled_agent_rewards

        return rewards

    def _get_rewards(self):
        self._read_reward_smls()
        return self._scale_rewards(self._rewards) if self.normalize_rewards else self._rewards
    @tenacity.retry(wait=tenacity.wait_fixed(0.01)
                          + tenacity.wait_random(0, 0.01),
                    )
    def _read_reward_smls(self):
        """
        This method cycles through the reward mem lists of the agents until they all have read the information.
        """

        # print('waiting to read reward smls')
        # agent_status = [self.agent_mem_lists[agent]['rewards'][0] for agent in self.agent_mem_lists] #We expect these to be True by now
        if not all([self.agent_mem_lists[agent]['rewards'][0] for agent in self.agent_mem_lists]):
            raise tenacity.TryAgain
        else:
            self._rewards = {}
            # print('reading reward smls')
            for i, agent_name in enumerate(self.agent_mem_lists):
                agent_reward = self.agent_mem_lists[agent_name]['rewards'][1]
                assert agent_reward == agent_reward, 'agent reward is nan of some type!'
                self._rewards[agent_name] = agent_reward
                self.agent_mem_lists[agent_name]['rewards'][0] = False #Set flag to false, obs were read

            assert all([self.agent_mem_lists[agent]['rewards'][0] for agent in self.agent_mem_lists]) == False, 'all agent rewards should be read by now and ready to be written'

            return True
        # print('read reward smls')

    @tenacity.retry(wait=tenacity.wait_fixed(0.01) + tenacity.wait_random(0, 0.01), )
    def wait_for_controller_smls(self):
        # we need the 'kill'[3] to be false
        # ToDo: technically this should be an any?
        if any([self.controller_smls[env_id]['kill'][3] for env_id in self.controller_smls]):
            raise tenacity.TryAgain
        else:
            # envs_reset = [self.controller_smls[env_id]['kill'][3] for env_id in self.controller_smls]
        # we'd expect them to be false now
            assert not any([self.controller_smls[env_id]['kill'][3] for env_id in self.controller_smls]), 'reset flag was not reset'
            return True


    @tenacity.retry(wait=tenacity.wait_fixed(0.01) + tenacity.wait_random(0, 0.01),)
    def wait_for_kill_smls(self):
        # we just set this to true, and need to wait for them to be false again
        # kill_signals_not_yet_read = [self.controller_smls[env_id]['kill'][1] for env_id in self.controller_smls] #should be set to false
        if any([self.controller_smls[env_id]['kill'][1] for env_id in self.controller_smls]):
            return tenacity.TryAgain
            # kill_signals_not_yet_read = []
            # for env_id in self.controller_smls:
            #     signal_read = self.controller_smls[env_id]['kill'][1]
            #     kill_signals_not_yet_read.append(signal_read)
        # print('trex processes killed')
        # make sure theyre all fallse nnow
        else:
            assert not any([self.controller_smls[env_id]['kill'][1] for env_id in self.controller_smls]), 'kill flag was not reset'
            return True

    def __startup_TREX_Core(self, config_name):
        #start the trex simulation and returns the multiprocessing pool object

        launch_lists = [prep_trex(config_name) for env in self.env_ids]
        augmented_launch_lists = add_envid_to_launchlist(launch_lists, self.env_ids
                                                        )

        new_launch_list = []
        for trex_launch_list in augmented_launch_lists:
            new_launch_list.extend(trex_launch_list)

        pool_size = int(mp.cpu_count()-5)  # Adjust based on needs
        pool = mp.Pool(processes=pool_size)
        trex_results = pool.map_async(run_subprocess, new_launch_list)  # this launches the TREX-Core sim in a non-blocking fashion (so it runs in the background)
        pool.close()

        return pool

    def _close_controller_smls(self):
        for env_id in self.env_ids:
            for sml_key in self.controller_smls[env_id].keys():
                self.controller_smls[env_id][sml_key].shm.close()
                self.controller_smls[env_id][sml_key].shm.unlink()

        del self.controller_smls
        print('closed simcontroller smls')
    def _close_agent_memlists(self):
        # print('closing interprocess memory', self.env_ids, flush=True)
        if len(self.env_ids) > 1:
            raise NotImplementedError('Multi-Environment TREX-Core not yet implemented')

        for agent in self.agent_mem_lists:
            for memlist in self.agent_mem_lists[agent]:
                self.agent_mem_lists[agent][memlist].shm.close()
                self.agent_mem_lists[agent][memlist].shm.unlink()

        del self.agent_mem_lists
        print('closed agent smls')

    def get_action_keys(self):
        return self.agent_action_array


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

    def ping(self):
        print('sucessfully pinged TREX env', self.env_ids, flush=True)
        return self.env_ids

    def _setup_spaces(self):
        '''
        This method sets up the action and observation spaces based on the values that are in the config
        For now, agents are assumed to be homogenous in the
        '''
        # ToDo: Separate into _setup_obs_spaces
        # ToDo: Separate into _setup_action_spaces
        # ToDo: then change action spaces to be able to be multidimensional

        self.observation_spaces = {}
        for agent in self.config['participants']:
            if self.config['participants'][agent]['trader']['type'] == 'gym_agent':
                try:
                    self.agents_obs_names[agent] = self.config['participants'][agent]['trader']['observations']
                except:
                    print('There was a problem loading the config observations')
                num_agent_obs = len(self.agents_obs_names[agent])
                agent_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, num_agent_obs,))
                self.observation_spaces[agent] = agent_obs_space


        self.action_spaces = {}
        if self.action_space_type == 'discrete':
            self.agent_action_translation = {}

        for agent in self.config['participants']:
            if self.config['participants'][agent]['trader']['type'] == 'gym_agent':

                #try:
                actions = self.config['participants'][agent]['trader']['actions']
                self.agent_action_array[agent] = [action for action in actions if actions[action]['heuristic'] == 'learned']

                min_action = [action['min'] for action in actions.values() if action['heuristic']=='learned']
                max_action = [action['max'] for action in actions.values() if action['heuristic']=='learned']
                assert len(min_action) == len(max_action) == len(self.agent_action_array[agent]), 'There was a problem with the action space'
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



    def _setup_interprocess_memory(self):
        """
        This method sets up the interprocess Shareable lists in memory for all the agents that have the
        designation gym_agent.
        Takes in nothing
        Returns: Dictionary {agent_identification_from_config : { obs_list :obs_list_object, action_list :action_list_object
        """
        from multiprocessing import shared_memory

        if len(self.env_ids) > 1:
            raise NotImplementedError('Multiple environments not yet supported')
        else:
            env_nbr = 0

        self.controller_smls = {}
        for env_id in self.env_ids:
            # reset_tuple = ('reset', False, False)
            # kill_tuple = ('kill', False, False)
            kill_list_name = 'sim_controller_kill_env_id_'+str(env_id)
            try:
                kill_list = shared_memory.ShareableList(['kill', False, 'reset', False], name=kill_list_name)
            except:
                print('found ', kill_list_name,' already in memory, attaching onto it.')
                kill_list = shared_memory.ShareableList(name=kill_list_name)
            self.controller_smls[env_id] = {'kill': kill_list}


        self.agent_mem_lists = {}
        for agent in self.agents:
            self.agent_mem_lists[agent] = {}
            # this is where we set up the shared memory object, each agent needs 2 objects actions, observations
            # todo: November 21 2022; for parallel runner there will need to be extra identifiers for sharelists to remain unique
            actions_name = agent +'_' + str(self.env_ids[env_nbr])+'_actions'
            # print('trex-env: ', 'env_id', self.env_ids[env_nbr], 'actions_name:', actions_name, flush=True)
            obs_name = agent +'_' + str(self.env_ids[env_nbr])+'_obs'
            # print('trex-env: ','env_id', self.env_ids[env_nbr], 'obs_name:', obs_name, flush=True)
            reward_name = agent+'_'+str(self.env_ids[env_nbr])+'_reward'
            # print('trex-env: ','env_id', self.env_ids[env_nbr], 'reward_name:', reward_name, flush=True)

            # Flattened gym spaces. Actions are like this:
            length_of_obs = len(self.agents_obs_names[agent]) + 1
            length_of_actions = len(self.agent_action_array[agent]) + 1

            # all smls have the follwing convention:
            # [0]: ready to be read if True, ready to be written if False
            # [1:]: the values
            try:
                actions_list = shared_memory.ShareableList([0.0]*length_of_actions, name=actions_name)
            except:
                print('found ', actions_name,' already in memory, attaching onto it.')
                actions_list = shared_memory.ShareableList(name=actions_name)
            self.agent_mem_lists[agent]['actions'] = actions_list
            # print(actions_name, flush=True)

            try:
                obs_list = shared_memory.ShareableList([0.0]*length_of_obs, name=obs_name)
            except:
                print('found ', obs_name,' already in memory, attaching onto it.')
                obs_list = shared_memory.ShareableList(name=obs_name)
            self.agent_mem_lists[agent]['obs'] = obs_list
            # print(obs_name, flush=True)
            try:
                reward_list = shared_memory.ShareableList([0.0, 0.0], name=reward_name)
            except:
                print('found ', reward_name,' already in memory, attaching onto it.')
                reward_list = shared_memory.ShareableList(name=reward_name)
            self.agent_mem_lists[agent]['rewards'] = reward_list
            # print(reward_name, flush=True)

        self._reset_interprocess_memory()

    def _force_nonblocking_sml(self): #this is a hack to make sure that the smls of the agents are not blocking any type of env command
        # print('setting flushstate for memlists') #this is  sowe can make sure that at least one step happens
        for agent in self.agent_mem_lists:
            self.agent_mem_lists[agent]['actions'][0] = True #can be read, doesnt matteranyways
            self.agent_mem_lists[agent]['obs'][0] = False #shoudl be written
            self.agent_mem_lists[agent]['rewards'][0] = False
    def _reset_interprocess_memory(self):
        for env_id in self.controller_smls:
            self.controller_smls[env_id]['kill'][1] = False # kill sim command
            self.controller_smls[env_id]['kill'][3] = False # reset sim command

        # [0]: ready to be read if True, ready to be written if False
        for agent in self.agent_mem_lists:
            self.agent_mem_lists[agent]['actions'][0] = False
            self.agent_mem_lists[agent]['obs'][0] = True
            self.agent_mem_lists[agent]['rewards'][0] = True



