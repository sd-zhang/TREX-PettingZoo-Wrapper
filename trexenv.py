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


class TrexEnv: #ToDo: make this inherit from PettingZoo or sth else?
    """

    """
    def __init__(self,
                 config_name=None, #ToDo: add a default here
                 run_name=hash(os.times()) % 100, #ToDo: add a default here
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


        # get the n agents from the config:
        self.n_agents = 0
        for agent in self.config['participants']:
            if self.config['participants'][agent]['trader']['type'] == 'gym_agent':
                self.n_agents += 1
        assert self.n_agents > 0, 'There are no gym_agents in the config, please pick a config with at least 1 gym_agent'

        #set up agent names
        self.agent_names = [agent for agent in self.config['participants'] if
                            self.config['participants'][agent]['trader']['type'] == 'gym_agent']

        # set up general env variables
        self.episode_length = int(np.floor(self.config['study']['days'] * 24 * 60 * 60 / self.config['study']['time_step_size']) + 1) #Because the length of an episode is given by the config
        self.episode_limit = int(np.floor(self.config['study']['generations'])) #number of max episodes
        self.t_env_steps = 0
        self.episode_current = 0
        self._seed = kwargs['seed'] if 'seed' in kwargs else 0
        if 'seed' in kwargs:
            print('setting seed to', kwargs['seed'], 'BEWARE that this is not fully enforced yet!')

        self.terminated = False
        self._obs = []
        self.agent_obs_array = {} #holds the observations of each agent
        self.agent_action_array = {} #holds the action types of each agent

        # set up spaces
        self.action_space_type = kwargs['action_space_type']
        if self.action_space_type == 'discrete':
            assert 'action_space_entries' in kwargs, 'action_space_entries must be specified in the environment yaml for discrete action space'
            self.action_space_entries = kwargs['action_space_entries']
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
        #ToDo: put some info about how to get action shape shit here
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
        for i, agent in enumerate(self.agent_mem_lists):

            agent_action = actions[i]
            #make sure agent action is a antive float

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
        # should_be_false_now = [self.agent_mem_lists[agent]['actions'][0] for agent in self.agent_mem_lists]
        self.write_to_action_smls(agent_actions_decoded)

        # terminated:
        # this will need to be able to get set on the end of each generation
        terminated = [True]*self.n_agents if self.t_env_steps >= self.episode_length else [False]*self.n_agents
        self.t_env_steps += 1

        # Reward:
        # Rewards are going to have to be sent over from the gym trader, which will be able to
        # get information from the reward
        rewards = self._read_reward_values()

        # info:
        # Imma keep it as a open dictionary for now:
        info = {}
        info['rewards'] = {agent: rewards[i] for i, agent in enumerate(self.agent_names)}  # include the rewards individually for each agent
        info['terminated'] = {agent: terminated[i] for i, agent in enumerate(self.agent_names)} #include terminated for each agent

        obs = self._get_obs()

        truncated = None #FixMe: Make this do what it is supposed to I guess

        return obs, rewards, all(terminated), truncated,  info

    def write_to_action_smls(self, agent_actions_decoded):
        #we want alll agent flages to be false
        while all([self.agent_mem_lists[agent]['actions'][0] for agent in self.agent_mem_lists]): #ToDo: this should be any, nno?
            time.sleep(0.01)

        # should_be_false_now = [self.agent_mem_lists[agent]['actions'][0] for agent in self.agent_mem_lists]
        assert all([not self.agent_mem_lists[agent]['actions'][0] for agent in self.agent_mem_lists]), 'actions were not read ready'
        # now that all past actions have been consumed, we can write the new actions
        for i, agent in enumerate(self.agent_mem_lists):
            agent_action = agent_actions_decoded[agent]
            for j, action in enumerate(agent_action):
                self.agent_mem_lists[agent]['actions'][j+1] = action
            self.agent_mem_lists[agent]['actions'][0] = True #set the can be read to true

        assert all([self.agent_mem_lists[agent]['actions'][0] for agent in self.agent_mem_lists]), 'actions were not written correctly'
        return True

    def reset(self):
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
        info = {}
        return obs, info

    def wait_for_controller_smls(self):
        while all([self.controller_smls[env_id]['kill'][3] for env_id in self.controller_smls]):
            time.sleep(0.01)
            # envs_reset = [self.controller_smls[env_id]['kill'][3] for env_id in self.controller_smls]
        return True

    def close(self):
        # gets rid of the smls
        # print('WARNING: this might be unreliable ATM, check that the processes are actually killed!')

        # here we send the kill command to the sim controller and wait for the confirmation flag
        # print('sending kill signal to trex processes')
        for env_id in self.env_ids:
            self.controller_smls[env_id]['kill'][1] = True #setting the command flag to kill
        self._force_nonblocking_sml()
        # print('waiting for trex processes to die')
        # kill_signals_not_yet_read = [self.controller_smls[env_id]['kill'][1] for env_id in self.controller_smls] #should be set to false
        self.wait_for_kill_smls()

        self._close_agent_memlists()
        self._close_controller_smls()

        self.trex_pool.terminate()
        self.terminated = True

    def wait_for_kill_smls(self):
        while all([self.controller_smls[env_id]['kill'][1] for env_id in self.controller_smls]):
            time.sleep(0.01)
            # kill_signals_not_yet_read = []
            # for env_id in self.controller_smls:
            #     signal_read = self.controller_smls[env_id]['kill'][1]
            #     kill_signals_not_yet_read.append(signal_read)
        # print('trex processes killed')

    def __startup_TREX_Core(self, config_name):
        #start the trex simulation and returns the multiprocessing pool object

        launch_lists = [prep_trex(config_name) for env in self.env_ids]
        augmented_launch_lists = add_envid_to_launchlist(launch_lists, self.env_ids
                                                        )

        new_launch_list = []
        for trex_launch_list in augmented_launch_lists:
            new_launch_list.extend(trex_launch_list)

        pool_size = int(mp.cpu_count() - 2)  # Adjust based on needs
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
    def get_agent_names(self):
        return self.agent_names

    def get_state_size(self):
        """
        This method is required for gym
        This method gets the size of the state global state used by get_env_info
        Global state is the individual agents state multiplied by the number of agents.
        :return:
        The size of the state information is a single int
        """
        state_size = int(self.observation_space[-1].shape[0]) * self.n_agents
        # print("State size ", state_size)
        return state_size

    def get_action_keys(self):
        return self.agent_action_array

    def get_obs_keys(self):
        return self.agent_obs_array
    def get_obs_spaces(self):
        """
        THIS METHOD IS REQUIRED FOR GYM
        This method returns the size of each individual agents observation space, assuming homogenous observation spaces!
        FixMe: this is an epymarl leftover and will need to be adjusted at some point
        """
        agent_obs_spaces = {}
        for i, agent in enumerate(self.agent_mem_lists):
            agent_obs_spaces[agent] = self.observation_space[i]
        return agent_obs_spaces

    def get_action_spaces(self):
        """
        THIS IS REQUIRED FFOR GYM
        Returns the total number of actions that an agent could ever take, assuming homogenous observation spaces!
        FixMe: this is an epymarl leftover and will need to be adjusted at some point
        :return:
        """
        agents_action_space = {}
        for i, agent in enumerate(self.agent_mem_lists):
            agents_action_space[agent] = self.action_space[i]
        return agents_action_space

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

        for agent in self.config['participants']:
            if self.config['participants'][agent]['trader']['type'] == 'gym_agent':
                try:
                    self.agent_obs_array[agent] = self.config['participants'][agent]['trader']['observations']
                except:
                    print('There was a problem loading the config observations')
        obs_list = [spaces.Box(low=-np.inf, high=np.inf, shape=(len(obs),)) for obs in self.agent_obs_array.values()]
        obs_tuple = tuple(obs_list)
        self.observation_space = spaces.Tuple(obs_tuple)


        agent_action_spaces = []
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

                agent_action_spaces.append(agent_action_space)
                # except:
                #    print("there was a problem loading the actions")

        self.action_space = spaces.Tuple(tuple(agent_action_spaces))

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
        for agent in self.config['participants']:
            if self.config['participants'][agent]['trader']['type'] == 'gym_agent':
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
                length_of_obs = len(self.agent_obs_array[agent]) + 1
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
    def _read_obs_values(self):
        """
        This method cycles through the mem lists of the agents until they all have all read the information.
        """
        self._obs = []

        # lets make sure all agent obs are ready to be read
        # agent_status = [self.agent_mem_lists[agent]['obs'][0] for agent in self.agent_mem_lists] #We expect these to be True by now
        while not all([self.agent_mem_lists[agent]['obs'][0] for agent in self.agent_mem_lists]):
            time.sleep(0.01)  # wait for 1ms
            # try:
            #     agent_status = [self.agent_mem_lists[agent]['obs'][0] for agent in self.agent_mem_lists]
            # except:
            #     pass
        #after having made sure all are true, we can read the values
        for i, agent_name in enumerate(self.agent_mem_lists):
        # agent is a dictionary 'obs', 'actions', 'rewards'

            agent_obs = [self.agent_mem_lists[agent_name]['obs'][j] for j in range(1,len(self.agent_mem_lists[agent_name]['obs']))] #get the values, THIS SEEMS TO WORK WITH SHAREABLE LISTS SO THIS IS WHAT WE DO
            self._obs.append(agent_obs)
            self.agent_mem_lists[agent_name]['obs'][0] = False #Set flag to false, obs were read and are ready to be written again

        assert all([self.agent_mem_lists[agent]['obs'][0] for agent in self.agent_mem_lists]) == False, 'all agent obs should be read by now and ready to be written'
        # print('self._obs after', self._obs)


    def _read_reward_values(self):
        """
        This method cycles through the reward mem lists of the agents until they all have read the information.
        """
        rewards_t = []

        # agent_status = [self.agent_mem_lists[agent]['rewards'][0] for agent in self.agent_mem_lists] #We expect these to be True by now
        while not all([self.agent_mem_lists[agent]['rewards'][0] for agent in self.agent_mem_lists]):
            time.sleep(0.01)
            # agent_status = [self.agent_mem_lists[agent]['rewards'][0] for agent in self.agent_mem_lists]

        for i, agent_name in enumerate(self.agent_mem_lists):
            rewards_t.append(self.agent_mem_lists[agent_name]['rewards'][1])
            self.agent_mem_lists[agent_name]['rewards'][0] = False #Set flag to false, obs were read

        assert all([self.agent_mem_lists[agent]['rewards'][0] for agent in self.agent_mem_lists]) == False, 'all agent rewards should be read by now and ready to be written'

        rewards_t = [float(reward) if reward is not None else np.nan for reward in rewards_t]
        return rewards_t

    def get_avail_actions(self):
        """
        #FixMe: needs fxing for the new action space
        This method will return a list of list that gives the available actions
        return: avail_actions -> list of [1]* n_agents

        """
        # For now, all actions are availiable at all time
        agent_action_space = self.action_space[-1]
        if self.action_space_type == 'continuous':
            action_space_shape = agent_action_space.shape
        elif self.action_space_type == 'discrete':
            action_space_shape = agent_action_space.n
        else:
            print('did not recognize action space type', self.action_space_type)
            raise NotImplementedError

        ACTIONS = [1]*action_space_shape
        avail_actions = [ACTIONS] *self.n_agents

        return avail_actions

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
        self._read_obs_values()
        return self._obs

