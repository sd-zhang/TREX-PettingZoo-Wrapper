import multiprocessing.managers

from TREX_env._utils.sml_utils import read_flag_x_times
from TREX_env._utils.trex_utils import prep_trex, run_subprocess, add_envid_to_launchlist
import TREX_Core._utils.runner
from gymnasium import spaces
import numpy as np
import os
import multiprocessing as mp


class TrexEnv: #ToDo: make this inherit from PettingZoo or sth else?
    """

    """
    def __init__(self,
                 config_name=None, #ToDo: add a default here
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
        self.episode_length = self.config['study']['days'] * 24 * 60 * 60 / self.config['study']['time_step_size'] #Because the length of an episode is given by the config
        self.episode_limit =  int(np.floor(self.config['study']['generations'])) #number of max episodes
        self.t_env_steps = 0
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
        self.mem_lists = self._setup_interprocess_memory()

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
        for i, agent in enumerate(self.mem_lists):


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
        agents_actions_consumed = [False for _ in range(self.n_agents)]
        while not all(agents_actions_consumed):
            for i, agent in enumerate(self.mem_lists):
                if self.mem_lists[agent]['actions'][0] == False: #agent action not yet consumed
                    agent_action = agent_actions_decoded[agent]
                    for j, action in enumerate(agent_action): #FixMe: for some reason slicing here doesnt want to work
                        self.mem_lists[agent]['actions'][j+1] = action
                    self.mem_lists[agent]['actions'][0] = True
                    agents_actions_consumed[i] = True



        # terminated:
        # this will need to be able to get set on the end of each generation
        if self.t_env_steps < self.episode_length:
            terminated = [0.0]*self.n_agents
        else:
            terminated = [1.0] * self.n_agents
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

    def reset(self):
        '''
        https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
        This method resets the trex environment.
        The reset would have to be able to kill all the TREX processes,
        then reboot them all and have the gym traders reconnect to the shared memory objects.
        TODO Peter: November 30, 2022; This is going to need to reset the TREX instance
        '''
        self.t_env_steps = 0
        print('Resetting TREX atm not resetting TREX-Core, it is only a t_env_steps reset!') #ToDo: get steven to write the appropriate feature for Core
        # print('reset trex_env', self.env_ids, flush=True)
        obs = self._get_obs()
        info = {}
        return obs, info

    def close(self):
        # gets rid of the smls
        print('closing trex_env', self.env_ids, flush=True)
        print('WARNING: this might be unreliable ATM, check that the processes are actually killed!')
        self.terminated = True
        self.trex_pool.terminate()
        self._close_interprocess_memory()

    def __startup_TREX_Core(self, config_name):
        #start the trex simulation and returns the multiprocessing pool object

        launch_lists = [prep_trex(config_name) for env in self.env_ids]
        augmented_launch_lists = add_envid_to_launchlist(launch_lists, self.env_ids
                                                        )

        new_launch_list = []
        for trex_launch_list in augmented_launch_lists:
            new_launch_list.extend(trex_launch_list)

        pool_size = int(mp.cpu_count() / 2)  # Adjust based on needs
        pool = mp.Pool(processes=pool_size)
        trex_results = pool.map_async(run_subprocess, new_launch_list)  # this launches the TREX-Core sim in a non-blocking fashion (so it runs in the background)
        pool.close()

        return pool

    def _close_interprocess_memory(self):
        # print('closing interprocess memory', self.env_ids, flush=True)
        if len(self.env_ids) > 1:
            raise NotImplementedError('Multi-Environment TREX-Core not yet implemented')

        for agent in self.mem_lists:
            for memlist in self.mem_lists[agent]:
                self.mem_lists[agent][memlist].shm.close()
                self.mem_lists[agent][memlist].shm.unlink()

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

    def get_obs_spaces(self):
        """
        THIS METHOD IS REQUIRED FOR GYM
        This method returns the size of each individual agents observation space, assuming homogenous observation spaces!
        FixMe: this is an epymarl leftover and will need to be adjusted at some point
        """
        agent_obs_spaces = {}
        for i, agent in enumerate(self.mem_lists):
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
        for i, agent in enumerate(self.mem_lists):
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
        agents_smls = {}
        if len (self.env_ids) > 1:
            raise NotImplementedError('Multiple environments not yet supported')
        else:
            env_nbr = 0


        for agent in self.config['participants']:
            if self.config['participants'][agent]['trader']['type'] == 'gym_agent':
                # this is where we set up the shared memory object, each agent needs 2 objects actions, observations
                # todo: November 21 2022; for parallel runner there will need to be extra identifiers for sharelists to remain unique
                actions_name = agent +'_' + str(self.env_ids[env_nbr])+'_actions'
                # print('trex-env: ', 'env_id', self.env_ids[env_nbr], 'actions_name:', actions_name, flush=True)
                obs_name = agent +'_' + str(self.env_ids[env_nbr])+'_obs'
                # print('trex-env: ','env_id', self.env_ids[env_nbr], 'obs_name:', obs_name, flush=True)
                reward_name = agent+'_'+str(self.env_ids[env_nbr])+'_reward'
                # print('trex-env: ','env_id', self.env_ids[env_nbr], 'reward_name:', reward_name, flush=True)

                # Flattened gym spaces. Actions are like this:
                # [bid price, bid quantity, solar ask price, solar ask quantity, bess ask price, bess ask quantity]
                length_of_obs = len(self.agent_obs_array[agent]) + 1
                length_of_actions = len(self.agent_action_array[agent]) + 1

                # observations = [0.0] * length_of_obs

                actions_list = shared_memory.ShareableList([0.0]*length_of_actions, name=actions_name)
                # print(actions_name, flush=True)
                obs_list = shared_memory.ShareableList([0.0]*length_of_obs, name=obs_name)
                # print(obs_name, flush=True)
                reward_list = shared_memory.ShareableList([0.0, 0.0], name=reward_name)
                # print(reward_name, flush=True)

                agents_smls[agent] = {
                    'obs':  obs_list,
                    'actions': actions_list,
                    'rewards': reward_list
                }

        return agents_smls

    def _read_obs_values(self):
        """
        This method cycles through the mem lists of the agents until they all have all read the information.
        """
        agent_status = [False] * self.n_agents

        self._obs = []
        while not all(agent_status):
            #print('Memlist before', self.mem_lists, flush=True)
            for i, agent_name in enumerate(self.mem_lists):
            # agent is a dictionary 'obs', 'actions', 'rewards'
                # if self.mem_lists[agent_name]['obs'][0] and not agent_status[i]: #if the flag is set and wwe have not read the values already
                if not agent_status[i]:
                    obs_ready = read_flag_x_times(self.mem_lists[agent_name]['obs'], name='obs')
                    if obs_ready:
                        agent_obs = [self.mem_lists[agent_name]['obs'][j] for j in range(1,len(self.mem_lists[agent_name]['obs']))] #get the values, THIS SEEMS TO WORK WITH SHAREABLE LISTS SO THIS IS WHAT WE DO
                        self._obs.append(agent_obs)
                        agent_status[i] = True #set the
                        # self.mem_lists[agent_name]['obs'][0] = False #Set flag to false

        # print('self._obs after', self._obs)

    def _read_reward_values(self):
        """
        This method cycles through the reward mem lists of the agents until they all have read the information.
        """
        # encode the agents as one hot vectors:
        agent_status = [False] * self.n_agents
        rewards_t = []
        while not all(agent_status):
            for i, agent_name in enumerate(self.mem_lists):
                # agent is a dictionary 'obs', 'actions', 'rewards'
                # if self.mem_lists[agent_name]['rewards'][0] and not agent_status[i]: #if the flag is set and wwe have not read the values already
                if not agent_status[i]:
                    reward_ready = read_flag_x_times(self.mem_lists[agent_name]['rewards'], name='rewards')
                    if reward_ready:
                        # rewards are good to read
                        rewards_t.append(self.mem_lists[agent_name]['rewards'][1])
                        agent_status[i] = True
                        # self.mem_lists[agent_name]['rewards'][0] = False


        #turn the reward into a float unless it is none, then turn it into a np.nan
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

