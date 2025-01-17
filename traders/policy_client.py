# from clients.participants.participants import Residential
# erp stands for Execute Remote Policy
# from TREX_env.utils.sml_utils import read_flag_x_times
from TREX_Core.utils.metrics import Metrics
import asyncio
# import serialize
import importlib
import numpy as np
import datetime

#ToDo: make all actions learnable (ask price, ask quan, bid_price, bid_quan, battery)
#ToDo: make the agent not wait until reward is a number, just pass through!


class Trader:
    """
    Class: Trader
    This class implements the gym compatible trader that is used in tandem with EPYMARL TREXEnv.

    """
    def __init__(self, **kwargs):
        """
        Initializes the trader using the parameters in the TREX_Core config that was selected.
        In particular, this sets up the connections to the shared lists that are established in EPYMARL TREXEnv or any
        other process that seeks to interact with TREX. These lists need to be already initialized before the gym trader
        attempts to connect with them.

        params: kwargs -> dictionary created from the config json file in TREX_Core.configs
        """
        # Some util stuffies
        # print('GOT TO THE GYM_AGENT INIT')
        # self.t_episode_steps = 0 # number of actions taken
        self.__participant = kwargs['trader_fns']
        self.__client = self.__participant['client']
        self.actions_dict_t = dict()
        # self.status = {
        #     'weights_loading': False,
        #     'weights_loaded': False,
        #     'weights_saving': False,
        #     'weights_saved': True
        # }

        ##### Setup the shared memory names based on config #####
        # #ToDo: find a way to add the env number here
        # self.name = self.__participant['id']
        #
        # assert 'env_id' in kwargs, 'Expected to find env_id in kwargs, needed to connect to shared memory manager'
        # env_info = kwargs['env_id']
        # #make sure 'env_id', 'smm_hash', 'smm_address' and 'smm_port' in kwargs
        #
        # env_id = kwargs['env_id']
        # self.action_list_name = self.name+'_' + str(env_id) +  '_actions'
        # # print('Gym agent', self.name, 'action list name', self.action_list_name, flush=True)
        # self.observation_list_name = self.name+'_' + str(env_id) +  '_obs'
        # # print('Gym agent', self.name, 'observation list name', self.observation_list_name, flush=True)
        # self.reward_list_name = self.name+'_' + str(env_id) +  '_reward'
        # print('Gym agent', self.name, 'reward list name', self.reward_list_name, flush=True)
        ''' 
        Shared lists get initialized on TREXENV side, so all that the agents have to do is connect to their respective 
        observation and action lists. Agents dont have to worry about making the actions pretty, they just have to send
        them into the buffer. 
        '''

        # self._check_sharedmemory()
        # self.shared_list_action = shared_memory.ShareableList(name=self.action_list_name)
        # self.shared_list_observation = shared_memory.ShareableList(name=self.observation_list_name)
        # self.shared_list_reward = shared_memory.ShareableList(name=self.reward_list_name)
        self.actions = dict()
        self.actions_event = asyncio.Event()

        #find the right default behaviors from kwargs['default_behaviors']
        self.observation_variables = kwargs['observations']

        #ToDo - Daniel - Think about a nicer way of doing this
        #decode actions, load heuristics if necessary
        # self.allowed_actions = kwargs['actions']
        # self.learned_actions = {}
        # self.heuristic_actions = {}
        # for action in kwargs['actions']:
        #
        #     # Deprecated
        #     if kwargs['actions'][action]['heuristic'] == 'learned':
        #         self.learned_actions[action] = None
        #     elif kwargs['actions'][action]['heuristic'] == 'netload':
        #         assert 'quantity' in action, 'Netload heuristic only works for quantity actions'
        #         self.heuristic_actions[action] = None
        #     elif kwargs['actions'][action]['heuristic'] == 'fixed':
        #         assert 'price' in action, 'Fixed heuristic only works for price actions'
        #         self.heuristic_actions[action] = None
        #     else:
        #         raise NotImplementedError('Only learned actions and netload quantities are supported in the gym agent. Please reassign ', action, 'to learned', flush=True)
        #
        # assert 'storage' in self.learned_actions, 'storage not in learned actions, WTF?'



        # TODO: Find out where the action space will be defined: I suspect its not here
        # Initialize the agent learning parameters for the agent (your choice)
        # self.bid_price = kwargs['bid_price'] if 'bid_price' in kwargs else None
        # self.ask_price = kwargs['ask_price'] if 'ask_price' in kwargs else None

        ####### Metrics tracking initialization ########
        self.track_metrics = kwargs['track_metrics'] if 'track_metrics' in kwargs else False
        self.metrics = Metrics(self.__participant['id'], track=self.track_metrics)
        if self.track_metrics:
            self.__init_metrics()

        ###### Reward function intialization from config #######
        reward_function = kwargs['reward_function'] if 'reward_function' in kwargs else None
        if reward_function:
            try:
                Reward = importlib.import_module('rewards.' + reward_function).Reward
            except ImportError:
                Reward = importlib.import_module('TREX_Core.rewards.' + reward_function).Reward
            self._rewards = Reward(
                self.__participant['timing'],
                self.__participant['ledger'],
                self.__participant['market_info'])



       #  print('init done')

    # @tenacity.retry(wait=tenacity.wait_fixed(3))
    # def _check_sharedmemory(self):
    #     """
    #     This method checks if all the shared memory arrays have been create by the EPYMARL process before having them
    #     initialized by the gym_agents.
    #     """
    #
    #     shared_action_list = shared_memory.ShareableList(name=self.action_list_name)
    #
    #     shared_observation_list = shared_memory.ShareableList(name=self.observation_list_name)
    #
    #     shared_reward_list = shared_memory.ShareableList(name=self.reward_list_name)
    #
    #     return True

    def __init_metrics(self):
        import sqlalchemy
        '''
        Pretty self explanitory, this method resets the metric lists in 'agent_metrics' as well as zeroing the metrics dictionary. 
        '''
        self.metrics.add('timestamp', sqlalchemy.Integer)
        self.metrics.add('actions_dict', sqlalchemy.JSON)
        self.metrics.add('next_settle_load', sqlalchemy.Integer)
        self.metrics.add('next_settle_generation', sqlalchemy.Integer)

        # if self.battery:
        #     self.metrics.add('battery_action', sqlalchemy.Integer)
        #     self.metrics.add('state_of_charge', sqlalchemy.Float)

    # Core Functions, learn and act, called from outside
    async def pre_process_obs(self):
        timing = self.__participant['timing']

        # self.__timing.update({
        #     'timezone': self.timezone,
        #     # 'timezone': message['timezone'],
        #     'duration': duration,
        #     'last_round': (start_time - duration, start_time),
        #     'current_round': (start_time, end_time),
        #     'last_settle': (start_time + duration * (close_steps - 1), start_time + duration * close_steps),
        #     'next_settle': (start_time + duration * close_steps, start_time + duration * (close_steps + 1)),
        #     'stale_round': (start_time - duration * 10, start_time - duration * 9)
        # })

        # print('entered preprocessing')

        # we need to make sure that the observation get put into the right order
        obs_t_dict = {key: None for key in self.observation_variables}

        if 'reward_time_lag' in self.observation_variables:
            # calculations for reward time offset
            n_rounds_act_to_r = (timing['next_settle'][0] - timing['last_settle'][0]) / timing['round_duration']
            n_rounds_obs_to_act = (timing['next_settle'][0] - timing['next_settle'][0]) / timing['round_duration'] # this is a remainder from earlier timing considerations, kept in for future flexibility
            n_rounds_obs_to_r = n_rounds_obs_to_act + n_rounds_act_to_r
            n_rounds_current_to_r = (timing['next_settle'][0] - timing['current_round'][0]) / timing['round_duration'] + n_rounds_obs_to_r
            obs_t_dict['reward_time_lag'] = n_rounds_current_to_r


        if 'generation_now' in self.observation_variables or 'load_now' in self.observation_variables or 'netload_now' in self.observation_variables:
            gen_now, load_now = await self.__participant['read_profile'](timing['current_round'])
            if 'generation_now' in self.observation_variables:
                obs_t_dict['generation_now'] = gen_now
            if 'load_now' in self.observation_variables:
                obs_t_dict['load_now'] = load_now
            if 'netload_now' in self.observation_variables:
                obs_t_dict['netload_now'] = load_now - gen_now

        if "t_settle" in self.observation_variables:
            obs_t_dict['t_settle'] = timing['next_settle'][0]
        if 'generation_settle' in self.observation_variables or 'load_settle' in self.observation_variables or 'netload_settle' in self.observation_variables:
            gen_settle, load_settle = await self.__participant['read_profile'](timing['next_settle'])
            if 'generation_settle' in self.observation_variables:
                obs_t_dict['generation_settle'] = gen_settle
            if 'load_settle' in self.observation_variables:
                obs_t_dict['load_settle'] = load_settle
            if 'netload_settle' in self.observation_variables:
                obs_t_dict['netload_settle'] = load_settle - gen_settle

        if "t_deliver" in self.observation_variables:
            obs_t_dict['t_deliver'] = timing['next_settle'][0] + timing['round_duration']
        if 'generation_deliver' in self.observation_variables or 'load_deliver' in self.observation_variables or 'netload_deliver' in self.observation_variables:
            gen_deliver, load_deliver = await self.__participant['read_profile'](timing['next_deliver'])
            if 'generation_deliver' in self.observation_variables:
                obs_t_dict['generation_deliver'] = gen_deliver
            if 'load_deliver' in self.observation_variables:
                obs_t_dict['load_deliver'] = load_deliver
            if 'netload_deliver' in self.observation_variables:
                obs_t_dict['netload_deliver'] = load_deliver - gen_deliver

        if 'SoC_settle' in self.observation_variables:
            storage_schedule = await self.__participant['storage']['check_schedule'](timing['next_settle'])
            soc = storage_schedule[timing['next_settle']]['projected_soc_end']
            obs_t_dict['SoC_settle'] = soc

        if 'actual_netload_now' in self.observation_variables:
            # get gen and load for current round
            gen_now, load_now = await self.__participant['read_profile'](timing['current_round'])
            # get battery schedule for current round
            storage_schedule = await self.__participant['storage']['check_schedule'](timing['current_round'])
            storage_schedule = storage_schedule[timing['current_round']]
            storage = storage_schedule['energy_scheduled']


            # calculate netload
            netload_now = load_now - gen_now + storage
            obs_t_dict['actual_netload_now'] = netload_now

        # ToDo: revert this once market issues are solved
        # collect the settle stats if necessary
        settle_stats = self.__participant['market_info']['settle_stats']
        participant = self.__participant
        if 'settled_time' in settle_stats:
            ts_settle_stats = settle_stats['settled_time']
            ts_settle_stats = str(tuple(ts_settle_stats))

            if ts_settle_stats in participant['market_info']:
                grid_stats = participant['market_info'][ts_settle_stats]
                grid_sell_price = grid_stats['grid']['sell_price']
                grid_buy_price = grid_stats['grid']['buy_price']
                assert grid_buy_price >= grid_sell_price, 'grid buy price should be higher than grid sell price'
            else:
                print('grid stats not available for participant', participant['id'], 'at timestep', timing['next_settle'], flush=True)
                # raise ValueError('Grid stats not available')
        else:
            print('settle stats not available for participant', participant['id'], 'at timestep', timing['next_settle'], flush=True)
            # raise ValueError('Settle stats not available')

        if 'avg_bid_price' in self.observation_variables:
            obs_t_dict['avg_bid_price'] = 0.068
        if 'avg_ask_price' in self.observation_variables:
            obs_t_dict['avg_ask_price'] = 0.1449

        if 'total_bid_quantity' in self.observation_variables:
            obs_t_dict['total_bid_quantity'] = 0
        if 'total_ask_quantity' in self.observation_variables:
            obs_t_dict['total_ask_quantity'] = 0


        # print('settle stats', settle_stats, flush=True)
        for obs in self.observation_variables:
            if obs in settle_stats:
                obs_t_dict[obs] = self.__participant['market_info']['settle_stats'][obs]

        if "daytime_sin" in self.observation_variables or "daytime_cos" in self.observation_variables:
            t_now = timing['next_settle'][0] #this should be a timestamp, so we convert it using datetime into the hour format
            t_now = datetime.datetime.fromtimestamp(t_now)
            t_now = t_now.hour + t_now.minute/60 + t_now.second/3600
            rad = 2 * np.pi * t_now / 24
            # print('t_now', t_now, flush=True)
            if 'daytime_sin' in self.observation_variables:
                sin = np.sin(rad).tolist()
                obs_t_dict['daytime_sin'] = sin
            if 'daytime_cos' in self.observation_variables:
                cos = np.cos(rad).tolist()
                obs_t_dict['daytime_cos'] = cos

        if "yeartime_sin" in self.observation_variables or "yeartime_cos" in self.observation_variables:
            t_now = timing['next_settle'][0] #this should be a timestamp, so we convert it using datetime into the hour format
            # t_now = datetime.datetime.fromtimestamp(t_now) #ToDo: stuff is in seconds rn anyways
            rad = 2 * np.pi * t_now / (365 * 24 * 60 * 60)
            # print('t_now', t_now, flush=True)
            if 'yeartime_sin' in self.observation_variables:
                sin = np.sin(rad).tolist()
                obs_t_dict['yeartime_sin'] = sin
            if 'yeartime_cos' in self.observation_variables:
                cos = np.cos(rad).tolist()
                obs_t_dict['yeartime_cos'] = cos


        obs_list = [obs_t_dict[obs] for obs in self.observation_variables]
        # print('obs_list', obs_list, flush=True)
        return obs_list

    async def send_obs_rewards(self):
        # obs = np.random.rand(1, 11)[0].tolist()
        # reward = np.random.rand()
        try:
            obs = await self.pre_process_obs()
        except:
            print('no obs yet probably step 0')
            obs = [0] * len(self.observation_variables)
        try:
            reward = await self._rewards.calculate()
        except:
            print('no reward yet probably step 0')
            reward = 0

        data = {'participant_id': self.__participant['id'], 'obs': obs, 'reward': reward}
        self.__client.publish('/'.join([self.__participant['market_id'], 'algorithm', 'obs_rewards']), data, qos=2)

        # self.__client.publish('/'.join([self.market_id, 'simulation', 'end_turn']), self.participant_id,
        #                       user_property=('to', self.market_sid))

    async def act(self, **kwargs):
        """


        """

        '''
        actions are none so far
        ACTIONS ARE FOR THE NEXT settle!!!!!

        actions = {
            'bess': {
                time_interval: scheduled_qty
            },
            'bids': {
                time_interval: {
                    'quantity': qty,
                    'price': dollar_per_kWh
                }
            },
            'asks': {
                source:{
                     time_interval: {
                        'quantity': qty,
                        'price': dollar_per_kWh?
                     }
                 }
            }
        sources inclued: 'solar', 'bess'
        Actions in the shared list 
        [bid price, bid quantity, solar ask price, solar ask quantity, bess ask price, bess ask quantity]
        }
        '''
        # print("in agent.act")
        ##### Initialize the actions

        # Timing information
        # timing = self.__participant['timing']
        # self.current_round = self.__participant['timing']['current_round']
        # self.round_duration = self.__participant['timing']['duration']
        # self.next_round = (self.current_round[0]+self.round_duration, self.current_round[1]+self.round_duration)
        # self.last_settle = self.__participant['timing']['last_settle']
        # self.next_settle = self.__participant['timing']['next_settle']
        # self.next_deliver = (self.next_settle[0] + self.round_duration, self.next_settle[1] + self.round_duration)
        # self.last_round = timing['last_round']
        #
        # obs_t = await self.pre_process_obs()
        # # print('Agent Observations', obs_t, flush=True)
        #
        # #### Send rewards into reward buffer:
        # reward = await self._rewards.calculate()
        # print(reward, 'at', self.t_episode_steps, 'in agent', flush=True)
        # print('reward', reward, flush=True)
        #if we get rewards we pass obs, etc to GYM
        # this is not the optimal way of doing this but it is going to allow us to keep everything outside of gym clean
        #ToDO: all - look for better solutions
        '''
        #########################################################################
        it is here that we wait for the action values to be written from Gym
        #########################################################################
        '''

        # if reward is not None: Deprecated behavior
    #     await self.write_obs_to_sml(obs_t)
    # #
    #     await self.write_r_to_sml(reward)
    # #
    #     await self.read_action_from_sml()
        # await self.get_heuristic_actions(ts_act=ts_act) #Deprecated
        # wait for the actions to come from EPYMARL

        # actions come in with a set order, they will need to be split up
        # send msg to request actions
        # self.actions_event.clear()
        # await asyncio.sleep(np.random.random())
        self.__client.publish('/'.join([self.__participant['market_id'], 'algorithm', 'get_actions']),
                              self.__participant['id'], qos=2)
                              # user_property=('to', self.market_sid))

        await self.actions_event.wait()
        self.actions_event.clear()
        # action_dict_t =

        #     }
        # print('actions in agent', action_dict_t, flush=True)
        # if self.track_metrics:
        #     await asyncio.gather(
        #         self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
        #         self.metrics.track('actions_dict', action_dict_t),
        #         # self.metrics.track('next_settle_load', load),
        #         # self.metrics.track('next_settle_generation', generation)
        #         )
        #
        #     await self.metrics.save(10000)
        # print("gym agent action_dict_t", action_dict_t)

        # self.t_episode_steps += 1
        return self.actions_dict_t

    async def get_actions_return(self, message):
        # await self.decode_actions(message)
        self.actions_dict_t = message
        # self.actions_dict_t = await self.decode_actions(message)
        # print(self.__participant['id'], self.actions_dict_t)
        self.actions_event.set()
        # pass

    async def step(self):
        # actions must come in the following format:
        # actions = {
        #     'bess': {
        #         time_interval: scheduled_qty
        #     },
        #     'bids': {
        #         time_interval: {
        #             'quantity': qty,
        #             'price': dollar_per_kWh
        #         }
        #     },
        #     'asks' {
        #         source: {
        #             time_interval: {
        #                 'quantity': qty,
        #                 'price': dollar_per_kWh?
        #             }
        #         }
        #     }
        #
        # print(self.__participant['id'], 'sending obs/rewards')
        await self.send_obs_rewards()
        # print(self.__participant['id'], 'waiting for actions')
        next_actions = await self.act()
        # print(self.__participant['id'], 'got actions')
        # print(self.__participant['id'], 'waiting for next round')
        return next_actions

    async def reset(self, **kwargs):
        self.__participant['ledger'].reset()
        # self.actions.clear()
        # self.t_episode_steps = 0
        return True

    async def decode_actions(self):
        """
        We decode the external actions into the action dict to be fed into the market.
        As we are expecting an external heuristic to be feeding the gym agent, we want:
        1. A bid price and quantity
        2. A solar ask price and quantity
        3. A storage action

        """

        #ToDo: check if self.next_settle outside of our simulaion
        gen_settle, load_settle = await self.__participant['read_profile'](self.next_settle)
        gen_deliver, load_deliver = await self.__participant['read_profile'](self.next_deliver)


        assert 'storage' in self.learned_actions, 'storage neither in learned actions'
        target_storage_charge = -(load_settle - gen_settle) + self.learned_actions['storage'] *5000  #FixMe: find a way to implement this into the PPO agent instead
        # target_storage_charge = min(max(target_storage_charge, -5000), 5000)
        storage_schedule = await self.__participant['storage']['check_schedule'](self.next_settle)
        # print(storage_schedule)
        min_max_storage_charge = storage_schedule[self.next_settle]['energy_potential']
        storage_charge = min(max(target_storage_charge, min_max_storage_charge[0]), min_max_storage_charge[1])

        if 'price_bid' not in self.learned_actions:
            assert 'price_bid' in self.heuristic_actions, 'price_bid neither in learned actions nor in heuristic actions'
            self.heuristic_actions['price_bid'] = 0.11 #ToDo: check that this has no influence on the simple Market
        else:
            assert self.learned_actions['price_bid'] is not None
        price_bid = self.learned_actions['price_bid'] if 'price_bid' in self.learned_actions else self.heuristic_actions['price_bid']

        if 'price_ask' not in self.learned_actions:
            assert 'price_ask' in self.heuristic_actions, 'price_ask neither in learned actions nor in heuristic actions'
            self.heuristic_actions['price_ask'] = 0.11
        else:
            assert self.learned_actions['price_ask'] is not None
        price_ask = self.learned_actions['price_ask'] if 'price_ask' in self.learned_actions else self.heuristic_actions['price_ask']

        net_load = load_settle - gen_settle + storage_charge #ToDo:make sure this is the right way around!
        if 'quantity_bid' not in self.learned_actions:
            assert 'quantity_bid' in self.heuristic_actions, 'quantity_bid neither in learned actions nor in heuristic actions'
            self.heuristic_actions['quantity_bid'] = max(0, net_load)  #FixMe: revert
        else:
            assert self.learned_actions['quantity_bid'] is not None
        quantity_bid = self.learned_actions['quantity_bid'] if 'quantity_bid' in self.learned_actions else self.heuristic_actions['quantity_bid']

        if 'quantity_ask' not in self.learned_actions:
            assert 'quantity_ask' in self.heuristic_actions, 'quantity_ask neither in learned actions nor in heuristic actions'
            self.heuristic_actions['quantity_ask'] = max(0, -net_load) #FixMe: revert
        else:
            assert self.learned_actions['quantity_ask'] is not None
        quantity_ask = self.learned_actions['quantity_ask'] if 'quantity_ask' in self.learned_actions else self.heuristic_actions['quantity_ask']


        decoded_action = dict()
        decoded_action['bids'] = { str(self.next_settle): {'quantity': quantity_bid, 'price': price_bid}}
        decoded_action['asks'] = {'solar': { str(self.next_settle): {'quantity': quantity_ask, 'price': price_ask}}}
        decoded_action['bess'] = { str(self.next_settle): storage_charge }

        return decoded_action
