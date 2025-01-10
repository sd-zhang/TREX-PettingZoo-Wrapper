# import multiprocessing as mp
# import os
import asyncio
import time
from anyio import from_thread
# import numpy as np
# import pandas as pd
import pettingzoo as pz
import threading
import random
from pubsub import pub
# import event

# import tenacity
# from _utils.trex_utils import prep_trex, run_subprocess, add_envid_to_launchlist
# from gymnasium import spaces

# import TREX_Core.runner.runner as trunner
# from mathutils import RunningMeanStdMinMax


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

    def __init__(self, config):
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
        self.render_mode = False

        #set up agent names
        self.agents = []
        self.possible_agents = self.agents #change if that ever becomes a thing

        self.market_id = 'training'
        self.client_connected = threading.Event()
        self.get_actions_event = threading.Event()
        self.get_obs_event = threading.Event()
        self.end_step_event = threading.Event()
        self.end_episode_event = threading.Event()

        self.obs = dict()
        self.reward = dict()
        self.step_count = 0
        self.poke_count = 0

        # self.agents_obs_names = {} #holds the observations of each agent

    def render(self, mode="human"):
        raise NotImplementedError
    # def async_sender(self, sender):
    #     async with sender:
    #         sender.send('event')
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
        print('ready to send actions')
        pub.sendMessage('pz_event')

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
        print('next')
        # self.get_actions_event.wait()
        # self.step_count += 1
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
        obs = dict()
        infos = dict()

        # TODO: make sure we can talk to the sim controller

        # TODO: send sim controller a signal that we're ready to start the episode

        # self.get_obs_event.wait()
        self.client_connected.wait()

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

    def ping(self):
        print('sucessfully pinged TREX env', self.env_ids, flush=True)
        return self.env_ids
