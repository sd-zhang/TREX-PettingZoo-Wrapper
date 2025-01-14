import json
import datetime
import numpy as np
# import time

# import os
from cuid2 import Cuid
import paho.mqtt.client as mqtt

import random
from TREX_Core._clients.markets.ns_common import NSDefault

import supersuit as ss
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
from _utils.ppo_recurrent_custom import RecurrentPPO
from _utils.custom_Monitor import Custom_VecMonitor
from _utils.custom_vec_normalize import VecNormalize
from _utils.schedule import exponential_schedule
import commentjson




class Client:

    def __init__(self, server_address, config):
        # Initialize client-server data
        self.server_address = server_address
        self.client = mqtt.Client(client_id=Cuid(length=10).generate(),
                                  callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        # self.config = config  # temporary
        self.config = config
        self.market_id = 'training'

        # TODO: initialize trex env

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tboard_logdir = f"runs/{current_time}"
        Env = importlib.import_module('trex_env')
        self.env = Env.Env(client=self.client,
                           config=self.config,
                           action_space_type='continuous',
                           action_space_entries=None,
                           baseline_offset_rewards=True,
                           one_hot_encode_agent_ids=True,
                           )

        num_bits = self.env.num_one_hot_bits
        agents_obs_keys = self.env.agents_obs_names
        # episode_length = env.episode_length
        agents = list(key for key in agents_obs_keys.keys())
        agents_obs_keys = agents_obs_keys[agents[0]]

        env = ss.pettingzoo_env_to_vec_env_v1(self.env)
        # print(self.env.actions)
        env2 = SB3VecEnvWrapper(env)
        env3 = Custom_VecMonitor(env2, filename=tboard_logdir, obs_names=agents_obs_keys)
        self.final_env = VecNormalize(env3,
                                norm_obs=True,
                                norm_reward=False,
                                num_bits=num_bits, #Daniel: The num bits corresponds to a 1hot encoding of agent id for our case, this is iIrc slight abuse but works
                                clip_obs=1e6, clip_reward=1e6, gamma=0.99, epsilon=1e-08)

        # TODO: initialize algorithm
        # Model = importlib.import_module('sb3_contrib').RecurrentPPO
        # self.model = Model()
        self.model = RecurrentPPO('MlpLstmPolicy',
                                  self.final_env,
                                  verbose=0,
                                  use_sde=False,
                                  tensorboard_log=tboard_logdir,
                                  device="cuda",
                                  # if you start having memory issues, moving to cpu might help at the expense of speed
                                  n_epochs=4,
                                  # target_kl=0.05,
                                  learning_rate=exponential_schedule(0.0003, 15, 0.69),
                                  n_steps=9 * 24,
                                  stats_window_size=1,
                                  ent_coef=0.00,
                                  # policy_kwargs=policy_dict,
                                  batch_size=3 * 24,
                                  recalculate_lstm_states=True,
                                  rewards_shift=2,
                                  self_bootstrap_dones=True,
                                  )

        # #TODO: pass NS stuff to super class
        # self.ns = NSDefault(self.env)

    def on_connect(self, client, userdata, flags, reason_code, properties):
        # market_id = self.config['market']['id']  # temporary

        print('connected')
        # pass
        # market_id = self.market.market_id
        # print('Connected market', market_id)
        # client.subscribe("/".join([market_id]), qos=0)
        # # client.subscribe("/".join([market_id, '+']), qos=0)
        # client.subscribe("/".join([market_id, market_id]), qos=0)
        # client.subscribe("/".join([market_id, 'join_market']), qos=0)
        # client.subscribe("/".join([market_id, 'bid']), qos=0)
        # client.subscribe("/".join([market_id, 'ask']), qos=0)
        # client.subscribe("/".join([market_id, 'settlement_delivered']), qos=0)
        # client.subscribe("/".join([market_id, 'meter']), qos=0)
        #
        # # client.subscribe("/".join([market_id, 'simulation', '+']), qos=0)
        client.subscribe("/".join([self.market_id, 'algorithm', 'get_actions']), qos=2)
        client.subscribe("/".join([self.market_id, 'algorithm', 'obs_rewards']), qos=2)
        # client.subscribe("/".join([market_id, 'algorithm', 'rewards']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'start_generation']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'end_generation']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'end_simulation']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'is_market_online']), qos=0)
        # participant_id = self.participant.participant_id
        # loop = asyncio.get_running_loop()
        # user_property=('to', self.market_sid))
        # loop.create_task(self.ns.on_connect())
        # self.client.publish('/'.join([self.market_id, 'algorithm', 'policy_sever_ready']), '')
        # time.sleep(5)
        self.env.client_connected.set()

        # print(self.env.client_connected)

    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        # self.ns.on_disconnect()
        print('pettingzoo disconnected')

    def on_message(self, client, userdata, message):
        message = {
            'topic': message.topic,
            'payload': message.payload.decode(),
            'properties': message.properties
        }
        # print(message)
        #
        # # await self.msg_queue.put(msg)
        self.process_message(message)
        return 0

    def process_message(self, message):
        # if self.market.run:
        topic_event = message['topic'].split('/')[-1]
        payload = message['payload']

        match topic_event:
            # market related events
            case 'get_actions':
                # TODO: can we just call model.predict() here? for individual agents?
                # or we call model.predict at once at the end of the episode and let the agents query in the next round?
                # print(payload)
                participant_id = payload
                # await asyncio.sleep(random.randint(1, 5))
                # actions = {'bif': (random.random(), random.random())}
                # print(actions)
                actions = json.dumps(self.env.actions['participant_id'])
                self.client.publish('/'.join([self.market_id, 'algorithm', participant_id, 'get_actions_return']),
                                    actions, qos=2)
                # pass
                # self.env.poke_count += 1
                # print('get actions received', self.env.poke_count)
                # self.env.get_actions_event.set()
            case 'obs_rewards':
                payload = json.loads(payload)
                self.env.obs_check(payload)

    def fake_model(self):
        fake_actions = {'b1': [-0.96259534]}
        # fake_actions = {'Building_1': [-0.96259534], 'Building_2': [-0.96259534]}
        # print('fake model')
        reset_out = self.final_env.reset()
        print('reset_out: ', reset_out)
        while True:
            self.final_env.step(fake_actions)

    def run_client(self):
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        # client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message

        # client.set_auth_credentials(token, None)
        self.client.connect(self.server_address, keepalive=60)

    def run(self):
        """Function to start the client and other background tasks

        Raises:
            SystemExit: [description]
        """
        self.client.loop_start()
        self.run_client()
        self.fake_model()
        # self.model.learn(total_timesteps=20 * 1e7)
        self.client.loop_stop()


if __name__ == '__main__':
    # import socket
    import argparse
    import importlib

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--host', default="localhost", help='')
    parser.add_argument('--port', default=1883, help='')
    parser.add_argument('--config')
    args = parser.parse_args()
    # server_address = ''.join(['http://', args.host, ':', str(args.port)])
    server_address = args.host
    # client = Client(server_address=server_address,
    #                 config=json.loads(args.config))
    def load_json_file(file_path):
        with open(file_path) as f:
            json_file = commentjson.load(f)
        return json_file

    config = load_json_file('../../_configs/erp_test.json')
    client = Client(server_address=server_address, config=config)
    client.run()
