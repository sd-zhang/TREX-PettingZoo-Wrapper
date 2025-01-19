import datetime
import json

import paho.mqtt.client as mqtt
import supersuit as ss
from cuid2 import Cuid
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from utils.custom_Monitor import Custom_VecMonitor
from utils.custom_vec_normalize import VecNormalize
from utils.ppo_recurrent_custom import RecurrentPPO
from utils.schedule import exponential_schedule


class Client:

    def __init__(self, server_address, config):
        # Initialize client-server data
        self.server_address = server_address
        self.client = mqtt.Client(client_id=Cuid(length=10).generate(),
                                  callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.config = config

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
        env = SB3VecEnvWrapper(env)
        env = Custom_VecMonitor(env, filename=tboard_logdir, obs_names=agents_obs_keys)
        self.final_env = VecNormalize(env,
                                      norm_obs=True,
                                      norm_reward=False,
                                      num_bits=num_bits,
                                      # Daniel: The num bits corresponds to a 1hot encoding of agent id for our case, this is iIrc slight abuse but works
                                      clip_obs=1e6, clip_reward=1e6, gamma=0.99, epsilon=1e-08)

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

    def on_connect(self, client, userdata, flags, reason_code, properties):

        print('connected')
        market_id = self.env.market_id
        # print("market_id", market_id)
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
        client.subscribe("/".join([market_id, 'algorithm', 'get_actions']), qos=2)
        client.subscribe("/".join([market_id, 'algorithm', 'obs_rewards']), qos=2)
        # client.subscribe("/".join([market_id, 'algorithm', 'rewards']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'start_generation']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'end_episode']), qos=2)
        client.subscribe("/".join([market_id, 'simulation', 'end_simulation']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'is_policy_server_online']), qos=0)
        # participant_id = self.participant.participant_id
        # loop = asyncio.get_running_loop()
        # user_property=('to', self.market_sid))
        # loop.create_task(self.ns.on_connect())
        # self.client.publish('/'.join([self.market_id, 'algorithm', 'policy_sever_ready']), '')
        # time.sleep(5)
        self.env.client_connected.set()
        # self.env.send_ready()

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
        self.process_message(message)
        return 0

    def process_message(self, message):
        # if self.market.run:
        topic_event = message['topic'].split('/')[-1]
        payload = message['payload']

        match topic_event:
            # market related events
            case 'get_actions':
                participant_id = payload
                actions = json.dumps(self.env.get_actions(participant_id))
                self.client.publish('/'.join([self.env.market_id, 'algorithm', participant_id, 'get_actions_return']),
                                    actions, qos=2)
            case 'obs_rewards':
                payload = json.loads(payload)
                self.env.obs_check(payload)
            case 'end_episode':
                if not self.env.get_obs_event.is_set():
                    self.env.send_ready()
            # case 'end_simulation':
            #     self.env.close()
            #     self.client.disconnect()
            case 'is_policy_server_online':
                if self.env.client_connected.is_set() and not self.env.get_obs_event.is_set():
                    self.env.send_ready()

    # def fake_model(self):
    #     fake_actions = {'b1': [-0.96259534]}
    #     # fake_actions = {'Building_1': [-0.96259534], 'Building_2': [-0.96259534]}
    #     # print('fake model')
    #     # reset_out = self.final_env.reset()
    #     # print('reset_out: ', reset_out)
    #     while True:
    #         time.sleep(1)
    #         # print('step')
    #         # self.final_env.step(fake_actions)

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
        # self.fake_model()
        self.model.learn(total_timesteps=self.config['study']['total_steps'])
        self.client.loop_stop()


if __name__ == '__main__':
    import argparse
    import importlib
    from TREX_Core.runner.runner import get_config

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--host', default="localhost", help='')
    parser.add_argument('--port', default=1883, help='')
    parser.add_argument('--config')
    args = parser.parse_args()
    # server_address = ''.join(['http://', args.host, ':', str(args.port)])
    server_address = args.host
    config = json.loads(args.config)

    # root_dir = 'E:/git/TREX-PettingZoo-Wrapper'
    # config = get_config('erp_test2', root_dir=root_dir)
    # config['market']['id'] = 'training'
    # from pprint import pprint
    # pprint(config)
    client = Client(server_address=server_address, config=config)
    client.run()
