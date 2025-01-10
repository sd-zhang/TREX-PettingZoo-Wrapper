import json
# import time

# import os
from cuid2 import Cuid
import paho.mqtt.client as mqtt

import random
from TREX_Core._clients.markets.ns_common import NSDefault


class Client:

    def __init__(self, server_address, config=None):
        # Initialize client-server data
        self.server_address = server_address
        self.client = mqtt.Client(client_id=Cuid(length=10).generate(),
                                  callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        # self.config = config  # temporary
        self.config = json.dumps({'test': 'config'})
        self.market_id = 'training'

        #TODO: initialize trex env
        Env = importlib.import_module('trex_env')
        self.env = Env.Env(client=self.client, config=config)

        #TODO: initialize algorithm
        # Model = importlib.import_module('sb3_contrib').RecurrentPPO
        # self.model = Model()

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
                actions = {'bif': (random.random(), random.random())}
                # print(actions)
                self.client.publish('/'.join([self.market_id, 'algorithm', participant_id, 'get_actions_return']),
                                    json.dumps(actions), qos=2)
                # pass
                # self.env.poke_count += 1
                # print('get actions received', self.env.poke_count)
                # self.env.get_actions_event.set()
            case 'obs_rewards':
                payload = json.loads(payload)
                self.env.obs_check(payload)

    def fake_model(self):
        fake_actions = dict()
        # print('fake model')
        self.env.reset()
        while True:
            self.env.step(fake_actions)

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

    client = Client(server_address=server_address)
    client.run()
