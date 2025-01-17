import asyncio
import json
# import time

import anyio
from anyio import to_thread, to_process
import os
from cuid2 import Cuid
from gmqtt import Client as MQTTClient
from pubsub import pub

import random
from TREX_Core._clients.markets.ns_common import NSDefault

# if os.name == 'posix':
#     import uvloop
#     uvloop.install()

STOP = asyncio.Event()

class Client:

    def __init__(self, server_address, config=None):
        # Initialize client-server data
        self.server_address = server_address
        self.client = MQTTClient(Cuid(length=10).generate())
        # self.config = config  # temporary
        self.config = json.dumps({'test': 'config'})
        self.market_id = 'training'

        #TODO: initialize trex env
        Env = importlib.import_module('trex_env')
        self.env = Env.Env(config=config)

        #TODO: initialize algorithm
        # Model = importlib.import_module('sb3_contrib').RecurrentPPO
        # self.model = Model()

        # #TODO: pass NS stuff to super class
        # self.ns = NSDefault(self.env)

        # self.tg = None
        # self.loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(self.loop)
        pub.subscribe(self.env_event_handler, 'pz_event')

    async def async_handler(self):
        self.client.publish('/'.join([self.market_id, 'algorithm', 'policy_sever_ready']), '', qos=0)
        # await asyncio.sleep(0.01)

    def env_event_handler(self):
        print('a')
        # self.loop.run_until_complete(self.async_handler())
        # self.client.publish('/'.join([self.market_id, 'algorithm', 'policy_sever_ready']), '', qos=2)
        asyncio.run(self.async_handler())

    def on_connect(self, client, flags, rc, properties):
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

    def obs_check(self, payload):
        """ every time obs/rewards come in this function is called
        when the entire set of obs/rewards are in, set get_obs_event
        """
        self.env.obs = payload['obs']
        self.env.reward = payload['reward']
        self.env.get_obs_event.set()

    async def process_message(self, message):
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
                                    actions, qos=2)
                # pass
                # self.env.poke_count += 1
                # print('get actions received', self.env.poke_count)
                # self.env.get_actions_event.set()
            case 'obs_rewards':
                payload = json.loads(payload)
                self.obs_check(payload)
                # print(payload)

                # TODO: check if all observations are received
                # TODO: after all obs/rewards are received, toggle event
                # self.env.get_obs_event.set()

    def on_disconnect(self, client, packet, exc=None):
        # self.ns.on_disconnect()
        print('pettingzoo disconnected')

    # def on_subscribe(self, client, mid, qos, properties):
    #     print('SUBSCRIBED')

    async def on_message(self, client, topic, payload, qos, properties):
        # print('market RECV MSG:', topic, payload.decode(), properties)
        message = {
            'topic': topic,
            'payload': payload.decode(),
            'properties': properties
        }

        # await self.msg_queue.put(msg)
        await self.process_message(message)
        return 0

    def fake_model(self):
        fake_actions = dict()
        # print('fake model')
        self.env.reset()
        while True:
            self.env.step(fake_actions)

    async def run_client(self, client):
        client.on_connect = self.on_connect
        client.on_disconnect = self.on_disconnect
        # client.on_subscribe = self.on_subscribe
        client.on_message = self.on_message

        # client.set_auth_credentials(token, None)
        await client.connect(self.server_address, keepalive=60)
        await STOP.wait()
        # await asyncio.wait()
        await client.disconnect()

    async def run(self):
        """Function to start the client and other background tasks

        Raises:
            SystemExit: [description]
        """
        # async with anyio.create_task_group() as tg:
        #     tg.start_soon(self.run_client, self.client)
        #     tg.start_soon(to_thread.run_sync, self.fake_model)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.run_client(self.client))
            tg.create_task(asyncio.to_thread(self.fake_model))


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
    #
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    # client.loop = loop
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(client.run())
    # anyio.run(client.run)
