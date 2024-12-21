import asyncio
import json
# from asyncio import Queue
import os
from cuid2 import Cuid
from gmqtt import Client as MQTTClient
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from TREX_Core._clients.markets.ns_common import NSDefault

if os.name == 'posix':
    import uvloop
    uvloop.install()

STOP = asyncio.Event()

class Client:
    def __init__(self, server_address, config=None):
        # Initialize client-server data
        self.server_address = server_address
        self.client = MQTTClient(Cuid(length=10).generate())
        # self.config = config  # temporary
        self.config = json.dumps({'test': 'config'})
        # print(self.config)

        #TODO: initialize trex env
        Env = importlib.import_module('trex_env')
        self.env = Env.Env(self.client, config=config)

        #TODO: initialize algorithm
        # Model = importlib.import_module('sb3_contrib').RecurrentPPO
        # self.model = Model()

        # #TODO: pass NS stuff to super class
        # self.ns = NSDefault(self.env)

    def on_connect(self, client, flags, rc, properties):
        # market_id = self.config['market']['id']  # temporary
        market_id = 'test'
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
        client.subscribe("/".join([market_id, 'algorithm', 'get_actions']), qos=2)
        client.subscribe("/".join([market_id, 'algorithm', 'obs_rewards']), qos=2)
        # client.subscribe("/".join([market_id, 'algorithm', 'rewards']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'start_generation']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'end_generation']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'end_simulation']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'is_market_online']), qos=0)
        # participant_id = self.participant.participant_id
        # loop = asyncio.get_running_loop()
        # loop.create_task(self.ns.on_connect())

    async def process_message(self, message):
        # if self.market.run:
        topic_event = message['topic'].split('/')[-1]
        payload = message['payload']

        match topic_event:
            # market related events
            case 'get_actions':
                self.env.get_actions_go.set()
            case 'obs_rewards':
                payload = json.loads(payload)
                participant_id = payload['participant_id']
                self.env.observations[participant_id] = payload['data']
                # TODO: check if all observations are received
                # TODO: after all obs/rewards are received, toggle event

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

    # async def fake_async(self, func):
    #     with ProcessPoolExecutor() as executor:
    #         loop = asyncio.get_running_loop()
    #         await loop.run_in_executor(executor, func)
    # #         executor.submit(func)

    def fake_model(self):
        fake_actions = dict()
        self.env.reset()
        while True:
            # await asyncio.sleep(0.1)
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
        # await client.disconnect()
    async def run(self):
        """Function to start the client and other background tasks

        Raises:
            SystemExit: [description]
        """
        # tasks = [
        #     # asyncio.create_task(keep_alive()),
        #     # asyncio.create_task(self.ns.listen(self.msg_queue)),
        #     asyncio.create_task(self.run_client(self.sio_client)),
        #     asyncio.create_task(self.market.loop())
        # ]
        #
        # # try:
        # await asyncio.gather(*tasks)

        # asyncio.create_task(self.run_client(self.client))

        # for python 3.11+


        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.run_client(self.client))
            # tg.create_task(self.fake_async(self.fake_model))
            tg.create_task(asyncio.to_thread(self.fake_model))
            # tg.create_task(self.keep_alive())

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

    asyncio.run(client.run())
