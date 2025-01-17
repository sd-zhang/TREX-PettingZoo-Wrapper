# import socketio
import asyncio
import json


class NSDefault():
    def __init__(self, env):
        # super().__init__(namespace='')
        self.env = env
    async def process_message(self, message):
        # if self.market.run:
        topic_event = message['topic'].split('/')[-1]
        payload = message['payload']

        match topic_event:
            # market related events
            case 'is_market_online':
                await self.on_is_market_online()

    # async def on_connect(self):
    #     # print()
    #     pass
        # await self.market.register()

    # def on_disconnect(self):
    #     self.market.server_online = False

    # async def on_participant_disconnected(self, client_id):
    #     return await self.market.participant_disconnected(client_id)

    async def on_bid(self, bid):
        bid = json.loads(bid)
        return await self.market.submit_bid(bid)

    async def on_ask(self, ask):
        ask = json.loads(ask)
        return await self.market.submit_ask(ask)

    async def on_settlement_delivered(self, message):
        message = json.loads(message)
        await self.market.settlement_delivered(message)

    async def on_meter_data(self, message):
        # print("meter data")
        message = json.loads(message)
        await self.market.meter_data(message)

    async def on_participant_connected(self, message):
        # print(type(client_data))
        client_data = json.loads(message)
        await self.market.participant_connected(client_data)

    async def on_is_market_online(self):
        await self.market.market_is_online()

    async def on_start_round(self, message):
        message = json.loads(message)
        await self.market.step(message['duration'], sim_params=message)

    async def on_start_generation(self, message):
        # message = json.loads(message)
        table_name = str(message) + '_' + self.market.market_id
        await self.market.open_db(table_name)

    async def on_end_generation(self, message):
        await self.market.end_sim_generation()

    async def on_end_simulation(self):
        self.market.run = False
        await self.market.end_sim_generation(last_generation=True)
        await self.market.kill()
        # print(self.market.run)