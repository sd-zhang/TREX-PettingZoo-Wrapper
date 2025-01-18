import numpy as np
from operator import itemgetter
import itertools
from cuid2 import Cuid
from typing import override
from TREX_Core.markets.base.DoubleAuction import Market as BaseMarket


# TODO: THIS IS A SPECIAL MARKET FOR DANIEL C MAY
# DO NOT USE IF YOUR NAME IS NOT DANIEL C MAY, This comment is made by Steven S. Zhang


class Market(BaseMarket):
    """MicroTE is a futures trading based market design for transactive energy as part of TREX

    The market mechanism here works more like standard futures contracts,
    where delivery time interval is submitted along with the bid or ask.

    bids and asks are organized by source type
    the addition of delivery time requires that the bids and asks to be further organized by time slot

    Bids/asks can be are accepted for any time slot starting from one step into the future to infinity
    The minimum close slot is determined by 'close_steps', where a close_steps of 2 is 1 step into the future
    The the minimum close time slot is the last delivery slot that will accept bids/asks

    """

    def __init__(self, market_id, **kwargs):
        super().__init__(market_id, **kwargs)

        # start new data buffers for Daniel C May
        self.__round_settle_stats_buf = {
            "settlement_price_buy": list(),
            "settlement_price_sell": list()
        }
        self.__round_settle_stats = dict()

        self.__round_bid_stats = dict()

        self.__round_ask_stats = dict()
        # end new data buffers for Daniel C May

    @override
    async def get_market_info(self):
        market_info = await super().get_market_info()

        # "settle_stats": self.__round_settle_stats
        # }
        # update start round message for Daniel C May
        # market.update(self.__round_settle_stats)
        # self.__round_settle_stats = dict()
        for key in self.__round_settle_stats_buf:
            self.__round_settle_stats_buf[key].clear()

            # #ToDo: include variances
            # self.__round_bid_stats = dict()
            # self.__round_ask_stats = dict()
        market_info["settle_stats"] = self.__round_settle_stats
        return market_info

    @override
    async def __match(self, time_delivery):
        """Matches bids with asks for a single source type in a time slot

        THe matching and settlement process closely resemble double auctions.
        For all bids/asks for a source in the delivery time slots, highest bids are matched with lowest asks
        and settled pairwise. Quantities can be partially settled. Unsettled quantities are discarded. Participants are only obligated to buy/sell quantities settled for the delivery period.

        Parameters
        ----------
        time_delivery : tuple
            Tuple containing the start and end timestamps in UNIX timestamp format indicating the interval for energy to be delivered.

        Notes
        -----
        Presently, the settlement price is hard-coded as the average price of the bid/ask pair. In the near future, dedicated, more sophisticated functions for determining settlement price will be implemented

        """

        if time_delivery not in self.__open:
            return

        if 'bid' in self.__open[time_delivery]:
            self.__open[time_delivery]['bid'][:] = \
                sorted([bid for bid in self.__open[time_delivery]['bid'] if bid['quantity'] > 0],
                       key=itemgetter('price'), reverse=True)
            bids = self.__open[time_delivery]['bid']

            # Calculatre for stats for bids
            # self.__round_bid_stats['max_bid_price'] = np.amax([bid['price'] for bid in bids])
            # self.__round_bid_stats['min_bid_price'] = np.amin([bid['price'] for bid in bids])
            self.__round_bid_stats['avg_bid_price'] = np.mean([bid['price'] for bid in bids])
            # self.__round_bid_stats['max_bid_quantity'] = max(bids, key=itemgetter('quantity'))['quantity']
            # self.__round_bid_stats['min_bid_quantity'] = min(bids, key=itemgetter('quantity'))['quantity']
            self.__round_bid_stats['avg_bid_quantity'] = np.mean([bid['quantity'] for bid in bids])
            self.__round_bid_stats['total_bid_quantity'] = sum([bid['quantity'] for bid in bids])
            self.__round_bid_stats['std_bid_quantity'] = np.std([bid['quantity'] for bid in bids])
        else:
            # self.__round_bid_stats['max_bid_price'] = self.__grid.buy_price()
            # self.__round_bid_stats['min_bid_price'] = self.__grid.buy_price()
            self.__round_bid_stats['avg_bid_price'] = self.__grid.buy_price()
            # self.__round_bid_stats['max_bid_quantity'] = 0
            # self.__round_bid_stats['min_bid_quantity'] = 0
            self.__round_bid_stats['avg_bid_quantity'] = 0
            self.__round_bid_stats['total_bid_quantity'] = 0
            self.__round_bid_stats['std_bid_quantity'] = 0

        # if 'asks exist, collect stats
        if 'ask' in self.__open[time_delivery]:
            self.__open[time_delivery]['ask'][:] = \
                sorted([ask for ask in self.__open[time_delivery]['ask'] if ask['quantity'] > 0],
                       key=itemgetter('price'), reverse=False)
            asks = self.__open[time_delivery]['ask']

            # Calculatre for stats for asks
            # self.__round_ask_stats['max_ask_price'] = np.amax([ask['price'] for ask in asks])
            # self.__round_ask_stats['min_ask_price'] = np.amin([ask['price'] for ask in asks])
            self.__round_ask_stats['avg_ask_price'] = np.mean([ask['price'] for ask in asks])
            # self.__round_ask_stats['max_ask_quantity'] = max(asks, key=itemgetter('quantity'))['quantity']
            # self.__round_ask_stats['min_ask_quantity'] = min(asks, key=itemgetter('quantity'))['quantity']
            self.__round_ask_stats['avg_ask_quantity'] = sum(ask['quantity'] for ask in asks) / len(asks)
            self.__round_ask_stats['total_ask_quantity'] = sum(ask['quantity'] for ask in asks)
            self.__round_ask_stats['std_ask_quantity'] = np.std([ask['quantity'] for ask in asks])
        else:
            # self.__round_ask_stats['max_ask_price'] = self.__grid.sell_price()
            # self.__round_ask_stats['min_ask_price'] = self.__grid.sell_price()
            self.__round_ask_stats['avg_ask_price'] = self.__grid.sell_price()
            # self.__round_ask_stats['max_ask_quantity'] = 0
            # self.__round_ask_stats['min_ask_quantity'] = 0
            self.__round_ask_stats['avg_ask_quantity'] = 0
            self.__round_ask_stats['total_ask_quantity'] = 0
            self.__round_ask_stats['std_ask_quantity'] = 0

        if {'ask', 'bid'} > self.__open[time_delivery].keys():
            return

        # remove zero-quantity bid and ask entries
        # sort bids by decreasing price and asks by increasing price

        for bid, ask, in itertools.product(bids, asks):
            if ask['price'] > bid['price']:
                continue

            if bid['participant_id'] == ask['participant_id']:
                continue

            # if bid['source'] != ask['source']:
            #     continue

            if bid['quantity'] <= 0 or ask['quantity'] <= 0:
                continue

            # Settle highest price bids with lowest price asks
            await self.settle(bid, ask, time_delivery)


    @override
    async def settle(self, bid: dict, ask: dict, time_delivery: tuple):
        # collect stats for Daniel C May
        try:
            quantity, settlement_price_buy, settlement_price_sell = await super().settle(bid, ask, time_delivery)
            self.__round_settle_stats_buf["settlement_price_sell"].append((settlement_price_sell, quantity))
            self.__round_settle_stats_buf["settlement_price_buy"].append((settlement_price_buy, quantity))
        except TypeError:
            return


    # Finish all processes and remove all unnecessary/ remaining records in preparation for a new time step, begin processes for next step
    @override
    async def step(self, timeout=60, sim_params=None):
        await super().step(timeout, sim_params)

        # calculate stats for Daniel C May
        settlements_sell = np.array(self.__round_settle_stats_buf["settlement_price_sell"])  # [price, quantity]
        settlements_buy = np.array(self.__round_settle_stats_buf["settlement_price_buy"])  # [price, quantity]

        total_sold_quantity = np.sum(settlements_sell[:, 1]) if settlements_sell.size > 0 else None
        total_bought_quantity = np.sum(settlements_buy[:, 1]) if settlements_buy.size > 0 else None
        assert total_sold_quantity == total_bought_quantity  # make sure this is consistent

        self.__round_settle_stats = {}
        self.__round_settle_stats["settled_time"] = tuple(self.__timing['last_settle'])
        self.__round_settle_stats["total_settled_quantity"] = total_sold_quantity if settlements_sell.size > 0 else 0
        # self.__round_settle_stats["avg_settlement_ask_quantity"] = np.average(settlements_sell[:, 1]) if settlements_sell != [] else 0
        # self.__round_settle_stats["avg_settlement_bid_quantity"] = np.average(settlements_buy[:, 1]) if settlements_buy != [] else 0
        # self.__round_settle_stats["max_settlement_ask_quantity"] = np.max(settlements_sell[:, 1]) if settlements_sell != [] else 0
        # self.__round_settle_stats["max_settlement_bid_quantity"] = np.max(settlements_buy[:, 1]) if settlements_buy != [] else 0
        # self.__round_settle_stats["avg_settlement_ask_price"] = np.average(settlements_sell[:, 0], weights=settlements_sell[:, 1]) if settlements_sell != [] else self.__grid.sell_price()
        # self.__round_settle_stats["avg_settlement_bid_price"] = np.average(settlements_buy[:, 0], weights=settlements_buy[:, 1]) if settlements_buy != [] else self.__grid.buy_price()
        # self.__round_settle_stats["max_settlement_ask_price"] = np.max(settlements_sell[:, 0]) if settlements_sell != [] else self.__grid.sell_price()
        # self.__round_settle_stats["max_settlement_bid_price"] = np.max(settlements_buy[:, 0]) if settlements_buy != [] else self.__grid.buy_price()
        # self.__round_settle_stats["min_settlement_ask_price"] = np.min(settlements_sell[:, 0]) if settlements_sell != [] else self.__grid.sell_price()
        # self.__round_settle_stats["min_settlement_bid_price"] = np.min(settlements_buy[:, 0]) if settlements_buy != [] else self.__grid.buy_price()
        # self.__round_settle_stats["stdDev_settlement_ask_price"] = np.std(settlements_sell[:, 0]) if settlements_sell != [] else 0
        # self.__round_settle_stats["stdDev_settlement_bid_price"] = np.std(settlements_buy[:, 0]) if settlements_buy != [] else 0

        self.__round_settle_stats.update(self.__round_bid_stats)
        self.__round_settle_stats.update(self.__round_ask_stats)
        # print(self.__round_settle_stats)
        # for key in self.__round_bid_stats:
        #     assert key not in self.__round_settle_stats
        #     self.__round_settle_stats[key] = self.__round_bid_stats[key]

        # for key in self.__round_ask_stats:
        #     assert key not in self.__round_settle_stats
        #     self.__round_settle_stats[key] = self.__round_ask_stats[key]

        # print(self.__round_settle_stats, flush=True)
        # await self.__client.emit('end_round', data="")
