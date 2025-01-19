from operator import itemgetter
from cuid2 import Cuid
from typing import override
from TREX_Core.markets.base.DoubleAuction import Market as BaseMarket

#TODO: THIS IS A SPECIAL MARKET FOR DANIEL C MAY
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
        self.__round_settle_stats = dict()
        for key in self.__round_settle_stats_buf:
            self.__round_settle_stats_buf[key].clear()

        # # ToDo: include variances
        # self.__round_bid_stats = dict()
        # self.__round_ask_stats = dict()

        market_info["settle_stats"] = self.__round_settle_stats

        return market_info

    async def __match(self, time_delivery):
        """Matches bids with asks for a single source type in a time slot

        THe matching and settlement process closely resemble double auctions.
        In Original market:
            For all bids/asks for a source in the delivery time slots, highest bids are matched with lowest ask and settled pairwise. Quantities can be partially settled.
            Unsettled quantities are discarded.
            Participants are only obligated to buy/sell quantities settled for the delivery period.

        In this market:
            all bids and asks are auto-assigned a price as a function of the supply : demand ratio (see Zhang et al.)
            Then this market uses an expectation formulation of the original settlement mechanism. The goal here is to provide more predictable settlements
            all bids and asks are sorted from lowest to highest quantity
            we then calculate the expected settlement if the original settlement mechanism was used


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
                       key=itemgetter('quantity'),
                       # reverse=True
                       )
            bids = self.__open[time_delivery]['bid'] #ToDo: make sure these are sorted from lowest to highest

        # if 'asks exist, collect stats
        if 'ask' in self.__open[time_delivery]:
            self.__open[time_delivery]['ask'][:] = \
                sorted([ask for ask in self.__open[time_delivery]['ask'] if ask['quantity'] > 0],
                       key=itemgetter('quantity'),
                       # reverse=False,
                       )
            asks = self.__open[time_delivery]['ask'] #ToDo: make sure these are sorted from lowest to highest

        if 'ask' not in self.__open[time_delivery]:
            return

        if 'bid' not in self.__open[time_delivery]:
            return


        #ToDo: change the settlement mechanism here
        # I think the best way to do this is to determine the quantities first
        # then call the settle method
        # we first sort bids and asks by quantity from low to high
        # we then assign the price for each bid and ask based on the supply and demand ratio
        # we then determine if we have more bidded quantity or asked quantity total,
        # the smaller one will be fully settled, the larger one will be partially settled
        # to determine the partial settlement, we:
        #   for every entry in the fully settled list
        #   we can calculate the expected settlement for each entry in the partially settled list by
        #   dividing the fully settled entry's quantity over the number of entries in the partial settlement list
        #   we then subtract the expected settlement from the partially settled entry's quantity
        #   if a quantity becomes negative, we have to redistribute this overflow over the remainder of the partially settled entries
        # the output format of this needs to be compatible with Steven's original __settle function, except for the addition of a settled quantity parameter

        # calculating total supply and demand, calculating and assigning prices
        total_supply_quantity = sum(ask['quantity'] for ask in asks)
        total_demand_quantity = sum(bid['quantity'] for bid in bids)
        round_bid_price, round_ask_price = await self.__determine_round_prices(supply=total_supply_quantity,
                                                                               demand=total_demand_quantity)

        # print(round_bid_price, round_ask_price)
        for bid in bids:
            bid['price'] = round_bid_price
        self.__round_bid_stats['avg_bid_price'] = round_bid_price
        # self.__round_bid_stats['avg_bid_quantity'] = np.mean([bid['quantity'] for bid in bids])
        self.__round_bid_stats['total_bid_quantity'] = total_demand_quantity
        # self.__round_bid_stats['std_bid_quantity'] = np.std([bid['quantity'] for bid in bids])

        for ask in asks:
            ask['price'] = round_ask_price
        self.__round_ask_stats['avg_ask_price'] = round_ask_price
        # self.__round_ask_stats['avg_ask_quantity'] = sum(ask['quantity'] for ask in asks) / len(asks)
        self.__round_ask_stats['total_ask_quantity'] = total_supply_quantity
        # self.__round_ask_stats['std_ask_quantity'] = np.std([ask['quantity'] for ask in asks])

        # determining which is the fully settled and which is the partially settled list
        if total_supply_quantity > total_demand_quantity:
            fully_settled = bids
            partially_settled = asks
            partially_settled_target = total_supply_quantity - total_demand_quantity
             #this is used to doublecheck later
        else:
            fully_settled = asks
            partially_settled = bids
            partially_settled_target = total_demand_quantity - total_supply_quantity

         #this is used to doublecheck later

        # calculating how each entry of fully settled settles on partially settled
        # we use the key settlement_quantity in partially settled as a tracker for the quantity of settlements this current round
        # print('---------------------')
        for fully_settled_entry in fully_settled:
            # print('to be fully settled entry before:', fully_settled_entry)
            # print('to be partially settled before:', partially_settled)

            settlement_quantity = fully_settled_entry['quantity'] / len(partially_settled)
            __settlement_quant = 0

            for index, partially_settled_entry in enumerate(partially_settled):
                partially_settled_quantity = min(settlement_quantity, partially_settled_entry['quantity'])
                overflow = settlement_quantity - partially_settled_entry['quantity']
                if overflow > 0: # we need to redistribute, this means raising the settlement quantity for all next entries
                    settlement_quantity += overflow / (len(partially_settled) - index - 1)

                # now that all are settled, we can invoke the settle method, depending on which one is the full settled list
                # setttle can determing the settlement quantity from the partially settled entry 'settlement quantity'
                if total_supply_quantity > total_demand_quantity:
                    bid = fully_settled_entry
                    ask = partially_settled_entry
                else:
                    bid = partially_settled_entry
                    ask = fully_settled_entry

                await self.__check_and_settle(bid, ask, time_delivery, round_bid_price, round_ask_price, partially_settled_quantity)

                __settlement_quant += partially_settled_quantity
                # now that we are settled for this pair
                # we can also update the fully settled entry

                # we can update the quantity of the partially settled entry
                # partially_settled_entry['quantity'] = partially_settled_entry['quantity'] -  partially_settled_entry['settlement_quantity']

            # we can check if an entry in the partially settled list has a remaining quantity of 0, if so, we can delete it and save some compute time
            partially_settled[:] = [entry for entry in partially_settled if entry['quantity'] > 0] #FixMe: check if this does not break the overflow calc

            # fully_settled_entry['quantity'] = fully_settled_entry['quantity'] - __settlement_quant
            # print(fully_settled_entry['quantity'], 'after deducting', __settlement_quant)

            # after all partially settled entries are settled, we can check if the fully settled entry is fully settled
            # if not, we have made a mistake
            # print('to be fully settled entry after:', fully_settled_entry)
            # print('to be partially settled after:', partially_settled)
            # assert np.isclose(fully_settled_entry['quantity'], 0), 'fully settled entry is not fully settled, please check the code'

        # assert np.isclose(sum(entry['quantity'] for entry in fully_settled), 0), 'fully settled list is not fully settled, please check the code'

        # assert np.isclose(sum(entry['quantity'] for entry in partially_settled), partially_settled_target), 'partial settlement target is not met, please check the code'



    async def __check_and_settle(self, bid, ask, time_delivery, round_bid_price, round_ask_price, settlement_q):
            if ask['price'] > bid['price']:
                return

            if bid['participant_id'] == ask['participant_id']:
                return

            # if bid['source'] != ask['source']:
            #     continue

            if bid['lock'] or ask['lock']:
                return

            if bid['quantity'] <= 0 or ask['quantity'] <= 0:
                return

            # if bid['participant_id'] not in self.__participants:
            #     bid['lock'] = True
            #     return
            #
            # if ask['participant_id'] not in self.__participants:
            #     ask['lock'] = True
            #     return

            # Settle highest price bids with lowest price asks
            #ToDo: we'll have to rewrite this also
            await self.__settle(bid, ask, time_delivery, round_bid_price, round_ask_price, settlement_q)

    async def __determine_round_prices(self, supply, demand):
        #get min price and max price
        # ToDo: replace these with the grid values at the current time
        grid_sell_price = self.__grid.sell_price()
        grid_buy_price = self.__grid.buy_price()

        # these are the static "baseline" prices if the ratio (demand/supply) was to be 0
        bid_price_max = 0.1426
        ask_price_max = 0.1299

        bid_slope = -0.0254
        ask_slope = -0.0280

        # calculating buy min price, because Steven's curve doesn't respect WBB at the min
        # this is determining at what ratio the ask price is at the min price, at which point we can deduce a min proce for the bid_price
        ask_min_price_ratio = (grid_sell_price - ask_price_max) / ask_slope
        bid_min_price = bid_price_max + bid_slope * ask_min_price_ratio
        ask_min_price = grid_sell_price


        # assert demand > 0, 'demand is smaller of equal to zero, cannot be'
        # assert supply > 0, 'supply is smaller of equal to zero, cannot be'

        ratio = supply / demand

        round_ask_price = ask_price_max + ask_slope * ratio
        round_ask_price = max(round_ask_price, ask_min_price)
        round_ask_price = min(round_ask_price, ask_price_max)

        round_bid_price = bid_price_max + bid_slope * ratio
        round_bid_price = max(round_bid_price, bid_min_price)
        round_bid_price = min(round_bid_price, bid_price_max)

        return round_bid_price, round_ask_price

    async def __settle(self, bid: dict, ask: dict, time_delivery: tuple, round_bid_price: float, round_ask_price: float, settlement_q: float, settlement_method=None, locking=False):
        """Performs settlement for bid/ask pairs found during the matching process.
            this market uses an expectation formulation of the original settlement mechanism. The goal here is to provide more predictable settlements
            all bids and asks are sorted from lowest to highest quantity
            we then calculate the expected settlement if the original settlement mechanism was used

        If bid/ask are valid, the bid/ask quantities are adjusted, a commitment record is created, and a settlement confirmation is sent to both participants.

        Parameters
        ----------
        bid: dict
            bid entry to be settled. Should be a reference to the open bid

        ask: dict
            bid entry to be settled. Should be a reference to the open ask

        time_delivery : tuple
            Tuple containing the start and end timestamps in UNIX timestamp format.

        locking: bool
        Optinal locking mode, which locks the bid and ask until a callback is received after settlement confirmation is sent. The default value is False.

        Currently, locking should be disabled in simulation mode, as waiting for callback causes some settlements to be incomplete, likely due a flaw in the implementation or a poor understanding of how callbacks affect the sequence of events to be executed in async mode.

        Notes
        -----
        It is possible to settle directly with the grid, although this feature is currently not used by the agents and is under consideration to be deprecated.


        """

        # grid is not allowed to interact through market
        if ask['source'] == 'grid':
            return

        # only proceed to settle if settlement quantity is positive
        quantity = settlement_q
        if quantity <= 0:
            return

        if locking:
            # lock the bid and ask until confirmations are received
            ask['lock'] = True
            bid['lock'] = True

        commit_id = Cuid().generate(6)
        settlement_time = self.__timing['current_round'][1]

        settlement_price_sell = round_ask_price
        settlement_price_buy = round_bid_price

        # collect stats for Daniel C May
        self.__round_settle_stats_buf["settlement_price_sell"].append((settlement_price_sell, quantity))
        self.__round_settle_stats_buf["settlement_price_buy"].append((settlement_price_buy, quantity))

        # Record successful settlements
        if time_delivery not in self.__settled:
            self.__settled[time_delivery] = {}


        #This is how the settlements are structured
        #ToDo: ask about how strict this record format is
        # since now we wont have strict 1 to 1 settlements ... unless we actually do that?
        record = {
            'quantity': quantity,
            'seller_id': ask['participant_id'],
            'buyer_id': bid['participant_id'],
            'energy_source': ask['source'],
            'settlement_price_sell': settlement_price_sell,
            'settlement_price_buy': settlement_price_buy,
            'time_purchase': settlement_time
        }

        self.__settled[time_delivery][commit_id] = {
            'time_settlement': settlement_time,
            'source': ask['source'],
            'record': record,
            'ask': ask,
            'seller_id': ask['participant_id'],
            'bid': bid,
            'buyer_id': bid['participant_id'],
            'lock': locking
        }

        # This is what gets emitted by the market
        message = {
            'commit_id': commit_id,
            'ask_id': ask['uuid'],
            'bid_id': bid['uuid'],
            'source': ask['source'],
            'quantity': quantity,
            'sell_price': settlement_price_sell,
            'buy_price': settlement_price_buy,
            'buyer_id': bid['participant_id'],
            'seller_id': ask['participant_id'],
            'time_delivery': time_delivery
        }

        if locking:
            await self.__client.emit('send_settlement', message,
                                     callback=self.__settle_confirm_lock)
        else:
            await self.__client.emit('send_settlement', message)
            bid['quantity'] = max(0, bid['quantity'] - self.__settled[time_delivery][commit_id]['record']['quantity']) #FiXMe: what do these two lines dooo?
            ask['quantity'] = max(0, ask['quantity'] - self.__settled[time_delivery][commit_id]['record']['quantity'])
        self.__status['round_settled'].append(commit_id)
