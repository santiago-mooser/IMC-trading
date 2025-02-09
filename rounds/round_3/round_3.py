from datamodel import OrderDepth, TradingState, Order, Symbol, ProsperityEncoder, Listing, Trade, Observation
from typing import List, Any, Tuple
import json
import numpy as np
import collections

class Strategy:
    def __init__(self, name: str, max_position: int):
        self.name: str = name
        self.cached_prices: list = []
        self.cached_means: list = []
        self.max_pos: int = max_position
        self.trade_count: int = 1

        self.prod_position: int = 0
        self.new_buy_orders: int = 0
        self.new_sell_orders: int = 0
        self.order_depth: OrderDepth = OrderDepth()

    def reset_from_state(self, state: TradingState):
        self.prod_position = state.position[self.name] if self.name in state.position.keys() else 0
        self.order_depth: OrderDepth = state.order_depths[self.name]

        self.new_buy_orders = 0
        self.new_sell_orders = 0

    def sell_product(self, best_bids, i, order_depth, orders):
        # Sell product at best bid
        best_bid_volume = order_depth.buy_orders[best_bids[i]]
        if self.prod_position - best_bid_volume >= -self.max_pos:
            orders.append(Order(self.name, best_bids[i], -best_bid_volume))
            self.prod_position += -best_bid_volume
            self.new_sell_orders += best_bid_volume

        else:
            # Sell as much as we can without exceeding the self.max_pos[product]
            vol = self.prod_position + self.max_pos
            orders.append(Order(self.name, best_bids[i], -vol))
            self.prod_position += -vol
            self.new_sell_orders += vol

    def buy_product(self, best_asks, i, order_depth, orders):
        # Buy product at best ask
        best_ask_volume = order_depth.sell_orders[best_asks[i]]
        if self.prod_position - best_ask_volume <= self.max_pos:
            orders.append(Order(self.name, best_asks[i], -best_ask_volume))
            self.prod_position += -best_ask_volume
            self.new_buy_orders += -best_ask_volume
        else:
            # Buy as much as we can without exceeding the self.max_pos[product]
            vol = self.max_pos - self.prod_position
            orders.append(Order(self.name, best_asks[i], vol))
            self.prod_position += vol
            self.new_buy_orders += vol

    def continuous_buy(self, order_depth: OrderDepth, orders: list):
        if len(order_depth.sell_orders) != 0:
            best_asks = sorted(order_depth.sell_orders.keys())

            i = 0
            while i < self.trade_count and len(best_asks) > i:
                if self.prod_position == self.max_pos:
                    break

                self.buy_product(best_asks, i, order_depth, orders)
                i += 1

    def continuous_sell(self, order_depth: OrderDepth, orders: list):
        if len(order_depth.buy_orders) != 0:
            best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

            i = 0
            while i < self.trade_count and len(best_bids) > i:
                if self.prod_position == -self.max_pos:
                    break

                self.sell_product(best_bids, i, order_depth, orders)
                i += 1

class CrossStrategy(Strategy):
    def __init__(self, name: str, min_req_price_difference: int, max_position: int):
        super().__init__(name, max_position)
        self.strategy_start_day = 2

        self.old_asks = []
        self.old_bids = []
        self.min_req_price_difference = min_req_price_difference

        # try the imbalance indicator: (total_bid_vol - total_ask_vol)/ (total_bid_vol + total_ask_vol), pos if bid vol is higher
        self.imbalance = 0
        self.direction = 0

        # Stoikov Model
        self.mm = StoikovMarketMaker(0.2, 5.4, 0, 20)

    def trade(self, trading_state: TradingState, orders: list):
        order_depth: OrderDepth = trading_state.order_depths[self.name]
        self.cache_prices(order_depth)
        if len(self.old_asks) < self.strategy_start_day or len(self.old_bids) < self.strategy_start_day:
            return

        avg_bid, avg_ask = self.calculate_prices(self.strategy_start_day)

        # imbalance is not useful, stop using it
        curr_imbalance =  0 # self.calculate_imbalance(self.strategy_start_day)
        if curr_imbalance > self.imbalance:
            # we may track the changes of the imbalance
            pass

        if curr_imbalance > 0.5:
            self.direction = 1
        elif curr_imbalance < -0.5:
            self.direction = -1
        else:
            self.direction = 0

        # update imbalance
        self.imbalance = curr_imbalance

        bid_volume = self.max_pos - self.prod_position
        ask_volume = -self.max_pos - self.prod_position
        #buy order
        # orders.append(Order(self.name, int(avg_bid + self.min_req_price_difference + self.direction), bid_volume))
        # #sell order
        # orders.append(Order(self.name, int(avg_ask - self.min_req_price_difference + self.direction), ask_volume))

        self.mm.update_inventory(self.prod_position)
        bid_quote, ask_quote = self.mm.calculate_quotes(avg_bid,avg_ask)

        if len(order_depth.sell_orders) != 0:
            best_asks = sorted(order_depth.sell_orders.keys())

            i = 0
            while i < self.trade_count and len(best_asks) > i and best_asks[i] - bid_quote <= self.min_req_price_difference + self.direction:
                if self.prod_position == self.max_pos:
                    break
                self.buy_product(best_asks, i, order_depth, orders)
                i += 1

        if len(order_depth.buy_orders) != 0:
            # Sort all the available buy orders by their price
            best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

            i = 0
            # Check if the lowest bid (buy order) is lower than the above defined fair value
            while i < self.trade_count and len(best_bids) > i and ask_quote - best_bids[i] <= self.min_req_price_difference - self.direction:
                if self.prod_position == -self.max_pos:
                    break
                self.sell_product(best_bids, i, order_depth, orders)

                i += 1

    def calculate_prices(self, days: int) -> Tuple[int, int]:
        # Calculate the average bid and ask price for the last days

        relevant_bids = []
        for bids in self.old_bids[-days:]:
            relevant_bids.extend([(value, bids[value]) for value in bids])
        relevant_asks = []
        for asks in self.old_asks[-days:]:
            relevant_asks.extend([(value, asks[value]) for value in asks])

        avg_bid = np.average([x[0] for x in relevant_bids], weights=[x[1] for x in relevant_bids])
        avg_ask = np.average([x[0] for x in relevant_asks], weights=[x[1] for x in relevant_asks])

        return avg_bid, avg_ask

    def calculate_imbalance(self, days: int) -> float:
        # Calculate the imbalance for the last days

        relevant_bids = []
        for bids in self.old_bids[-days:]:
            relevant_bids.extend([(value, bids[value]) for value in bids])
        relevant_asks = []
        for asks in self.old_asks[-days:]:
            relevant_asks.extend([(value, asks[value]) for value in asks])

        bid_vol = sum([x[1] for x in relevant_bids])
        ask_vol = sum([x[1] for x in relevant_asks])
        if bid_vol + ask_vol == 0:
            return 0
        imbalance = (bid_vol - ask_vol)/ (bid_vol + ask_vol)

        return imbalance

    def cache_prices(self, order_depth: OrderDepth):
        sell_orders = order_depth.sell_orders
        buy_orders = order_depth.buy_orders

        self.old_asks.append(sell_orders)
        self.old_bids.append(buy_orders)

class DiffStrategy(Strategy):
    def __init__(self, name: str, max_position: int, derivative_resolution: int, diff_thresh: int):
        super().__init__(name, max_position)
        self.derivative_resolution: int = derivative_resolution
        self.diff_thresh: int = diff_thresh

    def trade(self, trading_state: TradingState, orders: list):
        order_depth: OrderDepth = trading_state.order_depths[self.name]
        self.cache_purchased_prices(trading_state)
        self.calculate_means()

        diff = self.get_price_difference()

        if diff < -self.diff_thresh and len(order_depth.sell_orders) != 0:
            self.continuous_buy(order_depth, orders)

        if diff > self.diff_thresh and len(order_depth.buy_orders) != 0:
            self.continuous_sell(order_depth, orders)

    def get_price_difference(self) -> float:
        # Calculate the difference between the current mean and the mean from
        # self.derivative_resolution days ago
        if len(self.cached_means) < self.derivative_resolution + 1:
            old_mean = self.cached_means[0]
        else:
            old_mean = self.cached_means[-self.derivative_resolution]
        diff = self.cached_means[-1] - old_mean
        return diff

    def calculate_means(self):
        #
        if len(self.cached_prices) == 0:
            self.cached_means.append(0)

        else:
            relevant_prices = []
            for day_prices in self.cached_prices[max(-len(self.cached_prices), -1):]:
                for price in day_prices:
                    relevant_prices.append(price)
            prices = np.array([x[1] for x in relevant_prices])
            quantities = np.abs(np.array([x[0] for x in relevant_prices]))

            self.cached_means.append(np.average(prices, weights=quantities))

    def cache_purchased_prices(self, state: TradingState) -> None:
        # Caches prices of bought and sold products

        market_trades = state.market_trades
        own_trades = state.own_trades

        prod_trades: List[Trade] = own_trades.get(self.name, []) + market_trades.get(self.name, [])

        if len(prod_trades) > 0:
            prices = [(trade.quantity, trade.price) for trade in prod_trades]
            self.cached_prices.append(prices)

class FixedStrategy(Strategy):
    def __init__(self, name: str, max_pos: int):
        super().__init__(name, max_pos)
        self.pearls_price = 10000
        self.pearls_diff = 4

    def trade(self, trading_state: TradingState, orders: list):
        order_depth: OrderDepth = trading_state.order_depths[self.name]
        # Check if there are any SELL orders
        if len(order_depth.sell_orders) > 0:
            #
            # self.cache_prices(order_depth)
            # Sort all the available sell orders by their price
            best_asks = sorted(order_depth.sell_orders.keys())

            # Check if the lowest ask (sell order) is lower than the above defined fair value
            i = 0
            while i < self.trade_count and best_asks[i] < self.pearls_price:
                # Fill ith ask order if it's below the acceptable
                if self.prod_position == self.max_pos:
                    break
                self.buy_product(best_asks, i, order_depth, orders)
                i += 1
        if len(order_depth.buy_orders) != 0:
            best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

            i = 0
            while i < self.trade_count and best_bids[i] > self.pearls_price:
                if self.prod_position == -self.max_pos:
                    break
                self.sell_product(best_bids, i, order_depth, orders)
                i += 1

class FixedStrategy2(Strategy):
    def __init__(self, name: str, max_pos: int):
        super().__init__(name, max_pos)
        self.amethysts_price = 10000

    def trade(self, trading_state: TradingState, orders: list):
        order_depth: OrderDepth = trading_state.order_depths[self.name]
        # Check if there are any SELL orders
        bid_volume = self.max_pos - self.prod_position
        ask_volume = -self.max_pos - self.prod_position

        orders.append(Order(self.name, self.amethysts_price - 1, bid_volume))
        orders.append(Order(self.name, self.amethysts_price + 1, ask_volume))

class ObservationStrategy(Strategy):
    def __init__(self, name: str, max_position: int):
        super().__init__(name, max_position)
        self.cost = 0
        self.gear_timestamp_diff = 70000

        self.max_pos = max_position

        self.bidPrice = None
        self.askPrice = None
        self.transportFees = None
        self.exportTariff = None
        self.importTariff = None
        self.sunlight = None
        self.humidity = None

        self.sunlight_average = 2100
        self.ema_param = 0.5

        self.humidity_out = None
        self.sunlight_out = None

    def handle_observations(self, trading_state: TradingState):
        obs = trading_state.observations.conversionObservations[self.name]

        self.bidPrice = obs.bidPrice
        self.askPrice = obs.askPrice
        self.transportFees = obs.transportFees
        self.exportTariff = obs.exportTariff
        self.importTariff = obs.importTariff
        self.sunlight = obs.sunlight
        self.humidity = obs.humidity
        self.humidity_effect_on_production = 0

        self.humidity_out_timestamp = trading_state.timestamp

        if self.humidity > 80 or self.humidity < 60:
            self.humidity_out = True
        else:
            self.humidity_out = False

        if self.sunlight < 2500:
            self.sunlight_out = True
        else:
            self.sunlight_out = False

        if self.humidity < 60 or self.humidity > 80:
            self.humidity_out = True

        if self.humidity > 80:
            self.humidity_effect_on_production = int((80 - self.humidity) /5)
        elif self.humidity < 60:
            self.humidity_effect_on_production = int((self.humidity - 80) /5)

        # calculate ema of sunlight
        if self.sunlight_average == 0:
            self.sunlight_average = self.sunlight
        self.sunlight = self.sunlight * self.ema_param + self.sunlight_average * (1 - self.ema_param)


    def trade(self, trading_state: TradingState, orders: list):

        order_list =[]

        self.handle_observations(trading_state)
        self.update_cost(trading_state)
        order_depth: OrderDepth = trading_state.order_depths[self.name]

        if self.askPrice == self.bidPrice:
            return
        remote_bid = round(self.bidPrice)
        remote_ask = round(self.askPrice)

        # Using the remote bid and remote ask, we can arbitrage between the two markets without having to exhange between the two

        # If any local bid is higher than remote market, sell in local
        for i, local_bid in enumerate(order_depth.buy_orders):
            if local_bid >= remote_bid:
                order_list.append(Order(self.name, local_bid, -order_depth.buy_orders[local_bid]))

        # If the local ask is lower than the remote market, buy in local
        for i, local_ask in enumerate(order_depth.sell_orders):
            if local_ask <= remote_ask:
                order_list.append(Order(self.name, local_ask, order_depth.sell_orders[local_ask]))



        orders.extend(order_list)

    def conversion(self, trading_state, position) -> int:
        '''
        get the number of conversion for this round
        '''

        remote_bid = round(self.bidPrice - self.importTariff - self.transportFees)
        remote_ask = round(self.askPrice + self.exportTariff + self.transportFees)

        order_depth = trading_state.order_depths[self.name]

        buy_sell=0

        # conversions only allow us to sell in remote

        # if the local bid is lower than the remote market, buy in remote
        for i, local_bid in enumerate(order_depth.buy_orders):
            if local_bid >= remote_bid:
                buy_sell -= order_depth.buy_orders[local_bid]

        # If the local ask is higher than the remote market, sell in remote
        for i, local_ask in enumerate(order_depth.sell_orders):
            if local_ask <= remote_ask:
                buy_sell += order_depth.sell_orders[local_ask]


        # if any remote ask is lower than local bid, sell in remote
        for i, local_bid in enumerate(order_depth.buy_orders):
            if local_bid >= remote_bid:
                buy_sell += order_depth.buy_orders[local_bid]

        # if any remote bid is higher than local ask, buy in remote
        for i, local_ask in enumerate(order_depth.sell_orders):
            if local_ask <= remote_ask:
                buy_sell -= order_depth.sell_orders[local_ask]

        #  if there is remote ask that is higher than local ask, sell in remote
        for i, local_ask in enumerate(order_depth.sell_orders):
            if local_ask <= remote_ask:
                buy_sell += order_depth.sell_orders[local_ask]

        # finally, if there is remote bid that is lower than local bid, buy in remote
        for i, local_bid in enumerate(order_depth.buy_orders):
            if local_bid >= remote_bid:
                buy_sell -= order_depth.buy_orders[local_bid]

        return buy_sell



    def update_cost(self, trading_state: TradingState):
        '''
        update the cost of the current
        '''
        if self.name not in trading_state.market_trades:
            return

        market_trades = trading_state.market_trades[self.name]
        for trade in market_trades:
            if trade.buyer == "SUBMISSION":
                self.cost += trade.price * trade.quantity + 0.1 * trade.quantity
            else:
                self.cost -= trade.price * trade.quantity

class TimeBasedStrategy(CrossStrategy):
    def __init__(self, name, min_req_price_difference, max_position):
        super().__init__(name, min_req_price_difference, max_position)
        self.berries_ripe_timestamp = 350000
        self.berries_peak_timestamp = 500000
        self.berries_sour_timestamp = 650000

    def trade(self, trading_state: TradingState, orders: list):
        order_depth = trading_state.order_depths[self.name]
        if 0 < trading_state.timestamp - self.berries_ripe_timestamp < 5000:
            # print("BERRIES ALMOST RIPE")

            # start buying berries if they start being ripe
            if len(order_depth.sell_orders) != 0:
                best_asks = sorted(order_depth.sell_orders.keys())

                i = 0
                while i < self.trade_count and len(best_asks) > i:
                    if self.prod_position == -self.max_pos:
                        break
                    self.buy_product(best_asks, i, order_depth, orders)
                    i += 1

        elif 0 < trading_state.timestamp - self.berries_peak_timestamp < 5000:
            # print("BERRIES READY TO SELL")
            self.continuous_sell(order_depth, orders)
        else:
            super().trade(trading_state, orders)

class ArbitrageStrategy(Strategy):
    def __init__(self, name, min_req_price_difference: int, max_position: dict, zscore_threshold: float):
        super().__init__(name, max_position)

        self.min_price_diff = min_req_price_difference
        self.max_pos = max_position

        self.orders = []

        self.products = ["CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]
        self.coeffs = {}
        self.coeffs["GIFT_BASKET"] = [ 0.99637084, 8.54381396, -0.161608, 0.02234559, -0.03371232, 0.21668808, -0.21632859, 0.04613577, -0.02076174,  0.03116055, -0.03263618,  0.04016784, -0.02797431]
        self.coeffs["CHOCOLATE"] = [-0.00437733,  0.00308218,  0.01752576,  0.98360442, -0.90183322, -2.79369535,  0.02711808, -0.00315843,  0.11274011, -0.11258077]
        self.coeffs["ROSES"] = [-0.00981411,  0.00393729,  0.01444269,  0.99055514, -0.0615577, -0.3335764,  0.03956394, -0.04242976, -0.07749774,  0.07836905]
        self.coeffs["STRAWBERRIES"] = [-0.01862957,  0.0360459 ,  0.11510304,  0.86723537, -0.04803083, -1.15100102, -0.00741988, -0.02239479, -0.00789983,  0.0081571 ]

        self.fair_market_prices = {}
        for product in self.products:
            self.fair_market_prices[product] = 0

        self.midprices = {}
        for product in self.products:
            self.midprices[product] = []

        self.estimated_values = {}
        for product in self.products:
            self.estimated_values[product] = []

        self.intercepts = {}
        for product in self.products:
            self.intercepts[product] = 0

        self.imbalances = {}
        for product in self.products:
            self.imbalances[product] = []

        self.vwaps = {}
        for product in self.products:
            self.vwaps[product] = []

        self.spreads = {}
        for product in self.products:
            self.spreads[product] = []

        self.buy_volume = {}
        for product in self.products:
            self.buy_volume[product] = 0

        self.sell_volume = {}
        for product in self.products:
            self.sell_volume[product] = 0

        self.timestamp = 0
        self.midprice_diff = []

        self.z_score = []
        self.zscore_threshold = zscore_threshold

    def trade(self, trading_state: TradingState, orders: list):

        self.timestamp = trading_state.timestamp
        self.prod_position = trading_state.position.get(self.name, 0)

        # calculate the estimated value of each product, including basket
        for product in self.products:
            order_depth = trading_state.order_depths[product]

            # calculate the variables needed for our linear regression
            # 'mid_price_3','mid_price_2','mid_price_1','mid_price','imbalance_1','imbalance','spread_1','spread','vwap_1','vwap'

            self.midprices[product].append(self.calculate_mid_price(order_depth))
            if len(self.midprices[product]) > 4:
                self.midprices[product].pop(0)

            self.imbalances[product].append(self.calculate_imbalance(order_depth, product))
            if len(self.imbalances[product]) > 2:
                self.imbalances[product].pop(0)

            self.spreads[product].append(self.calculate_spread(order_depth))
            if len(self.spreads[product]) > 2:
                self.spreads[product].pop(0)

            self.vwaps[product].append(self.calculate_vwap(trading_state, product))
            if len(self.vwaps[product]) > 2:
                self.vwaps[product].pop(0)

            if product != "GIFT_BASKET":
                self.estimated_values[product].append(self.calculate_fair_market_price(product))
                if len(self.estimated_values[product]) > 2:
                    self.estimated_values[product].pop(0)

        if self.timestamp < 200:
            return

        # first we find out the underlying value of the securities
        underlying_value = self.calculate_basket_underlying_price()

        fair_value_of_basket = self.calculate_fair_market_price_basket()

        self.midprice_diff.append(underlying_value - fair_value_of_basket)

        current_z_score = (self.midprice_diff[-1] - np.mean(self.midprice_diff)) / np.std(self.midprice_diff)

        if len(self.midprice_diff) > 25:
            self.midprice_diff.pop(0)

        self.z_score.append(current_z_score)

        if len(self.z_score) < 2:
            return


        if len(self.z_score) > 2:
            self.z_score.pop(0)

        long_signal = (current_z_score < -self.zscore_threshold) and (self.z_score[-2] > -self.zscore_threshold)
        long_exit = (current_z_score > -self.zscore_threshold) and (self.z_score[-2] < -self.zscore_threshold)

        short_signal = (current_z_score > self.zscore_threshold) and (self.z_score[-2] < self.zscore_threshold)
        short_exit = (current_z_score < self.zscore_threshold) and (self.z_score[-2] > self.zscore_threshold)

        for product in self.products:

            if long_signal:

                if product == "GIFT_BASKET":
                    for price, vol in order_depth.sell_orders.items():
                        if price < fair_value_of_basket + self.min_price_diff:
                            self.buy_volume["GIFT_BASKET"] += vol
                            if self.buy_volume["GIFT_BASKET"] > self.max_pos["GIFT_BASKET"]:
                                vol = self.buy_volume["GIFT_BASKET"] - self.max_pos["GIFT_BASKET"]
                                self.orders.append(Order(self.name, price, vol))
                                break
                            self.orders.append(Order(self.name, price, vol))

                else:
                    order_depth = trading_state.order_depths[product]
                    for price, vol in order_depth.buy_orders.items():
                        if price < self.estimated_values[product][-1] + self.min_price_diff:
                            self.sell_volume[product] += vol
                            if self.sell_volume[product] > self.max_pos[product]:
                                vol = self.sell_volume[product] - self.max_pos[product]
                                self.orders.append(Order(self.name, price, -vol))
                                break
                            self.orders.append(Order(self.name, price, -vol))

            elif short_signal:

                if product == "GIFT_BASKET":
                    for price, vol in order_depth.buy_orders.items():
                        if price > fair_value_of_basket - self.min_price_diff:
                            self.sell_volume["GIFT_BASKET"] += vol
                            if self.sell_volume["GIFT_BASKET"] > self.max_pos["GIFT_BASKET"]:
                                vol = self.sell_volume["GIFT_BASKET"] - self.max_pos["GIFT_BASKET"]
                                self.orders.append(Order(self.name, price, -vol))
                                break
                            self.orders.append(Order(self.name, price, -vol))

                else:
                    order_depth = trading_state.order_depths[product]
                    for price, vol in order_depth.sell_orders.items():
                        if price > self.estimated_values[product][-1] - self.min_price_diff:
                            self.buy_volume[product] += vol
                            if self.buy_volume[product] > self.max_pos[product]:
                                vol = self.buy_volume[product] - self.max_pos[product]
                                self.orders.append(Order(self.name, price, vol))
                                break
                            self.orders.append(Order(self.name, price, vol))

            elif long_exit:
                if product == "GIFT_BASKET":
                    for price, vol in order_depth.sell_orders.items():
                        if price < fair_value_of_basket + self.min_price_diff:
                            self.buy_volume["GIFT_BASKET"] -= vol
                            self.orders.append(Order(self.name, price, -vol))
                else:
                    order_depth = trading_state.order_depths[product]
                    for price, vol in order_depth.buy_orders.items():
                        if price < self.estimated_values[product][-1] + self.min_price_diff:
                            self.sell_volume[product] -= vol
                            self.orders.append(Order(self.name, price, vol))

            elif short_exit:
                if product == "GIFT_BASKET":
                    for price, vol in order_depth.buy_orders.items():
                        if price > fair_value_of_basket - self.min_price_diff:
                            self.sell_volume["GIFT_BASKET"] -= vol
                            self.orders.append(Order(self.name, price, vol))
                else:
                    order_depth = trading_state.order_depths[product]
                    for price, vol in order_depth.sell_orders.items():
                        if price > self.estimated_values[product][-1] - self.min_price_diff:
                            self.buy_volume[product] -= vol
                            self.orders.append(Order(self.name, price, -vol))

    def tradersData(self, trading_state: TradingState):
        '''
        This function will return the traders data
        '''
        return self.z_score

    def orders(self):
        return self.orders

    def calculate_fair_market_price(self, product: str):
        '''
        This is where we will do the linear regression for the baskets.
        For now, it just returns the midpoint
        '''
        if self.timestamp < 200:
            return self.midprices[product][-1]

        intercept = self.intercepts[product]
        coeff = self.coeffs[product]

        if len(self.midprices[product]) != 4 or len(self.imbalances[product]) != 2 or len(self.spreads[product]) != 2 or len(self.vwaps[product]) != 2:
            return self.midprices[product][-1]

        features = []

        # ['mid_price_3','mid_price_2','mid_price_1','mid_price','imbalance_1','imbalance','spread_1','spread','vwap_1','vwap']
        for midprice in self.midprices[product][-4:]:
            features.append(midprice)
        for imbalance in self.imbalances[product][-2:]:
            features.append(imbalance)
        for spread in self.spreads[product][-2:]:
            features.append(spread)
        for vwap in self.vwaps[product][-2:]:
            features.append(vwap)

        # Calculate fair price
        fair_price = sum(np.multiply(features,coeff)) + intercept

        return int(fair_price)

    def calculate_fair_market_price_basket(self) -> int:

        # if we don't have enough data, just return midprice
        if self.timestamp < 200:
            return self.midprices["GIFT_BASKET"][-1]

        coeff = self.coeffs["GIFT_BASKET"]
        intercept = 0

        features = []

        # ['mid_price', 'imbalance', 'imbalance_1','spread', 'spread_1', 'vwap', 'vwap_1', 'chocolate_predicted_price', 'chocolate_predicted_price_1', 'strawberry_predicted_price', 'strawberry_predicted_price_1', 'roses_predicted_price_1', 'roses_predicted_price', ]

        features = [self.midprices["GIFT_BASKET"][-1]]

        for imbalance in self.imbalances["GIFT_BASKET"][-2:]:
            features.append(imbalance)

        for spread in self.spreads["GIFT_BASKET"][-2:]:
            features.append(spread)

        for vwap in self.vwaps["GIFT_BASKET"][-2:]:
            features.append(vwap)

        for estimated_value in self.estimated_values["CHOCOLATE"][-2:]:
            features.append(estimated_value)

        for estimated_value in self.estimated_values["STRAWBERRIES"][-2:]:
            features.append(estimated_value)

        for estimated_value in self.estimated_values["ROSES"][-2:]:
            features.append(estimated_value)


        # Calculate fair price
        fair_price = sum(np.multiply(features,coeff)) + intercept

        return int(fair_price)

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:

        market_bids = order_depth.buy_orders

        market_asks = order_depth.sell_orders

        best_bid = max(market_bids)
        best_ask = min(market_asks)

        return  (best_bid + best_ask)/2

    def calculate_imbalance(self, order_depth: OrderDepth, product: str) -> float:
        # Calculate the imbalance for the last days

        bid_vol = sum(order_depth.buy_orders.values())
        ask_vol = sum(order_depth.sell_orders.values())
        if bid_vol + ask_vol == 0:
            return 0
        imbalance = (bid_vol + ask_vol)/ (bid_vol - ask_vol)

        return imbalance

    def calculate_vwap(self, state : TradingState, product: str):
        """
        Volume-Weighted Average Price
        calculate from all the trades happened from last iteration
        """

        total_value = 0
        total_volume = 0

        # when there is not trad in the beginning
        if product not in state.market_trades:
            if self.vwaps[product] == []:
                return self.midprices[product][-1]
            else:
                return self.vwaps[product][-1]

        market_trades = state.market_trades[product]

        for trade in market_trades:
            total_value += trade.price * trade.quantity
            total_volume += trade.quantity

        # Ensure we don't divide by zero in case of no trades
        if total_volume != 0:
            vwap = total_value / total_volume
            return vwap
        else:
            return self.vwaps[product][-1]

    def calculate_spread(self, order_depth: OrderDepth) -> float:

        market_bids = order_depth.buy_orders

        market_asks = order_depth.sell_orders

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return  best_ask - best_bid

    def calculate_basket_underlying_price(self):
        '''
        Calculate the real underlying price of the basket by calculating the value of the underlying assets:
        4 chocolates, 6 strawberries & 1 rose
        '''

        strawberry_fair_price = self.estimated_values["STRAWBERRIES"][-1]
        chocolate_fair_price = self.estimated_values["CHOCOLATE"][-1]
        roses_fair_price = self.estimated_values["ROSES"][-1]

        return 4 * chocolate_fair_price + 6 * strawberry_fair_price + roses_fair_price


##############################################################################################
### RegressionStrategy, using regression to find the best fair price
##############################################################################################

class RegressionStrategy(Strategy):
    def __init__(self, name: str, min_req_price_difference: int, max_position: int, intercept, coef):
        super().__init__(name, max_position)

        self.prices = []
        self.imbalances = []
        self.vwaps = []
        self.midprice_ema10 = []
        self.spreads = []

        self.strategy_start_day = 4

        # New params
        self.strategy_window = 10
        # New strat
        self.orderbook_imbalance_delta = []
        self.retreat_parameter = 0.1

        self.fair_price = None

        self.old_asks = []
        self.old_bids = []

        self.min_req_price_difference = min_req_price_difference
        self.intercept = intercept
        self.coef = coef
        self.mm = StoikovMarketMaker(0.23348603634235995, 1.966874725882954, 0, 20)

    def trade(self, trading_state: TradingState, orders: list):
        order_depth: OrderDepth = trading_state.order_depths[self.name]

        self.cache_features(trading_state)

        if len(self.prices) < self.strategy_start_day:
            return

        if len(self.prices) == self.strategy_start_day:
            fair_price = self.calculate_fair_price()
            avg_bid, avg_ask = fair_price - 2, fair_price + 2
        else:
            avg_bid, avg_ask =  self.calculate_prices(self.strategy_start_day)
        # self.mm.update_inventory(self.prod_position)
        # avg_bid, avg_ask = self.mm.calculate_quotes(avg_bid,avg_ask)
        orders.extend(self.compute_orders_regression(order_depth, avg_bid, avg_ask))


        # if len(order_depth.sell_orders) != 0:
        #     best_asks = sorted(order_depth.sell_orders.keys())

        #     i = 0
        #     while i < self.trade_count and len(best_asks) > i and best_asks[i] - bid_quote <= self.min_req_price_difference:
        #         if self.prod_position == self.max_pos:
        #             break
        #         self.buy_product(best_asks, i, order_depth, orders)
        #         i += 1

        # if len(order_depth.buy_orders) != 0:
        #     # Sort all the available buy orders by their price
        #     best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

        #     i = 0
        #     # Check if the lowest bid (buy order) is lower than the above defined fair value
        #     while i < self.trade_count and len(best_bids) > i and ask_quote - best_bids[i] <= self.min_req_price_difference:
        #         if self.prod_position == -self.max_pos:
        #             break
        #         self.sell_product(best_bids, i, order_depth, orders)

        #         i += 1



    def cache_features(self, trading_state):
        """
        update features for the regression
        """
        order_depth: OrderDepth = trading_state.order_depths[self.name]

        sell_orders = order_depth.sell_orders
        buy_orders = order_depth.buy_orders

        if len(self.old_asks) == self.strategy_start_day:
            self.old_asks.pop(0)
        if len(self.old_bids) == self.strategy_start_day:
            self.old_bids.pop(0)

        self.old_asks.append(sell_orders)
        self.old_bids.append(buy_orders)

        # mid_prices t-3 to t
        if len(self.prices) == self.strategy_start_day:
            self.prices.pop(0)

        mid_price = self.calculate_mid_price(order_depth)
        self.prices.append(mid_price)


        # order imbalance t-1 to t
        if len(self.imbalances) == 2:
            self.imbalances.pop(0)

        imbalance = self.calculate_imbalance(order_depth)
        self.imbalances.append(imbalance)

        # order spreads t-1 to t
        if len(self.spreads) == 2:
            self.spreads.pop(0)

        spread = self.calculate_spread(order_depth)
        self.spreads.append(spread)


        # order vwaps t-1 to t
        if len(self.vwaps) == 2:
            self.vwaps.pop(0)

        vwap = self.calculate_vwap(trading_state)
        self.vwaps.append(vwap)


    def calculate_fair_price(self) -> float:
        """
        Calculate fair price using linear regression
        """
        features = self.prices + self.imbalances + self.spreads + self.vwaps
        # self.logger.log(features)
        # ema_10 = np.mean(self.old_asks[-10:])

        # Calculate fair price
        fair_price = sum(np.multiply(features,self.coef)) + self.intercept

        return round(fair_price)

    def calculate_prices(self, days: int) -> Tuple[int, int]:
        # Calculate the average bid and ask price for the last days

        relevant_bids = []
        for bids in self.old_bids[-days:]:
            relevant_bids.extend([(value, bids[value]) for value in bids])
        relevant_asks = []
        for asks in self.old_asks[-days:]:
            relevant_asks.extend([(value, asks[value]) for value in asks])

        avg_bid = np.average([x[0] for x in relevant_bids], weights=[x[1] for x in relevant_bids])
        avg_ask = np.average([x[0] for x in relevant_asks], weights=[x[1] for x in relevant_asks])

        return avg_bid, avg_ask

    def calculate_retreat(self, order_depth: OrderDepth, state: TradingState) -> float:
        return min(max(state.position[self.name], self.max_pos), -self.min_pos) * self.retreat_parameter

    def calculate_own_trades_imbalance(self, state: TradingState) -> float:
        # Calculate our own trades' imbalance
        own_trades = state.past_own_trades.get(self.name, [])
        buy_volume = 0
        sell_volume = 0

        for trade in own_trades:
            if trade.quantity > 0:
                buy_volume += trade.quantity
            else:
                sell_volume += trade.quantity

        return buy_volume - sell_volume

    def calculate_orderbook_imbalance_delta(self, order_depth: OrderDepth) -> None:
        # calculate the change in the orderbook imbalance
        if len(self.orderbook_imbalance) < self.strategy_start_day:
            return 0

        current_orderbook_imbalance = self.calculate_current_orderbook_imbalance(order_depth)

        oi_delta = current_orderbook_imbalance - self.orderbook_imbalance[-1]

        self.orderbook_imbalance.append(current_orderbook_imbalance)
        if len(self.orderbook_imbalance) > self.strategy_window:
            self.orderbook_imbalance.pop(0)

        self.orderbook_imbalance_delta.append(oi_delta)

    def calculate_midprice_ema10(self, order_depth: OrderDepth) -> None:
        # Calculate the ema10 of the real midprice
        midprice = self.calculate_real_midprice(order_depth)
        self.midprice_ema10.append(midprice)
        if len(self.midprice_ema10) > 10:
            self.midprice_ema10.pop(0)

    def calculate_real_midprice(self, order_depth: OrderDepth) -> int:
        # Calculate the midprice of the ask and bid with the highest volume
        max_bid = max(order_depth.buy_orders, key=order_depth.buy_orders.get)
        max_ask = max(order_depth.sell_orders, key=order_depth.sell_orders.get)
        return (max_bid + max_ask) / 2

    def calculate_imbalance(self, order_depth: OrderDepth) -> float:
        # Calculate the imbalance for the last days

        bid_vol = sum(order_depth.buy_orders.values())
        ask_vol = sum(order_depth.sell_orders.values())
        if bid_vol + ask_vol == 0:
            return 0
        imbalance = (bid_vol + ask_vol)/ (bid_vol - ask_vol)
        return imbalance

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:

        market_bids = order_depth.buy_orders

        market_asks = order_depth.sell_orders

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return  (best_bid + best_ask)/2

    def calculate_spread(self, order_depth: OrderDepth) -> float:

        market_bids = order_depth.buy_orders

        market_asks = order_depth.sell_orders

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return  best_ask - best_bid

    def calculate_vwap(self, state : TradingState):
        """
        Volume-Weighted Average Price
        calculate from all the trades happened from last iteration
        """

        total_value = 0
        total_volume = 0

        # when there is not trad in the beginning
        if self.name not in state.market_trades:
            if self.vwaps == []:
                return self.calculate_mid_price(state.order_depths[self.name])
            else:
                return self.vwaps[-1]

        market_trades = state.market_trades[self.name]

        for trade in market_trades:
            total_value += trade.price * trade.quantity
            total_volume += trade.quantity

        # Ensure we don't divide by zero in case of no trades
        if total_volume != 0:
            vwap = total_value / total_volume
            return vwap
        else:
            return self.vwaps[-1]

    def compute_orders_regression(self, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []


        osell = order_depth.sell_orders
        obuy = order_depth.buy_orders
        if len(osell) != 0:
            best_sell_pr = sorted(osell.keys())[0]
        if len(obuy) != 0:
            best_buy_pr = sorted(order_depth.buy_orders.keys(), reverse=True)[0]

        # osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        # obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        # sell_vol, best_sell_pr = self.values_extract(osell)
        # buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.prod_position

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.prod_position<0) and (ask == acc_bid+1))) and cpos < self.max_pos:
                order_for = min(-vol, self.max_pos - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(self.name, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < self.max_pos:
            num = self.max_pos - cpos
            orders.append(Order(self.name, bid_pr, num))
            cpos += num

        cpos = self.prod_position

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.prod_position>0) and (bid+1 == acc_ask))) and cpos > -self.max_pos:
                order_for = max(-vol, -self.max_pos-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(self.name, bid, order_for))

        if cpos > -self.max_pos:
            num = -self.max_pos-cpos
            orders.append(Order(self.name, sell_pr, num))
            cpos += num

        return orders

##############################################################################################
### StoikovMarketMaker: set prices based on the inventory
##############################################################################################

class StoikovMarketMaker:
    def __init__(self, alpha, beta, target_inventory, max_inventory):
        """
        Initialize the market maker parameters.
        :param alpha: Sensitivity of quote prices to inventory level.
        :param beta: Base spread factor.
        :param target_inventory: Desired inventory level.
        :param max_inventory: Maximum allowable inventory.
        """
        self.alpha = alpha
        self.beta = beta
        self.target_inventory = target_inventory
        self.max_inventory = max_inventory
        self.inventory = 0  # Initial inventory

    def update_inventory(self, inv):
        """
        Update the market maker's inventory based on the trade history.
        :param trades: List of executed trades (positive for buys, negative for sells).
        """
        self.inventory = inv
        # Ensure inventory stays within bounds
        self.inventory = max(min(self.inventory, self.max_inventory), -self.max_inventory)

    def calculate_quotes(self, best_bid, best_ask):
        """
        Calculate the bid and ask prices based on the current midpoint price and inventory level.
        :param midpoint: Current midpoint price.
        :return: Tuple of (bid_price, ask_price).
        """
        midpoint = (best_bid + best_ask) / 2
        inventory_factor = self.alpha * (self.inventory - self.target_inventory) / self.max_inventory
        spread = self.beta * (1 + abs(inventory_factor))
        bid_price = midpoint - spread / 2
        ask_price = midpoint + spread / 2
        return round(bid_price), round(ask_price)
##############################################################################################
### BASKET STRATEGY
##############################################################################################

class BasketStrategy(Strategy):
    def __init__(self, name: str, max_pos: int):
        super().__init__(name, max_pos)
        self.position = {'CHOCOLATE' : 0, 'ROSES' : 0, 'STRAWBERRIES' : 0, 'GIFT_BASKET' : 0}
        self.position_limit = {'CHOCOLATE' : 250, 'ROSES' : 60, 'STRAWBERRIES' : 350, 'GIFT_BASKET' : 20}
        self.orders = {'CHOCOLATE' : [], 'ROSES': [], 'STRAWBERRIES' : [], 'GIFT_BASKET' : []}
        self.prods = ['CHOCOLATE', 'ROSES', 'STRAWBERRIES', 'GIFT_BASKET']
        self.osell = {}
        self.obuy = {}
        self.best_sell = {}
        self.best_buy = {}
        self.worst_sell = {}
        self.worst_buy = {}
        self.mid_price = {}
        self.vol_buy = {}
        self.vol_sell = {}
        self.basket_std = 76 # the std of the diff

    def trade(self, trading_state: TradingState, orders: list):
        order_depth: OrderDepth = trading_state.order_depths
        # Check if there are any SELL orders
        bid_volume = self.max_pos - self.prod_position
        ask_volume = -self.max_pos - self.prod_position

        for p in self.prods:
            self.osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            self.obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            self.best_sell[p] = next(iter(self.osell[p]))
            self.best_buy[p] = next(iter(self.obuy[p]))

            self.worst_sell[p] = next(reversed(self.osell[p]))
            self.worst_buy[p] = next(reversed(self.obuy[p]))

            self.mid_price[p] = (self.best_sell[p] + self.best_buy[p])/2
            self.vol_buy[p], self.vol_sell[p] = 0, 0
            for price, vol in self.obuy[p].items():
                self.vol_buy[p] += vol
                if self.vol_buy[p] >= self.position_limit[p]/10:
                    break
            for price, vol in self.osell[p].items():
                self.vol_sell[p] += -vol
                if self.vol_sell[p] >= self.position_limit[p]/10:
                    break

            self.mid_price[p] = (self.best_sell[p] + self.best_buy[p])/2

        # the residual is the difference minus the mean
        res_buy = self.mid_price['GIFT_BASKET'] - self.mid_price['CHOCOLATE']*4 - self.mid_price['STRAWBERRIES']*6 - self.mid_price['ROSES'] - 379.49
        res_sell = self.mid_price['GIFT_BASKET'] - self.mid_price['CHOCOLATE']*4 - self.mid_price['STRAWBERRIES']*6 - self.mid_price['ROSES'] - 379.49

        res = self.mid_price['GIFT_BASKET'] - self.mid_price['CHOCOLATE']*4 - self.mid_price['STRAWBERRIES']*6 - self.mid_price['ROSES'] - 379.49
        trade_at = self.basket_std *0.5
        close_at = self.basket_std*(-1000)
        pb_pos = self.position['GIFT_BASKET']
        pb_neg = self.position['GIFT_BASKET']
        basket_buy_sig = 0
        basket_sell_sig = 0
        self.cont_buy_basket_unfill = 0
        self.cont_sell_basket_unfill = 0

        if self.position['GIFT_BASKET'] == self.max_pos:
            self.cont_buy_basket_unfill = 0
        if self.position['GIFT_BASKET'] == -self.max_pos:
            self.cont_sell_basket_unfill = 0
        do_bask = 0

        # if res_sell > trade_at:
        #     vol = self.position['GIFT_BASKET'] + self.max_pos
        #     self.cont_buy_basket_unfill = 0 # no need to buy rn
        #     assert(vol >= 0)
        #     if vol > 0:
        #         do_bask = 1
        #         basket_sell_sig = 1
        #         orders.append(Order('GIFT_BASKET', self.worst_buy['GIFT_BASKET'], -vol))
        #         self.cont_sell_basket_unfill += 2
        #         pb_neg -= vol
        # elif res_buy < -trade_at:
        #     vol = self.max_pos - self.position['GIFT_BASKET']
        #     self.cont_sell_basket_unfill = 0 # no need to sell rn
        #     assert(vol >= 0)
        #     if vol > 0:
        #         do_bask = 1
        #         basket_buy_sig = 1
        #         orders.append(Order('GIFT_BASKET', self.worst_sell['GIFT_BASKET'], vol))
        #         self.cont_buy_basket_unfill += 2
        #         pb_pos += vol


        if res > trade_at:
            vol = self.position['GIFT_BASKET'] + self.position_limit['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders.append(Order('GIFT_BASKET', self.worst_buy['GIFT_BASKET'], -vol))
        elif res < -trade_at:
            vol = self.position_limit['GIFT_BASKET'] - self.position['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders.append(Order('GIFT_BASKET', self.worst_sell['GIFT_BASKET'], vol))
        elif res < close_at and self.position['GIFT_BASKET'] < 0:
            vol = -self.position['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders.append(Order('GIFT_BASKET', self.worst_sell['GIFT_BASKET'], vol))
        elif res > -close_at and self.position['GIFT_BASKET'] > 0:
            vol = self.position['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                orders.append(Order('GIFT_BASKET', self.worst_buy['GIFT_BASKET'], -vol))


        return orders



##############################################################################################
### Apply strategy to products
##############################################################################################
# class Starfruit(RegressionStrategy):
#     def __init__(self):
#         super().__init__("STARFRUIT", min_req_price_difference=3, max_position=20,
#                          intercept = 1.2389442725070694,
#                          coef = [ 0.15033769,  0.17131938,  0.28916903,  0.37856112, -2.27925121, -3.1545027 ,  0.01490655,  0.0156036 ,  0.03238973, -0.02205384]
#                          )

# class Amethysts(FixedStrategy2):
#     def __init__(self):
#         super().__init__("AMETHYSTS", max_pos=20)

# class Orchids(ObservationStrategy):
#     def __init__(self):
#         super().__init__("ORCHIDS", max_position=20)

class Baskets(ArbitrageStrategy):
    def __init__(self):
        super().__init__("GIFT_BASKET", min_req_price_difference=5, zscore_threshold=1,
                         max_position={
                                "CHOCOLATE": 250,
                                "ROSES": 60,
                                "STRAWBERRIES": 350,
                                "GIFT_BASKET": 60
                         }
                         )

# class Baskets(BasketStrategy):
#     def __init__(self):
#         super().__init__("GIFT_BASKET",  max_pos=20)

# class Chocolate(RegressionStrategy):
#     def __init__(self):
#         super().__init__("CHOCOLATE", min_req_price_difference=3, max_position=250,
#                          intercept = 0,
#                          coef = [-0.00437733,  0.00308218,  0.01752576,  0.98360442, -0.90183322,
#                                 -2.79369535,  0.02711808, -0.00315843,  0.11274011, -0.11258077]
#                         )
# class Roses(RegressionStrategy):
#     def __init__(self):
#         super().__init__("ROSES", min_req_price_difference=3, max_position=60,
#                          intercept = 0,
#                          coef = [-0.00981411,  0.00393729,  0.01444269,  0.99055514, -0.0615577 ,
#                                     -0.3335764 ,  0.03956394, -0.04242976, -0.07749774,  0.07836905]
#                          )
# class Strawberries(RegressionStrategy):
#     def __init__(self):
#         super().__init__("STRAWBERRIES", min_req_price_difference=3, max_position=350,
#                          intercept = 0,
#                          coef = [-0.01862957,  0.0360459 ,  0.11510304,  0.86723537, -0.04803083,
#                                 -1.15100102, -0.00741988, -0.02239479, -0.00789983,  0.0081571 ]
#                         )
# class Chocolate(DiffStrategy):
#     def __init__(self):
#         super().__init__("CHOCOLATE", min_req_price_difference=3, max_position=250)
# class Roses(DiffStrategy):
#     def __init__(self):
#         super().__init__("ROSES", min_req_price_difference=3, max_position=60)
# class Strawberries(DiffStrategy):
#     def __init__(self):
#         super().__init__("STRAWBERRIES", min_req_price_difference=3, max_position=350,)
##############################################################################################
### Trader Class
##############################################################################################
class Trader:

    def __init__(self) -> None:
        self.products = {
            # "STARFRUIT": Starfruit(),
            # "AMETHYSTS": Amethysts(),
            # "ORCHIDS": Orchids(),
            "GIFT_BASKET": Baskets(),
            # "CHOCOLATE": Chocolate(),
            # "ROSES": Roses(),
            # "STRAWBERRIES": Strawberries()
        }
        self.logger = Logger()

    def run(self, state: TradingState) -> Tuple[dict[Symbol, list[Order]], int, str]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

        # Initialize the method output dict as an empty dict
        result = {}

        for product in state.order_depths.keys():
            if product in self.products.keys():
                orders: list[Order] = []

                self.products[product].reset_from_state(state)
                self.products[product].trade(trading_state=state, orders=orders)
                if product == "GIFT_BASKET":
                    orders = self.products["GIFT_BASKET"].orders()
                # !!!!!!!!!! #
                # Need to change this
                # since strategy now returns
                # more than one trade
                # !!!!!!!!!! #
                result[product] = orders
                # !!!!!!!!!! #

        traderData = str(self.products["GIFT_BASKET"].tradersData(state))


		# Sample conversion request.
        orchids_position = state.position.get("ORCHIDS", 0)
        # conversions = self.products["ORCHIDS"].conversion(state, orchids_position)
        conversions = 0
        self.logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData

    def conversions(self, state: TradingState) -> int:
        return self.products["ORCHIDS"].conversion(state)


    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

############################################
# This is the logger that we use for the   #
# visualizer                               #
############################################
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."
