from datamodel import OrderDepth, TradingState, Order, Symbol, ProsperityEncoder, Listing, Trade, Observation
from typing import List, Any, Tuple
import json
import numpy as np


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
        
        self.imbalance = 0
        
    def trade(self, trading_state: TradingState, orders: list):
        order_depth: OrderDepth = trading_state.order_depths[self.name]
        self.cache_prices(order_depth)
        if len(self.old_asks) < self.strategy_start_day or len(self.old_bids) < self.strategy_start_day:
            return

        avg_bid, avg_ask = self.calculate_prices(self.strategy_start_day)

        if len(order_depth.sell_orders) != 0:
            best_asks = sorted(order_depth.sell_orders.keys())

            i = 0
            while i < self.trade_count and len(best_asks) > i and best_asks[i] - avg_bid <= self.min_req_price_difference:
                if self.prod_position == self.max_pos:
                    break
                self.buy_product(best_asks, i, order_depth, orders)
                i += 1

        if len(order_depth.buy_orders) != 0:
            # Sort all the available buy orders by their price
            best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

            i = 0
            # Check if the lowest bid (buy order) is lower than the above defined fair value
            while i < self.trade_count and len(best_bids) > i and avg_ask - best_bids[i] <= self.min_req_price_difference:
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
    
    def calculate_imbalance(self, days: int):
        # Calculate the average bid and ask price for the last days

        relevant_bids = []
        for bids in self.old_bids[-days:]:
            relevant_bids.extend([(value, bids[value]) for value in bids])
        relevant_asks = []
        for asks in self.old_asks[-days:]:
            relevant_asks.extend([(value, asks[value]) for value in asks])

        bid_vol = np.sum([x[1] for x in relevant_bids])
        ask_vol = np.sum([x[1] for x in relevant_asks])
        self.imbalance = (bid_vol - ask_vol)/ (bid_vol + ask_vol)

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
        self.old_dolphins = -1
        self.dolphins_spotted = False
        self.dolphins_gone = False

        self.dolphin_action_time = 900
        self.gear_timestamp_diff = 70000

        self.dolphins_spotted_timestamp = -1
        self.dolphins_gone_timestamp = -1

    def handle_observations(self, trading_state: TradingState):
        for observation in trading_state.observations.keys():
            if observation == "DOLPHIN_SIGHTINGS":
                if self.old_dolphins == -1:
                    self.old_dolphins = trading_state.observations["DOLPHIN_SIGHTINGS"]
                    continue
                if trading_state.observations["DOLPHIN_SIGHTINGS"] - self.old_dolphins > 10:
                    # print("DOLPHINS SPOTTED")
                    self.dolphins_spotted = True
                    self.dolphins_spotted_timestamp = trading_state.timestamp
                if trading_state.observations["DOLPHIN_SIGHTINGS"] - self.old_dolphins < -10:
                    # print("DOLPHINS GONE")
                    self.dolphins_gone = True
                    self.dolphins_gone_timestamp = trading_state.timestamp
                self.old_dolphins = trading_state.observations["DOLPHIN_SIGHTINGS"]

    def trade(self, trading_state: TradingState, orders: list):

        self.handle_observations(trading_state)

        order_depth: OrderDepth = trading_state.order_depths[self.name]

        if self.dolphins_spotted and trading_state.timestamp - self.dolphins_spotted_timestamp < self.dolphin_action_time:
            # start buying gear if dolphins have been spotted
            # print(self.dolphins_spotted_timestamp)
            # print("BUYING GEAR")
            # print(trading_state.timestamp)
            self.continuous_buy(order_depth, orders)

        if self.dolphins_gone and trading_state.timestamp - self.dolphins_gone_timestamp < self.dolphin_action_time:
            # start selling gear if dolphins are going away
            # print("SELLING GEAR")
            self.continuous_sell(order_depth, orders)
        if self.dolphins_spotted and trading_state.timestamp - self.dolphins_spotted_timestamp > self.gear_timestamp_diff:
            # Start selling after dolphins have been spotted for long enough
            self.continuous_sell(order_depth, orders)

            if trading_state.timestamp - self.dolphins_spotted_timestamp - self.gear_timestamp_diff > self.dolphin_action_time:
                self.dolphins_spotted = False

        if self.dolphins_gone and trading_state.timestamp - self.dolphins_gone_timestamp > self.gear_timestamp_diff:
            # Start buying after dolphins have been gone for long enough
            self.continuous_buy(order_depth, orders)
            if trading_state.timestamp - self.dolphins_gone_timestamp - self.gear_timestamp_diff > self.dolphin_action_time:
                self.dolphins_gone = False

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


class Starfruit(CrossStrategy):
    def __init__(self):
        super().__init__("STARFRUIT", min_req_price_difference=3, max_position=20)

class Amethysts(FixedStrategy2):
    def __init__(self):
        super().__init__("AMETHYSTS", max_pos=20)

class Trader:

    def __init__(self) -> None:
        self.products = {
            # "STARFRUIT": Starfruit(),
            "AMETHYSTS": Amethysts()
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
                result[product] = orders

        

        traderData = ""

		# Sample conversion request.
        conversions = 0

        self.logger.flush(state, result, conversions, traderData)

        return result, 0, ""

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