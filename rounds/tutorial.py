from datamodel import OrderDepth, TradingState, Order, Symbol, ProsperityEncoder, Listing, Trade, Observation
from typing import List, Any, Tuple
import json
import copy
import statistics
import math

AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"
PRODUCTS = [AMETHYSTS, STARFRUIT]
PRODUCT_ENABLED = {
    AMETHYSTS : True,
    STARFRUIT : True,
}
POSITION_LIMIT = {
    AMETHYSTS : 20,
    STARFRUIT : 20,
}
POSITION = {}
for product in PRODUCTS:
    POSITION.update({product: 0})
DEFAULT_PRICES = {
    AMETHYSTS : 10000,
    STARFRUIT : 5000,
}

EMA_PARAM = 0.29

class Trader:
    def __init__(self) -> None:
        # general parameters to keep track of the state of the trader
        self.position = copy.deepcopy(POSITION)
        self.position_limit = POSITION_LIMIT
        self.product_enabled = PRODUCT_ENABLED

        # Values to compute pnl
        self.cash = 0
        self.timestamp = 0


        self.logger = Logger()

        # self.past_order_depths keeps the list of all past prices
        self.past_order_depths = dict()
        for product in PRODUCTS:
            self.past_order_depths[product] = []

        # self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        self.sma_prices = dict()

        for product in PRODUCTS:
            self.ema_prices[product] = None
            self.sma_prices[product] = DEFAULT_PRICES[product]

        self.ema_param = EMA_PARAM

    def run(self, state: TradingState) -> Tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        traderData = state.traderData
        self.timestamp = state.timestamp

        self.update_ema_prices(state)

        # Orders to be placed on exchange matching engine
        for product in state.order_depths:

            if not self.product_enabled[product]:
                continue

            orders: List[Order] = []

            # Run strategy for given product and retrieve orders (if any)
            orders = self.compute_orders(product, state)
            result[product] = orders

            order_depth: OrderDepth = state.order_depths[product]

            # Addorder depth to self.past_order_depths
            self.past_order_depths[product].append(order_depth)
            if len(self.past_order_depths[product]) > 10:
                self.past_order_depths[product].pop(0)

        # String value holding Trader state data if we want to pass data to next execution of run()
        # It will be delivered as TradingState.traderData on next execution.
        traderData = ""

		# Sample conversion request. Check more details below.
        conversions = 0
        self.logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def get_position(self, product, state : TradingState) -> int:
        return state.position.get(product, 0)

    def get_mid_price(self, product, state : TradingState) -> float:

        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price

        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2

    def update_ema_prices(self, state : TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            # Update ema price
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]

    def update_sma_prices(self, state : TradingState):
        """
        Update the simple moving average of the prices of each product.
        """
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            # Update sma price
            self.past_order_depths[product].append(mid_price)
            if len(self.past_order_depths[product]) > 10:
                self.past_order_depths[product].pop(0)

            self.sma_prices[product] = sum(self.past_order_depths[product]) / len(self.past_order_depths[product])

    def compute_orders(self, product: Symbol, state: TradingState) -> List[Order]:
        """
        Calculate the acceptable price for a product.
        """

        if product == AMETHYSTS:
            return self.amethysts_strategy(state)

        if product == STARFRUIT:
            return self.starfruit_strategy(state)


    def starfruit_strategy(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of STARFRUIT.
        """
        position_starfruit = self.get_position(STARFRUIT, state)

        bid_volume = self.position_limit[STARFRUIT] - position_starfruit
        ask_volume = - self.position_limit[STARFRUIT] - position_starfruit

        max_bid = min(state.order_depths[STARFRUIT].buy_orders, key=state.order_depths[STARFRUIT].buy_orders.get)
        min_ask = max(state.order_depths[STARFRUIT].sell_orders, key=state.order_depths[STARFRUIT].sell_orders.get)

        # # get the volatility and standard deviation from the previous 10 order depths
        # average_spread, standard_deviation = self.compute_volatility(STARFRUIT, state)

        # # Now calculate how many standard deviations from the mean the spread is
        # current_spread = min_ask - max_bid
        # if standard_deviation == 0:
        #     standard_deviation = self.starfruit_spread_std
        # spread_deviation = (average_spread - current_spread) / standard_deviation

        # # print(f"spread deviation: {spread_deviation}")

        # # Adjust spread based on volatility
        # # self.starfruit_spread_mean*(1-len(self.past_order_depths[STARFRUIT])/10) +
        # # dynamic_spread = (spread_deviation*self.starfruit_spread_deviation_coeff+average_spread*self.starfruit_spread_coeff)*len(self.past_order_depths[STARFRUIT])/10

        # dynamic_spread = max(min(spread_deviation/8, -1), 1)

        orders = []

        if position_starfruit == 0:
            bid_price = round(self.ema_prices[STARFRUIT] - 1)
            ask_price = round(self.ema_prices[STARFRUIT] + 1)

            if bid_price == ask_price or bid_price > ask_price:
                ask_price += 1
                bid_price -= 1


            # Not long nor short
            orders.append(Order(STARFRUIT, bid_price, bid_volume))
            orders.append(Order(STARFRUIT, ask_price, ask_volume))

        if position_starfruit > 0:
            bid_price = round(self.ema_prices[STARFRUIT] - 1)
            ask_price = round(self.ema_prices[STARFRUIT])


            if bid_price == ask_price or bid_price > ask_price:
                ask_price += 1

            # Long position
            orders.append(Order(STARFRUIT, bid_price, bid_volume))
            orders.append(Order(STARFRUIT, ask_price, ask_volume))

            #If there are any asks below ema, create a buy order for all of them
            for ask in state.order_depths[STARFRUIT].sell_orders:
                if ask < self.ema_prices[STARFRUIT]:
                    orders.append(Order(STARFRUIT, ask, -state.order_depths[STARFRUIT].sell_orders[ask]))

        if position_starfruit < 0:

            bid_price = round(self.ema_prices[STARFRUIT])
            ask_price = round(self.ema_prices[STARFRUIT])

            if bid_price == ask_price or bid_price > ask_price:
                bid_price -= 1


            # Short position
            orders.append(Order(STARFRUIT, bid_price, bid_volume))
            orders.append(Order(STARFRUIT, ask_price, ask_volume))

            # Also, if there are any bids above ema, create a sell order for all of them
            for bid in state.order_depths[STARFRUIT].buy_orders:
                if bid > self.ema_prices[STARFRUIT]:
                    orders.append(Order(STARFRUIT, bid, state.order_depths[STARFRUIT].buy_orders[bid]))

        return orders

    def amethysts_strategy(self, state : TradingState) -> List[Order]:
        orders = []
        position_amethysts = self.get_position(AMETHYSTS, state)

        bids_above_default = [bid for bid in state.order_depths[AMETHYSTS].buy_orders if bid > DEFAULT_PRICES[AMETHYSTS]]
        asks_below_default = [ask for ask in state.order_depths[AMETHYSTS].sell_orders if ask < DEFAULT_PRICES[AMETHYSTS]]

        bid_volume = self.position_limit[AMETHYSTS] - position_amethysts
        ask_volume = - self.position_limit[AMETHYSTS] - position_amethysts

        orders.append(Order(AMETHYSTS, DEFAULT_PRICES[AMETHYSTS] - 1, bid_volume))
        orders.append(Order(AMETHYSTS, DEFAULT_PRICES[AMETHYSTS] + 1, ask_volume))

        # If there are any bids above the default price, create a buy order for all of them
        for bid in bids_above_default:
            orders.append(Order(AMETHYSTS, bid, state.order_depths[AMETHYSTS].buy_orders[bid]))

        # If there are any asks below the default price, create a sell order for all of them
        for ask in asks_below_default:
            orders.append(Order(AMETHYSTS, ask, -state.order_depths[AMETHYSTS].sell_orders[ask]))

        return orders


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
