from datamodel import OrderDepth, TradingState, Order, Symbol, ProsperityEncoder, Listing, Trade, Observation
from typing import List, Any, Tuple
import json
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
DEFAULT_PRICES = {
    AMETHYSTS : 10_000,
    STARFRUIT : 5_000,
}


EMA_PARAM = 0.5

class Trader:
    def __init__(self) -> None:

        self.position_limit = POSITION_LIMIT
        self.product_enabled = PRODUCT_ENABLED

        # Values to compute pnl
        self.cash = 0
        self.round = 0

        self.logger = Logger()

        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()
        for product in PRODUCTS:
            self.past_prices[product] = []

        # self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        for product in PRODUCTS:
            self.ema_prices[product] = None

        self.ema_param = EMA_PARAM

    def run(self, state: TradingState) -> Tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        traderData = state.traderData
        self.update_ema_prices(state)

        # Orders to be placed on exchange matching engine
        for product in state.order_depths:

            if not self.product_enabled[product]:
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Run strategy for given product and retrieve orders (if any)
            orders = self.run_strategy(product, state)
            result[product] = orders

        # String value holding Trader state data if we want to pass data to next execution of run()
        # It will be delivered as TradingState.traderData on next execution.
        traderData = ""

		# Sample conversion request. Check more details below.
        conversions = 0

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

    def run_strategy(self, product: Symbol, state: TradingState) -> List[Order]:
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
        position_STARFRUIT = self.get_position(STARFRUIT, state)

        bid_volume = self.position_limit[STARFRUIT] - position_STARFRUIT
        ask_volume = - self.position_limit[STARFRUIT] - position_STARFRUIT

        orders = []

        if position_STARFRUIT == 0:
            # Not long nor short
            orders.append(Order(STARFRUIT, math.floor(self.ema_prices[STARFRUIT] - 1), bid_volume))
            orders.append(Order(STARFRUIT, math.ceil(self.ema_prices[STARFRUIT] + 1), ask_volume))

        if position_STARFRUIT > 0:
            # Long position
            orders.append(Order(STARFRUIT, math.floor(self.ema_prices[STARFRUIT] - 2), bid_volume))
            orders.append(Order(STARFRUIT, math.ceil(self.ema_prices[STARFRUIT]), ask_volume))

        if position_STARFRUIT < 0:
            # Short position
            orders.append(Order(STARFRUIT, math.floor(self.ema_prices[STARFRUIT]), bid_volume))
            orders.append(Order(STARFRUIT, math.ceil(self.ema_prices[STARFRUIT] + 2), ask_volume))

        return orders

    def amethysts_strategy(self, state : TradingState) -> List[Order]:
        orders = []
        position_amethysts = self.get_position(AMETHYSTS, state)


        max_bid = max(state.order_depths[AMETHYSTS].buy_orders, key=state.order_depths[AMETHYSTS].buy_orders.get)
        min_ask = min(state.order_depths[AMETHYSTS].sell_orders, key=state.order_depths[AMETHYSTS].sell_orders.get)

        bid_volume = self.position_limit[AMETHYSTS] - position_amethysts
        ask_volume = - self.position_limit[AMETHYSTS] - position_amethysts

        # 80% of hte time, create buy order 1 tick above best bid and 1 tick below best ask
        if self.round % 5 != 0:

            if max_bid == DEFAULT_PRICES[AMETHYSTS] -1:
                orders.append(Order(AMETHYSTS, max_bid, bid_volume))
            else:
                orders.append(Order(AMETHYSTS, max_bid + 1, bid_volume))

            if min_ask == DEFAULT_PRICES[AMETHYSTS] + 1:
                orders.append(Order(AMETHYSTS, min_ask, ask_volume))
            else:
                orders.append(Order(AMETHYSTS, min_ask - 1, ask_volume))
            return orders

        # 20% of the time, create buy order 1 ticks below default price and sell order 1 ticks above default price
        orders.append(Order(AMETHYSTS, DEFAULT_PRICES[AMETHYSTS] - 1, bid_volume))
        orders.append(Order(AMETHYSTS, DEFAULT_PRICES[AMETHYSTS] + 1, ask_volume))

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
