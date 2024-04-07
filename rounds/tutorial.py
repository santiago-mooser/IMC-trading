from datamodel import OrderDepth, TradingState, Order, Symbol, ProsperityEncoder, Listing, Trade, Observation
from typing import List, Any, Tuple
import json
import copy
import statistics
import math

# List of products and whether they are enabled
AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"
PRODUCTS = [AMETHYSTS, STARFRUIT]
PRODUCT_ENABLED = {
    AMETHYSTS : True,
    STARFRUIT : True,
}
# Position limit for each product
POSITION_LIMIT = {
    AMETHYSTS : 20,
    STARFRUIT : 20,
}

# General vars to keep track of the state of the trader
POSITION = {}
for product in PRODUCTS:
    POSITION.update({product: 0})
DEFAULT_PRICES = {
    AMETHYSTS : 10000,
    STARFRUIT : 5000,
}

# Variables to keep track of the exponential moving average of the prices
EMA_PARAM = 0.29
PRODUCT_EMA_ENABLED = {
    AMETHYSTS : True,
    STARFRUIT : True,
}

class Trader:

    def __init__(self) -> None:
        """
        If you need any variables initialized
        in the beginning of the round, put
        them here
        """
        # general parameters to keep track of the state of the trader
        self.position = copy.deepcopy(POSITION)
        self.position_limit = POSITION_LIMIT
        self.product_enabled = PRODUCT_ENABLED
        self.ema_enabled = PRODUCT_EMA_ENABLED

        # Variables used for strategies
        # self.past_order_depths keeps the list of all past prices, used in some strategies
        self.past_order_depths = dict()
        for product in PRODUCTS:
            self.past_order_depths[product] = []
        # self.ema_prices keeps an exponential moving average of prices for each product
        self.ema_prices = dict()
        self.ema_param = EMA_PARAM
        self.sma_prices = dict()
        for product in self.ema_enabled:
            self.ema_prices[product] = None
            self.sma_prices[product] = DEFAULT_PRICES[product]

        # Variables for logging
        self.timestamp = 0
        self.logger = Logger()


    def run(self, state: TradingState) -> Tuple[dict[Symbol, list[Order]], int, str]:
        """
        This is where the high-level logic of the trading will happen, such as
        iterating over the enabled products and calling the corresponding
        strategy function, or updating the state of variables used in the
        strategies. The run function is called every time the trader is executed.
        """
        result = {}
        traderData = state.traderData
        self.timestamp = state.timestamp
        self.position = state.position
        self.update_ema_prices(state)

        # Orders to be placed on exchange matching engine
        for product in state.order_depths:

            # If the product is disabled, quickly skip to the next product
            if not self.product_enabled[product]:
                continue

            # Initialize a new list of orders for the given product
            orders: List[Order] = []

            # Run strategy for given product and retrieve orders (if any)
            orders = self.compute_orders(product, state)
            result[product] = orders

            # Retrieve and add order depth to self.past_order_depths
            order_depth: OrderDepth = state.order_depths[product]
            self.past_order_depths[product].append(order_depth)
            if len(self.past_order_depths[product]) > 10:
                self.past_order_depths[product].pop(0)

        # String value holding Trader state data if we want to pass data to next execution of run()
        # It will be delivered as TradingState.traderData on next execution.
        traderData = self.logger.to_json({
            'past_order_depths': self.past_order_depths,
            'ema_prices': self.ema_prices,
            'sma_prices': self.sma_prices,
        })

		# Sample conversion request.
        conversions = 0

        self.logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def compute_orders(self, product: Symbol, state: TradingState) -> List[Order]:
        """
        This function literally just calls the correct function
        for a given product and returns the calculated orders for a product.
        """

        if product == AMETHYSTS:
            return self.amethysts_strategy(state)

        if product == STARFRUIT:
            return self.starfruit_strategy(state)


############################################
    ## Strategies for the various products #
############################################
    # These are the functions that are     #
    # called in order to price a product.  #
    # Call them from the computer_orders() #
    # function so that they are easier to  #
    # manage.                              #
############################################
    def starfruit_strategy(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of STARFRUIT.
        Starfruit is a volatile product, so we want to create orders
        that are close to the exponential moving average of the price, which should
        allow us to profit from the volatility.
        """
        position_starfruit = self.get_position(STARFRUIT, state)

        bid_volume = self.position_limit[STARFRUIT] - position_starfruit
        ask_volume = - self.position_limit[STARFRUIT] - position_starfruit

        orders = []

        if position_starfruit == 0:
            # If we have no position, we want to create a bid and an ask order
            bid_price = round(self.ema_prices[STARFRUIT] - 1)
            ask_price = round(self.ema_prices[STARFRUIT] + 1)

            if bid_price == ask_price or bid_price > ask_price:
                ask_price += 1
                bid_price -= 1
            # Not long nor short
            orders.append(Order(STARFRUIT, bid_price, bid_volume))
            orders.append(Order(STARFRUIT, ask_price, ask_volume))

        if position_starfruit > 0:
            # if we have a long position, we want to create a bid and an ask order
            # in order to close the position
            bid_price = round(self.ema_prices[STARFRUIT] - 1)
            ask_price = round(self.ema_prices[STARFRUIT])


            if bid_price == ask_price or bid_price > ask_price:
                ask_price += 1

            # Long position
            orders.append(Order(STARFRUIT, bid_price, bid_volume))
            orders.append(Order(STARFRUIT, ask_price, ask_volume))

            # If there are any asks below ema, create a buy order for all of them
            for ask in state.order_depths[STARFRUIT].sell_orders:
                if ask < self.ema_prices[STARFRUIT]:
                    orders.append(Order(STARFRUIT, ask, -state.order_depths[STARFRUIT].sell_orders[ask]))

        if position_starfruit < 0:
            # If we have a short position, we want to create a bid and an ask order
            # in order to close the position
            bid_price = round(self.ema_prices[STARFRUIT])
            ask_price = round(self.ema_prices[STARFRUIT]+1)

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
        """
        Calculate the orders for AMETHYSTS.
        These are basically just bonds so here we're just buying anything under the default price
        and selling anything above the default price.
        Since we're not playing against other players, we probably don't need to force a strictly
        winning strategy (i.e. always +1 or -1 from the default price) but we can just buy/sell
        whatever is available., which should give us more profits.
        """
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

############################################
    # Helper functions for the strategies #
############################################
    # If you need any helper functions in #
    # order to compute the trategy for the#
    # product, you can define them here   #
############################################

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
            if not self.ema_enabled[product]:
                continue
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
