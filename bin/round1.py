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
    AMETHYSTS : False,
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
NUM_PERIODS_FAST = 5 # Static time period parameter for the fast EMA
K_FAST = 2 / (NUM_PERIODS_FAST + 1) # Static smoothing factor parameter for fast EMA

NUM_PERIODS_SLOW = 10 # Static time period parameter for slow EMA
K_SLOW = 2 / (NUM_PERIODS_SLOW + 1) # Static smoothing factor parameter for slow EMA

EMA_PARAM = 0.29
PRODUCT_EMA_ENABLED = {
    AMETHYSTS : True,
    STARFRUIT : True,
}

# Variable for VWAP
PRODUCT_VWAP_ENABLED = {
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
        self.vwap_enabled = PRODUCT_VWAP_ENABLED
        self.avg_cost = dict()
        self.amethysts_positions = dict()
        self.imbalance = 0
        
        # Variables used for strategies
        # self.past_order_depths keeps the list of all past prices, used in some strategies
        self.past_order_depths = dict()
        for product in PRODUCTS:
            self.past_order_depths[product] = []
            self.avg_cost[product] = 0
        # self.ema_prices keeps an exponential moving average of prices for each product
        self.ema_prices = dict()
        self.ema_param = EMA_PARAM
        self.sma_prices = dict()
        self.ema_fast = dict()
        self.ema_slow = dict()
        self.vwap = dict()
        for product in self.ema_enabled:
            self.ema_prices[product] = None
            self.sma_prices[product] = DEFAULT_PRICES[product]
            self.ema_fast[product] = None
            self.ema_slow[product] = None
        for product in self.vwap_enabled:
            self.vwap[product] = DEFAULT_PRICES[product]
            self.total_value = dict()
            self.total_volume = dict()

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
        self.update_vwap(state)

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
            'vwap': self.vwap
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
        sell_orders = state.order_depths[STARFRUIT].sell_orders
        buy_orders = state.order_depths[STARFRUIT].buy_orders
        total_bid_volume = sum([order[1] for order in buy_orders.items()])
        total_ask_volume = sum([order[1] for order in sell_orders.items()])
        if (total_bid_volume + total_ask_volume) !=0:
            self.imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
                
        # assume the vwap is the fair price
        best_ask, best_ask_amount = list(state.order_depths[STARFRUIT].sell_orders.items())[0]
        best_bid, best_bid_amount = list(state.order_depths[STARFRUIT].buy_orders.items())[0]
        mid = round(self.get_mid_price(STARFRUIT, state))
        
        if self.imbalance > 0.1 and self.ema_slow[STARFRUIT] < self.ema_fast[STARFRUIT]:
            orders.append(Order(product, best_ask, bid_volume))
        elif self.imbalance < -0.1 and self.ema_slow[STARFRUIT] > self.ema_fast[STARFRUIT]: 
             
            orders.append(Order(product, best_bid, ask_volume))
            

        # if int(best_ask) < self.vwap[STARFRUIT]:
        #     # BUY
        #     orders.append(Order(product, best_ask, bid_volume))
        
        # if int(best_bid) > self.vwap[STARFRUIT]:
        #     # SELL
        #     orders.append(Order(product, best_bid, ask_volume))                    

        return orders

    def amethysts_strategy(self, state : TradingState) -> List[Order]:
        """
        Calculate the orders for AMETHYSTS.
        These are basically just bonds so here we're just buying anything under the default price
        and selling anything above the default price.
        Since we're not playing against other players, we probably don't need to force a strictly
        winning strategy (i.e. always +1 or -1 from the default price) but we can just buy/sell
        whatever is available., which should give us more profits.
        testing results: 242
        """
        
        for trade in state.own_trades.get(AMETHYSTS, []):
            if trade.buyer == 'SUBMISSION':
                self.amethysts_positions[trade.price] = self.amethysts_positions.get(trade.price, 0 )  + trade.quantity
            else:
                self.amethysts_positions[trade.price] = self.amethysts_positions.get(trade.price, 0 )  - trade.quantity
            
        orders = []
        position_amethysts = self.get_position(AMETHYSTS, state)
        # for k, v in self.amethysts_positions.items():
        #     if k < DEFAULT_PRICES[AMETHYSTS]:
        #         orders.append(Order(AMETHYSTS, k + 1, -v))
        #     if k > DEFAULT_PRICES[AMETHYSTS]:
        #         orders.append(Order(AMETHYSTS, k - 1, -v))


        bids_above_default = [bid for bid in state.order_depths[AMETHYSTS].buy_orders if bid > DEFAULT_PRICES[AMETHYSTS]]
        asks_below_default = [ask for ask in state.order_depths[AMETHYSTS].sell_orders if ask < DEFAULT_PRICES[AMETHYSTS]]

        bid_volume = self.position_limit[AMETHYSTS] - position_amethysts
        ask_volume = - self.position_limit[AMETHYSTS] - position_amethysts

        orders.append(Order(AMETHYSTS, round(DEFAULT_PRICES[AMETHYSTS],0) - 1, bid_volume))
        orders.append(Order(AMETHYSTS, round(DEFAULT_PRICES[AMETHYSTS],0) + 1, ask_volume))

        # # If there are any bids above the default price, create a buy order for all of them
        # for bid in bids_above_default:
        #     orders.append(Order(AMETHYSTS, bid, state.order_depths[AMETHYSTS].buy_orders[bid]))

        # # If there are any asks below the default price, create a sell order for all of them
        # for ask in asks_below_default:
        #     orders.append(Order(AMETHYSTS, ask, -state.order_depths[AMETHYSTS].sell_orders[ask]))

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
    
    def update_vwap(self, state : TradingState):
        """
        Volume-Weighted Average Price
        calculate from all the trades happened from last iteration
        """
        
        for product in PRODUCTS:
            if not self.vwap_enabled[product]:
                continue

            market_trades = state.market_trades[product]
            if product not in self.total_value:
                self.total_value[product] = 0
                self.total_volume[product] = 0
                
            for trade in market_trades:
                self.total_value[product] += trade.price * trade.quantity
                self.total_volume[product] += trade.quantity

            # Ensure we don't divide by zero in case of no trades
            if self.total_volume[product] != 0:
                vwap = self.total_value[product] / self.total_volume[product]
                self.vwap[product] = vwap
    
    
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
                self.ema_fast[product] = mid_price
                self.ema_slow[product] = mid_price
                
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]
                self.ema_fast[product] = K_FAST * mid_price + (1-K_FAST) * self.ema_fast[product]
                self.ema_slow[product] = K_SLOW * mid_price + (1-K_SLOW) * self.ema_slow[product]

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
