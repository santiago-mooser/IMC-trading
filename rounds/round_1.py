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

class StoikovMarketMaker(Strategy):
    def __init__(self, name: str, max_pos: int):
        super().__init__(name, max_pos)
        self.alpha = 0.5
        self.beta = 0.05
        self.target_inventory = 5
        self.max_inventory = max_pos
        self.inventory = 0  # Initial inventory
        self.mid_price = None
    def trade(self, trading_state: TradingState, orders: list):
        order_depth: OrderDepth = trading_state.order_depths[self.name]
        # Check if there are any SELL orders

        
    def calculate_mid_price(self, trading_state: TradingState) -> float:

        # if self.mid_price is None:
        #     default_price = DEFAULT_PRICES[self.name]

        market_bids = trading_state.order_depths[self.name].buy_orders
        # if len(market_bids) == 0:
        #     # There are no bid orders in the market (midprice undefined)
        #     return default_price

        market_asks = trading_state.order_depths[self.name].sell_orders
        # if len(market_asks) == 0:
        #     # There are no bid orders in the market (mid_price undefined)
        #     return default_price

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        self.mid_price =  (best_bid + best_ask)/2

    
##############################################################################################
### RegressionStrategy, using regression to find the best fair price
##############################################################################################
    
class RegressionStrategy(Strategy):
    def __init__(self, name: str, min_req_price_difference: int, max_position: int):
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
        self.intercept = 1.2389442725070694
        self.coef = [ 0.15033769,  0.17131938,  0.28916903,  0.37856112, -2.27925121, -3.1545027 ,  0.01490655,  0.0156036 ,  0.03238973, -0.02205384]
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
        self.mm.update_inventory(self.prod_position)
        avg_bid, avg_ask = self.mm.calculate_quotes(avg_bid,avg_ask)       
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
### Apply strategy to products
##############################################################################################
class Starfruit(RegressionStrategy):
    def __init__(self):
        super().__init__("STARFRUIT", min_req_price_difference=3, max_position=20)

class Amethysts(FixedStrategy2):
    def __init__(self):
        super().__init__("AMETHYSTS", max_pos=20)

##############################################################################################
### Trader Class
##############################################################################################
class Trader:

    def __init__(self) -> None:
        self.products = {
            "STARFRUIT": Starfruit(),
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