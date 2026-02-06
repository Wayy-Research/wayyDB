# wayyDB Python Scripting API

**High-Throughput Query Language for Live Trading Strategies**

Version 1.0 | February 2026

---

## Executive Summary

Python API designed for **microsecond-latency trading strategies** with:
- Real-time subscriptions to data feeds
- As-of joins for historical context
- Vectorized operations (NumPy-based)
- Zero-copy data access
- Lambda/event-driven execution model

---

## 1. Core API: Database & Tables

### 1.1 Database Connection

```python
import wayy_db as wdb

# Create or connect to database
db = wdb.Database("/data/markets")

# Get table
trades_table = db["trades"]
quotes_table = db["quotes"]
```

### 1.2 Table Interface

```python
# Inspect table
print(trades_table.num_rows)          # 1,234,567
print(trades_table.num_columns)       # 4
print(trades_table.column_names())    # ['timestamp', 'symbol', 'price', 'size']
print(trades_table.sorted_by)         # 'timestamp'

# Access column (zero-copy NumPy array)
prices = trades_table["price"].to_numpy()
sizes = trades_table["size"].to_numpy()

# Export to dict
data = trades_table.to_dict()  # {"timestamp": np.array, "price": np.array, ...}
```

---

## 2. Subscriptions: Real-Time Feeds

### 2.1 Subscribe to Symbol

```python
def on_aapl_trade(trade):
    """Callback fired for each AAPL trade"""
    print(f"AAPL: ${trade['price']:.2f} x {trade['size']}")

# Subscribe to real-time AAPL trades
db.subscribe("trades", symbol="AAPL", callback=on_aapl_trade)
```

### 2.2 Subscribe to Multiple Symbols

```python
def on_trade(trade):
    symbol = trade['symbol']
    price = trade['price']
    print(f"{symbol}: ${price}")

# Subscribe to basket of symbols
for sym in ["AAPL", "MSFT", "GOOGL"]:
    db.subscribe("trades", symbol=sym, callback=on_trade)
```

### 2.3 Subscribe with Filters

```python
def on_large_trade(trade):
    if trade['size'] > 10000:
        print(f"Large trade: {trade}")

# Only notify on size > 10,000
db.subscribe(
    "trades",
    symbol="AAPL",
    callback=on_large_trade,
    filter=lambda t: t['size'] > 10000
)

# Filter on price
db.subscribe(
    "trades",
    symbol="MSFT",
    callback=on_trade,
    filter=lambda t: 380 < t['price'] < 385
)
```

### 2.4 Subscribe to Aggregates

```python
# Get last 5 trades
def on_new_trade(trade):
    recent_5 = db["trades"].tail(5, symbol="AAPL")
    mavg = wdb.ops.mavg(recent_5["price"], window=5)
    print(f"5-trade MA: {mavg[-1]:.2f}")

db.subscribe("trades", symbol="AAPL", callback=on_new_trade)
```

---

## 3. Temporal Joins: Historical Context

### 3.1 As-Of Join (aj)

For each trade, get most recent quote:

```python
def on_trade(trade):
    # Get the most recent quote at this trade's timestamp
    quote = wdb.ops.aj(
        left=wdb.Table.single_row(trade),  # Single trade row as table
        right=db["quotes"],
        on=["symbol"],
        as_of="timestamp"
    )
    
    bid = quote["bid"][0]
    ask = quote["ask"][0]
    trade_price = trade["price"]
    
    # Calculate slippage
    slippage = trade_price - (bid + ask) / 2
    print(f"Slippage: {slippage:.4f}")

db.subscribe("trades", callback=on_trade)
```

### 3.2 Window Join (wj)

Get all quotes within 100ms of trade:

```python
def on_trade(trade):
    # Get all quotes from 100ms before to 0ms (now)
    quotes_window = wdb.ops.wj(
        left=wdb.Table.single_row(trade),
        right=db["quotes"],
        on=["symbol"],
        as_of="timestamp",
        before_ns=100_000_000,  # 100ms before
        after_ns=0
    )
    
    bid_avg = wdb.ops.avg(quotes_window["bid"])
    print(f"Avg bid in window: ${bid_avg:.2f}")

db.subscribe("trades", callback=on_trade)
```

---

## 4. Aggregations & Statistics

### 4.1 Basic Aggregations

```python
def on_trade(trade):
    # Quick stats on all AAPL trades
    aapl_trades = db["trades"].filter(symbol="AAPL")
    
    vwap = wdb.ops.vwap(aapl_trades["price"], aapl_trades["size"])
    avg_price = wdb.ops.avg(aapl_trades["price"])
    total_vol = wdb.ops.sum(aapl_trades["size"])
    
    print(f"VWAP: {vwap:.2f}, Avg: {avg_price:.2f}, Vol: {total_vol}")

db.subscribe("trades", symbol="AAPL", callback=on_trade)
```

### 4.2 Moving Averages

```python
def on_trade(trade):
    # 20-tick moving average
    recent_20 = db["trades"].tail(20, symbol="AAPL")
    mavg_20 = wdb.ops.mavg(recent_20["price"], window=20)
    
    current = trade["price"]
    ma = mavg_20[-1]
    
    if current > ma:
        print(f"Price ${current:.2f} above MA {ma:.2f}")
    else:
        print(f"Price ${current:.2f} below MA {ma:.2f}")

db.subscribe("trades", symbol="AAPL", callback=on_trade)
```

### 4.3 Exponential Moving Average

```python
def on_trade(trade):
    recent = db["trades"].tail(100, symbol="MSFT")
    ema = wdb.ops.ema(recent["price"], alpha=0.1)
    
    print(f"EMA: {ema[-1]:.2f}")

db.subscribe("trades", symbol="MSFT", callback=on_trade)
```

### 4.4 Standard Deviation

```python
def on_trade(trade):
    recent_50 = db["trades"].tail(50, symbol="GOOG")
    std = wdb.ops.std(recent_50["price"])
    mean = wdb.ops.avg(recent_50["price"])
    
    zscore = (trade["price"] - mean) / std
    print(f"Z-score: {zscore:.2f}")

db.subscribe("trades", symbol="GOOG", callback=on_trade)
```

---

## 5. Strategy Patterns

### 5.1 Mean Reversion Strategy

```python
class MeanReversionStrategy:
    def __init__(self, db, symbol, window=20, zscore_threshold=2.0):
        self.db = db
        self.symbol = symbol
        self.window = window
        self.threshold = zscore_threshold
        self.position = 0
        
        # Subscribe to real-time trades
        db.subscribe("trades", symbol=symbol, callback=self.on_trade)
    
    def on_trade(self, trade):
        # Get recent prices
        recent = self.db["trades"].tail(self.window, symbol=self.symbol)
        prices = recent["price"].to_numpy()
        
        # Calculate z-score
        mean = prices.mean()
        std = prices.std()
        zscore = (trade["price"] - mean) / std
        
        # Trading logic
        if zscore < -self.threshold and self.position == 0:
            print(f"BUY: {self.symbol} at {trade['price']} (z={zscore:.2f})")
            self.position = 1
            self.entry_price = trade["price"]
        
        elif zscore > self.threshold and self.position == 1:
            pnl = (trade["price"] - self.entry_price) * 100
            print(f"SELL: {self.symbol} at {trade['price']} (PnL: ${pnl:.2f})")
            self.position = 0

# Run strategy
strategy = MeanReversionStrategy(db, "AAPL", window=20, zscore_threshold=2.0)
```

### 5.2 Momentum Strategy

```python
class MomentumStrategy:
    def __init__(self, db, symbol, lookback=50, threshold=0.01):
        self.db = db
        self.symbol = symbol
        self.lookback = lookback
        self.threshold = threshold
        self.position = 0
        
        db.subscribe("trades", symbol=symbol, callback=self.on_trade)
    
    def on_trade(self, trade):
        recent = self.db["trades"].tail(self.lookback, symbol=self.symbol)
        prices = recent["price"].to_numpy()
        
        # Calculate returns
        returns = wdb.ops.pct_change(prices)
        momentum = returns.sum()  # Cumulative return
        
        if momentum > self.threshold and self.position == 0:
            print(f"BUY: momentum {momentum:.2%}")
            self.position = 1
        
        elif momentum < -self.threshold and self.position == 1:
            print(f"SELL: momentum {momentum:.2%}")
            self.position = 0

# Run strategy
strategy = MomentumStrategy(db, "MSFT", lookback=50, threshold=0.01)
```

### 5.3 Pair Trading Strategy

```python
class PairTradingStrategy:
    def __init__(self, db, symbol1, symbol2, lookback=50):
        self.db = db
        self.sym1, self.sym2 = symbol1, symbol2
        self.lookback = lookback
        self.position = 0
        
        db.subscribe("trades", symbol=symbol1, callback=self.on_trade)
        db.subscribe("trades", symbol=symbol2, callback=self.on_trade)
    
    def on_trade(self, trade):
        # Get recent prices for both
        prices1 = self.db["trades"].tail(self.lookback, symbol=self.sym1)["price"].to_numpy()
        prices2 = self.db["trades"].tail(self.lookback, symbol=self.sym2)["price"].to_numpy()
        
        # Normalize to price ratio
        ratio = prices1[-1] / prices2[-1]
        avg_ratio = (prices1 / prices2).mean()
        
        if ratio > avg_ratio * 1.02 and self.position == 0:
            print(f"SELL {self.sym1}, BUY {self.sym2} (ratio={ratio:.4f})")
            self.position = 1
        
        elif ratio < avg_ratio * 0.98 and self.position == 1:
            print(f"BUY {self.sym1}, SELL {self.sym2}")
            self.position = 0

# Run strategy
strategy = PairTradingStrategy(db, "AAPL", "MSFT", lookback=50)
```

---

## 6. Advanced Queries

### 6.1 Filter and Slice

```python
# Get last 100 AAPL trades
recent_aapl = db["trades"].tail(100, symbol="AAPL")

# Get all trades in price range
expensive = db["trades"].filter(symbol="AAPL", price_min=150, price_max=160)

# Get trades between timestamps
market_open = db["trades"].between(
    symbol="AAPL",
    start_ts=1674100800 * 1_000_000_000,  # 9:30 AM
    end_ts=1674104400 * 1_000_000_000     # 10:30 AM
)
```

### 6.2 Time-Based Queries

```python
# Last 5 minutes of trades
five_min_ago = int(time.time() * 1e9) - 300 * 1_000_000_000
recent_5min = db["trades"].since(symbol="AAPL", timestamp_ns=five_min_ago)

# VWAP for the day
open_time = 1674100800 * 1_000_000_000
day_trades = db["trades"].since(symbol="AAPL", timestamp_ns=open_time)
vwap = wdb.ops.vwap(day_trades["price"], day_trades["size"])
print(f"VWAP: {vwap:.2f}")
```

### 6.3 Cross-Table Joins

```python
# For each trade, get the bid-ask spread
trades = db["trades"]
quotes = db["quotes"]

# As-of join: each trade matched with most recent quote
matched = wdb.ops.aj(trades, quotes, on=["symbol"], as_of="timestamp")

# Calculate spreads
spreads = matched["ask"] - matched["bid"]
avg_spread = wdb.ops.avg(spreads)
print(f"Avg spread: ${avg_spread:.4f}")
```

---

## 7. Stream Processing

### 7.1 Windowed Aggregation

```python
class WindowedAggregator:
    def __init__(self, db, symbol, window_size_ms=1000):
        self.db = db
        self.symbol = symbol
        self.window_ms = window_size_ms
        self.last_window_ts = 0
        
        db.subscribe("trades", symbol=symbol, callback=self.on_trade)
    
    def on_trade(self, trade):
        current_ts = trade["timestamp"]
        
        # Check if we've moved to a new window
        if current_ts - self.last_window_ts >= self.window_ms * 1_000_000:
            # Emit previous window
            self._emit_window()
            self.last_window_ts = current_ts
    
    def _emit_window(self):
        # Get trades in window
        window_trades = self.db["trades"].between(
            symbol=self.symbol,
            start_ts=self.last_window_ts,
            end_ts=self.last_window_ts + self.window_ms * 1_000_000
        )
        
        # Aggregations
        ohlc = {
            "open": window_trades["price"].iloc[0],
            "high": wdb.ops.max(window_trades["price"]),
            "low": wdb.ops.min(window_trades["price"]),
            "close": window_trades["price"].iloc[-1],
            "volume": wdb.ops.sum(window_trades["size"])
        }
        
        print(f"Window OHLC: {ohlc}")

# Run aggregator
agg = WindowedAggregator(db, "AAPL", window_size_ms=1000)
```

---

## 8. Backtesting

### 8.1 Replay Strategy

```python
class BacktestEngine:
    def __init__(self, db, strategy_class, start_ts, end_ts):
        self.db = db
        self.strategy = strategy_class(db)
        self.start_ts = start_ts
        self.end_ts = end_ts
    
    def run(self):
        # Get all trades in backtest period
        trades = self.db["trades"].between(
            start_ts=self.start_ts,
            end_ts=self.end_ts
        )
        
        # Replay trades to strategy (in order)
        for row_id in range(trades.num_rows):
            trade = trades.row(row_id)
            self.strategy.on_trade(trade)
        
        # Print stats
        self._print_stats()
    
    def _print_stats(self):
        total_trades = self.strategy.trade_count
        pnl = self.strategy.total_pnl
        sharpe = self.strategy.sharpe_ratio()
        
        print(f"Trades: {total_trades}")
        print(f"Total PnL: ${pnl:.2f}")
        print(f"Sharpe Ratio: {sharpe:.2f}")

# Run backtest
backtest = BacktestEngine(
    db,
    MeanReversionStrategy,
    start_ts=1674086400 * 1_000_000_000,  # 2023-01-19 00:00:00
    end_ts=1674172800 * 1_000_000_000     # 2023-01-20 00:00:00
)
backtest.run()
```

---

## 9. Performance Considerations

### 9.1 Avoid in Hot Path

```python
# BAD: Rebuilds entire tail every callback
def on_trade(trade):
    recent = db["trades"].tail(100, symbol="AAPL")  # O(n) scan!
    # ...

# GOOD: Keep rolling window of recent trades
class SmartStrategy:
    def __init__(self, db, symbol):
        self.recent_prices = deque(maxlen=100)
        db.subscribe("trades", symbol=symbol, callback=self.on_trade)
    
    def on_trade(self, trade):
        self.recent_prices.append(trade["price"])
        mavg = np.mean(self.recent_prices)
        # O(1)!
```

### 9.2 Batch Operations

```python
# BAD: One join per trade (repeated work)
def on_trade(trade):
    quote = wdb.ops.aj(single_trade, db["quotes"], ...)

# GOOD: Batch trades, join once
trades_batch = []

def on_trade(trade):
    trades_batch.append(trade)
    if len(trades_batch) >= 100:
        matched = wdb.ops.aj(
            wdb.Table.from_batch(trades_batch),
            db["quotes"],
            on=["symbol"],
            as_of="timestamp"
        )
        process_matched(matched)
        trades_batch.clear()
```

---

## Summary: API Cheat Sheet

```python
# Connection
db = wdb.Database("/path")

# Tables
t = db["table_name"]
t.num_rows, t.num_columns, t.column_names()
col = t["column_name"].to_numpy()

# Subscriptions
db.subscribe(table, symbol=sym, callback=fn, filter=lambda x: ...)

# Temporal Joins
wdb.ops.aj(left, right, on=["col"], as_of="timestamp")
wdb.ops.wj(left, right, on=["col"], as_of="timestamp", before_ns=100e6, after_ns=0)

# Aggregations
wdb.ops.sum, avg, min, max, std, vwap

# Window Functions
wdb.ops.mavg, msum, mstd, ema, pct_change, diff

# Filtering
t.filter(symbol=sym, price_min=100, price_max=200)
t.tail(n, symbol=sym)
t.between(start_ts=ts1, end_ts=ts2)
```

All **optimized for microsecond latency and zero-copy access.**
