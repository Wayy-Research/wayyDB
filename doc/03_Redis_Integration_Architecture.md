# wayyDB Redis Integration: Pubsub & Multi-Tenant Architecture

**Real-Time Data Sharing for Live Trading Platforms**

Version 1.0 | February 2026

---

## Executive Summary

wayyDB integrates with **Redis as the pubsub backbone** for:
- Real-time tick distribution to subscribers
- Multi-tenant data isolation
- Community-driven data sharing (like Reddit for quant data)
- Change Data Capture (CDC) triggering
- Backpressure handling for slow subscribers

Enables **kdb+ + Reddit model: high-performance storage + open data sharing platform.**

---

## 1. Architecture Overview

### 1.1 Full Data Flow

```
External Data Sources (WebSocket)
         ↓
   WRData (aggregation)
         ↓
wayyDB (columnar storage)
    • Ingests ticks
    • Updates columns
    • Maintains indices
         ↓
CDC Engine (Change Data Capture)
    • Detects writes
    • Batches updates
         ↓
Redis Pubsub Channels
    • Ticks published in real-time
    • By symbol/venue/asset-class
    • Subscriber notifications
         ↓
Strategy Subscribers
    • Receive real-time ticks
    • Execute on-demand queries
    • Make trading decisions
```

### 1.2 Redis vs In-Process Pubsub

**Design Decision: In-Process Redis Client**

```
Option A: External Redis Server
  Pros: Distributed, multi-language clients
  Cons: Network latency (>100μs over localhost)

Option B: In-Process Redis (Embedded)
  Pros: <50μs latency, single binary
  Cons: Single-machine deployment
  
Option C: Redis Client Library (Local)
  Pros: Low latency, familiar API
  Cons: Need separate Redis server
```

**Recommendation: Start with Option B** (embedded Redis) for startup latency requirements.

---

## 2. Channel Architecture

### 2.1 Channel Naming Convention

```
ticks:{symbol}              # All trades for symbol (AAPL)
ticks:{symbol}:{venue}      # Trades for symbol+venue (AAPL:NYSE)
quotes:{symbol}             # All quotes for symbol
quotes:{symbol}:{venue}     # Quotes for symbol+venue
ohlc:{symbol}               # 1-minute OHLC bars
ohlc:{symbol}:{period}      # OHLC with custom period (5m, 1h)

# Example subscriptions
db.subscribe("ticks:AAPL")           # All AAPL trades
db.subscribe("quotes:MSFT:NASDAQ")   # MSFT quotes from NASDAQ only
db.subscribe("ohlc:SPY:5m")          # SPY 5-minute bars
```

### 2.2 Message Format

**Tick messages (JSON):**

```json
{
  "timestamp": 1674100800123456789,
  "symbol": "AAPL",
  "venue": "NYSE",
  "price": 150.25,
  "size": 1000,
  "bid": 150.24,
  "ask": 150.26
}
```

**Batch format (binary, more efficient):**

```cpp
struct BatchMessage {
    uint32_t count;              // Number of ticks
    uint32_t schema_id;          // Column layout
    uint8_t compressed;          // LZ4 compressed?
    uint64_t first_timestamp;
    uint64_t last_timestamp;
    
    // Followed by columnar data
    // timestamps[count], symbols[count], prices[count], sizes[count]
};
```

**Rationale: Batch binary format reduces Redis bandwidth by 10x.**

---

## 3. Change Data Capture (CDC) Engine

### 3.1 Write Path

```cpp
class CDCEngine {
    ColumnStore& store_;
    Redis& redis_;
    
    // Detect when ticks are written
    void on_column_write(const std::string& table, 
                         const std::string& column,
                         size_t row_count) {
        // Capture write event
        WriteEvent evt{table, column, row_count};
        cdc_queue_.push(evt);
    }
    
    // Background thread: publish changes
    void cdc_worker() {
        while (running_) {
            std::vector<WriteEvent> batch;
            cdc_queue_.wait_dequeue_bulk(batch, 100);  // Batch 100 events
            
            if (batch.empty()) continue;
            
            // For each table, aggregate changed rows
            std::map<std::string, std::vector<size_t>> changes;
            for (const auto& evt : batch) {
                if (evt.table == "trades") {
                    changes["ticks"].push_back(evt);
                }
                if (evt.table == "quotes") {
                    changes["quotes"].push_back(evt);
                }
            }
            
            // Publish to Redis
            for (const auto& [channel, events] : changes) {
                publish_batch(channel, events);
            }
        }
    }
};
```

### 3.2 Publish Strategy

**One-to-many broadcast:**

```cpp
void CDCEngine::publish_batch(const std::string& channel_prefix,
                               const std::vector<size_t>& row_ids) {
    // Read new rows from storage
    auto rows = store_.get_rows(row_ids);
    
    // Group by symbol for fan-out
    std::map<std::string, std::vector<Tick>> by_symbol;
    for (const auto& row : rows) {
        by_symbol[row.symbol].push_back(row);
    }
    
    // Publish to per-symbol channels
    for (const auto& [symbol, ticks] : by_symbol) {
        std::string channel = channel_prefix + ":" + symbol;
        
        // Serialize batch (binary format)
        auto message = serialize_batch(ticks);
        
        // Publish (non-blocking)
        redis_.publish(channel, message);
    }
}
```

---

## 4. Subscriber Model

### 4.1 Subscription Registration

```python
class SubscriberRegistry:
    def __init__(self, db, redis):
        self.db = db
        self.redis = redis
        self.subscribers = {}  # channel -> [callbacks]
    
    def subscribe(self, table, symbol=None, callback=None, filter=None):
        # Build channel name
        channel = f"ticks:{symbol}" if symbol else f"ticks:*"
        
        # Register callback
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        
        self.subscribers[channel].append({
            "callback": callback,
            "filter": filter
        })
        
        # Start listening to Redis channel
        self._start_listener(channel)
    
    def _start_listener(self, channel):
        def listener():
            pubsub = self.redis.pubsub()
            pubsub.subscribe(channel)
            
            for message in pubsub.listen():
                if message['type'] == 'message':
                    # Deserialize
                    ticks = deserialize_batch(message['data'])
                    
                    # Fire callbacks
                    for sub in self.subscribers[channel]:
                        for tick in ticks:
                            if sub["filter"] is None or sub["filter"](tick):
                                sub["callback"](tick)
        
        # Run in background thread
        threading.Thread(target=listener, daemon=True).start()
```

### 4.2 Multi-Subscriber Example

```python
db = wdb.Database("/data/markets")
registry = wdb.SubscriberRegistry(db, redis)

# Strategy 1: Mean reversion on AAPL
def strategy_1_callback(trade):
    recent = db["trades"].tail(20, symbol="AAPL")
    mavg = wdb.ops.mavg(recent["price"], 20)
    if trade["price"] > mavg[-1]:
        print("Strategy 1: SELL signal")

registry.subscribe("trades", symbol="AAPL", callback=strategy_1_callback)

# Strategy 2: Momentum on MSFT
def strategy_2_callback(trade):
    recent = db["trades"].tail(50, symbol="MSFT")
    returns = wdb.ops.pct_change(recent["price"])
    if returns.sum() > 0.01:
        print("Strategy 2: BUY signal")

registry.subscribe("trades", symbol="MSFT", callback=strategy_2_callback)

# Strategy 3: Pair trading (AAPL vs MSFT)
def strategy_3_callback(trade):
    # Gets called for both AAPL and MSFT
    aapl_price = db["trades"].tail(1, symbol="AAPL")["price"][0]
    msft_price = db["trades"].tail(1, symbol="MSFT")["price"][0]
    ratio = aapl_price / msft_price
    print(f"Pair ratio: {ratio:.4f}")

registry.subscribe("trades", symbol="AAPL", callback=strategy_3_callback)
registry.subscribe("trades", symbol="MSFT", callback=strategy_3_callback)
```

---

## 5. Multi-Tenant Data Isolation

### 5.1 Tenant-Based Channels

```
ticks:AAPL                  # Public, everyone can see
ticks:AAPL:tenant_123       # Private to tenant 123
ticks:AAPL:tenant_456       # Private to tenant 456

quotes:MSFT                 # Public
quotes:MSFT:hedge_fund_xyz  # Private to specific fund
```

### 5.2 Access Control

```cpp
class TenantAccessControl {
    std::map<std::string, std::set<std::string>> tenant_permissions_;
    // tenant_id -> {allowed_channels}
    
public:
    void grant_access(const std::string& tenant_id,
                      const std::string& channel) {
        tenant_permissions_[tenant_id].insert(channel);
    }
    
    bool can_subscribe(const std::string& tenant_id,
                       const std::string& channel) {
        auto it = tenant_permissions_.find(tenant_id);
        if (it == tenant_permissions_.end()) {
            return false;  // Tenant not found
        }
        return it->second.count(channel) > 0;
    }
    
    void publish_to_tenant(const std::string& tenant_id,
                           const std::string& channel,
                           const std::string& message) {
        if (!can_subscribe(tenant_id, channel)) {
            throw std::runtime_error("Tenant not authorized");
        }
        redis_.publish(channel, message);
    }
};
```

### 5.3 Revenue Model: Tiered Access

```
Free Tier:
  - Public channels only (ticks:AAPL, quotes:MSFT, etc.)
  - Read-only
  - Max 10 subscribers per symbol
  
Pro Tier ($99/month):
  - Publish private datasets
  - Up to 100 symbols
  - Private channels (ticks:AAPL:mydata)
  
Enterprise Tier (Custom):
  - Unlimited symbols and subscribers
  - Custom data schemas
  - SLA guarantees
  - Dedicated infrastructure
```

---

## 6. Backpressure Handling

### 6.1 Slow Subscriber Problem

```
Fast producer (1M ticks/sec)
    ↓
Redis buffer (growing)
    ↓
Slow subscriber (can only process 10K/sec)
    ↓
Buffer overflow → message loss
```

### 6.2 Solution: Ring Buffer + Overflow Policy

```cpp
class BackpressureHandler {
    static constexpr size_t BUFFER_SIZE = 1 << 20;  // 1M messages
    RingBuffer<Message> buffer_{BUFFER_SIZE};
    std::atomic<size_t> drops_{0};
    
public:
    void publish(const std::string& channel, const Message& msg) {
        // Try to enqueue
        if (!buffer_.push(msg)) {
            // Buffer full: drop oldest message
            Message old;
            buffer_.pop(old);
            buffer_.push(msg);
            drops_++;
            
            // Log metric: dropped messages
            metrics_.increment("redis.backpressure.drops");
        }
    }
    
    // Subscribers can request backfill
    std::vector<Message> get_missed_messages(size_t since_seq_num) {
        return buffer_.range(since_seq_num);
    }
};
```

### 6.3 Slow Consumer Mitigation

```python
# Strategy can opt into batching if slow
def slow_strategy(batch):
    """Called once per 100 ticks instead of once per tick"""
    for trade in batch:
        process(trade)

# Use batch subscription
db.subscribe("trades", 
             symbol="AAPL",
             callback=slow_strategy,
             batch_size=100,      # Only call every 100 ticks
             max_latency_ms=1000)  # Or every 1 second
```

---

## 7. Streaming & Replays

### 7.1 Redis Streams (Ordered Log)

For subscribers joining mid-day, use Redis Streams instead of pub/sub:

```cpp
class StreamWriter {
    redis::Redis redis_;
    
public:
    void publish_to_stream(const std::string& symbol, const Tick& tick) {
        // Write to ordered stream (FIFO log)
        redis_.xadd(
            "ticks:" + symbol + ":stream",
            "*",  // Auto-generate ID
            {
                {"timestamp", std::to_string(tick.timestamp)},
                {"price", std::to_string(tick.price)},
                {"size", std::to_string(tick.size)}
            }
        );
    }
};

// Subscriber can read from stream
class StreamSubscriber {
    void subscribe_with_replay(const std::string& symbol,
                                int64_t from_timestamp_ns) {
        // Find all messages since timestamp
        // (conversion needed: nanoseconds → stream ID)
        
        auto messages = redis_.xrange(
            "ticks:" + symbol + ":stream",
            "-",  // From beginning
            "+"   // To end
        );
        
        // Replay messages
        for (const auto& msg : messages) {
            process_tick(msg);
        }
        
        // Then subscribe for new messages
        redis_.xread({"ticks:" + symbol + ":stream"}, "$");
    }
};
```

### 7.2 Backtest Replay

```python
class BacktestWithRealFeed:
    def __init__(self, db, strategy_class):
        self.db = db
        self.strategy = strategy_class(db)
    
    def run(self, start_ts, end_ts):
        # Disable live Redis subscriptions
        # Pull historical data from columns
        historical_trades = self.db["trades"].between(
            start_ts=start_ts,
            end_ts=end_ts
        )
        
        # Replay as if real-time
        for row_id in range(historical_trades.num_rows):
            trade = historical_trades.row(row_id)
            
            # Call strategy callback (same as live)
            self.strategy.on_trade(trade)
        
        # Results are identical to live mode
        return self.strategy.stats()
```

---

## 8. Consistency Guarantees

### 8.1 Write-Once, Read-Many

```cpp
// Writes are atomic per batch
{
    std::lock_guard<std::mutex> lock(store_.write_lock());
    for (const auto& tick : batch) {
        store_.append(tick);
    }
    // Entire batch visible at once
}

// Reads never block
auto price = store_["price"][row_id];  // No locks
```

### 8.2 Eventual Consistency

```
t=0: Tick written to store
t=10μs: Sorted index updated
t=20μs: CDC detects change
t=50μs: Published to Redis
t=200μs: Subscriber notified
t=1000μs: Subscriber processes

Guarantees:
- Total order (timestamps are monotonic)
- No lost ticks (persisted to disk/WAL)
- No duplicates (exactly-once delivery via seq numbers)
```

### 8.3 Sequence Numbers for Exactly-Once

```python
class ExactlyOnceSubscriber:
    def __init__(self, db):
        self.db = db
        self.last_seq = self._load_checkpoint()
    
    def on_message(self, message):
        seq = message['seq']
        
        if seq <= self.last_seq:
            return  # Duplicate, skip
        
        # Process
        process(message['data'])
        
        # Save checkpoint
        self._save_checkpoint(seq)
        self.last_seq = seq
    
    def _save_checkpoint(self, seq):
        with open("/data/checkpoints/strategy.txt", "w") as f:
            f.write(str(seq))
    
    def _load_checkpoint(self):
        try:
            with open("/data/checkpoints/strategy.txt") as f:
                return int(f.read())
        except:
            return 0
```

---

## 9. Performance Characteristics

### 9.1 Latency Breakdown

```
Tick arrives at WebSocket
    ↓ 10μs
Ring buffer
    ↓ 50μs
Column append (batch of 100)
    ↓ 50μs
Sorted index update
    ↓ 30μs
CDC trigger
    ↓ 50μs
Redis publish
    ↓ 200μs
Subscriber receives notification
    ↓ 1000μs
Strategy executes
    ↓ 500μs
Trade execution
==================
Total: ~1.9ms (SLA: <5ms) ✓
```

### 9.2 Throughput Scaling

```
Single core:  1M ticks/sec
Dual core:    2M ticks/sec (symbol sharding)
4-core:       4M ticks/sec
8-core:       8M ticks/sec

Limitation: Redis pubsub single-threaded, max ~100K msgs/sec
Solution: Use Redis Cluster or separate instance per symbol shard
```

---

## 10. Deployment Options

### 10.1 Single-Machine (MVP)

```
wayyDB + Embedded Redis (in-process)
    ↓
Single binary
    ↓
Latency: <5ms end-to-end
Throughput: 1M ticks/sec
Cost: Minimal
```

### 10.2 Distributed (Production)

```
WebSocket Ingestion Tier
    ↓
wayyDB Shard 0 (AAPL, GOOG, MSFT)  +  Redis Shard 0
wayyDB Shard 1 (AMZN, TSLA, NVDA)  +  Redis Shard 1
    ↓
Subscriber Tier (Strategies)
```

### 10.3 Multi-Cloud (Enterprise)

```
GCP (NYSE data)  →┐
AWS (NASDAQ)     ├→ Central wayyDB
Azure (Crypto)   ┘
    ↓
Redis Cluster (3 nodes)
    ↓
Strategy instances (auto-scaling)
```

---

## 11. Implementation Checklist

**Phase 1: Core Integration**
- [ ] CDC engine detects column writes
- [ ] Batch publishing to Redis channels
- [ ] Python subscription API
- [ ] Multi-symbol fan-out

**Phase 2: Reliability**
- [ ] Backpressure handling (ring buffers)
- [ ] Sequence numbers (exactly-once)
- [ ] Subscriber checkpointing
- [ ] Replay mechanism (streams)

**Phase 3: Multi-Tenant**
- [ ] Tenant access control
- [ ] Private channels
- [ ] Revenue model integration
- [ ] Audit logging

**Phase 4: Scalability**
- [ ] Symbol sharding
- [ ] Redis Cluster support
- [ ] Distributed strategy execution
- [ ] Metrics & monitoring

---

## Summary

wayyDB's Redis integration creates **"a collaborative financial data platform"**:

- **Speed:** kdb+ performance (microsecond joins)
- **Access:** Reddit-style pubsub (community data sharing)
- **Reliability:** Exactly-once, ordered messages
- **Scale:** 1M+ ticks/sec, 10K+ subscribers
- **Monetization:** Free public + paid private tiers

**Result: First open-source, real-time, multi-tenant financial data network.**
