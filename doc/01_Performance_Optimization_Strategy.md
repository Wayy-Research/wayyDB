# wayyDB Performance Optimization Strategy

**Making It Blazing Fast: Microsecond Latency for Live Trading**

Version 1.0 | February 2026

---

## Executive Summary

Target: **1M+ ticks/second ingestion with <5ms end-to-end latency** (data arrives → strategy executes → trade fires).

Three optimization pillars:
1. **Lock-free ingestion** — Zero contention in hot path
2. **Batch processing** — Amortize overhead across 100 ticks
3. **Pre-computed indices** — Maintain sorts incrementally, not rebuild

---

## 1. Throughput Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| **Tick ingestion** | 1M/sec | 500K/sec |
| **As-of join latency** | <1ms | <5ms |
| **Redis publish latency** | <50μs | <200μs |
| **End-to-end** | <5ms | <20ms |
| **Memory per tick** | <100 bytes | <200 bytes |

### Breakdown: Where Does Each Millisecond Go?

```
WebSocket Tick Arrives
        ↓ (10μs)
Ring buffer (lock-free push)
        ↓ (50μs)
Batch accumulates (wait for 100 ticks)
        ↓ (30μs)
Column append (memory sequential write)
        ↓ (50μs)
Sorted index update (skip list insert O(log n))
        ↓ (30μs)
Change Data Capture trigger
        ↓ (50μs)
Redis publish (in-process, single message)
        ↓ (200μs)
Python subscriber wakes up
        ↓ (1000μs)
Strategy executes as-of join (binary search + compute)
        ↓ (500μs)
Trade execution
===================================
Total: ~2ms (budget: <5ms)
```

---

## 2. Lock-Free Ingestion Pipeline

### 2.1 SPSC Ring Buffer

**Single-Producer, Single-Consumer** — zero locks:

```cpp
template<typename T, size_t N>
class SPSCRingBuffer {
    alignas(64) T buffer_[N];
    alignas(64) std::atomic<size_t> write_pos_{0};
    alignas(64) std::atomic<size_t> read_pos_{0};
    
public:
    // Producer (WebSocket thread)
    bool push(const T& item) {
        size_t w = write_pos_.load(std::memory_order_relaxed);
        size_t next_w = (w + 1) % N;
        size_t r = read_pos_.load(std::memory_order_acquire);
        
        if (next_w == r) return false;  // Full, backpressure
        
        buffer_[w] = item;
        write_pos_.store(next_w, std::memory_order_release);
        return true;
    }
    
    // Consumer (ingestion thread)
    bool pop(T& item) {
        size_t r = read_pos_.load(std::memory_order_relaxed);
        size_t w = write_pos_.load(std::memory_order_acquire);
        
        if (r == w) return false;  // Empty
        
        item = buffer_[r];
        read_pos_.store((r + 1) % N, std::memory_order_release);
        return true;
    }
};
```

**Why lock-free?**
- Zero mutex overhead (no system calls, no context switches)
- Padding to cache line (64 bytes) prevents false sharing
- Relaxed memory order on reads (fast, only need acquire on boundaries)

---

## 3. Batch Processing

### 3.1 Batch Writer

Collect 100 ticks, write together:

```cpp
class BatchWriter {
    static constexpr size_t BATCH_SIZE = 100;
    std::vector<Tick> batch_;
    ColumnStore& store_;
    Redis& redis_;
    
public:
    void ingest(const Tick& tick) {
        batch_.push_back(tick);
        if (batch_.size() >= BATCH_SIZE) {
            flush();
        }
    }
    
    void flush() {
        if (batch_.empty()) return;
        
        // Single lock for entire batch (not per-tick!)
        auto lock = store_.write_lock();
        
        // Append all ticks to columns
        for (const auto& t : batch_) {
            store_["timestamp"].append(t.timestamp);
            store_["symbol"].append(t.symbol);
            store_["price"].append(t.price);
            store_["size"].append(t.size);
        }
        
        // Update sorted index once (incremental)
        store_.update_sorted_index(batch_);
        
        // Single Redis publish for batch
        redis_.publish("ticks:batch", serialize(batch_));
        
        batch_.clear();
    }
};
```

**Batching benefit:** 1 lock + 1 Redis publish for 100 ticks = 1/100 overhead per tick.

---

## 4. Incremental Index Maintenance

### 4.1 Skip List for Sorted Timestamps

Don't rebuild indices. Maintain during write using skip list (O(log n) insert):

```cpp
class SortedIndex {
    struct Node {
        int64_t key;
        size_t row_id;
        std::vector<Node*> forward;
        
        Node(int64_t k, size_t rid, int level) 
            : key(k), row_id(rid), forward(level + 1, nullptr) {}
    };
    
    Node* head_;
    int current_level_ = 0;
    static constexpr double P = 0.5;  // Promotion probability
    
public:
    void insert(int64_t timestamp, size_t row_id) {
        std::vector<Node*> update(current_level_ + 1);
        Node* current = head_;
        
        // Find insertion point
        for (int level = current_level_; level >= 0; --level) {
            while (current->forward[level] && 
                   current->forward[level]->key < timestamp) {
                current = current->forward[level];
            }
            update[level] = current;
        }
        
        // Create new node at random level
        int new_level = random_level();
        if (new_level > current_level_) {
            for (int i = current_level_ + 1; i <= new_level; ++i) {
                update[i] = head_;
            }
            current_level_ = new_level;
        }
        
        Node* node = new Node(timestamp, row_id, new_level);
        for (int level = 0; level <= new_level; ++level) {
            node->forward[level] = update[level]->forward[level];
            update[level]->forward[level] = node;
        }
    }
    
    // O(log n) binary search for as-of join
    size_t upper_bound(int64_t timestamp) {
        Node* current = head_;
        for (int level = current_level_; level >= 0; --level) {
            while (current->forward[level] && 
                   current->forward[level]->key <= timestamp) {
                current = current->forward[level];
            }
        }
        return current->row_id;
    }
};
```

**Why skip list?**
- O(log n) insert, no rebuild needed
- Probabilistic balance (no rebalancing operations)
- Cache-friendly node layout
- Superior to trees for this workload

---

## 5. SIMD Pre-Computation

### 5.1 Aggregate on Arrival

Compute moving averages/stats as ticks arrive:

```cpp
class SIMD_Aggregator {
    __m256d sum_vec = _mm256_setzero_pd();
    __m256d min_vec = _mm256_set1_pd(std::numeric_limits<double>::max());
    __m256d max_vec = _mm256_set1_pd(std::numeric_limits<double>::lowest());
    
public:
    void process_batch(const double* prices, size_t n) {
        // Process 4 prices at a time with AVX2
        for (size_t i = 0; i + 4 <= n; i += 4) {
            __m256d v = _mm256_loadu_pd(prices + i);
            sum_vec = _mm256_add_pd(sum_vec, v);
            min_vec = _mm256_min_pd(min_vec, v);
            max_vec = _mm256_max_pd(max_vec, v);
        }
    }
    
    double horizontal_sum() {
        // Reduce 4 doubles to 1
        __m128d v128 = _mm256_castpd256_pd128(sum_vec);
        __m128d v64 = _mm256_extractf128_pd(sum_vec, 1);
        v128 = _mm_add_pd(v128, v64);
        __m128d v32 = _mm_shuffle_pd(v128, v128, 1);
        return _mm_cvtsd_f64(_mm_add_sd(v128, v32));
    }
};
```

**Speedup: 4x for aggregations** (4 numbers processed per instruction).

---

## 6. Memory Management

### 6.1 Pre-Allocated Buffer Pool

Zero malloc in hot path:

```cpp
class MemoryPool {
    struct Block {
        void* ptr;
        size_t size;
        bool free;
    };
    std::vector<Block> blocks_;
    std::queue<Block*> free_list_;
    
public:
    MemoryPool(size_t block_size = 1 << 20, size_t num_blocks = 100) {
        for (size_t i = 0; i < num_blocks; ++i) {
            Block b;
            b.ptr = aligned_alloc(64, block_size);  // Cache-aligned
            b.size = block_size;
            b.free = true;
            blocks_.push_back(b);
            free_list_.push(&blocks_[i]);
        }
    }
    
    void* allocate() {
        if (!free_list_.empty()) {
            Block* b = free_list_.front();
            free_list_.pop();
            b->free = false;
            return b->ptr;
        }
        // Rare: pool exhausted, allocate new block
        return aligned_alloc(64, 1 << 20);
    }
    
    void deallocate(void* ptr) {
        // Return to free list (O(1), no actual free)
        for (auto& b : blocks_) {
            if (b.ptr == ptr) {
                b.free = true;
                free_list_.push(&b);
                return;
            }
        }
    }
};
```

**Result: Zero malloc/free in steady state.**

---

## 7. Write-Ahead Log (Optional Durability)

### 7.1 Append-Only Log

For production: durability without blocking ingestion:

```cpp
class DurabilityLog {
    int fd_;  // Memory-mapped file
    off_t write_offset_ = 0;
    
public:
    void append(const Tick& tick) {
        // Write to mmap'd region (not fsync)
        TickRecord rec{tick.timestamp, tick.symbol, tick.price, tick.size};
        ::memcpy((char*)mmap_ptr_ + write_offset_, &rec, sizeof(rec));
        write_offset_ += sizeof(rec);
        
        // Batch fsync every 10K records
        if (write_offset_ % (10000 * sizeof(TickRecord)) == 0) {
            ::msync(mmap_ptr_, write_offset_, MS_ASYNC);
        }
    }
};
```

**Key: MS_ASYNC** (don't block), not MS_SYNC.

---

## 8. Concurrency Model

### 8.1 Thread Layout

```
WebSocket Handler Thread 1
    ↓ (push to ring buffer)
Ring Buffer (SPSC, lock-free)
    ↓ (pop, batch)
Batch Writer Thread
    ↓ (one lock per batch)
ColumnStore (read: lock-free, write: single writer)
    ↓
Redis Pubsub Thread
    ↓
Strategy Subscriber Threads (each reads own columns, lock-free)
```

**Why single writer for storage?**
- Eliminates write-write contention
- Indices maintained by single thread (no conflicts)
- Readers never wait (copy-on-write or snapshots)

---

## 9. Benchmark Methodology

### 9.1 Test Harness

```cpp
// Generate synthetic TAQ-like data
std::vector<Tick> generate_ticks(size_t count) {
    std::vector<Tick> ticks;
    std::mt19937 gen(0);
    std::uniform_int_distribution<> symbols(0, 99);  // 100 symbols
    std::uniform_real_distribution<> price(100, 200);
    std::uniform_int_distribution<> size(1, 10000);
    
    int64_t ts = 0;
    for (size_t i = 0; i < count; ++i) {
        ts += gen() % 1000;  // 0-1000 nanosecond gaps
        ticks.push_back({
            ts,
            symbols(gen),
            price(gen),
            size(gen)
        });
    }
    return ticks;
}

// Benchmark ingestion
void benchmark_ingestion() {
    auto ticks = generate_ticks(1000000);
    BatchWriter writer;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& tick : ticks) {
        writer.ingest(tick);
    }
    writer.flush();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start
    ).count();
    
    double throughput = 1000000.0 / duration * 1e6;
    std::cout << "Throughput: " << throughput << " ticks/sec" << std::endl;
}
```

### 9.2 Expected Results

- **Throughput:** 1-2M ticks/sec (single core)
- **Latency:** p50 <100μs, p99 <500μs per batch
- **Memory:** ~50-80 bytes/tick (8 + 8 + 8 + 4 + overhead)

---

## 10. Scaling Beyond Single Core

### 10.1 Multi-Symbol Sharding

Shard by symbol (AAPL→CPU0, MSFT→CPU1, etc.):

```
WebSocket (one per exchange)
    ↓
Router: Distribute by symbol
    ↓
Worker CPU0 (AAPL/GOOG/META)  Worker CPU1 (MSFT/AMZN/TSLA)
    ↓                               ↓
Database Shard 0              Database Shard 1
    ↓                               ↓
Redis Pubsub                  Redis Pubsub
```

**Result: Linear scaling to 10+ cores, 10M+ ticks/sec.**

---

## Summary: The Speed Checklist

- [ ] Lock-free SPSC ring buffer (WebSocket → ingestion)
- [ ] Batch writer (100 ticks per flush)
- [ ] Skip list for incremental index maintenance
- [ ] SIMD aggregations (AVX2 vectorization)
- [ ] Pre-allocated memory pools (zero malloc)
- [ ] Optional WAL with async fsync
- [ ] Single-writer concurrency model
- [ ] Per-symbol sharding for multi-core
- [ ] Comprehensive benchmarking vs kdb+

**Target: 1M ticks/sec, <5ms end-to-end. Achievable.**
