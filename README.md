# WajPassant

A high-performance, UCI-compatible chess engine written in Rust. WajPassant is built around a custom **NNUE (Efficiently Updatable Neural Network)** and a deeply optimized, lock-free search architecture. It leverages self-play Reinforcement Learning (Expert Iteration) to continuously bootstrap its own tactical and positional intuition.

**Current Strength:** ~2280 - 2300 Elo *(National Master level, calibrated via fastchess against unhandicapped 2000+ Elo benchmarks)*

---

## Key Features

### Advanced Search Architecture
* **Lazy SMP (Symmetric Multiprocessing):** Multi-threaded search leveraging multi-core CPUs via a dedicated thread pool. Threads explore the tree independently while sharing discoveries instantly.
* **Lock-Free Transposition Table:** A custom micro-spinlock array implementation for the TT, completely eliminating OS scheduler bottlenecks and enabling millions of nodes per second (NPS) across threads without lock contention.
* **$O(1)$ MovePicker:** Lazy move selection using zero-allocation swap-remove mechanisms to prioritize TT hits, tactical captures, and killer moves instantly.
* **Thread-Local Heuristics:** Thread-isolated Killer Moves and History Hierarchy arrays to prevent global lock gridlock and ensure branch-specific accuracy.

### Pruning & Search Heuristics
* **Dynamic Null Move Pruning (NMP):** Scales reduction depth dynamically based on the current search depth to prune useless branches without suffering from tactical blindness.
* **History-Informed LMR (Late Move Reductions):** Integrates with the History Heuristic to dynamically adjust reductions for quiet moves based on their historical beta-cutoff success.
* **Quiescence Search & SEE:** Deep tactical resolution using MVV-LVA (Most Valuable Victim - Least Valuable Attacker) and Static Exchange Evaluation (SEE) to mathematically simulate material outcomes without mutating the board state.

### Neural Network Evaluation (NNUE)
* **Custom NNUE Backend:** Replaced classical static evaluation and Piece-Square Tables with a custom neural network loaded at runtime (`wajpassant.bin`), providing profound positional understanding, King safety evaluation, and complex material imbalance resolution.
* **Tapered Integration:** Flawlessly scales evaluation profiles from complex middlegames down to deep endgames.

### Reinforcement Learning Pipeline (Expert Iteration)
WajPassant includes a custom-built, Rayon-parallelized data mining environment designed to bootstrap intelligence via self-play:
* **Multi-Core AI Data Miner (`ai_datagen.rs`):** Orchestrates thousands of simultaneous self-play games at high search depths to generate high-quality datasets.
* **Expert Iteration:** The engine uses its heavily optimized search tree (the "Expert") to discover deep tactical truths, extracting millions of quiet positions. These positions are then used to train the next generation of the NNUE (the "Apprentice"), pushing the engine's Elo exponentially higher with each iteration.

---

## Project Structure

WajPassant is structured as a multi-binary Cargo workspace:

* `src/main.rs`: The primary engine binary, handling both the standard UCI loop and direct terminal CLI play.
* `src/search.rs` & `src/tt.rs`: The core of the engine, featuring the Negamax alpha-beta search, pruning heuristics, and the lock-free Transposition Table.
* `src/bin/ai_datagen.rs`: The heavily parallelized self-play data miner used for generating NNUE training datasets.
* `wajpassant.bin`: The compiled NNUE network weights loaded by the engine at runtime.

---

## Installation & Usage

### Building the Project
Ensure you have the latest stable version of [Rust and Cargo](https://rustup.rs/) installed.

```bash
git clone https://github.com/wwajj/wajpassant.git
cd wajpassant
cargo build --release
```

### Running the Engine
To start the engine in UCI mode (for testing in the terminal or plugging into a GUI like CuteChess, Arena, or En Croissant):
```bash
cargo run --release
```
Alternatively, to play against the engine directly in your terminal:
```bash
cargo run --release -- --cli
```
