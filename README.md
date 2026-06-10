# WajPassant

A high-performance, UCI-compatible chess engine written in Rust. WajPassant features a highly optimized bitboard board representation and a custom-built, parallelized Texel Tuner that uses supervised machine learning to independently derive advanced positional chess intuition.

**Current Strength:** ~2150 Elo *(Calibrated via `cutechess-cli` against handicapped Stockfish-2000)*
```bash
 --------------------------------------------------
Results of WajPassant vs Stockfish-2000 (10+0.1, NULL - 1t, NULL - 16MB, UHO_openings.epd):
Elo: 149.26 +/- 55.24, nElo: 148.57 +/- 48.15
LOS: 100.00 %, DrawRatio: 32.00 %, PairsRatio: 4.67
Games: 200, Wins: 136, Losses: 55, Draws: 9, Points: 140.5 (70.25 %)
Ptnml(0-2): [11, 1, 32, 8, 48], WL/DD Ratio: inf
-------------------------------------------------- 
```

## Key Features

### Core Engine & Architecture
* **Bitboard Representation:** Fast and efficient 64-bit integer board representation for piece tracking and move generation.
* **Zobrist Hashing:** Fast position fingerprinting for transpositions and state tracking.
* **UCI Protocol Support:** Fully compatible with modern chess GUIs (CuteChess, Arena, En Croissant, etc.).
* **Pre-calculated Lookup Tables:** Generates Attack and LMR (Late Move Reduction) tables at runtime for blazing-fast search speeds.

### Static Evaluation
The engine's evaluation function relies on mathematically derived knowledge rather than hardcoded human guesses:
* **Tapered Evaluation:** Smooth, phase-based interpolation between Midgame (MG) and Endgame (EG) states.
* **Piece-Square Tables (PSTs):** 768 parameters dictating piece placement bonuses/penalties, optimized for perfect horizontal symmetry.
* **Dynamic Material Weights:** Tuned piece values that reflect their true mathematical worth in different phases (e.g., valuing the Bishop pair in open endgames).
* **Positional Heuristics:** Mobility counting and castling right bonuses.

### The Texel Tuner Pipeline
WajPassant includes a custom-built optimization environment to train the engine's evaluation parameters on datasets of hundreds of thousands of games (e.g., UHO datasets).
* **Rayon-Parallelized MSE Calculation:** Evaluates millions of positions across all CPU cores in milliseconds.
* **Coordinate Descent Optimization:** Iteratively nudges weights to minimize Mean Squared Error against dataset outcomes.
* **Perspective-Corrected Objective:** Automatically maps absolute game outcomes to side-to-move perspectives using a Logistic Sigmoid function.
* **Automated Bootstrapping:** Capable of self-play generation (`generate_data.rs`) to continuously feed higher-quality datasets back into the tuner.

## Project Structure

WajPassant is structured as a multi-binary Cargo workspace:

* `src/main.rs`: The primary UCI chess engine binary.
* `src/bin/tuner.rs`: The Texel Tuner execution script. Reads `dataset.txt`, runs Coordinate Descent, and outputs optimized Rust arrays to `pst.txt`.
* `src/bin/generate_data.rs`: The self-play dataset generator used for reinforcement learning.

## Installation & Usage

### Building the Project
Ensure you have the latest stable version of [Rust and Cargo](https://rustup.rs/) installed.

```bash
git clone https://github.com/wwajj/wajpassant.git
cd wajpassant
cargo build --release
```

### Running the engine
To start the engine in UCI mode (for testing in the terminal or plugging into a GUI):
```bash
cargo run --release
```
Alternatively, to play against the engine using the CLI:
```bash
cargo run --release --cli
```
