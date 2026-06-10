//! High-performance, stack-allocated move storage.
//!
//! During move generation, a chess engine evaluates millions of positions per second.
//! Dynamically allocating memory (like using a standard `Vec`) for each position's
//! move list would be a massive performance bottleneck.
//!
//! The `MoveList` solves this by pre-allocating a fixed-size array directly on the
//! stack. Because the mathematical upper limit for legal moves in any chess position
//! is 218, a capacity of 256 guarantees we will never overflow while maintaining
//! lightning-fast memory access.

use crate::moves::Move;

/// A fixed-capacity array containing the moves generated for a single position.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MoveList {
    /// The pre-allocated array of moves.
    moves: [Move; 256],
    /// The total number of valid moves currently stored in the list.
    count: usize,
}

// Default trait implementation.
impl Default for MoveList {
    /// Constructs a `MoveList` initialized with 256 empty/null moves and a count of 0.
    fn default() -> Self {
        Self {
            moves: [Move::empty(); 256],
            count: 0,
        }
    }
}

// Inherent Methods
impl MoveList {
    /// Constructs a new, empty `MoveList`.
    #[inline(always)]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Appends a `Move` to the end of the list and increments the internal counter.
    ///
    /// **Panics:** If the engine attempts to push more than 256 moves in a single turn
    /// (only possible in debug mode).
    #[inline(always)]
    pub fn push(&mut self, mv: Move) {
        debug_assert!(
            self.count < 256,
            "FATAL: MoveList overflowed the 256 limit!"
        );
        self.moves[self.count] = mv;
        self.count += 1;
    }

    /// Returns a standard Rust iterator over only the valid moves currently in the list.
    /// This allows for idiomatic loops like `for mv in movelist.iter() { ... }`.
    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, Move> {
        self.moves[0..self.count].iter()
    }
}
