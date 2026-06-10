//! The History Heuristic module for move ordering optimization.
//!
//! This module implements a statistical dragnet that tracks the success and failure
//! of quiet moves across the entire search tree. By accumulating scores for moves that
//! frequently cause Beta cutoffs (and penalizing those that fail via History Malus),
//! the engine can intelligently sort unproven quiet moves based on their historical
//! reliability, significantly improving the safety and efficiency of Late Move Reductions (LMR).

use crate::board::Color;
use crate::moves::Move;

/// A 3-dimensional scorecard tracking the success and failure of quiet moves.
///
/// The table is dimensioned as `[Color][Start Square][Target Square]`.
/// It strictly maintains running totals, meaning scores accumulate across
/// different branches of the same Negamax search. Scores are bound by a gravity
/// limit to prevent integer overflow.
pub struct HistoryHierarchy {
    history_table: [[[i32; 64]; 64]; 2],
}

impl Default for HistoryHierarchy {
    /// Provides the default initialization with all scores set to 0.
    fn default() -> Self {
        Self {
            history_table: [[[0; 64]; 64]; 2],
        }
    }
}

impl HistoryHierarchy {
    /// Constructs a new, empty `HistoryHierarchy` table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates the historical score for a specific move using a gravity formula.
    /// Positive bonuses reward successful cutoffs, while negative bonuses (malus)
    /// penalize moves that failed to cause a cutoff. The score naturally tapers
    /// off as it approaches the maximum limit of 16,384 (2^14);
    pub fn write(&mut self, side: Color, mv: Move, bonus: i32) {
        let color = side as usize;
        let start = mv.get_start() as usize;
        let target = mv.get_target() as usize;

        let current_score = self.history_table[color][start][target];
        let limit = 16384;

        let new_score = current_score + bonus - (current_score * bonus.abs()) / limit;

        self.history_table[color][start][target] = new_score.clamp(-limit, limit);
    }

    /// Retrieves the accumulated historical score for a specific move.
    pub fn read(&self, side: Color, mv: Move) -> i32 {
        let color = side as usize;
        let start = mv.get_start() as usize;
        let target = mv.get_target() as usize;
        self.history_table[color][start][target]
    }

    /// Wipes the history table, resetting all scores to zero.
    pub fn clear(&mut self) {
        self.history_table = [[[0; 64]; 64]; 2];
    }
}
