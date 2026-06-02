//! The History Heuristic module for move ordering optimization.
//!
//! This module implements a statistical dragnet that tracks the success of quiet 
//! moves across the entire search tree. By accumulating scores for moves that 
//! frequently cause Beta cutoffs, the engine can intelligently sort unproven quiet 
//! moves based on their historical reliability, significantly improving the safety 
//! and efficiency of Late Move Reductions (LMR).

use crate::board::Color;
use crate::moves::Move;

/// A 3-dimensional scorecard tracking the success of quiet moves.
///
/// The table is dimensioned as `[Color][Start Square][Target Square]`.
/// It strictly maintains running totals, meaning scores accumulate across 
/// different branches of the same Negamax search.
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

    /// Rewards a quiet move by adding a bonus to its historical score.
    pub fn write(&mut self, side: Color, mv: Move, score: i32) {
        let color = side as usize;
        let start = mv.get_start() as usize;
        let target = mv.get_target() as usize;
        self.history_table[color][start][target] += score;
    }

    /// Retrieves the accumulated historical score for a specific move.
    pub fn read(&self, side: Color, mv: Move) -> i32 {
        let color = side as usize;
        let start = mv.get_start() as usize;
        let target = mv.get_target() as usize;
        self.history_table[color][start][target]
    }

    /// Wipes the history table, restting all scores to zero.
    pub fn clear(&mut self) {
        self.history_table = [[[0; 64]; 64]; 2];
    }
}
