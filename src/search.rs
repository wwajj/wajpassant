//! Minimax search with Alpha-Beta pruning.
//!
//! This module navigates the game tree using the Negamax framework. It evaluates
//! future positions and prunes mathematically dead branches to find the optimal
//! move for the current player.

use crate::board::Board;
use crate::moves::Move;
use crate::board::PieceType;

/// A score representing absolute victory (Checkmate).
pub const INFINITY: i32 = 50000;
pub const MATE_VALUE: i32 = 49000;

/// The root search function. Iterates through all legal moves in the current 
/// position, initiates the Negamax recursion, and returns the best move found.
pub fn search_best_move(board: &mut Board, depth: u8) -> Option<Move> {
    let moves = board.generate_all_moves();
    
    let mut best_move: Option<Move> = None;
    let mut best_score = -INFINITY;
    
    let mut alpha = -INFINITY;
    let beta = INFINITY;

    for mv in moves {
        if !board.make_move(mv) {
            continue;
        }

        let score = -negamax(board, depth - 1, -beta, -alpha);
        board.unmake_move(mv);

        if score > best_score {
            best_score = score;
            best_move = Some(mv);
        }

        if score > alpha {
            alpha = score;
        }
    }

    best_move
}

/// The recursive search function. 
fn negamax(board: &mut Board, depth: u8, mut alpha: i32, beta: i32) -> i32 {
    if depth == 0 {
        return board.evaluate();
    }

    let moves = board.generate_all_moves();
    let mut legal_moves = 0;

    for mv in moves {
        if !board.make_move(mv) {
            continue; 
        }
        legal_moves += 1;

        let score = -negamax(board, depth - 1, -beta, -alpha);
        board.unmake_move(mv);

        if score >= beta {
            return beta;
        }
        
        if score > alpha {
            alpha = score;
        }
    }

    if legal_moves == 0 {
        let us = board.side_to_move;
        let them = us.flip();
        let king_sq = board.pieces[us as usize][PieceType::King as usize].get_lsb();
        
        if board.is_square_attacked(king_sq, them) {
            return -MATE_VALUE;
        } else {
            return 0;
        }
    }

    alpha
}
