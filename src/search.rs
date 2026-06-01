//! Minimax search with Alpha-Beta pruning.
//!
//! This module navigates the game tree using the Negamax framework. It evaluates
//! future positions and prunes mathematically dead branches to find the optimal
//! move for the current player.
//! To improve pruning, this module also implements Quiescence Search to allow
//! the engine to continue searching through existing captures after reaching its
//! maximum depth, MVV-LVA sorting to better order nodes to search through, 
//! Mate Distance Scoring to checkmate more efficiently, and Iterative Deepening
//! to enable more efficient deeper searches 

use crate::board::Board;
use crate::moves::{Move, FLAG_EN_PASSANT, FLAG_CAPTURE};
use crate::board::PieceType;

/// A score representing absolute victory (Checkmate).
pub const INFINITY: i32 = 50000;
pub const MATE_VALUE: i32 = 49000;

/// Piece values for MVV_LVA lookup
const MVV_LVA_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 10000];

/// The root search function. Iterates through all legal moves in the current 
/// position, initiates the Negamax recursion, and returns the best move found.
pub fn search_best_move(board: &mut Board, depth: u8) -> Option<Move> {
    let mut best_move_so_far: Option<Move> = None;

    for current_depth in 1..=depth {
        let mut moves = board.generate_all_moves();
        moves.sort_unstable_by_key(|&mv| std::cmp::Reverse(score_move_iterative(board, mv, best_move_so_far)));

        let mut best_move: Option<Move> = None;
        let mut best_score = -INFINITY;
        
        let mut alpha = -INFINITY;
        let beta = INFINITY;

        for mv in moves {
            if !board.make_move(mv) {
                continue;
            }

            let score = -negamax(board, current_depth - 1, -beta, -alpha, 1);
            board.unmake_move(mv);

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }

            if score > alpha {
                alpha = score;
            }
        }
        best_move_so_far = best_move;
    }
    
    best_move_so_far
}

/// The recursive search function. 
fn negamax(board: &mut Board, depth: u8, mut alpha: i32, beta: i32, ply: i32) -> i32 {
    if depth == 0 {
        return quiescence_search(board, alpha, beta);
    }

    let mut moves = board.generate_all_moves();
    moves.sort_unstable_by_key(|&mv| std::cmp::Reverse(score_move(board, mv)));
    let mut legal_moves = 0;

    for mv in moves {
        if !board.make_move(mv) {
            continue; 
        }
        legal_moves += 1;

        let score = -negamax(board, depth - 1, -beta, -alpha, ply + 1);
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
            return -MATE_VALUE + ply;
        } else {
            return 0;
        }
    }

    alpha
}

/// Loops through all captures to calculate tactical positions at the end of
/// a negamax search
fn quiescence_search(board: &mut Board, mut alpha: i32, beta: i32) -> i32 {
    let stand_pat = board.evaluate();
    
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    let mut captures = board.generate_captures();
    captures.sort_unstable_by_key(|&mv| std::cmp::Reverse(score_move(board, mv)));

    for mv in captures {
        if !board.make_move(mv) {
            continue; 
        }

        let score = -quiescence_search(board, -beta, -alpha);
        board.unmake_move(mv);

        if score >= beta {
            return beta;
        }
        
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

/// Scores a move using the MVA-LVA (Most Valuable Victim - Least Valuable Attacker)
/// heuristic to optimize the alpha-beta pruning
fn score_move(board: &Board, mv: Move) -> i32 {
    if mv.is_promotion() { return 9000; }

    let flag = mv.get_flags();

    if flag == FLAG_EN_PASSANT {
        return (10 * 100) - 100;
    }

    if flag == FLAG_CAPTURE {
        let attacker = board.piece_at(mv.get_start()).unwrap() as usize;
        let victim = board.piece_at(mv.get_target()).unwrap() as usize;

        return (10 * MVV_LVA_VALUES[victim]) - MVV_LVA_VALUES[attacker];
    }

    0
}

/// Helper function to sort moves for Iterative Deepening.
/// If the move is the previously calculated best move, it gives it a high score,
/// otherwise it uses the score_move function
fn score_move_iterative(board: &Board, mv: Move, best_mv: Option<Move>) -> i32 {
    if let Some(best) = best_mv {
        if mv == best { return 1000000; }
    }

    score_move(board, mv)
}
