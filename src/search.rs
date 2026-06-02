//! Minimax search with Alpha-Beta pruning.
//!
//! This module navigates the game tree using the Negamax framework. It evaluates
//! future positions and prunes mathematically dead branches to find the optimal
//! move for the current player.

use std::sync::OnceLock;

use crate::board::{Board, PieceType};
use crate::hh::HistoryHierarchy;
use crate::moves::{Move, FLAG_EN_PASSANT, FLAG_CAPTURE};
use crate::tt::{TTFlag, TranspositionTable};

/// A global, safety initialized 64x64 array to hold LMR reduction values
pub static LMR_TABLE: OnceLock<[[u8; 64]; 64]> = OnceLock::new();

/// A score representing absolute victory (Checkmate).
pub const INFINITY: i32 = 50000;
pub const MATE_VALUE: i32 = 49000;

// Reduction factor for Null Move Pruning
const NMP_R: u8 = 2;

/// Piece values for MVV_LVA lookup
const MVV_LVA_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 10000];


/// position, initiates the Negamax recursion, and returns the best move found.
pub fn search_best_move(board: &mut Board, depth: u8) -> Option<Move> {
    let mut best_move_so_far: Option<Move> = None;
    let mut tt = TranspositionTable::new(32);
    let mut hh = HistoryHierarchy::new();

    for current_depth in 1..=depth {
        let mut moves = board.generate_all_moves();
        moves.sort_unstable_by_key(|&mv| std::cmp::Reverse(score_move_iterative(board, mv, best_move_so_far, &tt, &hh, depth as i32)));

        let mut best_move: Option<Move> = None;
        let mut best_score = -INFINITY;
        
        let mut alpha = -INFINITY;
        let beta = INFINITY;

        for mv in moves {
            if !board.make_move(mv) {
                continue;
            }

            let score = -negamax(board, current_depth - 1, -beta, -alpha, 1, &mut tt, &mut hh);
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
fn negamax(board: &mut Board, depth: u8, mut alpha: i32, beta: i32, ply: i32, tt: &mut TranspositionTable, hh: &mut HistoryHierarchy) -> i32 {
    if board.is_repetition() {
        return 0;
    }

    if depth == 0 {
        return quiescence_search(board, alpha, beta);
    }

    let hash = board.hash_history.last().copied().unwrap_or(0);
    if let Some(entry) = tt.read(hash, depth) {
        match entry.flag {
            TTFlag::Exact => return entry.score,
            TTFlag::Alpha if entry.score <= alpha => return alpha,
            TTFlag::Beta if entry.score >= beta => return beta,
            _ => {}
        }
    }

    let us = board.side_to_move;
    let them = us.flip();
    let pawn_bb = board.pieces[us as usize][PieceType::Pawn as usize];
    let king_bb = board.pieces[us as usize][PieceType::King as usize];
    let king_sq = king_bb.get_lsb();
    let in_check = board.is_square_attacked(king_sq, them);

    if (!in_check) && (board.occupancies[us as usize] != (pawn_bb | king_bb)) && (depth > NMP_R + 1) {
                if board.evaluate() >= beta {
                    board.make_null_move();
                    let nm_score = -negamax(board, depth - 1 - NMP_R, -beta, -beta + 1, ply + 1, tt, hh);
                    board.unmake_null_move();
                    
                    if nm_score >= beta { return beta; }
            }
    }

    let mut moves = board.generate_all_moves();
    let tt_move = tt.probe_move(hash);
    moves.sort_unstable_by_key(|&mv| std::cmp::Reverse(score_move_iterative(board, mv, tt_move, tt, hh, ply)));

    let mut legal_moves = 0;
    let mut best_move: Option<Move> = None;
    let mut tt_flag = TTFlag::Alpha;

    for mv in moves{
        if !board.make_move(mv) {
            continue; 
        }
        legal_moves += 1;

        let mut score;
        let gives_check = board.is_square_attacked(
            board.pieces[them as usize][PieceType::King as usize].get_lsb(),
            us
        );

        if (!mv.is_capture()) && (!mv.is_promotion()) && (!in_check) && (!gives_check)
            && (board.piece_at(mv.get_target()).unwrap_or(PieceType::Pawn) != PieceType::King) 
            && (legal_moves >= 5) && (depth > 3) {

            let mut lmr_r = LMR_TABLE.get().unwrap()[depth as usize][legal_moves.min(63)];
            if lmr_r >= depth { lmr_r = depth - 1; }

            score = -negamax(board, depth - 1 - lmr_r, -beta, -alpha, ply + 1, tt, hh);

            if score > alpha {
                score = -negamax(board, depth -1, -beta, -alpha, ply + 1, tt, hh);
            }
        } else {
            score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, tt, hh);
        }
        board.unmake_move(mv);

        if score >= beta {
            tt.write(hash, depth, beta, Some(mv), TTFlag::Beta);
            tt.write_killer(mv, ply);
            
            if !mv.is_capture() && !mv.is_promotion() {
                hh.write(board.side_to_move, mv, (depth as i32) * (depth as i32));
            }
            return beta;
        }
        
        if score > alpha {
            alpha = score;
            tt_flag = TTFlag::Exact;
            best_move = Some(mv);
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

    tt.write(hash, depth, alpha, best_move, tt_flag);

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
fn score_move_iterative(board: &Board, mv: Move, best_mv: Option<Move>, tt: &TranspositionTable, hh: &HistoryHierarchy, ply: i32) -> i32 {
    if Some(mv) == best_mv {
        return 1_000_000;
    }

    let tactical_score = score_move(board, mv);
    if tactical_score != 0 {
        return 100_000 + tactical_score;
    }

    let killer_score = tt.read_killer(mv, ply);
    if killer_score != 0 {
        return killer_score;
    }

    hh.read(board.side_to_move, mv)
}

/// Pre-calculates the Late Move Reduction values for all combinations of 
/// depth and move_index to avoid floating-point math during the search.
pub fn init_lmr_table() {
    print!("Initializing LMR values...\n");
    let mut table = [[0u8; 64]; 64];
    let divisor = 2.0;

    for depth in 0..64 {
        for i in 0..64 {
            // Prevent ln(0) which returns negative infinity
            let d_f64 = if depth > 0 { depth as f64 } else { 1.0 };
            let i_f64 = if i > 0 { i as f64 } else { 1.0 };

            // The industry standard LMR formula
            let reduction = (d_f64.ln() * i_f64.ln() / divisor) as u8;

            table[depth][i] = reduction;
        }
    }

    // Lock the table globally so the engine can read it instantly
    LMR_TABLE.set(table).expect("Failed to initialize LMR Table");
    print!("Successfully Generated LMR values!\n");
}
