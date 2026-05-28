//! Minimax search with Alpha-Beta pruning.
//!
//! This module navigates the game tree using the Negamax framework. It evaluates
//! future positions and prunes mathematically dead branches to find the optimal
//! move for the current player.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Instant, Duration};

use crate::board::{Board, PieceType};
use crate::eval::EvalParams;
use crate::hh::HistoryHierarchy;
use crate::moves::{Move, FLAG_EN_PASSANT, FLAG_CAPTURE};
use crate::tt::{TTFlag, TranspositionTable};

/// A global, safety initialized 64x64 array to hold LMR reduction values
pub static LMR_TABLE: OnceLock<[[u8; 64]; 64]> = OnceLock::new();

/// A score representing absolute victory (Checkmate).
pub const INFINITY: i32 = 50000;
pub const MATE_VALUE: i32 = 49000;

const QSEARCH_DEPTH: u8 = 0;
const MAX_EXTENSION_DEPTH: u8 = 8;

// Reduction factor for Null Move Pruning
const NMP_R: u8 = 2;

/// Piece values for MVV_LVA lookup
const MVV_LVA_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 10000];

/// Initiates the Negamax recursion and manages the time/abort lifecycle.
pub fn search_best_move(
    mut board: Board, 
    depth: u8, 
    abort_flag: Arc<AtomicBool>, 
    time_limit: Option<Duration>,
    quiet: bool
) -> Option<Move> {
    let mut best_move_so_far: Option<Move> = None;
    let mut tt = TranspositionTable::new(32);
    let mut hh = HistoryHierarchy::new();
    let params = EvalParams::new();
    
    let start_time = Instant::now();
    let mut nodes: u64 = 0;

    for current_depth in 1..=depth {
        let mut moves = board.generate_all_moves();
        moves.sort_unstable_by_key(|&mv| std::cmp::Reverse(score_move_iterative(&board, mv, best_move_so_far, &tt, &hh, current_depth as i32)));

        let mut best_move: Option<Move> = None;
        let mut best_score = -INFINITY;
        
        let mut alpha = -INFINITY;
        let beta = INFINITY;

        for mv in moves {
            if !board.make_move(mv, &params) {
                continue;
            }

            let score = -negamax(
                &mut board, current_depth - 1, -beta, -alpha, 1, 
                &mut tt, &mut hh, &abort_flag, start_time, time_limit,
                &mut nodes, &params
            );
            board.unmake_move(mv);

            if abort_flag.load(Ordering::Relaxed) {
                break;
            }

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }

            if score > alpha {
                alpha = score;
            }
        }

        if abort_flag.load(Ordering::Relaxed) {
            break;
        }
        
        best_move_so_far = best_move;

        let elapsed = start_time.elapsed().as_millis();
        let nps = if elapsed > 0 {
            (nodes * 1000) / elapsed as u64
        } else {
            nodes
        };

        if !quiet {
            if best_score.abs() > MATE_VALUE {
                let mate_in_plies = INFINITY - best_score.abs();
                let mate_in_moves = (mate_in_plies + 1) / 2;
                let sign = if best_score > 0 { 1 } else { -1 };
                println!(
                    "info depth {} score mate {} nodes {} nps {} time {}",
                    current_depth, mate_in_moves * sign, nodes, nps, elapsed
                );
            } else {
                println!(
                    "info depth {} score cp {} nodes {} nps {} time {}",
                    current_depth, best_score, nodes, nps, elapsed
                );
            }
        }
    }

    best_move_so_far
}

/// The recursive search function. 
fn negamax(
    board: &mut Board, depth: u8, mut alpha: i32, beta: i32, ply: i32, 
    tt: &mut TranspositionTable, hh: &mut HistoryHierarchy,
    abort_flag: &Arc<AtomicBool>, start_time: Instant, time_limit: Option<Duration>,
    nodes: &mut u64, params: &EvalParams
) -> i32 {
    
    *nodes += 1;
    if *nodes & 2047 == 0 {
        if abort_flag.load(Ordering::Relaxed) {
            return 0; 
        }
        if let Some(limit) = time_limit {
            if start_time.elapsed() >= limit {
                abort_flag.store(true, Ordering::Relaxed);
                return 0;
            }
        }
    }

    if board.is_repetition() {
        return 0;
    }

    if depth == 0 {
        return quiescence_search(board, alpha, beta, tt, params);
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
    let eval = board.evaluate(alpha, beta, params);

    if (!in_check) && (board.occupancies[us as usize] != (pawn_bb | king_bb)) && (depth > NMP_R + 1) {
        if eval >= beta {
            board.make_null_move();
            let nm_score = -negamax(
                board, depth - 1 - NMP_R, -beta, -beta + 1, ply + 1, 
                tt, hh, abort_flag, start_time, time_limit, nodes, params
            );
            board.unmake_null_move();
            
            if abort_flag.load(Ordering::Relaxed) { return 0; }
            
            if nm_score >= beta { return beta; }
        }
    }

    let mut moves = board.generate_all_moves();
    let tt_move = tt.probe_move(hash);
    moves.sort_unstable_by_key(|&mv| std::cmp::Reverse(score_move_iterative(board, mv, tt_move, tt, hh, ply)));

    let mut legal_moves = 0;
    let mut best_move: Option<Move> = None;
    let mut tt_flag = TTFlag::Alpha;
    let futility_margin = 150 + (depth as i32 * 100);

    for mv in moves {
        if !board.make_move(mv, params) {
            continue; 
        }
        legal_moves += 1;


        let mut score;
        let gives_check = board.is_square_attacked(
            board.pieces[them as usize][PieceType::King as usize].get_lsb(),
            us
        );
        let extension = if gives_check && ply < MAX_EXTENSION_DEPTH as i32 { 1 } else { 0 };

        let futility_zone = (eval + futility_margin) < alpha;

        if (depth <= 2) && (!in_check) && (!gives_check) && (!mv.is_capture())
            && (!mv.is_promotion()) && (eval.abs() < MATE_VALUE - 100)
            && futility_zone {
                board.unmake_move(mv);
                continue;
        }

        if (!mv.is_capture()) && (!mv.is_promotion()) && (!in_check) && (!gives_check)
            && (board.piece_at(mv.get_target()).unwrap_or(PieceType::Pawn) != PieceType::King) 
            && (legal_moves >= 5) && (depth > 3) {

            let mut lmr_r = LMR_TABLE.get().unwrap()[depth as usize][legal_moves.min(63)];
            if lmr_r >= depth { lmr_r = depth - 1; }

            score = -negamax(
                board, depth - 1 - lmr_r, -beta, -alpha, ply + 1, 
                tt, hh, abort_flag, start_time, time_limit, nodes, params
            );

            if score > alpha {
                score = -negamax(
                    board, depth - 1, -beta, -alpha, ply + 1, 
                    tt, hh, abort_flag, start_time, time_limit, nodes, params
                );
            }
        } 
        else {
            score = -negamax(
                board, depth - 1 + extension, -beta, -alpha, ply + 1, 
                tt, hh, abort_flag, start_time, time_limit, nodes, params
            );
        }
        
        board.unmake_move(mv);

        if abort_flag.load(Ordering::Relaxed) { return 0; }

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
        if in_check {
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
fn quiescence_search(board: &mut Board, mut alpha: i32, beta: i32,
    tt: &mut TranspositionTable, params: &EvalParams) -> i32 {
    let hash = board.hash_history.last().copied().unwrap_or(0);
    let mut tt_move = None;
    if let Some(entry) = tt.read(hash, QSEARCH_DEPTH) {
        tt_move = entry.best_move;
        match entry.flag {
            TTFlag::Exact => return entry.score,
            TTFlag::Alpha if entry.score <= alpha => return alpha,
            TTFlag::Beta if entry.score >= beta => return beta,
            _ => {}
        }
    }

    let stand_pat = board.evaluate(alpha, beta, params);
    
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    let mut captures = board.generate_captures();
    captures.sort_unstable_by_key(|&mv| {
        let mut score = score_move(board, mv);

        if Some(mv) == tt_move {
            score += 1_000_000;
        }

        std::cmp::Reverse(score)
    });

    let mut tt_flag = TTFlag::Alpha;
    let mut best_score = alpha;
    let mut best_q_move: Option<Move> = None;

    for mv in captures {
        if (static_exchange_evaluation(board, mv) < 0) && (!mv.is_promotion()) { continue; }

        if !board.make_move(mv, params) {
            continue; 
        }

        let score = -quiescence_search(board, -beta, -alpha, tt, params);
        board.unmake_move(mv);

        if score >= beta {
            tt.write(hash, QSEARCH_DEPTH, beta, Some(mv), TTFlag::Beta);
            return beta;
        }
        
        if score > alpha {
            alpha = score;
            best_score = score;
            tt_flag = TTFlag::Exact;
            best_q_move = Some(mv);
        }
    }

    tt.write(hash, QSEARCH_DEPTH, best_score, best_q_move, tt_flag);
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
/// Follows a strict hierarchy: TT Hit -> Tactics -> Killers -> History Heuristic.
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
            let d_f64 = if depth > 0 { depth as f64 } else { 1.0 };
            let i_f64 = if i > 0 { i as f64 } else { 1.0 };

            let reduction = (d_f64.ln() * i_f64.ln() / divisor) as u8;

            table[depth][i] = reduction;
        }
    }

    LMR_TABLE.set(table).expect("Failed to initialize LMR Table");
    print!("Successfully Generated LMR values!\n");
}

/// Mathematically simulates the material ouctome of a sequence of captures on
/// a single square without mutating the actual board state
pub fn static_exchange_evaluation(board: &Board, mv: Move) -> i32 {
    let mut gain: [i32; 32] = [0; 32];

    let start = mv.get_start();
    let target = mv.get_target();
    let mut attacker = board.side_to_move;
    let mut attacker_piece = board.piece_at(start).unwrap();
    let victim_piece = board.piece_at(target).unwrap_or(PieceType::Pawn);

    gain[0] = MVV_LVA_VALUES[victim_piece as usize];
    let mut sim_occupancy = board.occupancies[2];
    sim_occupancy = sim_occupancy.clear_bit(start);

    let mut depth = 1;
    loop {
        attacker = attacker.flip();
        gain[depth] = MVV_LVA_VALUES[attacker_piece as usize] - gain[depth - 1];
        
        if let Some(smallest_attacker) = board.get_smallest_attacker(target, attacker, sim_occupancy) {
            attacker_piece = smallest_attacker.0;
            sim_occupancy = sim_occupancy.clear_bit(smallest_attacker.1);
        } else {
            break;
        }

        depth += 1;
    }

    for d in (1..depth).rev() {
        gain[d - 1] = gain[d - 1].min(-gain[d]);
    }

    return gain[0]
}
