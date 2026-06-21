//! Minimax search with Alpha-Beta pruning.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::board::{Board, PieceType};
use crate::eval::EvalParams;
use crate::hh::HistoryHierarchy;
use crate::moves::{FLAG_CAPTURE, FLAG_EN_PASSANT, Move};
use crate::tt::{TTFlag, TranspositionTable};

pub static LMR_TABLE: OnceLock<[[u8; 64]; 64]> = OnceLock::new();
pub const INFINITY: i32 = 50000;
pub const MATE_VALUE: i32 = 49000;
const QSEARCH_DEPTH: u8 = 0;
const MAX_EXTENSION_DEPTH: u8 = 8;
const MVV_LVA_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 10000];

pub struct MovePicker {
    moves: Vec<Move>,
    scores: Vec<i32>,
}

impl MovePicker {
    pub fn new(
        board: &Board,
        tt_move: Option<Move>,
        killers: &[[Option<Move>; 2]; 64],
        hh: &HistoryHierarchy,
        ply: i32,
    ) -> Self {
        let moves = board.generate_all_moves();
        let mut scores = Vec::with_capacity(moves.len());

        for &mv in &moves {
            scores.push(score_move_iterative(board, mv, tt_move, killers, hh, ply));
        }

        Self { moves, scores }
    }

    pub fn next(&mut self) -> Option<Move> {
        if self.moves.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_score = self.scores[0];

        for i in 1..self.moves.len() {
            if self.scores[i] > best_score {
                best_score = self.scores[i];
                best_idx = i;
            }
        }

        self.scores.swap_remove(best_idx);
        Some(self.moves.swap_remove(best_idx))
    }
}

pub fn search_best_move(
    mut board: Board,
    depth: u8,
    abort_flag: Arc<AtomicBool>,
    time_limit: Option<Duration>,
    quiet: bool,
    tt: Arc<TranspositionTable>,
) -> Option<Move> {
    let mut best_move_so_far: Option<Move> = None;
    let mut best_score_so_far: i32 = 0;

    let mut hh = HistoryHierarchy::new();
    let mut killers = [[None; 2]; 64];
    let params = EvalParams::new();

    let start_time = Instant::now();
    let mut nodes: u64 = 0;

    for current_depth in 1..=depth {
        let mut moves = board.generate_all_moves();
        moves.sort_unstable_by_key(|&mv| {
            std::cmp::Reverse(score_move_iterative(
                &board,
                mv,
                best_move_so_far,
                &killers,
                &hh,
                current_depth as i32,
            ))
        });

        let mut delta = 30;
        let mut alpha = -INFINITY;
        let mut beta = INFINITY;

        if current_depth >= 4 && best_score_so_far.abs() < MATE_VALUE - 100 {
            alpha = best_score_so_far - delta;
            beta = best_score_so_far + delta;
        }

        loop {
            let mut best_move: Option<Move> = None;
            let mut best_score = -INFINITY;
            let mut current_alpha = alpha;

            for mv in &moves {
                if !board.make_move(*mv, &params) {
                    continue;
                }

                let score = -negamax(
                    &mut board,
                    current_depth - 1,
                    -beta,
                    -current_alpha,
                    1,
                    &tt,
                    &mut hh,
                    &mut killers,
                    &abort_flag,
                    start_time,
                    time_limit,
                    &mut nodes,
                    &params,
                );
                board.unmake_move(*mv);

                if abort_flag.load(Ordering::Relaxed) {
                    break;
                }

                if score > best_score {
                    best_score = score;
                    best_move = Some(*mv);
                }

                if score > current_alpha {
                    current_alpha = score;
                }
            }

            if abort_flag.load(Ordering::Relaxed) {
                break;
            }

            if best_score <= alpha {
                alpha -= delta;
                delta += delta / 2;
            } else if best_score >= beta {
                beta += delta;
                delta += delta / 2;
            } else {
                best_score_so_far = best_score;
                best_move_so_far = best_move;
                break;
            }
        }

        if abort_flag.load(Ordering::Relaxed) {
            break;
        }

        let elapsed = start_time.elapsed().as_millis();
        let nps = if elapsed > 0 {
            (nodes * 1000) / elapsed as u64
        } else {
            nodes
        };

        if !quiet {
            if best_score_so_far.abs() > MATE_VALUE {
                let mate_in_plies = INFINITY - best_score_so_far.abs();
                let mate_in_moves = (mate_in_plies + 1) / 2;
                let sign = if best_score_so_far > 0 { 1 } else { -1 };
                println!(
                    "info depth {} score mate {} nodes {} nps {} time {}",
                    current_depth,
                    mate_in_moves * sign,
                    nodes,
                    nps,
                    elapsed
                );
            } else {
                println!(
                    "info depth {} score cp {} nodes {} nps {} time {}",
                    current_depth, best_score_so_far, nodes, nps, elapsed
                );
            }
        }
    }

    best_move_so_far
}

pub fn negamax(
    board: &mut Board,
    depth: u8,
    mut alpha: i32,
    beta: i32,
    ply: i32,
    tt: &TranspositionTable,
    hh: &mut HistoryHierarchy,
    killers: &mut [[Option<Move>; 2]; 64],
    abort_flag: &Arc<AtomicBool>,
    start_time: Instant,
    time_limit: Option<Duration>,
    nodes: &mut u64,
    params: &EvalParams,
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
        return quiescence_search(board, alpha, beta, tt, params, nodes);
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

    if (depth <= 5) && (!in_check) && (ply > 0) & (eval.abs() < MATE_VALUE - 100) {
        let margin = depth as i32 * 75;
        if eval - margin >= beta {
            return eval;
        }
    }

    if (!in_check) && (board.occupancies[us as usize] != (pawn_bb | king_bb)) && (depth > 2) {
        if eval >= beta {
            board.make_null_move();
            let nmp_r = 3 + (depth / 6);
            let nm_score = -negamax(
                board,
                depth.saturating_sub(1 + nmp_r),
                -beta,
                -beta + 1,
                ply + 1,
                tt,
                hh,
                killers,
                abort_flag,
                start_time,
                time_limit,
                nodes,
                params,
            );
            board.unmake_null_move();

            if abort_flag.load(Ordering::Relaxed) {
                return 0;
            }

            if nm_score >= beta {
                return beta;
            }
        }
    }

    let mut tt_move = tt.probe_move(hash);

    if depth >= 4 && tt_move.is_none() && !in_check {
        let iid_depth = depth - 2;
        let _ = negamax(
            board, iid_depth, alpha, beta, ply, tt, hh, killers, abort_flag, start_time,
            time_limit, nodes, params,
        );

        tt_move = tt.probe_move(hash);
    }

    let mut move_picker = MovePicker::new(board, tt_move, killers, hh, ply);

    let mut legal_moves = 0;
    let mut best_move: Option<Move> = None;
    let mut tt_flag = TTFlag::Alpha;
    let mut quiet_moves_played: [Option<Move>; 64] = [None; 64];
    let mut quiet_count = 0;
    let futility_margin = 150 + (depth as i32 * 100);

    while let Some(mv) = move_picker.next() {
        if !board.make_move(mv, params) {
            continue;
        }
        legal_moves += 1;

        let is_quiet = !mv.is_capture() && !mv.is_promotion();
        if is_quiet && quiet_count < 64 {
            quiet_moves_played[quiet_count] = Some(mv);
            quiet_count += 1;
        }

        let mut score;
        let gives_check = board.is_square_attacked(
            board.pieces[them as usize][PieceType::King as usize].get_lsb(),
            us,
        );
        let extension = if gives_check && ply < MAX_EXTENSION_DEPTH as i32 {
            1
        } else {
            0
        };

        let futility_zone = (eval + futility_margin) < alpha;

        if (depth <= 2)
            && (!in_check)
            && (!gives_check)
            && (!mv.is_capture())
            && (!mv.is_promotion())
            && (eval.abs() < MATE_VALUE - 100)
            && futility_zone
        {
            board.unmake_move(mv);
            continue;
        }

        if legal_moves == 1 {
            score = -negamax(
                board,
                depth - 1 + extension,
                -beta,
                -alpha,
                ply + 1,
                tt,
                hh,
                killers,
                abort_flag,
                start_time,
                time_limit,
                nodes,
                params,
            );
        } else {
            let do_lmr = (!mv.is_capture())
                && (!mv.is_promotion())
                && (!in_check)
                && (!gives_check)
                && (board.piece_at(mv.get_target()).unwrap_or(PieceType::Pawn) != PieceType::King)
                && (legal_moves >= 5)
                && (depth > 3);

            if do_lmr {
                let mut lmr_r = LMR_TABLE.get().unwrap()[depth as usize][legal_moves.min(63)];
                let history_score = hh.read(board.side_to_move, mv);

                if history_score > 1000 {
                    lmr_r = lmr_r.saturating_sub(1);
                } else if history_score < -1000 {
                    lmr_r += 1;
                }

                if lmr_r >= depth {
                    lmr_r = depth - 1;
                }

                score = -negamax(
                    board,
                    depth - 1 - lmr_r,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    tt,
                    hh,
                    killers,
                    abort_flag,
                    start_time,
                    time_limit,
                    nodes,
                    params,
                );

                if score > alpha {
                    score = -negamax(
                        board,
                        depth - 1 + extension,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        tt,
                        hh,
                        killers,
                        abort_flag,
                        start_time,
                        time_limit,
                        nodes,
                        params,
                    );
                }
            } else {
                score = -negamax(
                    board,
                    depth - 1 + extension,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    tt,
                    hh,
                    killers,
                    abort_flag,
                    start_time,
                    time_limit,
                    nodes,
                    params,
                );
            }

            if score > alpha && score < beta {
                score = -negamax(
                    board,
                    depth - 1 + extension,
                    -beta,
                    -alpha,
                    ply + 1,
                    tt,
                    hh,
                    killers,
                    abort_flag,
                    start_time,
                    time_limit,
                    nodes,
                    params,
                );
            }
        }

        board.unmake_move(mv);

        if abort_flag.load(Ordering::Relaxed) {
            return 0;
        }

        if score >= beta {
            tt.write(hash, depth, beta, Some(mv), TTFlag::Beta);

            if is_quiet {
                let p = (ply as usize).min(63);
                if killers[p][0] != Some(mv) {
                    killers[p][1] = killers[p][0];
                    killers[p][0] = Some(mv);
                }
            }

            if is_quiet {
                let bonus = (depth as i32) * (depth as i32);
                hh.write(board.side_to_move, mv, bonus);

                for i in 0..quiet_count {
                    if let Some(played_mv) = quiet_moves_played[i] {
                        if played_mv != mv {
                            hh.write(board.side_to_move, played_mv, -bonus);
                        }
                    }
                }
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

fn quiescence_search(
    board: &mut Board,
    mut alpha: i32,
    beta: i32,
    tt: &TranspositionTable,
    params: &EvalParams,
    nodes: &mut u64,
) -> i32 {
    *nodes += 1;
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
        let victim = board.piece_at(mv.get_target()).unwrap_or(PieceType::Pawn);
        let victim_value = MVV_LVA_VALUES[victim as usize];
        let delta_margin = 200;

        if (stand_pat + victim_value + delta_margin < alpha) && (!mv.is_promotion()) {
            continue;
        }

        if (static_exchange_evaluation(board, mv) < 0) && (!mv.is_promotion()) {
            continue;
        }

        if !board.make_move(mv, params) {
            continue;
        }

        let score = -quiescence_search(board, -beta, -alpha, tt, params, nodes);
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

fn score_move(board: &Board, mv: Move) -> i32 {
    if mv.is_promotion() {
        return 9000;
    }

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

fn score_move_iterative(
    board: &Board,
    mv: Move,
    best_mv: Option<Move>,
    killers: &[[Option<Move>; 2]; 64],
    hh: &HistoryHierarchy,
    ply: i32,
) -> i32 {
    if Some(mv) == best_mv {
        return 1_000_000;
    }

    let tactical_score = score_move(board, mv);
    if tactical_score != 0 {
        return 100_000 + tactical_score;
    }

    if !mv.is_capture() && !mv.is_promotion() {
        let p = (ply as usize).min(63);
        if killers[p][0] == Some(mv) {
            return 90000;
        } else if killers[p][1] == Some(mv) {
            return 80000;
        }
    }

    hh.read(board.side_to_move, mv)
}

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

    LMR_TABLE
        .set(table)
        .expect("Failed to initialize LMR Table");
    print!("Successfully Generated LMR values!\n");
}

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

        if let Some(smallest_attacker) =
            board.get_smallest_attacker(target, attacker, sim_occupancy)
        {
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

    return gain[0];
}
