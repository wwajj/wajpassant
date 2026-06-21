//! The Universal Chess Interface (UCI) Protocol Listener.
//!
//! This module allows the engine to communicate with standard chess GUIs.
//! It utilizes multithreading and atomic variables to ensure the engine can
//! listen for "stop" commands and manage time controls without freezing.

use std::io::{self, BufRead};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

use crate::board::{Board, Color, PieceType};
use crate::eval::EvalParams;
use crate::moves::Move;
use crate::search::search_best_move;
use crate::tt::TranspositionTable;

pub fn uci_loop() {
    let stdin = io::stdin();
    let params = EvalParams::new();
    let mut board = Board::default();
    board.init_eval(&params);

    let tt = Arc::new(TranspositionTable::new(32));

    let mut abort_flag = Arc::new(AtomicBool::new(false));
    let mut search_thread: Option<thread::JoinHandle<()>> = None;

    for line in stdin.lock().lines() {
        let input = line.unwrap_or_default();
        let cmd = input.trim();

        if cmd.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = cmd.split_whitespace().collect();

        match tokens[0] {
            "uci" => {
                println!("id name WajPassant");
                println!("id author Rohan Bharadwaj");
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                board = Board::default();
                board.init_eval(&params);

                tt.clear();
            }
            "position" => {
                parse_position(&mut board, &tokens, &params);
            }
            "go" => {
                abort_flag.store(true, Ordering::Relaxed);
                if let Some(handle) = search_thread.take() {
                    let _ = handle.join();
                }

                abort_flag = Arc::new(AtomicBool::new(false));
                search_thread = Some(parse_go(
                    &board,
                    &tokens,
                    Arc::clone(&abort_flag),
                    Arc::clone(&tt),
                ));
            }
            "stop" => {
                abort_flag.store(true, Ordering::Relaxed);
                if let Some(handle) = search_thread.take() {
                    let _ = handle.join();
                }
            }
            "quit" => {
                abort_flag.store(true, Ordering::Relaxed);
                if let Some(handle) = search_thread.take() {
                    let _ = handle.join();
                }
                break;
            }
            _ => {}
        }
    }
}

/// Parses the "position" command from the GUI.
fn parse_position(board: &mut Board, tokens: &[&str], params: &EvalParams) {
    let mut current_idx = 1;

    if current_idx < tokens.len() && tokens[current_idx] == "startpos" {
        *board = Board::default();
        board.init_eval(params);
        current_idx += 1;
    } else if current_idx < tokens.len() && tokens[current_idx] == "fen" {
        current_idx += 1;
        let mut fen = String::new();
        while current_idx < tokens.len() && tokens[current_idx] != "moves" {
            fen.push_str(tokens[current_idx]);
            fen.push(' ');
            current_idx += 1;
        }
        *board = Board::from_fen(fen.trim(), params);
    }

    if current_idx < tokens.len() && tokens[current_idx] == "moves" {
        current_idx += 1;
        while current_idx < tokens.len() {
            let move_str = tokens[current_idx];
            if let Some(mv) = parse_uci_move(board, move_str) {
                board.make_move(mv, params);
            }
            current_idx += 1;
        }
    }
}

/// Parses the "go" command, calculates time, and spawns the search thread pool.
fn parse_go(
    board: &Board,
    tokens: &[&str],
    abort_flag: Arc<AtomicBool>,
    tt: Arc<TranspositionTable>,
) -> thread::JoinHandle<()> {
    let mut depth: u8 = 64;
    let mut wtime: u64 = 0;
    let mut btime: u64 = 0;
    let mut winc: u64 = 0;
    let mut binc: u64 = 0;
    let mut movestogo: u64 = 40;
    let mut movetime: u64 = 0;

    let mut i = 1;
    while i < tokens.len() {
        match tokens[i] {
            "depth" => {
                depth = tokens[i + 1].parse().unwrap_or(64);
                i += 1;
            }
            "wtime" => {
                wtime = tokens[i + 1].parse().unwrap_or(0);
                i += 1;
            }
            "btime" => {
                btime = tokens[i + 1].parse().unwrap_or(0);
                i += 1;
            }
            "winc" => {
                winc = tokens[i + 1].parse().unwrap_or(0);
                i += 1;
            }
            "binc" => {
                binc = tokens[i + 1].parse().unwrap_or(0);
                i += 1;
            }
            "movestogo" => {
                movestogo = tokens[i + 1].parse().unwrap_or(40);
                i += 1;
            }
            "movetime" => {
                movetime = tokens[i + 1].parse().unwrap_or(0);
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    let mut allocated_time: Option<Duration> = None;

    if movetime > 0 {
        allocated_time = Some(Duration::from_millis(movetime));
    } else if !tokens.contains(&"depth") {
        let our_time = if board.side_to_move == Color::White {
            wtime
        } else {
            btime
        };
        let our_inc = if board.side_to_move == Color::White {
            winc
        } else {
            binc
        };

        if our_time > 0 {
            let time_for_move = (our_time / (movestogo + 5)) + (our_inc / 2);
            allocated_time = Some(Duration::from_millis(time_for_move));
        }
    }

    let num_threads = 8;

    for _ in 1..num_threads {
        let search_board = board.clone();
        let thread_abort = Arc::clone(&abort_flag);
        let thread_tt = Arc::clone(&tt);
        let thread_time = allocated_time;

        thread::spawn(move || {
            let _ = search_best_move(
                search_board,
                depth,
                thread_abort,
                thread_time,
                true,      // quiet = true
                thread_tt, // Pass the Arc directly
            );
        });
    }

    let search_board = board.clone();
    let main_abort = Arc::clone(&abort_flag);
    let main_tt = Arc::clone(&tt);

    thread::spawn(move || {
        if let Some(mv) = search_best_move(
            search_board,
            depth,
            Arc::clone(&main_abort),
            allocated_time,
            false,
            main_tt,
        ) {
            main_abort.store(true, Ordering::Relaxed);
            println!("bestmove {}", format_uci_move(mv));
        } else {
            main_abort.store(true, Ordering::Relaxed);
            println!("bestmove 0000");
        }
    })
}

// --- Helper Functions ---

fn parse_uci_move(board: &mut Board, move_str: &str) -> Option<Move> {
    let clean_str = move_str.trim().to_lowercase();
    if clean_str.len() < 4 {
        return None;
    }

    let chars: Vec<char> = clean_str.chars().collect();
    let start_file = chars[0] as i32 - 'a' as i32;
    let start_rank = chars[1] as i32 - '1' as i32;
    let target_file = chars[2] as i32 - 'a' as i32;
    let target_rank = chars[3] as i32 - '1' as i32;

    let start_sq = (start_rank * 8 + start_file) as usize;
    let target_sq = (target_rank * 8 + target_file) as usize;

    let mut requested_promo = None;
    if chars.len() == 5 {
        requested_promo = match chars[4] {
            'q' => Some(PieceType::Queen),
            'r' => Some(PieceType::Rook),
            'b' => Some(PieceType::Bishop),
            'n' => Some(PieceType::Knight),
            _ => None,
        };
    }

    for mv in board.generate_all_moves() {
        if mv.get_start() as usize == start_sq && mv.get_target() as usize == target_sq {
            if mv.is_promotion() {
                if mv.get_promotion_piece() == requested_promo {
                    return Some(mv);
                }
            } else {
                return Some(mv);
            }
        }
    }
    None
}

/// Kept public just in case, but formatting is now mostly handled inside search.rs
pub fn format_uci_move(mv: Move) -> String {
    let files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    let ranks = ['1', '2', '3', '4', '5', '6', '7', '8'];

    let start = mv.get_start() as usize;
    let target = mv.get_target() as usize;

    let mut out = format!(
        "{}{}{}{}",
        files[start % 8],
        ranks[start / 8],
        files[target % 8],
        ranks[target / 8]
    );

    if mv.is_promotion() {
        let promo_char = match mv.get_promotion_piece().unwrap() {
            PieceType::Queen => 'q',
            PieceType::Rook => 'r',
            PieceType::Bishop => 'b',
            PieceType::Knight => 'n',
            _ => '?',
        };
        out.push(promo_char);
    }
    out
}
