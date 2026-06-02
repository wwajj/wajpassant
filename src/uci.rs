//! The Universal Chess Interface (UCI) Protocol Listener.
//!
//! This module allows the engine to communicate with standard chess GUIs 
//! by listening to stdin and responding via stdout.

use std::io::{self, BufRead};
use crate::board::{Board, PieceType};
use crate::search::search_best_move;
use crate::moves::Move;

/// The main infinite loop that listens for GUI commands.
pub fn uci_loop() {
    let stdin = io::stdin();
    let mut board = Board::default();
    board.init_eval();

    for line in stdin.lock().lines() {
        let input = line.unwrap_or_default();
        let cmd = input.trim();
        
        if cmd.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = cmd.split_whitespace().collect();

        match tokens[0] {
            "uci" => {
                println!("id name WajPassant 1.0");
                println!("id author Rohan Bharadwaj");
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                board = Board::default();
                board.init_eval();
            }
            "position" => {
                parse_position(&mut board, &tokens);
            }
            "go" => {
                parse_go(&mut board, &tokens);
            }
            "quit" => {
                break;
            }
            _ => {
            }
        }
    }
}

/// Parses the "position" command from the GUI.
fn parse_position(board: &mut Board, tokens: &[&str]) {
    let mut current_idx = 1;

    if current_idx < tokens.len() && tokens[current_idx] == "startpos" {
        *board = Board::default();
        board.init_eval();
        current_idx += 1;
    } else if current_idx < tokens.len() && tokens[current_idx] == "fen" {
        current_idx += 1;
        let mut fen = String::new();
        while current_idx < tokens.len() && tokens[current_idx] != "moves" {
            fen.push_str(tokens[current_idx]);
            fen.push(' ');
            current_idx += 1;
        }
        *board = Board::from_fen(fen.trim()); 
        board.init_eval();
    }

    if current_idx < tokens.len() && tokens[current_idx] == "moves" {
        current_idx += 1;
        while current_idx < tokens.len() {
            let move_str = tokens[current_idx];
            if let Some(mv) = parse_uci_move(board, move_str) {
                board.make_move(mv);
            }
            current_idx += 1;
        }
    }
}

/// Parses the "go" command and initiates the search.
fn parse_go(board: &mut Board, tokens: &[&str]) {
    let mut depth = 7; 

    for i in 0..tokens.len() {
        if tokens[i] == "depth" && i + 1 < tokens.len() {
            if let Ok(d) = tokens[i + 1].parse::<u8>() {
                depth = d;
            }
        }
    }

    if let Some(best_move) = search_best_move(board, depth) {
        println!("bestmove {}", format_uci_move(best_move));
    } else {
        println!("bestmove 0000"); 
    }
}

// --- Helper Functions ---

/// Converts a UCI string (e.g., "e2e4", "e7e8q") to our internal Move struct.
fn parse_uci_move(board: &mut Board, move_str: &str) -> Option<Move> {
    let clean_str = move_str.trim().to_lowercase();
    if clean_str.len() < 4 { return None; }

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

/// Converts our internal Move struct back into a UCI string for the GUI.
fn format_uci_move(mv: Move) -> String {
    let files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    let ranks = ['1', '2', '3', '4', '5', '6', '7', '8'];

    let start = mv.get_start() as usize;
    let target = mv.get_target() as usize;

    let mut out = format!("{}{}{}{}", files[start % 8], ranks[start / 8], files[target % 8], ranks[target / 8]);

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
