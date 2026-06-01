//! Command Line Interface for playing against the engine directly in the terminal.

use std::io::{self, Write};
use crate::board::{Board, Color, PieceType};
use crate::search::search_best_move;
use crate::moves::Move;

pub fn cli_loop() {
    let mut board = Board::default();
    board.init_eval();

    println!("WajPassant CLI");
    print!("Do you want to play as White (w) or Black (b)? ");
    io::stdout().flush().unwrap();
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    
    let user_color = if input.trim().to_lowercase() == "b" {
        println!("You are playing Black. WajPassant is White.");
        Color::Black
    } else {
        println!("You are playing White. WajPassant is Black.");
        Color::White
    };

    loop {
        // Print the board from the user's perspective
        print_board_cli(&board, user_color); 

        if board.side_to_move == user_color {
            // --- HUMAN TURN ---
            print!("Enter your move (e.g., e2e4): ");
            io::stdout().flush().unwrap();
            
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let move_str = input.trim();

            if move_str == "quit" || move_str == "exit" {
                break;
            }

            if let Some(user_move) = parse_user_move(&mut board, move_str) {
                if !board.make_move(user_move) {
                    println!("Move left King in check. Try again.");
                }
            } else {
                println!("Invalid move format or illegal move. Try again.");
            }
        } else {
            // --- ENGINE TURN ---
            println!("WajPassant is thinking at Depth 5...");
            
            if let Some(best_move) = search_best_move(&mut board, 5) {
                println!("WajPassant plays: {}", format_move(best_move)); 
                board.make_move(best_move);
            } else {
                println!("Game Over! WajPassant has no legal moves.");
                break;
            }
        }
    }
}

/// Prints the board perfectly flipped to the user's perspective.
fn print_board_cli(board: &Board, perspective: Color) {
    let ranks: Vec<usize> = if perspective == Color::White {
        (0..8).rev().collect()
    } else {
        (0..8).collect()
    };

    let files: Vec<usize> = if perspective == Color::White {
        (0..8).collect()
    } else {
        (0..8).rev().collect()
    };

    println!();
    for &rank in &ranks {
        print!(" {} | ", rank + 1);
        for &file in &files {
            let sq = rank * 8 + file;
            let mut piece_char = '.';
            
            let chars = ['P', 'N', 'B', 'R', 'Q', 'K'];
            for side in 0..2 {
                for pt in 0..6 {
                    if (board.pieces[side][pt].0 & (1u64 << sq)) != 0 {
                        let c = chars[pt];
                        piece_char = if side == Color::White as usize { c } else { c.to_ascii_lowercase() };
                    }
                }
            }
            print!("{} ", piece_char);
        }
        println!();
    }
    
    print!("   -----------------\n     ");
    for &file in &files {
        print!("{} ", (b'a' + file as u8) as char);
    }
    println!("\n");
}

/// Parses a standard coordinate string (e.g., "e2e4") into an internal Move struct.
fn parse_user_move(board: &mut Board, move_str: &str) -> Option<Move> {
    let clean_str = move_str.trim().to_lowercase();
    if clean_str.len() < 4 || clean_str.len() > 5 { return None; }

    let chars: Vec<char> = clean_str.chars().collect();
    let start_file = chars[0] as i32 - 'a' as i32;
    let start_rank = chars[1] as i32 - '1' as i32;
    let target_file = chars[2] as i32 - 'a' as i32;
    let target_rank = chars[3] as i32 - '1' as i32;

    if start_file < 0 || start_file > 7 || start_rank < 0 || start_rank > 7 ||
       target_file < 0 || target_file > 7 || target_rank < 0 || target_rank > 7 {
        return None;
    }

    let start_sq = (start_rank * 8 + start_file) as usize;
    let target_sq = (target_rank * 8 + target_file) as usize;

    let mut requested_promo = None;
    if chars.len() == 5 {
        requested_promo = match chars[4] {
            'q' => Some(PieceType::Queen),
            'r' => Some(PieceType::Rook),
            'b' => Some(PieceType::Bishop),
            'n' => Some(PieceType::Knight),
            _ => return None,
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

/// Formats a Move struct back into an algebraic string (e.g., "e2e4")
fn format_move(mv: Move) -> String {
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
