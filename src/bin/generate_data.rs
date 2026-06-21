use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use wajpassant::attacks::init_attacks;
use wajpassant::board::{Board, Color, PieceType};
use wajpassant::eval::EvalParams;
use wajpassant::search::{init_lmr_table, search_best_move, static_exchange_evaluation};
use wajpassant::tt::TranspositionTable;
use wajpassant::zobrist::init_zobrist;

fn is_quiet(board: &Board) -> bool {
    let us = board.side_to_move;
    let them = us.flip();
    let king_sq = board.pieces[us as usize][PieceType::King as usize].get_lsb();

    // Must not be in check
    if board.is_square_attacked(king_sq, them) {
        return false;
    }

    // Must not have any highly profitable captures pending
    let captures = board.generate_captures();
    for mv in captures {
        // If there's a capture that wins material (SEE > 0), it's not quiet
        if static_exchange_evaluation(board, mv) > 0 {
            return false;
        }
    }

    true
}

fn main() {
    init_attacks();
    init_lmr_table();
    init_zobrist();

    let dataset_path = format!("{}/src/bin/dataset.txt", env!("CARGO_MANIFEST_DIR"));
    let out_file = Arc::new(Mutex::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(dataset_path)
            .unwrap(),
    ));

    let openings_path = format!("{}/src/bin/UHO_openings.epd", env!("CARGO_MANIFEST_DIR"));
    let file = File::open(openings_path).expect("Could not find UHO_openings.epd");
    let reader = BufReader::new(file);
    let mut openings: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

    let mut rng = rand::thread_rng();
    openings.shuffle(&mut rng);

    let games_to_play = 5000;

    let pb = ProgressBar::new(games_to_play as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} games ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    (0..games_to_play).into_par_iter().for_each(|game| {
        let start_fen = &openings[game % openings.len()];
        let params = EvalParams::new();
        let mut board = Board::from_fen(start_fen, &params);

        let mut game_history: Vec<String> = Vec::new();
        let mut game_result = 0.5;

        // Initialize TT ONCE per game, 2MB is plenty for depth 4
        let tt = Arc::new(TranspositionTable::new(2));

        // Play the game
        for ply in 0..200 {
            // Early Draw Detection
            if board.is_repetition() || board.halfmove_clock >= 100 {
                game_result = 0.5;
                break;
            }

            // Evaluation Adjudication
            let current_eval = board.evaluate(-5000, 5000, &params);
            if current_eval > 1000 {
                game_result = if board.side_to_move == Color::White {
                    1.0
                } else {
                    0.0
                };
                break;
            } else if current_eval < -1000 {
                game_result = if board.side_to_move == Color::White {
                    0.0
                } else {
                    1.0
                };
                break;
            }

            let abort = Arc::new(AtomicBool::new(false));

            // Pass the Arc clone, and set quiet to TRUE
            let best_move = search_best_move(board.clone(), 4, abort, None, true, Arc::clone(&tt));

            if let Some(mv) = best_move {
                if ply > 12 && is_quiet(&board) {
                    game_history.push(board.to_fen());
                }

                board.make_move(mv, &params);
            } else {
                let in_check = board.is_square_attacked(
                    board.pieces[board.side_to_move as usize][PieceType::King as usize].get_lsb(),
                    board.side_to_move.flip(),
                );

                if in_check {
                    game_result = if board.side_to_move == Color::White {
                        0.0
                    } else {
                        1.0
                    };
                } else {
                    game_result = 0.5;
                }
                break;
            }
        }

        let mut batch = String::new();
        for fen in game_history {
            batch.push_str(&format!("{} | {}\n", fen, game_result));
        }

        // Write the extracted quiet positions and the ultimate result
        let mut file = out_file.lock().unwrap();
        file.write_all(batch.as_bytes()).unwrap();

        pb.inc(1);
    });

    pb.finish_with_message("Dataset generation complete!");
}
