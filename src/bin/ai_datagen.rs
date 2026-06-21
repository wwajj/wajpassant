use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::SystemTime;

use wajpassant::attacks::init_attacks;
use wajpassant::board::{Board, Color, PieceType};
use wajpassant::eval::EvalParams;
use wajpassant::hh::HistoryHierarchy;
use wajpassant::moves::Move;
use wajpassant::search::init_lmr_table;
use wajpassant::search::negamax;
use wajpassant::tt::TranspositionTable;
use wajpassant::zobrist::init_zobrist;

const TOTAL_GAMES: u64 = 100_000;

fn main() {
    init_attacks();
    init_lmr_table();
    init_zobrist();
    println!("WajPassant Multi-Core AI Data Miner initialized.");

    // ==========================================
    // --- THE CONSUMER (WRITER THREAD) ---
    // ==========================================
    // We create a Multi-Producer, Single-Consumer (mpsc) channel.
    let (tx, rx) = mpsc::channel::<String>();

    let writer_thread = thread::spawn(move || {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("wajpassant_training_data.txt")
            .expect("Failed to open target text data file");

        let pb = ProgressBar::new(TOTAL_GAMES as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} games (ETA: {eta}) | {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut total_positions_mined = 0;

        // The receiver listens until the sender channel is completely dropped
        for game_data_block in rx {
            // Count how many newlines (positions) are in this block
            let positions_in_game = game_data_block.matches('\n').count();
            total_positions_mined += positions_in_game;

            // Write the chunk to the file
            file.write_all(game_data_block.as_bytes()).unwrap();

            pb.set_message(format!("Mined {} unique positions", total_positions_mined));
            pb.inc(1); // Increment the game counter
        }

        pb.finish_with_message(format!(
            "Mining operation complete! Captured {} records.",
            total_positions_mined
        ));
    });

    // ==========================================
    // --- THE PRODUCERS (RAYON WORKERS) ---
    // ==========================================

    // .into_par_iter() magically splits the games across all available CPU cores.
    // .for_each_with(tx) gives every thread a clone of the transmitter to send data back.
    (0..TOTAL_GAMES)
        .into_par_iter()
        .for_each_with(tx, |sender, _| {
            let params = EvalParams::new();
            let mut board = Board::from_fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                &params,
            );

            // 4MB is optimal for depth 7 without thrashing the system allocator across 100k loops
            let tt = TranspositionTable::new(4);
            let mut hh = HistoryHierarchy::new();
            let mut killers = [[None; 2]; 64]; // Thread-local killer moves
            let mut game_positions = Vec::with_capacity(100);

            let mut ply_count = 0;
            let game_result = loop {
                if board.halfmove_clock >= 100 || board.is_repetition() {
                    break "0.5";
                }

                let chosen_move: Option<Move>;

                if ply_count < 8 {
                    chosen_move = get_random_opening_move(&mut board, &params);
                } else {
                    chosen_move =
                        run_search_for_datagen(&mut board, &params, &tt, &mut hh, &mut killers);
                }

                if chosen_move.is_none() {
                    let king_sq = board.pieces[board.side_to_move as usize]
                        [PieceType::King as usize]
                        .get_lsb();
                    if board.is_square_attacked(king_sq, board.side_to_move.flip()) {
                        break if board.side_to_move == Color::White {
                            "0.0"
                        } else {
                            "1.0"
                        };
                    } else {
                        break "0.5";
                    }
                }

                let mv = chosen_move.unwrap();

                if ply_count >= 8 && board.phase > 4 {
                    let mut eval = board.evaluate(-50000, 50000, &params);
                    if board.side_to_move == Color::Black {
                        eval = -eval;
                    }
                    game_positions.push((board.to_fen(), eval));
                }

                board.make_move(mv, &params);
                ply_count += 1;
            };

            // Format the entire game's data into a single String block
            let mut game_data_block = String::with_capacity(8192);
            for (fen, eval) in game_positions.iter() {
                game_data_block.push_str(&format!("{} | {} | {}\n", fen, eval, game_result));
            }

            // Send the block to the secretary thread to be written to the file
            if !game_data_block.is_empty() {
                sender.send(game_data_block).unwrap();
            }
        });

    // The main thread waits for the writer thread to finish emptying the channel buffer
    writer_thread.join().unwrap();
}

/// Grabs a truly random legal move for the first 8 plies to diversify positions.
fn get_random_opening_move(board: &mut Board, params: &EvalParams) -> Option<Move> {
    let raw_moves = board.generate_all_moves();
    let mut legal_moves = Vec::new();

    for mv in raw_moves {
        if board.make_move(mv, params) {
            board.unmake_move(mv);
            legal_moves.push(mv);
        }
    }

    if legal_moves.is_empty() {
        return None;
    }

    // High-entropy mixer blending system time and the board hash
    let mut seed = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    // Fast Xorshift64
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    seed = seed.wrapping_add(board.calculate_hash());

    let safe_index = (seed as usize) % legal_moves.len();
    Some(legal_moves[safe_index])
}

/// Executes a fixed depth 7 search using your core Negamax engine stack
fn run_search_for_datagen(
    board: &mut Board,
    params: &EvalParams,
    tt: &TranspositionTable, // Lock-free, passed as read-only reference
    hh: &mut HistoryHierarchy,
    killers: &mut [[Option<Move>; 2]; 64], // Newly required local array
) -> Option<Move> {
    hh.clear();
    *killers = [[None; 2]; 64];

    let abort_flag = Arc::new(AtomicBool::new(false));
    let mut best_move: Option<Move> = None;
    let target_depth = 7;

    for depth in 1..=target_depth {
        let _score = negamax(
            board,
            depth,
            -50000,
            50000,
            0,
            tt,
            hh,
            killers, // Pass local array down the tree
            &abort_flag,
            std::time::Instant::now(),
            None,
            &mut 0,
            params,
        );

        if let Some(tt_move) = tt.probe_move(board.calculate_hash()) {
            best_move = Some(tt_move);
        }
    }

    best_move
}
