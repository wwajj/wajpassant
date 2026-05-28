use std::env;
use std::fs::File;
use wajpassant::attacks::init_attacks;
use wajpassant::board::Board; 
use wajpassant::cli::cli_loop;
use wajpassant::eval::EvalParams;
use wajpassant::search::init_lmr_table;
use wajpassant::uci::uci_loop;
use wajpassant::zobrist::init_zobrist;

fn main() {
    init_attacks();
    init_lmr_table();
    init_zobrist();

    let args: Vec<String> = env::args().collect();
    let params = EvalParams::new();

    if args.contains(&String::from("--cli")) {
        cli_loop();
    } 
    // Handle standard perft
    else if let Some(pos) = args.iter().position(|arg| arg == "--perft") {
        if let Some(depth_str) = args.get(pos + 1) {
            if let Ok(depth) = depth_str.parse::<u8>() {
                let mut board = match args.get(pos + 2) {
                    Some(fen) => Board::from_fen(fen, &params),
                    None => Board::default(),
                };
                let nodes = board.perft(depth);
                println!("Perft depth {}: {} nodes", depth, nodes);
            }
        }
    }
    // Handle perft-debug
    // Usage: --perft-debug <depth> <output_file> [FEN]
    else if let Some(pos) = args.iter().position(|arg| arg == "--perft-debug") {
        if let (Some(depth_str), Some(file_path)) = (args.get(pos + 1), args.get(pos + 2)) {
            if let Ok(depth) = depth_str.parse::<u8>() {
                if let Ok(mut file) = File::create(file_path) {
                    let mut board = match args.get(pos + 3) {
                        Some(fen) => Board::from_fen(fen, &params),
                        None => Board::default(),
                    };
                    
                    let nodes = board.perft_debug(depth, String::new(), &mut file);
                    println!("Debug Perft depth {}: {} nodes. Written to {}", depth, nodes, file_path);
                } else {
                    eprintln!("Error: Could not create file at {}", file_path);
                }
            } else {
                eprintln!("Error: Depth must be a number.");
            }
        } else {
            eprintln!("Error: --perft-debug requires <depth> and <output_file>.");
        }
    } 
    else {
        uci_loop();
    }
}
