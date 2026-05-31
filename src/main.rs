use std::time::Instant;

use wajpassant::board::Board; 

fn main() {
    println!("🚀 Booting up Rust Chess Engine...");

    wajpassant::magics::init_magics();
    wajpassant::attacks::init_attacks(); 

    let start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    
    let mut board = Board::from_fen(start_fen);
    
    println!("\nStarting Position:");
    board.print();

    let depth = 6; 
    println!("\nRunning Perft test at Depth {}...", depth);

    let start_time = Instant::now();
    let nodes = board.perft(depth);
    let elapsed = start_time.elapsed();
    let seconds = elapsed.as_secs_f64();
    let nps = if seconds > 0.0 {
        (nodes as f64 / seconds) as u64
    } else {
        0
    };

    println!("\n========================================");
    println!("PERFT RESULTS");
    println!("========================================");
    println!("Depth : {}", depth);
    println!("Nodes : {}", nodes);
    println!("Time  : {:.3} seconds", seconds);
    println!("NPS   : {} nodes/sec", nps);
    println!("========================================\n");
}
