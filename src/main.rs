use std::env;
use wajpassant::attacks::init_attacks;
use wajpassant::cli::cli_loop;
use wajpassant::uci::uci_loop;
use wajpassant::zobrist::init_zobrist;

fn main() {
    init_attacks();
    init_zobrist();

    let args: Vec<String> = env::args().collect();

    if args.contains(&String::from("--cli")) {
        cli_loop();
    } else {
        uci_loop();
    }
}
