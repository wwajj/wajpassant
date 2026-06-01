use std::env;
use wajpassant::attacks::init_attacks;
use wajpassant::cli::cli_loop;
use wajpassant::uci::uci_loop;

fn main() {
    init_attacks();

    let args: Vec<String> = env::args().collect();

    if args.contains(&String::from("--cli")) {
        cli_loop();
    } else {
        uci_loop();
    }
}
