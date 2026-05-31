use wajpassant::attacks::init_attacks;
use wajpassant::uci::uci_loop;

fn main() {
    init_attacks();
    uci_loop();
}
