use wajpassant::board::Board;

fn main() {
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    let kiwipete = Board::from_fen(fen);
    kiwipete.print();
}
