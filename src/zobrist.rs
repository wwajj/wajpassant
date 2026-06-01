use std::sync::OnceLock;

// Global Locks
pub static ZOBRIST_PIECES: OnceLock<[[u64; 64]; 12]> = OnceLock::new();
pub static ZOBRIST_SIDE: OnceLock<u64> = OnceLock::new();
pub static ZOBRIST_CASTLING: OnceLock<[u64; 16]> = OnceLock::new();
pub static ZOBRIST_EN_PASSANT: OnceLock<[u64; 8]> = OnceLock::new();

/// A fast, deterministic 64-bit Pseudo-Random Number Generator using the Xorshift64 algorithm.
/// 
/// Keeps a local state variable and applies three bit-shifts and XORs to mix the bits.
/// This guarantees the exact same sequence of numbers on every engine boot.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Initializes all global Zobrist random number tables.
pub fn init_zobrist() {
    print!("Generating Zorbist Hash Tables...\n");
    let mut seed = 1070372u64;

    let mut pieces = [[0u64; 64]; 12];
    let mut castling = [0u64; 16];
    let mut en_passant = [0u64; 8];

    for p in 0..12 {
        for sq in 0..64 {
            pieces[p][sq] = xorshift64(&mut seed);
        }
    }

    let side = xorshift64(&mut seed);

    for i in 0..16 {
        castling[i] = xorshift64(&mut seed);
    }

    for i in 0..8 {
        en_passant[i] = xorshift64(&mut seed);
    }

    ZOBRIST_PIECES.set(pieces).unwrap_or_else(|_| panic!("Failed to set ZOBRIST_PIECES"));
    ZOBRIST_SIDE.set(side).unwrap_or_else(|_| panic!("Failed to set ZOBRIST_SIDE"));
    ZOBRIST_CASTLING.set(castling).unwrap_or_else(|_| panic!("Failed to set ZOBRIST_CASTLING"));
    ZOBRIST_EN_PASSANT.set(en_passant).unwrap_or_else(|_| panic!("Failed to set ZOBRIST_EN_PASSANT"));
    print!("Zorbist Hash Tables Successfully Generated...\n");
}
