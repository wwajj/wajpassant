use crate::bitboard::{Bitboard, Square, SQUARES, NOT_H_FILE, NOT_A_FILE};

const BISHOP_BLOCKERS: u64 = 2_u64.pow(9);
const ROOK_BLOCKERS: u64 = 2_u64.pow(12);

pub static mut WHITE_PAWN_ATTACKS: [Bitboard; 64] = [Bitboard::empty(); 64];
pub static mut BLACK_PAWN_ATTACKS: [Bitboard; 64] = [Bitboard::empty(); 64];
pub static mut KNIGHT_ATTACKS: [Bitboard; 64] = [Bitboard::empty(); 64];
pub static mut BISHOP_ATTACKS: [[Bitboard; BISHOP_BLOCKERS as usize]; 64] = [[Bitboard::empty(); BISHOP_BLOCKERS as usize]; 64];
pub static mut ROOK_ATTACKS: [[Bitboard; ROOK_BLOCKERS as usize]; 64] = [[Bitboard::empty(); ROOK_BLOCKERS as usize]; 64];
pub static mut KING_ATTACKS: [Bitboard; 64] = [Bitboard::empty(); 64];

pub static mut ROOK_MASKS: [Bitboard; 64] = [Bitboard::empty(); 64];
pub static mut BISHOP_MASKS: [Bitboard; 64] = [Bitboard::empty(); 64];

pub const NOT_AB_FILE: u64 = 0xFCFCFCFCFCFCFCFC;
pub const NOT_GH_FILE: u64 = 0x3F3F3F3F3F3F3F3F;

const KNIGHT_OFFSETS: [i8; 8] = [6, -6, 10, -10, 15, -15, 17, -17];
const KING_OFFSETS: [i8; 8] = [1, -1, 7, -7, 8, -8, 9, -9];

// populates knight and king attacks
pub fn init_attacks() {
    for sq_idx in 0..64 {
        let bb = Bitboard::empty().set_bit(SQUARES[sq_idx]);

        // white pawn logic
        let mut attacks = Bitboard::empty();
        attacks |= (bb << 9) & Bitboard(NOT_A_FILE);
        attacks |= (bb << 7) & Bitboard(NOT_H_FILE);
        unsafe {
            WHITE_PAWN_ATTACKS[sq_idx] = attacks;
        }

        // black pawn logic
        attacks = Bitboard::empty();
        attacks |= (bb >> 7) & Bitboard(NOT_A_FILE);
        attacks |= (bb >> 9) & Bitboard(NOT_H_FILE);
        unsafe {
            BLACK_PAWN_ATTACKS[sq_idx] = attacks;
        }

        // knight logic
        attacks = Bitboard::empty();
        for offset in KNIGHT_OFFSETS {
            let new_sq = offset + (sq_idx as i8);
            if new_sq > 63 || new_sq < 0 { continue; }

            let mut shifted = if offset > 0 {
                bb << (offset as usize)
            } else {
                bb >> (offset.abs() as usize)
            };

            shifted &= match offset {
                17 | -15 => Bitboard(NOT_A_FILE),
                10 | -6  => Bitboard(NOT_AB_FILE),
                15 | -17 => Bitboard(NOT_H_FILE),
                6  | -10 => Bitboard(NOT_GH_FILE),
                _ => Bitboard::full(),
            };

            attacks |= shifted;
        }
        unsafe {
            KNIGHT_ATTACKS[sq_idx] = attacks;
        }

        // king logic
        attacks = Bitboard::empty(); 

        for offset in KING_OFFSETS {
            let new_sq = offset + (sq_idx as i8);
            if new_sq > 63 || new_sq < 0 { continue; }

            let mut shifted = if offset > 0 {
                bb << (offset as usize)
            } else {
                bb >> (offset.abs() as usize)
            };

            shifted &= match offset {
                -7 |  1 |  9 => Bitboard(NOT_A_FILE),
                 7 | -1 | -9 => Bitboard(NOT_H_FILE),
                _ => Bitboard::full(),
            };

            attacks |= shifted;
        }
        unsafe {
            KING_ATTACKS[sq_idx] = attacks;
        }

    }
}

// calculates blocker combinations for bishops
pub fn get_bishop_attacks_slow(sq: Square, blockers: Bitboard) -> Bitboard {
    let mut attacks = Bitboard::empty();
    let rank = (sq as i32) / 8;
    let file = (sq as i32) % 8;

    // up-right
    let (mut r, mut f) = (rank + 1, file + 1);
    while r <= 7 && f <= 7 {
        let target_sq = (r * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        r += 1;
        f += 1;
    }

    // up-left
    let (mut r, mut f) = (rank + 1, file - 1);
    while r <= 7 && f >= 0 {
        let target_sq = (r * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        r += 1;
        f -= 1;
    }

    // down-right
    let (mut r, mut f) = (rank - 1, file + 1);
    while r >= 0 && f <= 7 {
        let target_sq = (r * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        r -= 1;
        f += 1;
    }

    // down-left
    let (mut r, mut f) = (rank - 1, file - 1);
    while r >= 0 && f >= 0 {
        let target_sq = (r * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        r -= 1;
        f -= 1;
    }
    attacks
}


// calculates blocker combinations for rooks
pub fn get_rook_attacks_slow(sq: Square, blockers: Bitboard) -> Bitboard {
    let mut attacks = Bitboard::empty();
    let rank = (sq as i32) / 8;
    let file = (sq as i32) % 8;

    // up
    let mut r = rank + 1;
    while r <= 7 {
        let target_sq = (r * 8 + file) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        r += 1;
    }

    // down
    let mut r = rank - 1;
    while r >= 0 {
        let target_sq = (r * 8 + file) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        r -= 1;
    }

    // right
    let mut f = file + 1;
    while f <= 7 {
        let target_sq = (rank * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        f += 1;
    }

    // left
    let mut f = file - 1;
    while f >= 0 {
        let target_sq = (rank * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        f -= 1;
    }
    attacks
}
