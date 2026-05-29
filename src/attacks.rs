//! Pre-calculated attack tables and Magic Bitboard generation.
//!
//! This module is responsible for generating and storing the attack maps for every 
//! piece on the board. For Leapers (Knights, Kings) and Pawns, it stores absolute 
//! attack bitboards. For Sliders (Bishops, Rooks, Queens), it implements the 
//! Magic Bitboard hashing technique for instantaneous attack retrieval during the game.

use crate::bitboard::{Bitboard, Square, SQUARES, NOT_A_FILE, NOT_H_FILE};
use crate::magics::{BISHOP_MAGICS, ROOK_MAGICS};

// --- Constants ---

/// Maximum number of bits needed to represent Bishop blockers (2^9).
const BISHOP_BLOCKERS: u64 = 2_u64.pow(9);
/// Maximum number of bits needed to represent Rook blockers (2^12).
const ROOK_BLOCKERS: u64 = 2_u64.pow(12);

/// Mask to clear the A and B files.
pub const NOT_AB_FILE: u64 = 0xFCFCFCFCFCFCFCFC;
/// Mask to clear the G and H files.
pub const NOT_GH_FILE: u64 = 0x3F3F3F3F3F3F3F3F;

/// 1D array index offsets for Knight jumps.
const KNIGHT_OFFSETS: [i8; 8] = [6, -6, 10, -10, 15, -15, 17, -17];
/// 1D array index offsets for King steps.
const KING_OFFSETS: [i8; 8] = [1, -1, 7, -7, 8, -8, 9, -9];

// --- Static Attack Tables ---

pub static mut WHITE_PAWN_ATTACKS: [Bitboard; 64] = [Bitboard::empty(); 64];
pub static mut BLACK_PAWN_ATTACKS: [Bitboard; 64] = [Bitboard::empty(); 64];
pub static mut KNIGHT_ATTACKS: [Bitboard; 64] = [Bitboard::empty(); 64];
pub static mut KING_ATTACKS: [Bitboard; 64] = [Bitboard::empty(); 64];

pub static mut BISHOP_MASKS: [Bitboard; 64] = [Bitboard::empty(); 64];
pub static mut ROOK_MASKS: [Bitboard; 64] = [Bitboard::empty(); 64];

pub static mut BISHOP_ATTACKS: [[Bitboard; BISHOP_BLOCKERS as usize]; 64] = [[Bitboard::empty(); BISHOP_BLOCKERS as usize]; 64];
pub static mut ROOK_ATTACKS: [[Bitboard; ROOK_BLOCKERS as usize]; 64] = [[Bitboard::empty(); ROOK_BLOCKERS as usize]; 64];

// --- Initialization ---

/// Populates all static attack arrays and Magic Bitboard tables.
pub fn init_attacks() {
    for sq_idx in 0..64 {
        let curr_sq = SQUARES[sq_idx];
        let bb = Bitboard::empty().set_bit(curr_sq);

        // White Pawn Logic
        let mut attacks = Bitboard::empty();
        attacks |= (bb << 9) & Bitboard(NOT_A_FILE);
        attacks |= (bb << 7) & Bitboard(NOT_H_FILE);
        unsafe {
            WHITE_PAWN_ATTACKS[sq_idx] = attacks;
        }

        // Black Pawn Logic
        let mut attacks = Bitboard::empty();
        attacks |= (bb >> 7) & Bitboard(NOT_A_FILE);
        attacks |= (bb >> 9) & Bitboard(NOT_H_FILE);
        unsafe {
            BLACK_PAWN_ATTACKS[sq_idx] = attacks;
        }

        // Knight Logic
        let mut attacks = Bitboard::empty();
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

        // King Logic
        let mut attacks = Bitboard::empty(); 

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

        // Set Slider Masks
        let bishop_mask = mask_bishop(curr_sq);
        let rook_mask = mask_rook(curr_sq);

        unsafe {
            BISHOP_MASKS[sq_idx] = bishop_mask;
            ROOK_MASKS[sq_idx] = rook_mask;
        }

        // Bishop Magic Initialization (Carry-Rippler)
        let mask = bishop_mask.0;
        let magic = BISHOP_MAGICS[sq_idx];
        let shift = 64 - bishop_mask.count();
        let mut occupancy = Bitboard::empty();

        loop {
            let move_map = get_bishop_attacks_slow(curr_sq, occupancy);
            let magic_index = (occupancy.0.wrapping_mul(magic) >> shift) as usize;
            unsafe {
                BISHOP_ATTACKS[sq_idx][magic_index] = move_map;
            }
            occupancy = Bitboard(occupancy.0.wrapping_sub(1) & mask);
            if occupancy == Bitboard::empty() { break; }
        }

        // Rook Magic Initialization (Carry-Rippler)
        let mask = rook_mask.0;
        let magic = ROOK_MAGICS[sq_idx];
        let shift = 64 - rook_mask.count();
        let mut occupancy = Bitboard::empty();

        loop {
            let move_map = get_rook_attacks_slow(curr_sq, occupancy);
            let magic_index = (occupancy.0.wrapping_mul(magic) >> shift) as usize;
            unsafe {
                ROOK_ATTACKS[sq_idx][magic_index] = move_map;
            }
            occupancy = Bitboard(occupancy.0.wrapping_sub(1) & mask);
            if occupancy == Bitboard::empty() { break; }
        }
    }
}

// --- Mask Generators ---

/// Calculates the relevant blocker squares for a Bishop, ignoring the outer board edges.
pub fn mask_bishop(sq: Square) -> Bitboard {
    let mut attacks = Bitboard::empty();
    let rank = (sq as i32) / 8;
    let file = (sq as i32) % 8;

    // Up-right
    let (mut r, mut f) = (rank + 1, file + 1);
    while r <= 6 && f <= 6 {
        let target_sq = (r * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        r += 1;
        f += 1;
    }

    // Up-left
    let (mut r, mut f) = (rank + 1, file - 1);
    while r <= 6 && f >= 1 {
        let target_sq = (r * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        r += 1;
        f -= 1;
    }

    // Down-right
    let (mut r, mut f) = (rank - 1, file + 1);
    while r >= 1 && f <= 6 {
        let target_sq = (r * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        r -= 1;
        f += 1;
    }

    // Down-left
    let (mut r, mut f) = (rank - 1, file - 1);
    while r >= 1 && f >= 1 {
        let target_sq = (r * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        r -= 1;
        f -= 1;
    }
    attacks
}

/// Calculates the relevant blocker squares for a Rook, ignoring the outer board edges.
pub fn mask_rook(sq: Square) -> Bitboard {
    let mut attacks = Bitboard::empty();
    let rank = (sq as i32) / 8;
    let file = (sq as i32) % 8;

    // Up
    let mut r = rank + 1;
    while r <= 6 {
        let target_sq = (r * 8 + file) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        r += 1;
    }

    // Down
    let mut r = rank - 1;
    while r >= 1 {
        let target_sq = (r * 8 + file) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        r -= 1;
    }

    // Right
    let mut f = file + 1;
    while f <= 6 {
        let target_sq = (rank * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        f += 1;
    }

    // Left
    let mut f = file - 1;
    while f >= 1 {
        let target_sq = (rank * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        f -= 1;
    }
    attacks
}

// --- Slow Attack Generators (For Initialization Only) ---

/// Traces Bishop rays outward until they hit a specific blocker configuration.
pub fn get_bishop_attacks_slow(sq: Square, blockers: Bitboard) -> Bitboard {
    let mut attacks = Bitboard::empty();
    let rank = (sq as i32) / 8;
    let file = (sq as i32) % 8;

    // Up-right
    let (mut r, mut f) = (rank + 1, file + 1);
    while r <= 7 && f <= 7 {
        let target_sq = (r * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        r += 1;
        f += 1;
    }

    // Up-left
    let (mut r, mut f) = (rank + 1, file - 1);
    while r <= 7 && f >= 0 {
        let target_sq = (r * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        r += 1;
        f -= 1;
    }

    // Down-right
    let (mut r, mut f) = (rank - 1, file + 1);
    while r >= 0 && f <= 7 {
        let target_sq = (r * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        r -= 1;
        f += 1;
    }

    // Down-left
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

/// Traces Rook rays outward until they hit a specific blocker configuration.
pub fn get_rook_attacks_slow(sq: Square, blockers: Bitboard) -> Bitboard {
    let mut attacks = Bitboard::empty();
    let rank = (sq as i32) / 8;
    let file = (sq as i32) % 8;

    // Up
    let mut r = rank + 1;
    while r <= 7 {
        let target_sq = (r * 8 + file) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        r += 1;
    }

    // Down
    let mut r = rank - 1;
    while r >= 0 {
        let target_sq = (r * 8 + file) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        r -= 1;
    }

    // Right
    let mut f = file + 1;
    while f <= 7 {
        let target_sq = (rank * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        f += 1;
    }

    // Left
    let mut f = file - 1;
    while f >= 0 {
        let target_sq = (rank * 8 + f) as usize;
        attacks |= Bitboard(1u64) << target_sq;
        if (blockers & (Bitboard(1u64) << target_sq)) != Bitboard::empty() { break; }
        f -= 1;
    }
    attacks
}

// --- Fast Magic Lookups (Used during Game Search) ---

/// Instantly retrieves the Bishop attack bitboard using Magic hashing.
#[inline(always)]
pub fn get_bishop_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    unsafe {
        let sq_idx = sq as usize;
        let mask = BISHOP_MASKS[sq_idx].0;
        let blockers = occupancy.0 & mask;
        let magic = BISHOP_MAGICS[sq_idx];
        
        let shift = 64 - BISHOP_MASKS[sq_idx].count(); 
        
        let magic_index = (blockers.wrapping_mul(magic) >> shift) as usize;
        BISHOP_ATTACKS[sq_idx][magic_index]
    }
}

/// Instantly retrieves the Rook attack bitboard using Magic hashing.
#[inline(always)]
pub fn get_rook_attacks(sq: Square, occupancy: Bitboard) -> Bitboard {
    unsafe {
        let sq_idx = sq as usize;
        let mask = ROOK_MASKS[sq_idx].0;
        let blockers = occupancy.0 & mask;
        let magic = ROOK_MAGICS[sq_idx];
        
        let shift = 64 - ROOK_MASKS[sq_idx].count();
        
        let magic_index = (blockers.wrapping_mul(magic) >> shift) as usize;
        ROOK_ATTACKS[sq_idx][magic_index]
    }
}
