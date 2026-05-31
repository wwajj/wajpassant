//! Magic Bitboard generation and hashing key discovery.
//!
//! This module uses a Pseudo-Random Number Generator (PRNG) to brute-force 
//! perfect hashing keys (Magic Numbers) at startup. These keys allow us to 
//! map any blocker configuration on a slider's path to a unique array index 
//! without any destructive collisions.

use crate::bitboard::{Bitboard, Square, SQUARES};
use crate::attacks::{mask_bishop, mask_rook, get_bishop_attacks_slow, get_rook_attacks_slow};

// --- Global Magic Arrays ---
/// 64 perfect hashing keys for the Bishop, generated at runtime.
pub static mut BISHOP_MAGICS: [u64; 64] = [0; 64];
/// 64 perfect hashing keys for the Rook, generated at runtime.
pub static mut ROOK_MAGICS: [u64; 64] = [0; 64];

// --- PRNG (Pseudo-Random Number Generator) ---
/// A lightweight xorshift PRNG used to find sparse magic numbers.
struct PRNG { 
    state: u64 
}

impl PRNG {
    /// Generates a random 64-bit integer.
    fn rand64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state = self.state.wrapping_mul(2685821657736338717);
        self.state
    }

    /// Generates a sparse random 64-bit integer (very few active 1s).
    /// Magic numbers hash significantly better when they have low bit density.
    fn rand_fewbits(&mut self) -> u64 {
        self.rand64() & self.rand64() & self.rand64()
    }
}

// --- The Magic Finder Algorithm ---

/// Brute-forces a magic number for a specific square and piece type.
pub fn find_magic(sq: Square, shift: u32, is_bishop: bool) -> u64 {
    let mask = if is_bishop { mask_bishop(sq) } else { mask_rook(sq) };
    let num_bits = mask.count();
    let num_blocker_configs = 1 << num_bits;
    
    let mut blockers = vec![Bitboard::empty(); num_blocker_configs];
    let mut attacks = vec![Bitboard::empty(); num_blocker_configs];
    
    let mut occupancy = Bitboard::empty();
    let mut i = 0;
    
    loop {
        blockers[i] = occupancy;
        attacks[i] = if is_bishop { 
            get_bishop_attacks_slow(sq, occupancy) 
        } else { 
            get_rook_attacks_slow(sq, occupancy) 
        };
        
        i += 1;
        
        occupancy = Bitboard(occupancy.0.wrapping_sub(1) & mask.0);
        if occupancy == Bitboard::empty() { break; }
    }

    let mut prng = PRNG { state: 123456789 }; 
    let mut used_attacks = vec![Bitboard::empty(); 1 << (64 - shift)];

    loop {
        let magic = prng.rand_fewbits();
        
        if (mask.0.wrapping_mul(magic) & 0xFF00_0000_0000_0000).count_ones() < 6 {
            continue;
        }

        let mut success = true;
        
        used_attacks.fill(Bitboard::empty()); 

        for i in 0..num_blocker_configs {
            let index = (blockers[i].0.wrapping_mul(magic) >> shift) as usize;
            
            if used_attacks[index] == Bitboard::empty() {
                used_attacks[index] = attacks[i];
            } else if used_attacks[index] != attacks[i] {
                success = false;
                break;
            }
        }

        if success {
            return magic;
        }
    }
}

// --- Initialization ---

/// Brute-forces and populates the `BISHOP_MAGICS` and `ROOK_MAGICS` arrays.
pub fn init_magics() {
    println!("Generating Magic Numbers...");
    
    for sq_idx in 0..64 {
        let sq = SQUARES[sq_idx];
        
        let bishop_shift = 64 - mask_bishop(sq).count();
        let rook_shift = 64 - mask_rook(sq).count();
        
        unsafe {
            BISHOP_MAGICS[sq_idx] = find_magic(sq, bishop_shift, true);
            ROOK_MAGICS[sq_idx] = find_magic(sq, rook_shift, false);
        }
    }
    
    println!("Magic Numbers Generated Successfully!");
}
