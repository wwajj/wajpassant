//! Bit-packed move representation and manipulation.
//!
//! This module defines the `Move` struct, which encodes all necessary information
//! about a chess move (start square, target square, and special flags like captures
//! or promotions) into a highly efficient, single 16-bit integer.

use crate::bitboard::{SQUARES, Square};
use crate::board::PieceType;

// --- Move Flags ---
// These flags occupy the top 4 bits (12-15) of the 16-bit Move integer.

/// Flag for a standard, non-capturing move.
pub const FLAG_QUIET: u16 = 0;
/// Flag for a pawn moving two squares forward from its starting position.
pub const FLAG_DOUBLE_PAWN: u16 = 1;
/// Flag for King-side castling.
pub const FLAG_KING_CASTLE: u16 = 2;
/// Flag for Queen-side castling.
pub const FLAG_QUEEN_CASTLE: u16 = 3;
/// Flag for a standard capture.
pub const FLAG_CAPTURE: u16 = 4;
/// Flag for an En Passant pawn capture.
pub const FLAG_EN_PASSANT: u16 = 5;

/// Flag for a quiet pawn promotion to a Knight.
pub const FLAG_PROMO_N: u16 = 8;
/// Flag for a quiet pawn promotion to a Bishop.
pub const FLAG_PROMO_B: u16 = 9;
/// Flag for a quiet pawn promotion to a Rook.
pub const FLAG_PROMO_R: u16 = 10;
/// Flag for a quiet pawn promotion to a Queen.
pub const FLAG_PROMO_Q: u16 = 11;

/// Flag for a capturing pawn promotion to a Knight.
pub const FLAG_CAPTURE_PROMO_N: u16 = 12;
/// Flag for a capturing pawn promotion to a Bishop.
pub const FLAG_CAPTURE_PROMO_B: u16 = 13;
/// Flag for a capturing pawn promotion to a Rook.
pub const FLAG_CAPTURE_PROMO_R: u16 = 14;
/// Flag for a capturing pawn promotion to a Queen.
pub const FLAG_CAPTURE_PROMO_Q: u16 = 15;

/// Standard array of flags to iterate over when a quiet pawn promotion occurs.
pub const PROMO_LIST: [u16; 4] = [FLAG_PROMO_N, FLAG_PROMO_B, FLAG_PROMO_R, FLAG_PROMO_Q];
/// Standard array of flags to iterate over when a capturing pawn promotion occurs.
pub const CAPTURE_PROMO_LIST: [u16; 4] = [
    FLAG_CAPTURE_PROMO_N,
    FLAG_CAPTURE_PROMO_B,
    FLAG_CAPTURE_PROMO_R,
    FLAG_CAPTURE_PROMO_Q,
];

/// A highly optimized 16-bit chess move.
///
/// **Bit Layout:**
/// * `0-5`: Start Square (6 bits, 0-63)
/// * `6-11`: Target Square (6 bits, 0-63)
/// * `12-15`: Move Flag (4 bits, 0-15)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Move(pub u16);

// Display trait implementation
impl std::fmt::Display for Move {
    /// Formats the move into standard algebraic notation (e.g., "e2e4", "e7e8q").
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A value of 0 indicates an empty/null move
        if self.0 == 0 {
            return write!(f, "0000");
        }

        let start_idx = self.0 & 0x3F;
        let target_idx = (self.0 >> 6) & 0x3F;

        let start_file = (b'a' + (start_idx as u8 % 8)) as char;
        let start_rank = (b'1' + (start_idx as u8 / 8)) as char;

        let target_file = (b'a' + (target_idx as u8 % 8)) as char;
        let target_rank = (b'1' + (target_idx as u8 / 8)) as char;

        if let Some(promo) = self.get_promotion_char() {
            write!(
                f,
                "{}{}{}{}{}",
                start_file, start_rank, target_file, target_rank, promo
            )
        } else {
            write!(
                f,
                "{}{}{}{}",
                start_file, start_rank, target_file, target_rank
            )
        }
    }
}

// Inherent Methods
impl Move {
    /// Constructs a new 16-bit `Move` from a start square, target square, and flag.
    #[inline(always)]
    pub fn build(start: Square, target: Square, flag: u16) -> Self {
        let start_u16 = start as u16;
        let target_u16 = target as u16;

        Move(start_u16 | (target_u16 << 6) | (flag << 12))
    }

    /// Constructs an empty/null Move (internally represented as `0`).
    #[inline(always)]
    pub fn empty() -> Self {
        Self(0)
    }

    /// Extracts the starting `Square` from the move.
    #[inline(always)]
    pub fn get_start(&self) -> Square {
        let sq_idx = self.0 & 0x3F;
        SQUARES[sq_idx as usize]
    }

    /// Extracts the target `Square` from the move.
    #[inline(always)]
    pub fn get_target(&self) -> Square {
        let sq_idx = (self.0 >> 6) & 0x3F;
        SQUARES[sq_idx as usize]
    }

    /// Extracts the 4-bit flag integer from the move.
    #[inline(always)]
    pub fn get_flags(&self) -> u16 {
        (self.0 >> 12) & 0x0F
    }

    /// Returns `true` if the move is any type of capture (standard, en passant, or capturing promotion).
    #[inline(always)]
    pub fn is_capture(&self) -> bool {
        (self.get_flags() & FLAG_CAPTURE) != 0
    }

    /// Returns `true` if the move results in a pawn promotion.
    #[inline(always)]
    pub fn is_promotion(&self) -> bool {
        self.get_flags() >= FLAG_PROMO_N
    }

    /// Returns `true` if the move is a King-side or Queen-side castle.
    #[inline(always)]
    pub fn is_castling(&self) -> bool {
        let f = self.get_flags();
        f == FLAG_KING_CASTLE || f == FLAG_QUEEN_CASTLE
    }

    /// Determines the correct `PieceType` to place on the board if the move is a promotion.
    pub fn get_promotion_piece(&self) -> Option<PieceType> {
        match self.get_flags() {
            FLAG_PROMO_N | FLAG_CAPTURE_PROMO_N => Some(PieceType::Knight),
            FLAG_PROMO_B | FLAG_CAPTURE_PROMO_B => Some(PieceType::Bishop),
            FLAG_PROMO_R | FLAG_CAPTURE_PROMO_R => Some(PieceType::Rook),
            FLAG_PROMO_Q | FLAG_CAPTURE_PROMO_Q => Some(PieceType::Queen),
            _ => None,
        }
    }

    /// Returns the character representation of the promoted piece.
    pub fn get_promotion_char(&self) -> Option<char> {
        match self.get_flags() {
            FLAG_PROMO_N | FLAG_CAPTURE_PROMO_N => Some('n'),
            FLAG_PROMO_B | FLAG_CAPTURE_PROMO_B => Some('b'),
            FLAG_PROMO_R | FLAG_CAPTURE_PROMO_R => Some('r'),
            FLAG_PROMO_Q | FLAG_CAPTURE_PROMO_Q => Some('q'),
            _ => None,
        }
    }
}
