use crate::bitboard::{Square, SQUARES};
use crate::board::{PieceType};

pub const FLAG_QUIET: u16 = 0;
pub const FLAG_DOUBLE_PAWN: u16 = 1;
pub const FLAG_KING_CASTLE: u16 = 2;
pub const FLAG_QUEEN_CASTLE: u16 = 3;
pub const FLAG_CAPTURE: u16 = 4;
pub const FLAG_EN_PASSANT: u16 = 5;
pub const FLAG_PROMO_N: u16 = 8;
pub const FLAG_PROMO_B: u16 = 9;
pub const FLAG_PROMO_R: u16 = 10;
pub const FLAG_PROMO_Q: u16 = 11;
pub const FLAG_CAPTURE_PROMO_N: u16 = 12;
pub const FLAG_CAPTURE_PROMO_B: u16 = 13;
pub const FLAG_CAPTURE_PROMO_R: u16 = 14;
pub const FLAG_CAPTURE_PROMO_Q: u16 = 15;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Move(pub u16);

impl Move {
    // constructor
    pub fn build(start: Square, target: Square, flag: u16) -> Self {
        // Start is bits 0-5, Target is bits 6-11, Flag is bits 12-15
        let start_u16 = start as u16;
        let target_u16 = target as u16;
        
        Move(start_u16 | (target_u16 << 6) | (flag << 12))
    }

    // constructs empty Move 
    pub fn empty() -> Self {
        Self(0)
    }

    // gets the start Square from Move
    pub fn get_start(&self) -> Square {
        let sq_idx = self.0 & 0x3F; 
        SQUARES[sq_idx as usize]
    }

    // gets the target Square from Move
    pub fn get_target(&self) -> Square {
        let sq_idx = (self.0 >> 6) & 0x3F; 
        SQUARES[sq_idx as usize]
    }

    // gets Flags from Move 
    pub fn get_flags(&self) -> u16 {
        (self.0 >> 12) & 0x0F
    }

    // returns true if capture Flag, false otherwise
    pub fn is_capture(&self) -> bool {
        (self.get_flags() & FLAG_CAPTURE) != 0
    }

    // returns true if promotion Flag, false otherwise
    pub fn is_promotion(&self) -> bool {
        self.get_flags() >= FLAG_PROMO_N
    }

    // returns true if castling Flag, false otherwise
    pub fn is_castling(&self) -> bool {
        let f = self.get_flags();
        f == FLAG_KING_CASTLE || f == FLAG_QUEEN_CASTLE
    }

    // extracts what Piece to put on Board after promotion
    pub fn get_promotion_piece(&self) -> Option<PieceType> {
        match self.get_flags() {
            FLAG_PROMO_N | FLAG_CAPTURE_PROMO_N => return Some(PieceType::Knight),
            FLAG_PROMO_B | FLAG_CAPTURE_PROMO_B => return Some(PieceType::Bishop),
            FLAG_PROMO_R | FLAG_CAPTURE_PROMO_R => return Some(PieceType::Rook),
            FLAG_PROMO_Q | FLAG_CAPTURE_PROMO_Q => return Some(PieceType::Queen),
            _ => return None,
        };
    }

    // extracts Piece as char after promotion
    pub fn get_promotion_char(&self) -> Option<char> {
        match self.get_flags() {
            FLAG_PROMO_N | FLAG_CAPTURE_PROMO_N => Some('n'),
            FLAG_PROMO_B | FLAG_CAPTURE_PROMO_B => Some('b'),
            FLAG_PROMO_R | FLAG_CAPTURE_PROMO_R => return Some('r'),
            FLAG_PROMO_Q | FLAG_CAPTURE_PROMO_Q => return Some('q'),
            _ => return None,
        }
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
            write!(f, "{}{}{}{}{}", start_file, start_rank, target_file, target_rank, promo)
        } else {
            write!(f, "{}{}{}{}", start_file, start_rank, target_file, target_rank)
        }
    }
}
