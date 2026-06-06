//! The core state representation of the chess engine.
//!
//! This module defines the `Board` struct, which is responsible for tracking
//! the positions of all pieces using bitboards, as well as maintaining the
//! game state (side to move, castling rights, en passant, and move clocks).

use std::io::Write;

use crate::attacks::{WHITE_PAWN_ATTACKS, BLACK_PAWN_ATTACKS, KNIGHT_ATTACKS, KING_ATTACKS, get_bishop_attacks, get_rook_attacks};
use crate::bitboard::{Bitboard, Square, SQUARES, NOT_H_FILE, NOT_A_FILE};
use crate::eval::{PST_MG, PST_EG, MATERIAL_MG, MATERIAL_EG, PHASE_WEIGHTS, MAX_PHASE, MOBILITY_WEIGHTS_MG, MOBILITY_WEIGHTS_EG};
use crate::moves::*;
use crate::movegen::{THIRD_RANK, SIXTH_RANK};
use crate::zobrist::{ZOBRIST_PIECES, ZOBRIST_SIDE, ZOBRIST_CASTLING, ZOBRIST_EN_PASSANT};

/// Starting position bitboard masks for White pieces.
pub const WHITE_START: u64 = 0x000000000000FFFF;
pub const WHITE_PAWNS: u64 = 0x000000000000FF00;
pub const WHITE_KNIGHTS: u64 = 0x0000000000000042;
pub const WHITE_BISHOPS: u64 = 0x0000000000000024;
pub const WHITE_ROOKS: u64 = 0x0000000000000081;
pub const WHITE_QUEENS: u64 = 0x0000000000000008;
pub const WHITE_KINGS: u64 = 0x0000000000000010;

/// Starting position bitboard masks for Black pieces.
pub const BLACK_START: u64 = 0xFFFF000000000000;
pub const BLACK_PAWNS: u64 = 0x00FF000000000000;
pub const BLACK_KNIGHTS: u64 = 0x4200000000000000;
pub const BLACK_BISHOPS: u64 = 0x2400000000000000;
pub const BLACK_ROOKS: u64 = 0x8100000000000000;
pub const BLACK_QUEENS: u64 = 0x0800000000000000;
pub const BLACK_KINGS: u64 = 0x1000000000000000; 

// Array used to strip castling rights.
pub const CASTLING_PERM: [u8; 64] = [
    13, 15, 15, 15, 12, 15, 15, 14, // Rank 1
    15, 15, 15, 15, 15, 15, 15, 15, // Rank 2
    15, 15, 15, 15, 15, 15, 15, 15, // Rank 3
    15, 15, 15, 15, 15, 15, 15, 15, // Rank 4
    15, 15, 15, 15, 15, 15, 15, 15, // Rank 5
    15, 15, 15, 15, 15, 15, 15, 15, // Rank 6
    15, 15, 15, 15, 15, 15, 15, 15, // Rank 7
     7, 15, 15, 15,  3, 15, 15, 11, // Rank 8
];

/// Represents the two players in a chess game.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    /// Returns the opposite color, effectively switching the turn.
    pub fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

/// Represents the six piece types in chess.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum PieceType {
    Pawn   = 0,
    Knight = 1,
    Bishop = 2,
    Rook   = 3,
    Queen  = 4,
    King   = 5,
}

impl PieceType {
    /// Converts a raw array index (0..5) back into a strongly-typed `PieceType`.
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Pawn),
            1 => Some(Self::Knight),
            2 => Some(Self::Bishop),
            3 => Some(Self::Rook),
            4 => Some(Self::Queen),
            5 => Some(Self::King),
            _ => None,
        }
    }
}

/// A snapshot of the irreversible aspects of the board state.
///
/// Pushed to the board's history stack before a move is made, allowing
/// the engine to perfectly restore the position during `unmake_move`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct UndoRecord {
    /// The target square for a valid En Passant capture, if any.
    pub en_passant: Option<Square>,
    /// Bitmask representing castling availability (WK=1, WQ=2, BK=4, BQ=8).
    pub castling_rights: u8,
    /// The number of halfmoves since the last capture or pawn advance (for the 50-move rule).
    pub halfmove_clock: u16,
    /// PieceType captured by move.
    pub captured_piece: Option<PieceType>,
    /// evaluation tracking
    pub mg_score: i32,
    pub eg_score: i32,
    pub phase: i32,
}

// Inherent methods
impl UndoRecord {
    /// Constructs an `UndoRecord`
    pub fn new(en_passant: Option<Square>, castling_rights: u8, halfmove_clock: u16, captured_piece:Option<PieceType>, mg_score: i32, eg_score: i32, phase: i32) -> Self {
        Self {
            en_passant: en_passant,
            castling_rights: castling_rights,
            halfmove_clock: halfmove_clock,
            captured_piece: captured_piece,
            mg_score: mg_score,
            eg_score: eg_score,
            phase: phase,
        }
    }
}

/// The master representation of the chess board state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    /// 2D array of bitboards separated by [Color][PieceType].
    pub pieces: [[Bitboard; 6]; 2],
    /// Master occupancies: [White, Black, Total].
    pub occupancies: [Bitboard; 3],
    /// The player whose turn it is to move.
    pub side_to_move: Color,
    /// The target square for a valid En Passant capture, if any.
    pub en_passant: Option<Square>,
    /// Bitmask representing castling availability (WK=1, WQ=2, BK=4, BQ=8).
    pub castling_rights: u8,
    /// The number of halfmoves since the last capture or pawn advance (for the 50-move rule).
    pub halfmove_clock: u16,
    /// The number of full moves in the game. Increments after Black's move.
    pub fullmove_number: u16,
    /// The history stack used to unmake moves
    pub history: Vec<UndoRecord>,
    /// Evaluation tracking
    pub mg_score: i32,
    pub eg_score: i32,
    pub phase: i32,
    // History stack to track positions over time
    pub hash_history: Vec<u64>,
}

// Standard traits 
impl Default for Board {
    /// Constructs a `Board` fully populated at the standard chess starting position.
    fn default() -> Self {
        let pieces = [
            [
                Bitboard::new(WHITE_PAWNS),
                Bitboard::new(WHITE_KNIGHTS),
                Bitboard::new(WHITE_BISHOPS),
                Bitboard::new(WHITE_ROOKS),
                Bitboard::new(WHITE_QUEENS),
                Bitboard::new(WHITE_KINGS),
            ],
            [
                Bitboard::new(BLACK_PAWNS),
                Bitboard::new(BLACK_KNIGHTS),
                Bitboard::new(BLACK_BISHOPS),
                Bitboard::new(BLACK_ROOKS),
                Bitboard::new(BLACK_QUEENS),
                Bitboard::new(BLACK_KINGS),
            ]
        ];

        let occupancies = [
            Bitboard::new(WHITE_START),
            Bitboard::new(BLACK_START),
            Bitboard::new(WHITE_START | BLACK_START),
        ];

        let mut board = Self {
            pieces,
            occupancies,
            side_to_move: Color::White,
            en_passant: None,
            castling_rights: 0b1111,
            halfmove_clock: 0,
            fullmove_number: 1,
            history: Vec::with_capacity(256),
            mg_score: 0,
            eg_score: 0,
            phase: 0,
            hash_history: Vec::new(),
        };

        let initial_hash = board.calculate_hash();
        board.hash_history.push(initial_hash);

        board
    }
}

// Inherent methods
impl Board {
    /// Constructs a completely empty `Board` with no pieces.
    pub fn empty() -> Self {
        let mut board = Self {
            pieces: [
                [Bitboard::empty(); 6],
                [Bitboard::empty(); 6]
            ],
            occupancies: [Bitboard::empty(); 3],
            side_to_move: Color::White,
            en_passant: None,
            castling_rights: 0,
            halfmove_clock: 0,
            fullmove_number: 1,
            history: Vec::with_capacity(256),
            mg_score: 0,
            eg_score: 0,
            phase: 0,
            hash_history: Vec::new(),
        };

        let initial_hash = board.calculate_hash();
        board.hash_history.push(initial_hash);

        board
    }

    /// Parses a standard FEN string and returns a populated `Board`.
    pub fn from_fen(fen: &str) -> Self {
        let mut board = Self::empty();

        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.is_empty() { return board };

        let mut rank = 7;
        let mut file = 0;

        for c in parts[0].chars() {
            if c == '/' {
                rank -= 1;
                file = 0;
            } else if c.is_digit(10) {
                file += c.to_digit(10).unwrap() as usize;
            } else {
                let side = if c.is_uppercase() { Color::White } else { Color::Black};
                let piece_type = match c.to_ascii_lowercase() {
                    'p' => PieceType::Pawn,
                    'n' => PieceType::Knight,
                    'b' => PieceType::Bishop,
                    'r' => PieceType::Rook,
                    'q' => PieceType::Queen,
                    'k' => PieceType::King,
                    _ => panic!("Invalid FEN piece character."),
                };

                let sq = SQUARES[rank * 8 + file];
                board.pieces[side as usize][piece_type as usize] =
                    board.pieces[side as usize][piece_type as usize].set_bit(sq);

                file += 1;
            }
        }

        if parts.len() > 1 {
            board.side_to_move = if parts[1] == "w" { Color::White } else { Color::Black };
        }

        if parts.len() > 2 {
            for c in parts[2].chars() {
                match c {
                    'K' => board.castling_rights |= 1,
                    'Q' => board.castling_rights |= 2,
                    'k' => board.castling_rights |= 4,
                    'q' => board.castling_rights |= 8,
                    '-' => break,
                    _ => {}
                }
            }
        }

        if parts.len() > 3 && parts[3] != "-" {
            let file_char = parts[3].chars().nth(0).unwrap();
            let rank_char = parts[3].chars().nth(1).unwrap();

            let file_idx = (file_char as u8 - b'a') as usize;
            let rank_idx = (rank_char as u8 - b'1') as usize;

            board.en_passant = Some(SQUARES[rank_idx * 8 + file_idx]);
        }

        if parts.len() > 4 {
            board.halfmove_clock = parts[4].parse().unwrap_or(0);
        }
        if parts.len() > 5 {
            board.fullmove_number = parts[5].parse().unwrap_or(1);
        }

        board.update_occupancies();
        board.init_eval();

        let initial_hash = board.calculate_hash();
        board.hash_history.push(initial_hash);

        board
    }

    /// Converts the current `Board` state into a standard FEN string.
    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        for rank in (0..8).rev() {
            let mut empty_count = 0;
            for file in 0..8 {
                let sq = SQUARES[rank * 8 + file];
                
                if let Some(pt) = self.piece_at(sq) {
                    if empty_count > 0 {
                        fen.push_str(&empty_count.to_string());
                        empty_count = 0;
                    }
                    
                    let mut c = match pt {
                        PieceType::Pawn   => 'p',
                        PieceType::Knight => 'n',
                        PieceType::Bishop => 'b',
                        PieceType::Rook   => 'r',
                        PieceType::Queen  => 'q',
                        PieceType::King   => 'k',
                    };
                    
                    if self.occupancies[Color::White as usize].is_occupied(sq) {
                        c = c.to_ascii_uppercase();
                    }
                    fen.push(c);
                } else {
                    empty_count += 1;
                }
            }
            if empty_count > 0 {
                fen.push_str(&empty_count.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push_str(match self.side_to_move {
            Color::White => " w ",
            Color::Black => " b ",
        });

        let mut castling = String::new();
        if (self.castling_rights & 1) != 0 { castling.push('K'); }
        if (self.castling_rights & 2) != 0 { castling.push('Q'); }
        if (self.castling_rights & 4) != 0 { castling.push('k'); }
        if (self.castling_rights & 8) != 0 { castling.push('q'); }
        if castling.is_empty() { castling.push('-'); }
        fen.push_str(&format!("{} ", castling));

        match self.en_passant {
            Some(sq) => fen.push_str(&format!("{:?} ", sq).to_lowercase()),
            None => fen.push_str("- "),
        }

        fen.push_str(&format!("{} {}", self.halfmove_clock, self.fullmove_number));
        fen
    }

    /// Prints the board as a formatted 8x8 console grid alongside relevant game state variables.
    pub fn print(&self) {
        let piece_chars = ['P', 'N', 'B', 'R', 'Q', 'K'];

        println!();

        for rank in (0..8).rev() {
            print!("{}  ", rank + 1);

            for file in 0..8 {
                let sq = SQUARES[rank * 8 + file];
                let mut piece_printed = false;

                for side in 0..2 {
                    for piece_idx in 0..6 {
                        if self.pieces[side][piece_idx].is_occupied(sq) {
                            let mut c = piece_chars[piece_idx];

                            if side == Color::Black as usize {
                                c = c.to_ascii_lowercase();
                            }
                            print!("{} ", c);
                            piece_printed = true;
                            break;
                        }
                    }
                    if piece_printed { break; }
                }
                if !piece_printed {
                    print!(". ");
                }
            }
            println!();
        }

        println!("\n   A B C D E F G H\n");
        println!("  Side to move: {}", match self.side_to_move {
            Color::White => "White",
            Color::Black => "Black",
        });

        println!("  Castling    : {}{}{}{}",
            if (self.castling_rights & 1) != 0 {"K"} else {"-"},
            if (self.castling_rights & 2) != 0 {"Q"} else {"-"},
            if (self.castling_rights & 4) != 0 {"k"} else {"-"},
            if (self.castling_rights & 8) != 0 {"q"} else {"-"},
        );

        println!("  En Passant  : {}", match self.en_passant {
            Some(sq) => format!("{:?}", sq),
            None => "-".to_string(),
        });
    } 

    /// Probes the bitboards to return the specific `PieceType` residing on a given square.
    pub fn piece_at(&self, sq: Square) -> Option<PieceType> {
        if !self.occupancies[2].is_occupied(sq) {
            return None;
        }

        let color = if self.occupancies[0].is_occupied(sq) { Color::White } else { Color::Black };

        for (piece_idx, bb) in self.pieces[color as usize].iter().enumerate() {
            if bb.is_occupied(sq) {
                return PieceType::from_index(piece_idx);
            }
        }

        None
    }

    /// Completely recalculates the 3 master occupancy bitboards based on the individual piece boards.
    pub fn update_occupancies(&mut self) {
        self.occupancies[0] = Bitboard::empty();
        self.occupancies[1] = Bitboard::empty();
        
        for pt in 0..6 {
            self.occupancies[0] |= self.pieces[Color::White as usize][pt];
            self.occupancies[1] |= self.pieces[Color::Black as usize][pt];
        }
        
        self.occupancies[2] = self.occupancies[0] | self.occupancies[1];
    }

    /// Removes a piece from the board and synchronizes the master occupancy masks.
    pub fn remove_piece(&mut self, sq: Square, side: Color, pt: PieceType) {
        self.pieces[side as usize][pt as usize] = self.pieces[side as usize][pt as usize].clear_bit(sq);

        self.occupancies[side as usize] = self.occupancies[side as usize].clear_bit(sq);
        self.occupancies[2] = self.occupancies[2].clear_bit(sq);
    }

    /// Adds a piece to the board and synchronizes the master occupancy masks.
    pub fn add_piece(&mut self, sq: Square, side: Color, pt: PieceType) {
        self.pieces[side as usize][pt as usize] = self.pieces[side as usize][pt as usize].set_bit(sq);

        self.occupancies[side as usize] = self.occupancies[side as usize].set_bit(sq);
        self.occupancies[2] = self.occupancies[2].set_bit(sq);
    }

    /// Determines if a square is under attack by the given color
    /// returns `true` if the square is under attack, `false` otherwise
    pub fn is_square_attacked(&self, sq: Square, attacker: Color) -> bool {
        let sq_idx = sq as usize;
        let attacker_idx = attacker as usize;
        let occ = self.occupancies[2];

        if (unsafe { KNIGHT_ATTACKS[sq_idx] } & self.pieces[attacker_idx][PieceType::Knight as usize]).0 != 0 {
            return true;
        };
        if (unsafe { KING_ATTACKS[sq_idx] } & self.pieces[attacker_idx][PieceType::King as usize]).0 != 0 {
            return true;
        };

        let diagonal_attackers = self.pieces[attacker_idx][PieceType::Bishop as usize]
                               | self.pieces[attacker_idx][PieceType::Queen as usize];
        if (get_bishop_attacks(sq, occ) & diagonal_attackers).0 != 0 {
            return true;
        }

        let orthogonal_attackers = self.pieces[attacker_idx][PieceType::Rook as usize]
                                 | self.pieces[attacker_idx][PieceType::Queen as usize];
        if (get_rook_attacks(sq, occ) & orthogonal_attackers).0 != 0 {
            return true;
        }

        let pawns = self.pieces[attacker_idx][PieceType::Pawn as usize];
        let sq_bb = Bitboard(1u64 << sq_idx as u64);

        if attacker == Color::White {
            let attacker_mask = ((sq_bb & Bitboard(NOT_H_FILE)) >> 7)
                              | ((sq_bb & Bitboard(NOT_A_FILE)) >> 9);
            if (attacker_mask & pawns).0 != 0 { return true; }
        } else {
            let attacker_mask = ((sq_bb & Bitboard(NOT_A_FILE)) << 7)
                              | ((sq_bb & Bitboard(NOT_H_FILE)) << 9);
            if (attacker_mask & pawns).0 != 0 { return true; }
        }

        false
    }

    /// Executes a move on the board
    /// Returns `true` if the move is legal, `false` if the move leaves the King in check
    pub fn make_move(&mut self, mv: Move) -> bool {
        let start = mv.get_start();
        let target = mv.get_target();
        let flag = mv.get_flags();

        let moved = self.piece_at(start).unwrap();
        let captured = self.piece_at(target);
        
        let record = UndoRecord::new(
            self.en_passant,
            self.castling_rights,
            self.halfmove_clock,
            captured,
            self.mg_score, 
            self.eg_score, 
            self.phase,
        );
        self.history.push(record);

        let us = self.side_to_move;
        let them = us.flip();

        let move_mask = (1u64 << start as u64) | (1u64 << target as u64);
        self.pieces[us as usize][moved as usize] ^= Bitboard(move_mask);

        self.remove_eval(us, moved, start);
        self.add_eval(us, moved, target);

        let target_mask = Bitboard(1u64 << target as u64);
        match flag {
            FLAG_QUIET | FLAG_DOUBLE_PAWN => {},
            
            FLAG_CAPTURE => {
                self.pieces[them as usize][captured.unwrap() as usize] ^= target_mask;
                self.remove_eval(them, captured.unwrap(), target);
            },
            FLAG_PROMO_N | FLAG_PROMO_B | FLAG_PROMO_R | FLAG_PROMO_Q => {
                let promo_piece = mv.get_promotion_piece().unwrap();
                self.pieces[us as usize][PieceType::Pawn as usize] ^= target_mask;
                self.pieces[us as usize][promo_piece as usize] ^= target_mask;
                
                // We moved a pawn to the target square in step 3, so remove it, then add the promoted piece
                self.remove_eval(us, PieceType::Pawn, target);
                self.add_eval(us, promo_piece, target);
            },
            FLAG_CAPTURE_PROMO_N | FLAG_CAPTURE_PROMO_B | FLAG_CAPTURE_PROMO_R | FLAG_CAPTURE_PROMO_Q => {
                let promo_piece = mv.get_promotion_piece().unwrap();
                self.pieces[them as usize][captured.unwrap() as usize] ^= target_mask;
                self.pieces[us as usize][PieceType::Pawn as usize] ^= target_mask;
                self.pieces[us as usize][promo_piece as usize] ^= target_mask;
                
                self.remove_eval(them, captured.unwrap(), target);
                self.remove_eval(us, PieceType::Pawn, target);
                self.add_eval(us, promo_piece, target);
            },
            FLAG_EN_PASSANT => {
                let capture_sq = if us == Color::White { (target as usize) - 8 } else { (target as usize) + 8 };
                self.pieces[them as usize][PieceType::Pawn as usize] ^= Bitboard(1u64 << capture_sq as u64);
                
                self.remove_eval(them, PieceType::Pawn, SQUARES[capture_sq]);
            },
            FLAG_KING_CASTLE => {
                let rook_mask = 0b1010_0000;
                let shift = if us == Color::White { 0 } else { 56 };
                self.pieces[us as usize][PieceType::Rook as usize] ^= Bitboard(rook_mask << shift);
                
                let (r_start, r_target) = if us == Color::White { (7, 5) } else { (63, 61) }; // H1->F1 or H8->F8
                self.remove_eval(us, PieceType::Rook, SQUARES[r_start]);
                self.add_eval(us, PieceType::Rook, SQUARES[r_target]);
            },
            FLAG_QUEEN_CASTLE => {
                let rook_mask = 0b0000_1001;
                let shift = if us == Color::White { 0 } else { 56 };
                self.pieces[us as usize][PieceType::Rook as usize] ^= Bitboard(rook_mask << shift);
                
                let (r_start, r_target) = if us == Color::White { (0, 3) } else { (56, 59) }; // A1->D1 or A8->D8
                self.remove_eval(us, PieceType::Rook, SQUARES[r_start]);
                self.add_eval(us, PieceType::Rook, SQUARES[r_target]);
            },
            _ => panic!("Invalid move flag encountered during make move."),
        };

        if (mv.is_capture()) || (moved == PieceType::Pawn) {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        if us == Color::Black {
            self.fullmove_number += 1;
        }

        self.castling_rights &= CASTLING_PERM[start as usize];
        self.castling_rights &= CASTLING_PERM[target as usize];

        if flag == FLAG_DOUBLE_PAWN {
            let ep_sq = if us == Color::White { (target as usize) - 8 } else { (target as usize) + 8 };
            self.en_passant = Some(SQUARES[ep_sq as usize]);
        } else {
            self.en_passant = None;
        }

        self.side_to_move = self.side_to_move.flip();
        self.update_occupancies();

        let new_hash = self.calculate_hash();
        self.hash_history.push(new_hash);

        let king_sq = self.pieces[us as usize][PieceType::King as usize].get_lsb();
        if self.is_square_attacked(king_sq, them) { 
            self.unmake_move(mv);
            return false;
        };

        true
    }

    /// Reverses the last move made on the board, restoring the previous state
    pub fn unmake_move(&mut self, mv:Move) {
        self.hash_history.pop();
        let record = self.history.pop().expect("Tried to unmake a move with empty UndoRecord Vector");

        self.side_to_move = self.side_to_move.flip();
        let us = self.side_to_move;
        let them = self.side_to_move.flip();

        let start = mv.get_start();
        let target = mv.get_target();
        let flag = mv.get_flags();

        let moved = if mv.is_promotion() {
            PieceType::Pawn
        } else {
            self.piece_at(target).expect("No piece found on target square during unmake move")
        };

        let move_mask = Bitboard((1u64 << start as u64) | (1u64 << target as u64));
        self.pieces[us as usize][moved as usize] ^= move_mask;

        let target_mask = Bitboard(1u64 << target as u64);
        match flag {
            FLAG_QUIET | FLAG_DOUBLE_PAWN => {}, 
            FLAG_CAPTURE => {
                self.pieces[them as usize][record.captured_piece.unwrap() as usize] ^= target_mask;
            },
            FLAG_PROMO_N | FLAG_PROMO_B | FLAG_PROMO_R | FLAG_PROMO_Q => {
                let promo_piece = mv.get_promotion_piece().unwrap();
                self.pieces[us as usize][PieceType::Pawn as usize] ^= target_mask;
                self.pieces[us as usize][promo_piece as usize] ^= target_mask;
            },
            FLAG_CAPTURE_PROMO_N | FLAG_CAPTURE_PROMO_B | FLAG_CAPTURE_PROMO_R | FLAG_CAPTURE_PROMO_Q => {
                let promo_piece = mv.get_promotion_piece().unwrap();
                self.pieces[them as usize][record.captured_piece.unwrap() as usize] ^= target_mask;
                self.pieces[us as usize][PieceType::Pawn as usize] ^= target_mask;
                self.pieces[us as usize][promo_piece as usize] ^= target_mask;
            },
            FLAG_EN_PASSANT => {
                let capture_sq = if us == Color::White { 
                    (target as usize) - 8 
                } else { 
                    (target as usize) + 8 
                };
                self.pieces[them as usize][PieceType::Pawn as usize] ^= Bitboard(1u64 << capture_sq);
            },
            FLAG_KING_CASTLE => {
                let rook_mask = 0b1010_0000;
                let shift = if us == Color::White { 0 } else { 56 };
                self.pieces[us as usize][PieceType::Rook as usize] ^= Bitboard(rook_mask << shift);
            },
            FLAG_QUEEN_CASTLE => {
                let rook_mask = 0b0000_1001;
                let shift = if us == Color::White { 0 } else { 56 };
                self.pieces[us as usize][PieceType::Rook as usize] ^= Bitboard(rook_mask << shift);
            },
            _ => panic!("Invalid move flag encountered during unmake_move."),
        }

        self.en_passant = record.en_passant;
        self.castling_rights = record.castling_rights;
        self.halfmove_clock = record.halfmove_clock;
        if us == Color:: Black {
            self.fullmove_number -= 1;
        }

        self.mg_score = record.mg_score;
        self.eg_score = record.eg_score;
        self.phase = record.phase;

        self.update_occupancies();
    }

    /// Extracts individual target squares from an attack bitboard and creates `Move` structs.
    #[inline(always)]
    fn serialize_moves(&self, moves: &mut Vec<Move>, start: Square, mut attacks: Bitboard, enemies: Bitboard) {
        while attacks.0 != 0 {
            let target_idx = attacks.pop_lsb();
            let target = SQUARES[target_idx as usize];
            
            let flag = if enemies.is_occupied(target) { FLAG_CAPTURE } else { FLAG_QUIET };

            moves.push(Move::build(start, target, flag));
        }
    }

    /// Generates all pseudo-legal moves for the current player in the given position.
    ///
    /// This function  generates moves that follow piece movement rules, but it does 
    /// NOT verify if the move leaves the King in check. 
    pub fn generate_all_moves(&self) -> Vec<Move> {
        let mut moves = Vec::with_capacity(256);

        let us = self.side_to_move as usize;
        let them = self.side_to_move.flip() as usize;

        let occ_us = self.occupancies[us];
        let occ_them = self.occupancies[them];
        let occ_all = self.occupancies[2];

        let mut knights = self.pieces[us][PieceType::Knight as usize];
        while knights.0 != 0 {
            let start_idx = knights.pop_lsb();
            let start_sq = SQUARES[start_idx as usize];
            let attacks = unsafe { KNIGHT_ATTACKS[start_idx as usize] } & !occ_us;
            self.serialize_moves(&mut moves, start_sq, attacks, occ_them);
        }

        let mut kings = self.pieces[us][PieceType::King as usize];
        while kings.0 != 0 {
            let start_idx = kings.pop_lsb();
            let start_sq = SQUARES[start_idx as usize];
            let attacks = unsafe { KING_ATTACKS[start_idx as usize] } & !occ_us;
            self.serialize_moves(&mut moves, start_sq, attacks, occ_them);
        }

        let mut diagonal_sliders = self.pieces[us][PieceType::Bishop as usize] 
                                 | self.pieces[us][PieceType::Queen as usize];
        while diagonal_sliders.0 != 0 {
            let start_idx = diagonal_sliders.pop_lsb();
            let start_sq = SQUARES[start_idx as usize];
            let attacks = get_bishop_attacks(start_sq, occ_all) & !occ_us;
            self.serialize_moves(&mut moves, start_sq, attacks, occ_them);
        }

        let mut orthogonal_sliders = self.pieces[us][PieceType::Rook as usize] 
                                   | self.pieces[us][PieceType::Queen as usize];
        while orthogonal_sliders.0 != 0 {
            let start_idx = orthogonal_sliders.pop_lsb();
            let start_sq = SQUARES[start_idx as usize];
            let attacks = get_rook_attacks(start_sq, occ_all) & !occ_us;
            self.serialize_moves(&mut moves, start_sq, attacks, occ_them);
        }

        let mut pawns = self.pieces[us][PieceType::Pawn as usize];

        while pawns.0 != 0 {
            let start_idx = pawns.pop_lsb();
            let start_sq = SQUARES[start_idx as usize];
            let start_bb = Bitboard(1u64 << (start_idx as u64));

            if us == Color::White as usize {
                let rank = (start_idx as usize) / 8;

                let single_push = Bitboard(start_bb.0 << 8) & !occ_all;
                if single_push.0 != 0 {
                    let target_idx = (start_idx as usize) + 8;
                    let target_sq = SQUARES[target_idx as usize];

                    if target_idx / 8 == 7 { 
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_Q));
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_R));
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_B));
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_N));
                    } else { 
                        moves.push(Move::build(start_sq, target_sq, FLAG_QUIET));

                        if rank == 1 {
                            let double_push = Bitboard(start_bb.0 << 16) & !occ_all;
                            if double_push.0 != 0 {
                                let d_sq = SQUARES[(start_idx as usize) + 16];
                                moves.push(Move::build(start_sq, d_sq, FLAG_DOUBLE_PAWN));
                            }
                        }
                    }
                }

                let attacks = Bitboard(((start_bb.0 & NOT_A_FILE) << 7) | ((start_bb.0 & NOT_H_FILE) << 9));
                let mut valid_captures = attacks & occ_them;

                while valid_captures.0 != 0 {
                    let target_idx = valid_captures.pop_lsb();
                    let target_sq = SQUARES[target_idx as usize];

                    if (target_idx as usize) / 8 == 7 {
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_Q));
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_R));
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_B));
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_N));
                    } else {
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE));
                    }
                }

                if let Some(ep_sq) = self.en_passant {
                    let ep_bb = Bitboard(1u64 << (ep_sq as usize) as u64);
                    if (attacks & ep_bb).0 != 0 {
                        moves.push(Move::build(start_sq, ep_sq, FLAG_EN_PASSANT));
                    }
                }

            } else {
                let rank = (start_idx as usize) / 8;

                let single_push = Bitboard(start_bb.0 >> 8) & !occ_all;
                if single_push.0 != 0 {
                    let target_idx = (start_idx as usize) - 8;
                    let target_sq = SQUARES[target_idx as usize];

                    if target_idx / 8 == 0 { 
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_Q));
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_R));
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_B));
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_N));
                    } else { 
                        moves.push(Move::build(start_sq, target_sq, FLAG_QUIET));

                        if rank == 6 {
                            let double_push = Bitboard(start_bb.0 >> 16) & !occ_all;
                            if double_push.0 != 0 {
                                let d_sq = SQUARES[(start_idx as usize) - 16];
                                moves.push(Move::build(start_sq, d_sq, FLAG_DOUBLE_PAWN));
                            }
                        }
                    }
                }

                let attacks = Bitboard(((start_bb.0 & NOT_A_FILE) >> 9) | ((start_bb.0 & NOT_H_FILE) >> 7));
                let mut valid_captures = attacks & occ_them;

                while valid_captures.0 != 0 {
                    let target_idx = valid_captures.pop_lsb();
                    let target_sq = SQUARES[target_idx as usize];

                    if (target_idx as usize) / 8 == 0 {
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_Q));
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_R));
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_B));
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_N));
                    } else {
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE));
                    }
                }

                if let Some(ep_sq) = self.en_passant {
                    let ep_bb = Bitboard(1u64 << (ep_sq as usize) as u64);
                    if (attacks & ep_bb).0 != 0 {
                        moves.push(Move::build(start_sq, ep_sq, FLAG_EN_PASSANT));
                    }
                }
            }
        }

        if us == Color::White as usize {
            let them_color = Color::Black;
            if (self.castling_rights & 1) != 0 {
                if !occ_all.is_occupied(SQUARES[5]) && !occ_all.is_occupied(SQUARES[6]) {
                    if !self.is_square_attacked(SQUARES[4], them_color)
                        && !self.is_square_attacked(SQUARES[5], them_color)
                        && !self.is_square_attacked(SQUARES[6], them_color) {
                        moves.push(Move::build(SQUARES[4], SQUARES[6], FLAG_KING_CASTLE));
                    }
                }
            }
            if (self.castling_rights & 2) != 0 {
                if !occ_all.is_occupied(SQUARES[1]) 
                    && !occ_all.is_occupied(SQUARES[2]) 
                    && !occ_all.is_occupied(SQUARES[3]) {
                    if !self.is_square_attacked(SQUARES[4], them_color)
                        && !self.is_square_attacked(SQUARES[3], them_color)
                        && !self.is_square_attacked(SQUARES[2], them_color) {
                        moves.push(Move::build(SQUARES[4], SQUARES[2], FLAG_QUEEN_CASTLE));
                    }
                }
            }
        } else {
            let them_color = Color::White;
            if (self.castling_rights & 4) != 0 {
                if !occ_all.is_occupied(SQUARES[61]) && !occ_all.is_occupied(SQUARES[62]) {
                    if !self.is_square_attacked(SQUARES[60], them_color)
                        && !self.is_square_attacked(SQUARES[61], them_color)
                        && !self.is_square_attacked(SQUARES[62], them_color) {
                        moves.push(Move::build(SQUARES[60], SQUARES[62], FLAG_KING_CASTLE));
                    }
                }
            }
            if (self.castling_rights & 8) != 0 {
                if !occ_all.is_occupied(SQUARES[57]) 
                    && !occ_all.is_occupied(SQUARES[58]) 
                    && !occ_all.is_occupied(SQUARES[59]) {
                    if !self.is_square_attacked(SQUARES[60], them_color)
                        && !self.is_square_attacked(SQUARES[59], them_color)
                        && !self.is_square_attacked(SQUARES[58], them_color) {
                        moves.push(Move::build(SQUARES[60], SQUARES[58], FLAG_QUEEN_CASTLE));
                    }
                }
            }
        }
        moves
    }

    /// Recursively walks the move tree to a given depth and counts the number of leaf nodes.
    ///
    /// This is a mathematical correctness test (Performance Test) used to verify 
    /// the flawless execution of move generation, `make_move`, and `unmake_move`.
    pub fn perft(&mut self, depth: u8) -> u64 {
        if depth == 0 {
            return 1;
        }

        let mut nodes: u64 = 0;
        
        let moves = self.generate_all_moves(); 

        for mv in moves {
            if self.make_move(mv) {
                nodes += self.perft(depth - 1);
                self.unmake_move(mv);
            }
        }

        nodes
    }

    /// A debugging version of Perft that writes every move sequence to a text file.
    pub fn perft_debug(&mut self, depth: u8, path: String, file: &mut std::fs::File) -> u64 {
        if depth == 0 {
            writeln!(file, "{}", path).unwrap();
            return 1;
        }

        let mut nodes: u64 = 0;
        let moves = self.generate_all_moves(); 

        for mv in moves {
            if self.make_move(mv) {
                let start_idx = mv.get_start() as usize;
                let target_idx = mv.get_target() as usize;
                
                let s_file = (b'a' + (start_idx % 8) as u8) as char;
                let s_rank = (b'1' + (start_idx / 8) as u8) as char;
                let t_file = (b'a' + (target_idx % 8) as u8) as char;
                let t_rank = (b'1' + (target_idx / 8) as u8) as char;
                
                let move_str = format!("{}{}{}{}", s_file, s_rank, t_file, t_rank);
                
                let new_path = if path.is_empty() {
                    move_str
                } else {
                    format!("{} {}", path, move_str)
                };

                nodes += self.perft_debug(depth - 1, new_path, file);
                
                self.unmake_move(mv);
            }
        }

        nodes
    }

    /// Calculates the static evaluation from scratch.
    pub fn init_eval(&mut self) {
        self.mg_score = 0;
        self.eg_score = 0;
        self.phase = 0;

        for side in 0..2 {
            let sign = if side == Color::White as usize { 1 } else { -1 };
            
            for pt in 0..6 {
                let mut bb = self.pieces[side][pt];
                
                while bb.0 != 0 {
                    let sq = bb.get_lsb() as usize;
                    bb = bb.clear_bit(SQUARES[sq]);

                    let lookup_sq = if side == Color::Black as usize { sq ^ 56 } else { sq };

                    self.mg_score += sign * (MATERIAL_MG[pt] + PST_MG[pt][lookup_sq]);
                    self.eg_score += sign * (MATERIAL_EG[pt] + PST_EG[pt][lookup_sq]);

                    self.phase += PHASE_WEIGHTS[pt];
                }
            }
        }

        if self.phase > MAX_PHASE { self.phase = MAX_PHASE; }
    }

    /// Returns the tapered evaluation of the current position in centipawns.
    pub fn evaluate(&self, alpha: i32, beta: i32) -> i32 {
        let p = self.phase.min(MAX_PHASE);

        let mut bonus = 0;
        if self.has_castled(Color::White) { bonus += 50; }
        if self.has_castled(Color::Black) { bonus -= 50; }

        if self.phase > 15 {
            if !self.can_castle(Color::White) { bonus -= 30; }
            if !self.can_castle(Color::Black) { bonus += 30; }
        }
        
        let mut score = ((self.mg_score * p + self.eg_score * (MAX_PHASE - p)) / MAX_PHASE) + bonus;
        let perspective_base = if self.side_to_move == Color::White { score } else { -score };

        let margin = 150;
        
        if perspective_base + margin <= alpha {
            return perspective_base;
        }
        if perspective_base - margin >= beta {
            return perspective_base;
        }

        let mut white_mobility_mg = 0;
        let mut white_mobility_eg = 0;
        let mut black_mobility_mg = 0;
        let mut black_mobility_eg = 0;

        for pt in 1..6 {
            let piece = PieceType::from_index(pt).unwrap();

            let w_count = self.get_piece_moves_bb(Color::White, piece).count() as i32;
            let b_count = self.get_piece_moves_bb(Color::Black, piece).count() as i32;

            white_mobility_mg += w_count * MOBILITY_WEIGHTS_MG[pt];
            white_mobility_eg += w_count * MOBILITY_WEIGHTS_EG[pt];
            
            black_mobility_mg += b_count * MOBILITY_WEIGHTS_MG[pt];
            black_mobility_eg += b_count * MOBILITY_WEIGHTS_EG[pt];
        }
        let w_mob_tapered = (white_mobility_mg * p + white_mobility_eg * (MAX_PHASE - p)) / MAX_PHASE;
        let b_mob_tapered = (black_mobility_mg * p + black_mobility_eg * (MAX_PHASE - p)) / MAX_PHASE;

        score += w_mob_tapered - b_mob_tapered;

        if self.side_to_move == Color::White {
            score
        } else {
            -score
        }
    }

    /// Incrementally adds a piece's value to the global board evaluation.
    #[inline(always)]
    fn add_eval(&mut self, side: Color, pt: PieceType, sq: Square) {
        let sign = if side == Color::White { 1 } else { -1 };
        let lookup_sq = if side == Color::Black { (sq as usize) ^ 56 } else { sq as usize };
        let pt_idx = pt as usize;

        self.mg_score += sign * (MATERIAL_MG[pt_idx] + PST_MG[pt_idx][lookup_sq]);
        self.eg_score += sign * (MATERIAL_EG[pt_idx] + PST_EG[pt_idx][lookup_sq]);
        self.phase += PHASE_WEIGHTS[pt_idx];
    }

    /// Incrementally removes a piece's value from the global board evaluation.
    #[inline(always)]
    fn remove_eval(&mut self, side: Color, pt: PieceType, sq: Square) {
        let sign = if side == Color::White { 1 } else { -1 };
        let lookup_sq = if side == Color::Black { (sq as usize) ^ 56 } else { sq as usize };
        let pt_idx = pt as usize;

        self.mg_score -= sign * (MATERIAL_MG[pt_idx] + PST_MG[pt_idx][lookup_sq]);
        self.eg_score -= sign * (MATERIAL_EG[pt_idx] + PST_EG[pt_idx][lookup_sq]);
        self.phase -= PHASE_WEIGHTS[pt_idx];
    }

    /// Generates all possible moves that involve a capture
    pub fn generate_captures(&self) -> Vec<Move> {
        let mut moves = Vec::with_capacity(32);

        let us = self.side_to_move as usize;
        let them = self.side_to_move.flip() as usize;

        let occ_them = self.occupancies[them];
        let occ_all = self.occupancies[2];

        let mut knights = self.pieces[us][PieceType::Knight as usize];
        while knights.0 != 0 {
            let start_idx = knights.pop_lsb();
            let start_sq = SQUARES[start_idx as usize];

            let attacks = unsafe { KNIGHT_ATTACKS[start_idx as usize] } & occ_them;
            self.serialize_moves(&mut moves, start_sq, attacks, occ_them);
        }

        let mut kings = self.pieces[us][PieceType::King as usize];
        while kings.0 != 0 {
            let start_idx = kings.pop_lsb();
            let start_sq = SQUARES[start_idx as usize];

            let attacks = unsafe { KING_ATTACKS[start_idx as usize] } & occ_them;
            self.serialize_moves(&mut moves, start_sq, attacks, occ_them);
        }

        let mut diagonal_sliders = self.pieces[us][PieceType::Bishop as usize] 
                                 | self.pieces[us][PieceType::Queen as usize];
        while diagonal_sliders.0 != 0 {
            let start_idx = diagonal_sliders.pop_lsb();
            let start_sq = SQUARES[start_idx as usize];

            let attacks = get_bishop_attacks(start_sq, occ_all) & occ_them;
            self.serialize_moves(&mut moves, start_sq, attacks, occ_them);
        }

        let mut orthogonal_sliders = self.pieces[us][PieceType::Rook as usize] 
                                   | self.pieces[us][PieceType::Queen as usize];
        while orthogonal_sliders.0 != 0 {
            let start_idx = orthogonal_sliders.pop_lsb();
            let start_sq = SQUARES[start_idx as usize];

            let attacks = get_rook_attacks(start_sq, occ_all) & occ_them;
            self.serialize_moves(&mut moves, start_sq, attacks, occ_them);
        }

        let mut pawns = self.pieces[us][PieceType::Pawn as usize];

        while pawns.0 != 0 {
            let start_idx = pawns.pop_lsb();
            let start_sq = SQUARES[start_idx as usize];
            let start_bb = Bitboard(1u64 << (start_idx as u64));

            if us == Color::White as usize {
                let attacks = Bitboard(((start_bb.0 & NOT_A_FILE) << 7) | ((start_bb.0 & NOT_H_FILE) << 9));
                let mut valid_captures = attacks & occ_them;

                while valid_captures.0 != 0 {
                    let target_idx = valid_captures.pop_lsb();
                    let target_sq = SQUARES[target_idx as usize];

                    if (target_idx as usize) / 8 == 7 {
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_Q));
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_R));
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_B));
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_N));
                    } else {
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE));
                    }
                }

                if (start_idx as usize) / 8 == 6 {
                    let push = Bitboard(start_bb.0 << 8);
                    if (push.0 & occ_all.0) == 0 {
                        let target_sq = SQUARES[((start_idx as usize) + 8) as usize];
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_Q));
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_R));
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_B));
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_N));
                    }
                }

                if let Some(ep_sq) = self.en_passant {
                    let ep_bb = Bitboard(1u64 << (ep_sq as usize) as u64);
                    if (attacks & ep_bb).0 != 0 {
                        moves.push(Move::build(start_sq, ep_sq, FLAG_EN_PASSANT));
                    }
                }

            } else {
                let attacks = Bitboard(((start_bb.0 & NOT_A_FILE) >> 9) | ((start_bb.0 & NOT_H_FILE) >> 7));
                let mut valid_captures = attacks & occ_them;

                while valid_captures.0 != 0 {
                    let target_idx = valid_captures.pop_lsb();
                    let target_sq = SQUARES[target_idx as usize];

                    if (target_idx as usize) / 8 == 0 {
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_Q));
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_R));
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_B));
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE_PROMO_N));
                    } else {
                        moves.push(Move::build(start_sq, target_sq, FLAG_CAPTURE));
                    }
                }

               if (start_idx as usize) / 8 == 1 {
                    let push = Bitboard(start_bb.0 >> 8);
                    if (push.0 & occ_all.0) == 0 {
                        let target_sq = SQUARES[((start_idx as usize) - 8) as usize];
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_Q));
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_R));
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_B));
                        moves.push(Move::build(start_sq, target_sq, FLAG_PROMO_N));
                    }
                } 
                
                if let Some(ep_sq) = self.en_passant {
                    let ep_bb = Bitboard(1u64 << (ep_sq as usize) as u64);
                    if (attacks & ep_bb).0 != 0 {
                        moves.push(Move::build(start_sq, ep_sq, FLAG_EN_PASSANT));
                    }
                }
            }
        }
        
        moves
    }

    /// Computes the Zobrist hash of the current board position from scratch.
    pub fn calculate_hash(&self) -> u64 {
        let mut hash: u64 = 0;

        for i in 0..12 {
            let mut bb = self.pieces[i / 6][i % 6];

            while bb.0 != 0 {
                let sq = bb.pop_lsb();
                hash ^= ZOBRIST_PIECES.get().unwrap()[i][sq as usize];
            }
        }

        if self.side_to_move == Color::Black {
            hash ^= ZOBRIST_SIDE.get().unwrap();
        }

        hash ^= ZOBRIST_CASTLING.get().unwrap()[self.castling_rights as usize];

        if let Some(ep_sq) = self.en_passant {
            hash ^= ZOBRIST_EN_PASSANT.get().unwrap()[(ep_sq as usize) % 8];
        }

        hash
    }

    /// Checks if the current board is a repitition of previous moves
    pub fn is_repetition(&self) -> bool {
        if self.hash_history.len() < 4 {
            return false;
        }

        let current_hash = match self.hash_history.last() {
            Some(&h) => h,
            None => return false,
        };

        let mut i = self.hash_history.len() as i32 - 3;
        while i >= 0 {
            if self.hash_history[i as usize] == current_hash {
                return true;
            }
            i -= 2;
        }

        false
    }

    /// Makes a null move, changing the side to move and updating the Zorbist
    /// hash without touching the piece bitboards
    pub fn make_null_move(&mut self) {
        let history = UndoRecord::new(
            self.en_passant,
            self.castling_rights,
            self.halfmove_clock,
            None,
            self.mg_score,
            self.eg_score,
            self.phase,
        );
        self.history.push(history);

        let mut new_hash = *self.hash_history.last().unwrap();
        if let Some(ep_sq) = self.en_passant {
            new_hash ^= ZOBRIST_EN_PASSANT.get().unwrap()[(ep_sq as usize) % 8];
        }
        new_hash ^= ZOBRIST_SIDE.get().unwrap();
        self.hash_history.push(new_hash);

        self.side_to_move = self.side_to_move.flip();
        self.en_passant = None;
        self.halfmove_clock += 1;
    }

    /// Unmakes a null move, rewinding the board state to how it was before
    /// the null move was made
    pub fn unmake_null_move(&mut self) {
        self.hash_history.pop();
        let history = self.history.pop().unwrap();

        self.side_to_move = self.side_to_move.flip();
        self.en_passant = history.en_passant;
        self.castling_rights = history.castling_rights;
        self.halfmove_clock = history.halfmove_clock;
        self.mg_score = history.mg_score;
        self.eg_score = history.eg_score;
        self.phase = history.phase;
    }


    /// Finds the absolute weakest piece of a given color that is currently
    /// attacking a square
    pub fn get_smallest_attacker(&self, target: Square, us: Color,
        sim_occupancy: Bitboard) -> Option<(PieceType, Square)> {

        if us == Color::White {
            let mut attack_bb:Bitboard = unsafe {
                self.pieces[us as usize][PieceType::Pawn as usize] & BLACK_PAWN_ATTACKS[target as usize]
            };
            attack_bb &= sim_occupancy;
            if attack_bb.0 != 0 {
                return Some((PieceType::Pawn, attack_bb.get_lsb()));
            }
 
        } else {
            let mut attack_bb = unsafe {
                self.pieces[us as usize][PieceType::Pawn as usize] & WHITE_PAWN_ATTACKS[target as usize]
            };
            attack_bb &= sim_occupancy;
            if attack_bb.0 != 0 {
                return Some((PieceType::Pawn, attack_bb.get_lsb()));
            }
        }

        let mut attack_bb = unsafe {
            KNIGHT_ATTACKS[target as usize] & self.pieces[us as usize][PieceType::Knight as usize]
        };
        attack_bb &= sim_occupancy;
        if attack_bb.0 != 0 {
            return Some((PieceType::Knight, attack_bb.get_lsb()));
        }

        let bishop_attacks = get_bishop_attacks(target, sim_occupancy);
        let attack_bb = bishop_attacks & self.pieces[us as usize][PieceType::Bishop as usize] & sim_occupancy;
        if attack_bb.0 != 0 {
            return Some((PieceType::Bishop, attack_bb.get_lsb()));
        }

        let rook_attacks = get_rook_attacks(target, sim_occupancy);
        let attack_bb = rook_attacks & self.pieces[us as usize][PieceType::Rook as usize] & sim_occupancy;
        if attack_bb.0 != 0 {
            return Some((PieceType::Rook, attack_bb.get_lsb()));
        }

        let mut attack_bb = bishop_attacks | rook_attacks;
        attack_bb &= self.pieces[us as usize][PieceType::Queen as usize] & sim_occupancy;
        if attack_bb.0 != 0 {
            return Some((PieceType::Queen, attack_bb.get_lsb()));
        }

        let mut attack_bb = unsafe {
            KING_ATTACKS[target as usize]
        };
        attack_bb &= self.pieces[us as usize][PieceType::King as usize] & sim_occupancy;
        if attack_bb.0 != 0 {
            return Some((PieceType::King, attack_bb.get_lsb()));
        }

        None
    }

pub fn get_piece_moves_bb(&self, side: Color, piece: PieceType) -> Bitboard {
    let mut moves = Bitboard::empty();
    let friendly_occupancy = if side == Color::White { self.occupancies[0] } else { self.occupancies[1] };
    let total_occupancy = self.occupancies[2];
    let mut piece_bb = self.pieces[side as usize][piece as usize];

    match piece {
        PieceType::Pawn => {
            if side == Color::White {
                while piece_bb.0 != 0 {
                    let sq = piece_bb.pop_lsb();
                    let sq_bb = Bitboard::from_square(sq);
                    let attacks = unsafe {
                        WHITE_PAWN_ATTACKS[sq as usize] & !friendly_occupancy
                    };
                    let single_pushes = (sq_bb << 8) & !total_occupancy;
                    let double_pushes = ((single_pushes & Bitboard(THIRD_RANK)) << 8) & !total_occupancy;

                    moves |= attacks | single_pushes | double_pushes
                }
            } else {
                while piece_bb.0 != 0 {
                    let sq = piece_bb.pop_lsb();
                    let sq_bb = Bitboard::from_square(sq);
                    let attacks = unsafe {
                    BLACK_PAWN_ATTACKS[sq as usize] & !friendly_occupancy
                    };

                    let single_pushes = (sq_bb >> 8) & !total_occupancy;
                    let double_pushes = ((single_pushes & Bitboard(SIXTH_RANK)) >> 8) & !total_occupancy;

                    moves |= attacks | single_pushes | double_pushes
                }
            }
        },
        PieceType::Knight => {
            while piece_bb.0 != 0 {
                let sq = piece_bb.pop_lsb();
                moves |= unsafe {
                    KNIGHT_ATTACKS[sq as usize] & !friendly_occupancy
                };
            }
        },
        PieceType::Bishop => {
            while piece_bb.0 != 0 {
                let sq = piece_bb.pop_lsb();
                moves |= get_bishop_attacks(sq, total_occupancy) & !friendly_occupancy;
            }
        },
        PieceType::Rook => {
            while piece_bb.0 != 0 {
                let sq = piece_bb.pop_lsb();
                moves |= get_rook_attacks(sq, total_occupancy) & !friendly_occupancy;
            }
        },
        PieceType::Queen => {
            while piece_bb.0 != 0 {
                let sq = piece_bb.pop_lsb();
                moves |= (get_bishop_attacks(sq, total_occupancy) | get_rook_attacks(sq, total_occupancy)) & !friendly_occupancy;
            }
        },
        PieceType::King => {
            while piece_bb.0 != 0 {
                let sq = piece_bb.pop_lsb();
                moves |= unsafe {
                    KING_ATTACKS[sq as usize] & !friendly_occupancy
                }
            }
        },
    }

    moves
    }

    /// Returns true if the King has moved to a castled position
    pub fn has_castled(&self, us: Color) -> bool {
        let king_bb = self.pieces[us as usize][PieceType::King as usize];
        if us == Color::White {
            let castled_pos = (1u64 << 6) | (1u64 << 2);
            (king_bb.0 & castled_pos != 0) & (self.castling_rights & 0b0011 == 0)
        } else {
            let castled_pos = (1u64 << 62) | (1u64 << 58);
            (king_bb.0 & castled_pos != 0) & (self.castling_rights & 0b1100 == 0)
        }
    }

    /// Retruns true if the player still retains at least one castling right
    pub fn can_castle(&self, us: Color) -> bool {
        if us == Color::White {
            (self.castling_rights & 0b0011) != 0
        } else {
            (self.castling_rights & 0b1100) != 0
        }
    }
}
