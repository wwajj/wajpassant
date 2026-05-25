use crate::bitboard::{Bitboard, Square, SQUARES};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Color {
    White = 0,
    Black = 1,
}
impl Color {
    // switches turn
    pub fn flip(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

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
    // converts a raw number back into a PieceType.
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

pub const WHITE_START: u64 = 0x000000000000FFFF;
pub const WHITE_PAWNS: u64 = 0x000000000000FF00;
pub const WHITE_KNIGHTS: u64 = 0x0000000000000042;
pub const WHITE_BISHOPS: u64 = 0x0000000000000024;
pub const WHITE_ROOKS: u64 = 0x0000000000000081;
pub const WHITE_QUEENS: u64 = 0x0000000000000008;
pub const WHITE_KINGS: u64 = 0x0000000000000010;
pub const BLACK_START: u64 = 0xFFFF000000000000;
pub const BLACK_PAWNS: u64 = 0x00FF000000000000;
pub const BLACK_KNIGHTS: u64 = 0x4200000000000000;
pub const BLACK_BISHOPS: u64 = 0x2400000000000000;
pub const BLACK_ROOKS: u64 = 0x8100000000000000;
pub const BLACK_QUEENS: u64 = 0x0800000000000000;
pub const BLACK_KINGS: u64 = 0x1000000000000000; 

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    pieces: [[Bitboard; 6]; 2],
    occupancies: [Bitboard; 3],
    side_to_move: Color,
    en_passant: Option<Square>,
    castling_rights: u8,
    halfmove_clock: u16,
    fullmove_number: u16,
}

impl Board {
    // constructs empty Board
    pub fn empty() -> Self {
        Self {
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
        }
    }

    // constructs Board at starting position
    pub fn default() -> Self {
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

        Self {
            pieces,
            occupancies,
            side_to_move: Color::White,
            en_passant: None,
            castling_rights: 0b1111,
            halfmove_clock: 0,
            fullmove_number: 1,
        }
    }

    // constructs Board from FEN string
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
                    _ => panic!("Invalid FEN piece character"),
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
        board
    }

    // constructs FEN String from Board
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

    // prints the Board in a formatted grid alongside necessary information
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

    // takes a Square and checks Bitboards and returns what piece is sitting there
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

    // helper method to recalculate the 3 master occupancy bitboards
    pub fn update_occupancies(&mut self) {
        self.occupancies[0] = Bitboard::empty();
        self.occupancies[1] = Bitboard::empty();
        
        for pt in 0..6 {
            self.occupancies[0] |= self.pieces[Color::White as usize][pt];
            self.occupancies[1] |= self.pieces[Color::Black as usize][pt];
        }
        
        self.occupancies[2] = self.occupancies[0] | self.occupancies[1];
    }

    // removes a piece from the board and updates master masks
    pub fn remove_piece(&mut self, sq: Square, side: Color, pt: PieceType) {
        self.pieces[side as usize][pt as usize] = self.pieces[side as usize][pt as usize].clear_bit(sq);

        self.occupancies[side as usize] = self.occupancies[side as usize].clear_bit(sq);
        self.occupancies[2] = self.occupancies[2].clear_bit(sq);
    }

    // adds a piece to the board and updates the master masks
    pub fn add_piece(&mut self, sq: Square, side: Color, pt: PieceType) {
        self.pieces[side as usize][pt as usize] = self.pieces[side as usize][pt as usize].set_bit(sq);

        self.occupancies[side as usize] = self.occupancies[side as usize].set_bit(sq);
        self.occupancies[2] = self.occupancies[2].set_bit(sq);
    }
}

impl Default for Board {
    fn default() -> Self {
        Board::default()
    }
}
