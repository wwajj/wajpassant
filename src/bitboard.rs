use std::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr, BitAndAssign, BitOrAssign, BitXorAssign};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Square {
    A1 = 0,  B1 = 1,  C1 = 2,  D1 = 3,  E1 = 4,  F1 = 5,  G1 = 6,  H1 = 7,
    A2 = 8,  B2 = 9,  C2 = 10, D2 = 11, E2 = 12, F2 = 13, G2 = 14, H2 = 15,
    A3 = 16, B3 = 17, C3 = 18, D3 = 19, E3 = 20, F3 = 21, G3 = 22, H3 = 23,
    A4 = 24, B4 = 25, C4 = 26, D4 = 27, E4 = 28, F4 = 29, G4 = 30, H4 = 31,
    A5 = 32, B5 = 33, C5 = 34, D5 = 35, E5 = 36, F5 = 37, G5 = 38, H5 = 39,
    A6 = 40, B6 = 41, C6 = 42, D6 = 43, E6 = 44, F6 = 45, G6 = 46, H6 = 47,
    A7 = 48, B7 = 49, C7 = 50, D7 = 51, E7 = 52, F7 = 53, G7 = 54, H7 = 55,
    A8 = 56, B8 = 57, C8 = 58, D8 = 59, E8 = 60, F8 = 61, G8 = 62, H8 = 63,
}
pub const SQUARES: [Square; 64] = [
    Square::A1, Square::B1, Square::C1, Square::D1, Square::E1, Square::F1, Square::G1, Square::H1,
    Square::A2, Square::B2, Square::C2, Square::D2, Square::E2, Square::F2, Square::G2, Square::H2,
    Square::A3, Square::B3, Square::C3, Square::D3, Square::E3, Square::F3, Square::G3, Square::H3,
    Square::A4, Square::B4, Square::C4, Square::D4, Square::E4, Square::F4, Square::G4, Square::H4,
    Square::A5, Square::B5, Square::C5, Square::D5, Square::E5, Square::F5, Square::G5, Square::H5,
    Square::A6, Square::B6, Square::C6, Square::D6, Square::E6, Square::F6, Square::G6, Square::H6,
    Square::A7, Square::B7, Square::C7, Square::D7, Square::E7, Square::F7, Square::G7, Square::H7,
    Square::A8, Square::B8, Square::C8, Square::D8, Square::E8, Square::F8, Square::G8, Square::H8,
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Bitboard(pub u64);
pub const NOT_H_FILE: u64 = 0x7F7F7F7F7F7F7F7F;
pub const NOT_A_FILE: u64 = 0xFEFEFEFEFEFEFEFE;

impl Bitboard {
    // constructor
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    // constructs Bitboard of 0s
    pub fn empty() -> Self {
        Self(0)
    }

    // constructs Bitboard of 1s
    pub fn full() -> Self {
        Self(u64::MAX)
    }

    // sets Square bit of empty Bitboard to 1 
    pub fn from_square(sq: Square) -> Self {
        let value = 1u64 << (sq as u64);
        Self(value)
    }

    // changs the bit at given Square to 1
    pub fn set_bit(&self, sq: Square) -> Self {
        let value = self.0 | (1u64 << (sq as u64));
        Self(value)
    }

    // changes the bit at given Square to 0
    pub fn clear_bit(&self, sq: Square) -> Self {
        let value = self.0 & !(1u64 <<  (sq as u64));
        Self(value)
    }

    // flips bit at given Square
    pub fn toggle_bit(&self, sq: Square) -> Self {
        let value = self.0 ^ (1u64 << (sq as u64));
        Self(value)
    }

    // returns True if the bit at the Square is 1, false if 0
    pub fn is_occupied(&self, sq: Square) -> bool {
        ((self.0 >> (sq as u64)) & 1) == 1
    }

    // returns the total number of bits set to 1 on the Bitboard
    pub fn count(&self) -> u32 {
        self.0.count_ones()
    }

    // returns the Square of the LSB
    // throws an error for an empty board
    pub fn get_lsb(&self) -> Square {
        let index = self.0.trailing_zeros() as usize; 
        SQUARES[index]
    }

    // finds the LSB, returns its Square, and clears that bit to 0
    pub fn pop_lsb(&mut self) -> Square {
        let sq = self.get_lsb();
        self.0 &= self.0 - 1;
        sq
    }

    // shifts all bits up one rank
    pub fn shift_north(&self) -> Self {
        let value = self.0 << 8;
        Self(value)
    }

    // shifts all bits down one rank
    pub fn shift_south(&self) -> Self {
        let value = self.0 >> 8;
        Self(value)
    }

    // shifts all bits right one file
    pub fn shift_east(&self) -> Self {
        let value: u64 = (self.0 & NOT_H_FILE) << 1;
        Self(value)
    }

    // shifts all bits left one file
    pub fn shift_west(&self) -> Self {
        let value: u64 = (self.0 & NOT_A_FILE) >> 1;
        Self(value)
    }

    // prints the Bitboard as a formatted 8x8 grid
    pub fn print(&self) {
        println!();

        for rank in (0..8).rev() {
            print!("{}  ", rank + 1);

            for file in 0..8 {
                let sq = rank * 8 + file;
                if self.0 & (1u64 << sq) != 0 {
                    print!("X ");
                } else {
                    print!(". ");
                }
            }
            println!()
        }

        println!("\n    A B C D E F G H\n")
    }
}

impl BitAnd for Bitboard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Bitboard(self.0 & rhs.0)
    }
}


impl BitOr for Bitboard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Bitboard(self.0 | rhs.0)
    }
}

impl BitXor for Bitboard {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl Not for Bitboard {
    type Output = Self;

    fn not(self) -> Self::Output {
        Bitboard(!self.0)
    }
}

impl Shl<usize> for Bitboard {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        Bitboard(self.0 << rhs)
    }
}

impl Shr<usize> for Bitboard {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output {
        Bitboard(self.0 >> rhs)
    }
}

impl BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}
