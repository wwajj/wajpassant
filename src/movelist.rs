use crate::moves::{Move};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MoveList {
    moves: [Move; 256],
    count: usize,
}

impl MoveList {
    // constructor
    pub fn empty() -> Self {
        Self {
            moves: [Move::empty(); 256],
            count: 0,
        }
    }

    // adds a Move to the moves array
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.count < 256, "MoveList overflow");
        self.moves[self.count] = mv;

        self.count += 1;
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Move> {
        self.moves[0..self.count].iter()
    }
}
