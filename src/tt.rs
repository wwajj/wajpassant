use crate::moves::Move;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum TTFlag {
    Exact,
    Alpha,
    Beta,
}

#[derive(Copy, Clone)]
pub struct TTEntry {
    pub zobrist: u64,
    pub depth: u8,
    pub score: i32,
    pub best_move: Option<Move>,
    pub flag: TTFlag,
}

// A blank entry to fill the table with at startup
impl Default for TTEntry {
    fn default() -> Self {
        Self {
            zobrist: 0,
            depth: 0,
            score: 0,
            best_move: None,
            flag: TTFlag::Exact,
        }
    }
}

pub struct TranspositionTable {
    pub entries: Vec<TTEntry>,
    pub killers: [[Option<Move>; 2]; 64],
}

impl TranspositionTable {
    /// Creates a new Transposition Table with a fixed number of slots.
    pub fn new(size_in_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<TTEntry>();
        let num_entries = (size_in_mb * 1024 * 1024) / entry_size;

        Self {
            entries: vec![TTEntry::default(); num_entries],
            killers: [[None; 2]; 64],
        }
    }

    /// Saves a position to the Transposition Table
    pub fn write(&mut self, zobrist: u64, depth: u8, score: i32, best_move: Option<Move>, flag: TTFlag) {
        let index = (zobrist % self.entries.len() as u64) as usize;

        if depth >= self.entries[index].depth {
            self.entries[index] = TTEntry {
                zobrist,
                depth,
                score,
                best_move,
                flag,
            }
        }
    }

    /// Attempts to retrieve a position from the table
    /// Returns Some(TTEntry) if a valid, deep-enough match is found
    pub fn read(&self, zobrist: u64, depth: u8) -> Option<TTEntry> {
        let index = (zobrist % self.entries.len() as u64) as usize;
        let entry = self.entries[index];

        if entry.zobrist == zobrist {
            if entry.depth >= depth { return Some(entry) };
        }

        None
    }

    /// Extracts just the best move from a previous search, regardless of depth
    pub fn probe_move(&self, zobrist: u64) -> Option<Move> {
        let index = (zobrist % self.entries.len() as u64) as usize;
        let entry = self.entries[index];

        if entry.zobrist == zobrist {
            return entry.best_move;
        }

        None
    }

    /// Clears all the entries
    pub fn clear(&mut self) {
        self.entries.fill(TTEntry::default());
    }

    /// Writes a killer move
    pub fn write_killer(&mut self, mv: Move, ply: i32) {
        if mv.is_capture() || mv.is_promotion() {
            return;
        }

        let p = (ply as usize).min(63);

        if self.killers[p][0] != Some(mv) {
            self.killers[p][1] = self.killers[p][0];
            self.killers[p][0] = Some(mv);
        }
    }

    /// Reads a killer move
    pub fn read_killer(&self, mv: Move, ply: i32) -> i32 {
        let p = (ply as usize).min(63);

        if !mv.is_capture() && !mv.is_promotion() {
            if self.killers[p][0] == Some(mv) {
                return 90000;
            } else if self.killers[p][1] == Some(mv) {
                return 80000;
            }
        }

        0
    }

    /// Clears all killer moves
    pub fn clear_killers(&mut self) {
        self.killers = [[None; 2]; 64]
    }
}
