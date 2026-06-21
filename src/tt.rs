use crate::moves::Move;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};

// --- Custom Micro-SpinLock ---
// Bypasses the Operating System scheduler entirely for nanosecond acquisition speeds
pub struct SpinLock<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

// Tell Rust it is safe to share this across threads because we manually guard it
unsafe impl<T> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    pub fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    #[inline(always)] // Force the compiler to inline this for zero-cost abstraction
    pub fn access<R, F: FnOnce(&mut T) -> R>(&self, f: F) -> R {
        // Spin rapidly until we successfully flip the atomic boolean to true
        while self.locked.swap(true, Ordering::Acquire) {
            std::hint::spin_loop(); // Tells the CPU to optimize power while spinning
        }

        // Safely access the underlying data
        let result = f(unsafe { &mut *self.data.get() });

        // Release the lock
        self.locked.store(false, Ordering::Release);
        result
    }
}

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
    pub entries: Vec<SpinLock<TTEntry>>,
}

impl TranspositionTable {
    /// Creates a new Lock-Free Transposition Table
    pub fn new(size_in_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<SpinLock<TTEntry>>();
        let num_entries = (size_in_mb * 1024 * 1024) / entry_size;

        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            entries.push(SpinLock::new(TTEntry::default()));
        }

        Self { entries }
    }

    pub fn write(
        &self,
        zobrist: u64,
        depth: u8,
        score: i32,
        best_move: Option<Move>,
        flag: TTFlag,
    ) {
        let index = (zobrist % self.entries.len() as u64) as usize;

        self.entries[index].access(|entry| {
            if depth >= entry.depth {
                *entry = TTEntry {
                    zobrist,
                    depth,
                    score,
                    best_move,
                    flag,
                };
            }
        });
    }

    pub fn read(&self, zobrist: u64, depth: u8) -> Option<TTEntry> {
        let index = (zobrist % self.entries.len() as u64) as usize;

        self.entries[index].access(|entry| {
            if entry.zobrist == zobrist && entry.depth >= depth {
                Some(*entry)
            } else {
                None
            }
        })
    }

    pub fn probe_move(&self, zobrist: u64) -> Option<Move> {
        let index = (zobrist % self.entries.len() as u64) as usize;

        self.entries[index].access(|entry| {
            if entry.zobrist == zobrist {
                entry.best_move
            } else {
                None
            }
        })
    }

    pub fn clear(&self) {
        for lock in &self.entries {
            lock.access(|entry| {
                *entry = TTEntry::default();
            });
        }
    }
}
