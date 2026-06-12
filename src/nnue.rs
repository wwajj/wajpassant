//! The NNUE (Efficiently Updatable Neural Network) module.

use std::fs::File;
use std::io::Read;
use std::sync::OnceLock;

use crate::bitboard::{Bitboard, Square};
use crate::board::{Color, PieceType};

// The standard size of the HalfKP hidden layer
pub const HIDDEN_LAYER_SIZE: usize = 256;
// The size of the input features, padded for memory alignment
pub const INPUT_FEATURES: usize = 41024;

// The quantization divisor
pub const SCALE: i32 = 400;
pub const QA: i32 = 255;
pub const QB: i32 = 64;

/// Global storage for the quantized network parameters
pub static GLOBAL_WEIGHTS: OnceLock<Box<NNUEWeights>> = OnceLock::new();

/// Represents the first hidden layer of the neural network from a single perspective.
/// We use `i16` because the network weights are quantized integers, which allows
/// for blazing fast SIMD / AVX2 matrix arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Accumulator {
    pub values: [i16; HIDDEN_LAYER_SIZE],
}

impl Accumulator {
    /// Creates a blank accumulator initialized to zero.
    pub fn new() -> Self {
        Self {
            values: [0; HIDDEN_LAYER_SIZE],
        }
    }

    /// Adds a single feature column to this accumulator layer
    #[inline(always)]
    pub fn add_feature(&mut self, weights: &NNUEWeights, feature_idx: usize) {
        let offset = feature_idx * HIDDEN_LAYER_SIZE;
        for i in 0..HIDDEN_LAYER_SIZE {
            self.values[i] += weights.feature_weights[offset + i];
        }
    }

    /// Subtracts a single feature column from this accumulator layer
    #[inline(always)]
    pub fn remove_feature(&mut self, weights: &NNUEWeights, feature_idx: usize) {
        let offset = feature_idx * HIDDEN_LAYER_SIZE;
        for i in 0..HIDDEN_LAYER_SIZE {
            self.values[i] -= weights.feature_weights[offset + i];
        }
    }
}

/// The state of the neural network during the search.
/// Because evaluating a position requires the perspective of the side to move,
/// we track White's perspective and Black's perspective simultaneously.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NNUEState {
    pub white_acc: Accumulator,
    pub black_acc: Accumulator,
}

impl NNUEState {
    pub fn new() -> Self {
        Self {
            white_acc: Accumulator::new(),
            black_acc: Accumulator::new(),
        }
    }

    /// Complete state reconstruction from a raw board state
    /// Triggered on initialization or whenever a king moves.
    pub fn refresh(&mut self, pieces: &[[Bitboard; 6]; 2], weights: &NNUEWeights) {
        self.white_acc.values = weights.feature_biases;
        self.black_acc.values = weights.feature_biases;

        let w_king = pieces[Color::White as usize][PieceType::King as usize].get_lsb();
        let b_king = pieces[Color::Black as usize][PieceType::King as usize].get_lsb();

        for side in 0..2 {
            let piece_color = if side == 0 {
                Color::White
            } else {
                Color::Black
            };
            for pt_idx in 0..5 {
                let pt = PieceType::from_index(pt_idx).unwrap();
                let mut bb = pieces[side][pt_idx];

                while bb.0 != 0 {
                    let sq = bb.pop_lsb();
                    if let Some(w_idx) =
                        NNUEWeights::get_feature_index(w_king, Color::White, sq, piece_color, pt)
                    {
                        self.white_acc.add_feature(weights, w_idx);
                    }
                    if let Some(b_idx) =
                        NNUEWeights::get_feature_index(b_king, Color::Black, sq, piece_color, pt)
                    {
                        self.black_acc.add_feature(weights, b_idx);
                    }
                }
            }
        }
    }
}

/// Holds the pre-trained weights and biases for the Neural Network
pub struct NNUEWeights {
    // Input Layer
    pub feature_weights: [i16; INPUT_FEATURES * HIDDEN_LAYER_SIZE],
    pub feature_biases: [i16; HIDDEN_LAYER_SIZE],
    // Output Layer
    pub output_weights: [i16; HIDDEN_LAYER_SIZE * 2],
    pub output_bias: i32,
}

impl NNUEWeights {
    /// Translates a specific piece on the board into a HalfKP feature index
    /// which tells the accumulator exactly which column of weights to use
    #[inline(always)]
    pub fn get_feature_index(
        king_sq: Square,
        king_color: Color,
        piece_sq: Square,
        piece_color: Color,
        pt: PieceType,
    ) -> Option<usize> {
        if pt == PieceType::King {
            return None;
        }

        let pt_idx = pt as usize;
        let color_offset = if king_color == piece_color { 0 } else { 5 };
        let piece_feature = (color_offset + pt_idx) * 64 + (piece_sq as usize);

        Some((king_sq as usize) * 640 + piece_feature)
    }

    /// Loads a compiled NNUE binary file from the disk and populates the global network.
    /// Returns true if successful, false if the file was not found or invalid.
    pub fn load_from_file(path: &str) -> bool {
        println!("Attempting to load NNUE network from {}...", path);

        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(_) => {
                println!("Warning: NNUE file not found. Falling back to classical evaluation.");
                return false;
            }
        };

        let mut weights = unsafe {
            let layout = std::alloc::Layout::new::<NNUEWeights>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut NNUEWeights;
            Box::from_raw(ptr)
        };

        let read_i16 = |f: &mut std::fs::File| -> std::io::Result<i16> {
            let mut buf = [0u8; 2];
            f.read_exact(&mut buf)?;
            Ok(i16::from_le_bytes(buf))
        };

        let read_i32 = |f: &mut std::fs::File| -> std::io::Result<i32> {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            Ok(i32::from_le_bytes(buf))
        };

        for i in 0..(INPUT_FEATURES * HIDDEN_LAYER_SIZE) {
            match read_i16(&mut file) {
                Ok(val) => weights.feature_weights[i] = val,
                Err(_) => return false,
            }
        }

        for i in 0..HIDDEN_LAYER_SIZE {
            match read_i16(&mut file) {
                Ok(val) => weights.feature_biases[i] = val,
                Err(_) => return false,
            }
        }

        for i in 0..(HIDDEN_LAYER_SIZE * 2) {
            match read_i16(&mut file) {
                Ok(val) => weights.output_weights[i] = val,
                Err(_) => return false,
            }
        }

        match read_i32(&mut file) {
            Ok(val) => weights.output_bias = val,
            Err(_) => return false,
        }

        if GLOBAL_WEIGHTS.set(weights).is_err() {
            println!("Error: Tried to load NNUE weights more than once.");
            return false;
        }

        println!("NNUE Network successfully loaded and verified.");
        true
    }
}

/// Executes the final neural network forward pass to produce a centipawn evaluation.
pub fn evaluate_nnue(state: &NNUEState, side_to_move: Color, weights: &NNUEWeights) -> i32 {
    let (active_acc, inactive_acc) = if side_to_move == Color::White {
        (&state.white_acc, &state.black_acc)
    } else {
        (&state.black_acc, &state.white_acc)
    };

    let mut output: i32 = weights.output_bias;

    for i in 0..HIDDEN_LAYER_SIZE {
        let activated = active_acc.values[i].clamp(0, 255) as i32;
        output += activated * (weights.output_weights[i] as i32);
    }

    for i in 0..HIDDEN_LAYER_SIZE {
        let activated = inactive_acc.values[i].clamp(0, 127) as i32;
        output += activated * (weights.output_weights[HIDDEN_LAYER_SIZE + i] as i32);
    }

    (output * SCALE) / (QA * QB)
}
