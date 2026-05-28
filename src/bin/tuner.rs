use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;
use rayon::prelude::*;

use wajpassant::attacks::init_attacks;
use wajpassant::board::Board;
use wajpassant::eval::EvalParams;
use wajpassant::search::init_lmr_table;
use wajpassant::zobrist::init_zobrist;

const K: f64 = 1.13;

struct DataPoint {
    board: Board,
    result: f64,
}

fn sigmoid(score: i32) -> f64 {
    1.0 / (1.0 + 10.0f64.powf(-K * (score as f64) / 400.0))
}

/// Calculates the average Mean Squared Error across all loaded positions
fn calculate_mse(dataset: &[DataPoint], params: &EvalParams) -> f64 {
    let total_error: f64 = dataset.par_iter().map(|data| {
        // Evaluate the position using the *current* weights being tested
        let score = data.board.evaluate_from_scratch(params);
        let prediction = sigmoid(score);
        let mut target = data.result;
        if data.board.side_to_move == wajpassant::board::Color::Black {
            target = 1.0 - target;
        }

        (prediction - target).powi(2)
    }).sum();

    total_error / dataset.len() as f64
}

/// Returns the horizontally mirrored square index
pub fn mirror_sq(sq: usize) -> usize {
    let rank = sq / 8;
    let file = sq % 8;
    let mirrored_file = 7 - file;
    rank * 8 + mirrored_file
}

fn main() {
    // Initialize Engine Tables
    init_attacks();
    init_lmr_table();
    init_zobrist();

    // Load the Dataset
    let dataset_path = format!("{}/src/bin/dataset.txt", env!("CARGO_MANIFEST_DIR"));
    println!("Loading dataset from {}...", dataset_path);
    
    let file = File::open(&dataset_path).expect("Could not find dataset.txt");
    let reader = BufReader::new(file);
    let mut dataset: Vec<DataPoint> = Vec::new();
    let mut params = EvalParams::default();

    for line in reader.lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split(" | ").collect();
        if parts.len() == 2 {
            let fen = parts[0];
            let result: f64 = parts[1].parse().unwrap();
            dataset.push(DataPoint {
                board: Board::from_fen(fen, &params),
                result,
            });
        }
    }
    println!("Loaded {} positions.", dataset.len());

    // Establish the Baseline
    let mut best_mse = calculate_mse(&dataset, &params);
    println!("Baseline MSE: {:.6}\n", best_mse);

    let piece_names = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];
    let step = 1; 

    // The Coordinate Descent Optimization Loop
    for iteration in 1..=10 { 
        println!("--- Epoch {} ---", iteration);
        let mut improved_this_epoch = false;
        let start_time = Instant::now();

        // --- Tune Midgame Material ---
        for pt in 0..6 {
            // Try nudging UP
            params.material_mg[pt] += step;
            let mut mse = calculate_mse(&dataset, &params);
            
            if mse < best_mse {
                best_mse = mse;
                improved_this_epoch = true;
                // Keep stepping UP until it stops improving
                loop {
                    params.material_mg[pt] += step;
                    let next_mse = calculate_mse(&dataset, &params);
                    if next_mse < best_mse {
                        best_mse = next_mse;
                    } else {
                        params.material_mg[pt] -= step; // Revert the bad step
                        break;
                    }
                }
            } else {
                // Going UP was bad. Revert and try going DOWN
                params.material_mg[pt] -= 2 * step; 
                mse = calculate_mse(&dataset, &params);
                
                if mse < best_mse {
                    best_mse = mse;
                    improved_this_epoch = true;
                    // Keep stepping DOWN until it stops improving
                    loop {
                        params.material_mg[pt] -= step;
                        let next_mse = calculate_mse(&dataset, &params);
                        if next_mse < best_mse {
                            best_mse = next_mse;
                        } else {
                            params.material_mg[pt] += step; // Revert the bad step
                            break;
                        }
                    }
                } else {
                    // Both UP and DOWN were worse. Revert to original.
                    params.material_mg[pt] += step;
                }
            }
            println!("MG {}: {}", piece_names[pt], params.material_mg[pt]);
        }

        // --- Tune Endgame Material ---
        for pt in 0..6 {
            params.material_eg[pt] += step;
            let mut mse = calculate_mse(&dataset, &params);
            
            if mse < best_mse {
                best_mse = mse;
                improved_this_epoch = true;
                loop {
                    params.material_eg[pt] += step;
                    let next_mse = calculate_mse(&dataset, &params);
                    if next_mse < best_mse { best_mse = next_mse; } else { params.material_eg[pt] -= step; break; }
                }
            } else {
                params.material_eg[pt] -= 2 * step; 
                mse = calculate_mse(&dataset, &params);
                if mse < best_mse {
                    best_mse = mse;
                    improved_this_epoch = true;
                    loop {
                        params.material_eg[pt] -= step;
                        let next_mse = calculate_mse(&dataset, &params);
                        if next_mse < best_mse { best_mse = next_mse; } else { params.material_eg[pt] += step; break; }
                    }
                } else {
                    params.material_eg[pt] += step;
                }
            }
            println!("EG {}: {}", piece_names[pt], params.material_eg[pt]);
        }

        // --- Tune Midgame PSTs (Horizontal Symmetry) ---
        for pt in 0..6 {
            for sq in 0..64 {
                let file = sq % 8;
                if file > 3 { continue; } // Only tune files A, B, C, D

                let mirror = mirror_sq(sq);

                // Try nudging UP
                params.pst_mg[pt][sq] += step;
                if sq != mirror { params.pst_mg[pt][mirror] += step; }
                
                let mut mse = calculate_mse(&dataset, &params);
                
                if mse < best_mse {
                    best_mse = mse;
                    improved_this_epoch = true;
                    loop {
                        params.pst_mg[pt][sq] += step;
                        if sq != mirror { params.pst_mg[pt][mirror] += step; }
                        
                        let next_mse = calculate_mse(&dataset, &params);
                        if next_mse < best_mse { best_mse = next_mse; } else { 
                            params.pst_mg[pt][sq] -= step; 
                            if sq != mirror { params.pst_mg[pt][mirror] -= step; }
                            break; 
                        }
                    }
                } else {
                    // Going UP was bad. Revert and try DOWN
                    params.pst_mg[pt][sq] -= 2 * step;
                    if sq != mirror { params.pst_mg[pt][mirror] -= 2 * step; }
                    
                    mse = calculate_mse(&dataset, &params);
                    if mse < best_mse {
                        best_mse = mse;
                        improved_this_epoch = true;
                        loop {
                            params.pst_mg[pt][sq] -= step;
                            if sq != mirror { params.pst_mg[pt][mirror] -= step; }
                            
                            let next_mse = calculate_mse(&dataset, &params);
                            if next_mse < best_mse { best_mse = next_mse; } else { 
                                params.pst_mg[pt][sq] += step; 
                                if sq != mirror { params.pst_mg[pt][mirror] += step; }
                                break; 
                            }
                        }
                    } else {
                        // Both were worse. Revert to original.
                        params.pst_mg[pt][sq] += step;
                        if sq != mirror { params.pst_mg[pt][mirror] += step; }
                    }
                }
            }
            println!("Finished MG PST for {}", piece_names[pt]);
        }

        // --- Tune Endgame PSTs (Horizontal Symmetry) ---
        for pt in 0..6 {
            for sq in 0..64 {
                let file = sq % 8;
                if file > 3 { continue; } 

                let mirror = mirror_sq(sq);

                params.pst_eg[pt][sq] += step;
                if sq != mirror { params.pst_eg[pt][mirror] += step; }
                
                let mut mse = calculate_mse(&dataset, &params);
                
                if mse < best_mse {
                    best_mse = mse;
                    improved_this_epoch = true;
                    loop {
                        params.pst_eg[pt][sq] += step;
                        if sq != mirror { params.pst_eg[pt][mirror] += step; }
                        let next_mse = calculate_mse(&dataset, &params);
                        if next_mse < best_mse { best_mse = next_mse; } else { 
                            params.pst_eg[pt][sq] -= step; 
                            if sq != mirror { params.pst_eg[pt][mirror] -= step; }
                            break; 
                        }
                    }
                } else {
                    params.pst_eg[pt][sq] -= 2 * step;
                    if sq != mirror { params.pst_eg[pt][mirror] -= 2 * step; }
                    mse = calculate_mse(&dataset, &params);
                    if mse < best_mse {
                        best_mse = mse;
                        improved_this_epoch = true;
                        loop {
                            params.pst_eg[pt][sq] -= step;
                            if sq != mirror { params.pst_eg[pt][mirror] -= step; }
                            let next_mse = calculate_mse(&dataset, &params);
                            if next_mse < best_mse { best_mse = next_mse; } else { 
                                params.pst_eg[pt][sq] += step; 
                                if sq != mirror { params.pst_eg[pt][mirror] += step; }
                                break; 
                            }
                        }
                    } else {
                        params.pst_eg[pt][sq] += step;
                        if sq != mirror { params.pst_eg[pt][mirror] += step; }
                    }
                }
            }
            println!("Finished EG PST for {}", piece_names[pt]);
        }

        println!("Epoch {} finished in {:.2?}. Best MSE: {:.6}\n", iteration, start_time.elapsed(), best_mse);

        if !improved_this_epoch {
            println!("Convergence reached! Weights are fully optimized for this dataset.");
            break;
        }
    }
    // ==========================================
    // --- OUTPUT FINAL PARAMETERS TO FILE ---
    // ==========================================
    let out_path = format!("{}/src/bin/pst.txt", env!("CARGO_MANIFEST_DIR"));
    let mut out_file = File::create(&out_path).expect("Could not create pst.txt");
    let const_names = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"];

    println!("Writing optimized parameters to {}...", out_path);

    // Write Material Weights
    writeln!(out_file, "// === OPTIMIZED MATERIAL WEIGHTS ===").unwrap();
    writeln!(out_file, "pub const MATERIAL_MG: [i32; 6] = {:?};", params.material_mg).unwrap();
    writeln!(out_file, "pub const MATERIAL_EG: [i32; 6] = {:?};\n", params.material_eg).unwrap();

    // Write Midgame PSTs
    writeln!(out_file, "// === OPTIMIZED MIDGAME PSTS ===").unwrap();
    for pt in 0..6 {
        writeln!(out_file, "pub const {}_MG_PST: [i32; 64] = [", const_names[pt]).unwrap();
        for chunk in params.pst_mg[pt].chunks(8) {
            write!(out_file, "    ").unwrap();
            for &val in chunk {
                write!(out_file, "{:>4}, ", val).unwrap(); // {:>4} right-aligns for clean columns
            }
            writeln!(out_file).unwrap();
        }
        writeln!(out_file, "];\n").unwrap();
    }

    // Write Endgame PSTs
    writeln!(out_file, "// === OPTIMIZED ENDGAME PSTS ===").unwrap();
    for pt in 0..6 {
        writeln!(out_file, "pub const {}_EG_PST: [i32; 64] = [", const_names[pt]).unwrap();
        for chunk in params.pst_eg[pt].chunks(8) {
            write!(out_file, "    ").unwrap();
            for &val in chunk {
                write!(out_file, "{:>4}, ", val).unwrap();
            }
            writeln!(out_file).unwrap();
        }
        writeln!(out_file, "];\n").unwrap();
    }

    println!("Tuning complete!");
}
