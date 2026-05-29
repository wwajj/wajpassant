use crate::attacks::{KNIGHT_ATTACKS, KING_ATTACKS, get_bishop_attacks, get_rook_attacks};
use crate::bitboard::{Bitboard, Square, SQUARES, NOT_A_FILE, NOT_H_FILE};
use crate::board::{Color, PieceType, Board};
use crate::movelist::MoveList;
use crate::moves::*;

pub const FIRST_RANK: u64 = 0x00000000000000FF;
pub const THIRD_RANK: u64 = 0x0000000000FF0000;
pub const SIXTH_RANK: u64 = 0x0000FF0000000000;
pub const EIGHTH_RANK: u64 = 0xFF00000000000000;

pub const PROMO_LIST: [u16; 4] = [FLAG_PROMO_N, FLAG_PROMO_B, FLAG_PROMO_R, FLAG_PROMO_Q];
pub const CAPTURE_PROMO_LIST: [u16; 4] = [FLAG_CAPTURE_PROMO_N, FLAG_CAPTURE_PROMO_B, FLAG_CAPTURE_PROMO_R, FLAG_CAPTURE_PROMO_Q];

pub fn generate_legal_moves(board: &Board, mvlist: &mut MoveList) {
    let us = board.side_to_move;
    let them = us.flip();
    let friendly_occupancy = board.occupancies[us as usize];
    let enemy_occupancy = board.occupancies[them as usize];
    let total_occupancy = board.occupancies[2];

    generate_pawn_moves(board, mvlist, us, enemy_occupancy, total_occupancy);
    generate_knight_moves(board, mvlist, us, friendly_occupancy, enemy_occupancy, total_occupancy);
    generate_bishop_moves(board, mvlist, us, friendly_occupancy, enemy_occupancy, total_occupancy);
    generate_rook_moves(board, mvlist, us, friendly_occupancy, enemy_occupancy, total_occupancy);
    generate_queen_moves(board, mvlist, us, friendly_occupancy, enemy_occupancy, total_occupancy);
    generate_king_moves(board, mvlist, us, friendly_occupancy, enemy_occupancy, total_occupancy);
}

pub fn generate_pawn_moves(board: &Board, mvlist: &mut MoveList, us: Color, enemy_occupancy: Bitboard, total_occupancy: Bitboard) {
    if us == Color::White {
        let pieces = board.pieces[us as usize][PieceType::Pawn as usize];
        let pushes = pieces.shift_north() & !total_occupancy;
        let promotion_pushes = pushes & Bitboard(EIGHTH_RANK);
        let normal_pushes = pushes & Bitboard(!EIGHTH_RANK);
        let double_pushes = (normal_pushes & Bitboard(THIRD_RANK)).shift_north() & !total_occupancy;

        let left_diagonal = ((pieces & Bitboard(NOT_A_FILE)) << 7) & enemy_occupancy;
        let right_diagonal = ((pieces & Bitboard(NOT_H_FILE)) << 9) & enemy_occupancy;
        let left_promotion_diagonal = left_diagonal & Bitboard(EIGHTH_RANK);
        let left_normal_diagonal = left_diagonal & Bitboard(!EIGHTH_RANK);
        let right_promotion_diagonal = right_diagonal & Bitboard(EIGHTH_RANK);
        let right_normal_diagonal = right_diagonal & Bitboard(!EIGHTH_RANK);

        if let Some (ep_sq) = board.en_passant {
            let ep_bb = Bitboard(1u64 << ep_sq as usize);
            let ep_left = ((pieces & Bitboard(NOT_A_FILE)) << 7) & ep_bb;
            let ep_right = ((pieces & Bitboard(NOT_H_FILE)) << 9) & ep_bb;

            let mut bb = ep_left;
            while bb.0 != 0 {
                let target_idx = bb.pop_lsb() as usize;
                let start_idx = target_idx - 7;

                mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], FLAG_EN_PASSANT));
            }

            let mut bb = ep_right;
            while bb.0 != 0 {
                let target_idx = bb.pop_lsb() as usize;
                let start_idx = target_idx - 9;

                mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], FLAG_EN_PASSANT));
            }
        }

        let mut bb = promotion_pushes;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx - 8;

            for flag in PROMO_LIST {
                mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], flag));
            }
        }

        let mut bb = normal_pushes;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx - 8;

            mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], FLAG_QUIET));
        }

        let mut bb = double_pushes;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx - 16;

            mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], FLAG_DOUBLE_PAWN));
        }

        let mut bb = left_promotion_diagonal;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx - 7;

            for flag in CAPTURE_PROMO_LIST {
                mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], flag));
            }
        }

        let mut bb = left_normal_diagonal;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx - 7;

            mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], FLAG_CAPTURE));
        }
 
        let mut bb = right_promotion_diagonal;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx - 9;

            for flag in CAPTURE_PROMO_LIST {
                mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], flag));
            }
        }

        let mut bb = right_normal_diagonal;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx - 9;

            mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], FLAG_CAPTURE));
        }
    } else {
        let pieces = board.pieces[1][PieceType::Pawn as usize];
        let pushes = pieces.shift_north() & !total_occupancy;
        let promotion_pushes = pushes & Bitboard(FIRST_RANK);
        let normal_pushes = pushes & Bitboard(!FIRST_RANK);
        let double_pushes = (normal_pushes & Bitboard(SIXTH_RANK)).shift_south() & !total_occupancy;

        let left_diagonal = ((pieces & Bitboard(NOT_H_FILE)) >> 7) & enemy_occupancy;
        let right_diagonal = ((pieces & Bitboard(NOT_A_FILE)) >> 9) & enemy_occupancy;
        let left_promotion_diagonal = left_diagonal & Bitboard(FIRST_RANK);
        let left_normal_diagonal = left_diagonal & Bitboard(!FIRST_RANK);
        let right_promotion_diagonal = right_diagonal & Bitboard(FIRST_RANK);
        let right_normal_diagonal = right_diagonal & Bitboard(!FIRST_RANK);

        if let Some (ep_sq) = board.en_passant {
            let ep_bb = Bitboard(1u64 << ep_sq as usize);
            let ep_left = ((pieces & Bitboard(NOT_H_FILE)) >> 7) & ep_bb;
            let ep_right = ((pieces & Bitboard(NOT_A_FILE)) >> 9) & ep_bb;

            let mut bb = ep_left;
            while bb.0 != 0 {
                let target_idx = bb.pop_lsb() as usize;
                let start_idx = target_idx + 7;

                mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], FLAG_EN_PASSANT));
            }

            let mut bb = ep_right;
            while bb.0 != 0 {
                let target_idx = bb.pop_lsb() as usize;
                let start_idx = target_idx + 9;

                mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], FLAG_EN_PASSANT));
            }
        }

        let mut bb = promotion_pushes;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx + 8;

            for flag in PROMO_LIST {
                mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], flag));
            }
        }

        let mut bb = normal_pushes;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx + 8;

            mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], FLAG_QUIET));
        }

        let mut bb = double_pushes;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx + 16;

            mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], FLAG_DOUBLE_PAWN));
        }

        let mut bb = left_promotion_diagonal;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx + 7;

            for flag in CAPTURE_PROMO_LIST {
                mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], flag));
            }
        }

        let mut bb = left_normal_diagonal;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx + 7;

            mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], FLAG_CAPTURE));
        }
 
        let mut bb = right_promotion_diagonal;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx + 9;

            for flag in CAPTURE_PROMO_LIST {
                mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], flag));
            }
        }

        let mut bb = right_normal_diagonal;
        while bb.0 != 0 {
            let target_idx = bb.pop_lsb() as usize;
            let start_idx = target_idx + 9;

            mvlist.push(Move::build(SQUARES[start_idx], SQUARES[target_idx], FLAG_CAPTURE));
        }
    }
}

pub fn generate_knight_moves(board: &Board, mvlist: &mut MoveList, us: Color, friendly_occupancy: Bitboard, enemy_occupancy: Bitboard, total_occupancy: Bitboard) {
    let mut pieces = board.pieces[us as usize][PieceType::Knight as usize];
    while pieces.0 != 0 {
        let start = pieces.pop_lsb();
        let all_moves = unsafe {
            KNIGHT_ATTACKS[start as usize] & !friendly_occupancy
        };
        let mut captures = all_moves & enemy_occupancy;
        let mut quiets = all_moves & !total_occupancy;

        while captures.0 != 0 {
            let target = captures.pop_lsb();
            mvlist.push(Move::build(start, target, FLAG_CAPTURE));
        }

        while quiets.0 != 0 {
            let target = quiets.pop_lsb();
            mvlist.push(Move::build(start, target, FLAG_QUIET));
        }
    }
}

pub fn generate_king_moves(board: &Board, mvlist: &mut MoveList, us: Color, friendly_occupancy: Bitboard, enemy_occupancy: Bitboard, total_occupancy: Bitboard) {
    let mut pieces = board.pieces[us as usize][PieceType::King as usize];
    while pieces.0 != 0 {
        let start = pieces.pop_lsb();
        let all_moves = unsafe {
            KING_ATTACKS[start as usize] & !friendly_occupancy
        };
        let mut captures = all_moves & enemy_occupancy;
        let mut quiets = all_moves & !total_occupancy;

        while captures.0 != 0 {
            let target = captures.pop_lsb();
            mvlist.push(Move::build(start, target, FLAG_CAPTURE));
        }

        while quiets.0 != 0 {
            let target = quiets.pop_lsb();
            mvlist.push(Move::build(start, target, FLAG_QUIET));
        }
    }

    if us == Color::White {
        if board.castling_rights & 1 != 0 {
            if total_occupancy.0 & 0x60 == 0 {
                mvlist.push(Move::build(Square::E1, Square::G1, FLAG_KING_CASTLE));
            }
        }

        if board.castling_rights & 2 != 0 {
            if total_occupancy.0 & 0x0E == 0 {
                mvlist.push(Move::build(Square::E1, Square::C1, FLAG_QUEEN_CASTLE));
            }
        }
    } else {
        if board.castling_rights & 4 != 0 {
            if total_occupancy.0 & 0x6000000000000000 == 0 {
                mvlist.push(Move::build(Square::E8, Square::G8, FLAG_KING_CASTLE));
            }
        }

        if board.castling_rights & 8 != 0 {
            if total_occupancy.0 & 0x0E00000000000000 == 0 {
                mvlist.push(Move::build(Square::E8, Square::C8, FLAG_QUEEN_CASTLE));
            }
        }
    }
}

pub fn generate_bishop_moves(board: &Board, mvlist: &mut MoveList, us: Color, friendly_occupancy: Bitboard, enemy_occupancy: Bitboard, total_occupancy: Bitboard) {
    let mut pieces = board.pieces[us as usize][PieceType::Bishop as usize];
    
    while pieces.0 != 0 {
        let start = pieces.pop_lsb();
        let all_moves = get_bishop_attacks(start, total_occupancy);
        let valid_moves = all_moves & !friendly_occupancy;
        let mut captures = valid_moves & enemy_occupancy;
        let mut quiets = valid_moves & !total_occupancy;
        
        while captures.0 != 0 {
            let target = captures.pop_lsb();
            mvlist.push(Move::build(start, target, FLAG_CAPTURE));
        }
        
        while quiets.0 != 0 {
            let target = quiets.pop_lsb();
            mvlist.push(Move::build(start, target, FLAG_QUIET));
        }
    }
}

pub fn generate_rook_moves(board: &Board, mvlist: &mut MoveList, us: Color, friendly_occupancy: Bitboard, enemy_occupancy: Bitboard, total_occupancy: Bitboard) {
    let mut pieces = board.pieces[us as usize][PieceType::Rook as usize];
    
    while pieces.0 != 0 {
        let start = pieces.pop_lsb();
        let all_moves = get_rook_attacks(start, total_occupancy);
        let valid_moves = all_moves & !friendly_occupancy;
        let mut captures = valid_moves & enemy_occupancy;
        let mut quiets = valid_moves & !total_occupancy;
        
        while captures.0 != 0 {
            let target = captures.pop_lsb();
            mvlist.push(Move::build(start, target, FLAG_CAPTURE));
        }
        
        while quiets.0 != 0 {
            let target = quiets.pop_lsb();
            mvlist.push(Move::build(start, target, FLAG_QUIET));
        }
    }
}

pub fn generate_queen_moves(board: &Board, mvlist: &mut MoveList, us: Color, friendly_occupancy: Bitboard, enemy_occupancy: Bitboard, total_occupancy: Bitboard) {
    let mut pieces = board.pieces[us as usize][PieceType::Queen as usize];
    
    while pieces.0 != 0 {
        let start = pieces.pop_lsb();
        
        let bishop_moves = get_bishop_attacks(start, total_occupancy);
        let rook_moves = get_rook_attacks(start, total_occupancy);
        let all_moves = bishop_moves | rook_moves;
        
        let valid_moves = all_moves & !friendly_occupancy;
        
        let mut captures = valid_moves & enemy_occupancy;
        let mut quiets = valid_moves & !total_occupancy;
        
        while captures.0 != 0 {
            let target = captures.pop_lsb();
            mvlist.push(Move::build(start, target, FLAG_CAPTURE));
        }
        
        while quiets.0 != 0 {
            let target = quiets.pop_lsb();
            mvlist.push(Move::build(start, target, FLAG_QUIET));
        }
    }
}
