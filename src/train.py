import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ==========================================
# 1. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class WajPassantNNUE(nn.Module):
    def __init__(self):
        super(WajPassantNNUE, self).__init__()

        # Hidden Layer: 41024 inputs -> 256 neurons
        self.feature_layer = nn.Linear(41024, 256)

        # Output Layer: 512 inputs (Active + Inactive) -> 1 Evaluation
        self.output_layer = nn.Linear(512, 1)

    def forward(self, active_features, inactive_features):
        # Pass inputs through the first layer and apply Clipped ReLU (clamp 0 to 1)
        acc_active = torch.clamp(self.feature_layer(active_features), 0.0, 1.0)
        acc_inactive = torch.clamp(self.feature_layer(inactive_features), 0.0, 1.0)

        # Concatenate the active and inactive accumulators
        combined = torch.cat([acc_active, acc_inactive], dim=1)

        # Calculate final centipawn score
        return self.output_layer(combined)


# ==========================================
# 2. THE DATA LOADER
# ==========================================
class ChessDataset(Dataset):
    def __init__(self, filepath):
        print("Loading dataset into memory... this might take a minute.")
        self.data = []
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split(" | ")
                if len(parts) == 3:
                    eval_score = max(-3000, min(3000, int(parts[1])))
                    result = float(parts[2])
                    self.data.append((parts[0], eval_score, result))
        print(f"Loaded {len(self.data)} positions!")

    def __len__(self):
        return len(self.data)

    def get_feature_index(self, king_sq, king_color, piece_sq, piece_color, pt_char):
        """
        A 1-to-1 Python translation of the Rust get_feature_index function.
        """
        # Assuming standard Rust Enum order: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4
        pt_map = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4}
        pt_idx = pt_map[pt_char]

        color_offset = 0 if king_color == piece_color else 5
        piece_feature = (color_offset + pt_idx) * 64 + piece_sq

        return (king_sq * 640) + piece_feature

    def fen_to_features(self, fen):
        """
        Translates a FEN string into the White and Black HalfKP accumulators.
        """
        white_acc = torch.zeros(41024)
        black_acc = torch.zeros(41024)

        board_state = fen.split(" ")[0]
        side_to_move = fen.split(" ")[1]

        # 1. Map FEN to standard 0-63 squares (A1 = 0, H8 = 63)
        pieces = []
        rank = 7  # FEN starts at Rank 8 (index 7)
        file = 0  # FEN starts at File A (index 0)

        for char in board_state:
            if char == "/":
                rank -= 1
                file = 0
            elif char.isdigit():
                file += int(char)
            else:
                sq = rank * 8 + file
                pieces.append((char, sq))
                file += 1

        # 2. Locate the Kings
        wk_sq = next((sq for p, sq in pieces if p == "K"), None)
        bk_sq = next((sq for p, sq in pieces if p == "k"), None)

        # 3. Populate Accumulators
        for p, sq in pieces:
            pt_char = p.lower()
            if pt_char == "k":  # Ignore kings as per Rust logic
                continue

            piece_color = "w" if p.isupper() else "b"

            # White's perspective
            if wk_sq is not None:
                w_idx = self.get_feature_index(wk_sq, "w", sq, piece_color, pt_char)
                white_acc[w_idx] = 1.0

            # Black's perspective
            if bk_sq is not None:
                b_idx = self.get_feature_index(bk_sq, "b", sq, piece_color, pt_char)
                black_acc[b_idx] = 1.0

        # 4. Assign Active/Inactive based on whose turn it is
        if side_to_move == "w":
            return white_acc, black_acc
        else:
            return black_acc, white_acc

    def __getitem__(self, idx):
        fen, eval_score, result = self.data[idx]
        active, inactive = self.fen_to_features(fen)

        side_to_move = fen.split(" ")[1]

        # Convert absolute result (1.0 = White wins) to relative result
        if side_to_move == "b":
            relative_result = 1.0 - result
        else:
            relative_result = result

        # Map relative result (0 to 1) into the same scale as the eval (-1 to 1)
        wdl_score = (relative_result - 0.5) * 2.0

        # Target blend (Eval + WDL)
        target_val = (eval_score / 400.0) * 0.5 + (wdl_score) * 0.5

        target = torch.tensor([target_val], dtype=torch.float32)
        return active, inactive, target


# ==========================================
# 3. THE RUST BINARY EXPORTER (VECTORIZED)
# ==========================================
def export_to_bin(model, filename="wajpassant.bin"):
    print(f"\nExporting quantized weights to {filename}...")

    QA = 255.0
    QB = 64.0

    with open(filename, "wb") as f:
        # 1. Feature Weights [41024 * 256] -> i16
        fw = model.feature_layer.weight.detach().transpose(0, 1)
        fw_quantized = torch.clamp(torch.round(fw * QA), -32768, 32767).to(torch.int16)
        f.write(fw_quantized.cpu().numpy().tobytes())

        # 2. Feature Biases [256] -> i16
        fb = model.feature_layer.bias.detach()
        fb_quantized = torch.clamp(torch.round(fb * QA), -32768, 32767).to(torch.int16)
        f.write(fb_quantized.cpu().numpy().tobytes())

        # 3. Output Weights [512] -> i16
        ow = model.output_layer.weight.detach().transpose(0, 1)
        ow_quantized = torch.clamp(torch.round(ow * QB), -32768, 32767).to(torch.int16)
        f.write(ow_quantized.cpu().numpy().tobytes())

        # 4. Output Bias [1] -> i32
        ob = model.output_layer.bias.detach()
        # Ensure we only grab the first scalar value if bias is shaped as [1]
        if ob.numel() == 1:
            ob = ob[0]
        ob_quantized = torch.clamp(
            torch.round(ob * (QA * QB)), -2147483648, 2147483647
        ).to(torch.int32)
        f.write(ob_quantized.cpu().numpy().tobytes())

    print("Export complete! Ready for Rust.")


# ==========================================
# 4. THE TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Igniting training sequence on: {device}")

    model = WajPassantNNUE().to(device)
    dataset = ChessDataset("bin/wajpassant_training_data.txt")

    # Massive batches for Apple Silicon
    dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        # --- TQDM PROGRESS BAR WRAPPER ---
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for active, inactive, target in progress_bar:
            active, inactive, target = (
                active.to(device),
                inactive.to(device),
                target.to(device),
            )

            optimizer.zero_grad()
            output = model(active, inactive)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Dynamically update the suffix of the progress bar with the live loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        print(
            f"--- Epoch {epoch+1} Complete. Average Loss: {total_loss/len(dataloader):.4f} ---\n"
        )

    export_to_bin(model, "wajpassant.bin")
