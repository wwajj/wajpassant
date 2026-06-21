import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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
        QA = 255.0

        scaled_active = self.feature_layer(active_features) * QA
        scaled_inactive = self.feature_layer(inactive_features) * QA

        acc_active = torch.clamp(scaled_active, 0.0, QA)
        acc_inactive = torch.clamp(scaled_inactive, 0.0, QA)

        combined = torch.cat([acc_active, acc_inactive], dim=1)

        return self.output_layer(combined) / QA


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
        pt_map = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4}
        pt_idx = pt_map[pt_char]

        color_offset = 0 if king_color == piece_color else 5
        piece_feature = (color_offset + pt_idx) * 64 + piece_sq

        return (king_sq * 640) + piece_feature

    def fen_to_features(self, fen):
        white_acc = torch.zeros(41024)
        black_acc = torch.zeros(41024)

        board_state = fen.split(" ")[0]
        side_to_move = fen.split(" ")[1]

        pieces = []
        rank = 7
        file = 0

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

        wk_sq = next((sq for p, sq in pieces if p == "K"), None)
        bk_sq = next((sq for p, sq in pieces if p == "k"), None)

        for p, sq in pieces:
            pt_char = p.lower()
            if pt_char == "k":
                continue

            piece_color = "w" if p.isupper() else "b"

            if wk_sq is not None:
                w_idx = self.get_feature_index(wk_sq, "w", sq, piece_color, pt_char)
                white_acc[w_idx] = 1.0

            if bk_sq is not None:
                b_king_flipped = bk_sq ^ 56
                sq_flipped = sq ^ 56

                b_idx = self.get_feature_index(
                    b_king_flipped, "b", sq_flipped, piece_color, pt_char
                )
                black_acc[b_idx] = 1.0

        if side_to_move == "w":
            return white_acc, black_acc
        else:
            return black_acc, white_acc

    def __getitem__(self, idx):
        fen, eval_score, result = self.data[idx]
        active, inactive = self.fen_to_features(fen)

        side_to_move = fen.split(" ")[1]

        if side_to_move == "b":
            relative_result = 1.0 - result
        else:
            relative_result = result

        wdl_score = (relative_result - 0.5) * 2.0
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
        fw = model.feature_layer.weight.detach().transpose(0, 1)
        fw_quantized = torch.clamp(torch.round(fw * QA), -32768, 32767).to(torch.int16)
        f.write(fw_quantized.cpu().numpy().tobytes())

        fb = model.feature_layer.bias.detach()
        fb_quantized = torch.clamp(torch.round(fb * QA), -32768, 32767).to(torch.int16)
        f.write(fb_quantized.cpu().numpy().tobytes())

        ow = model.output_layer.weight.detach().transpose(0, 1)
        ow_quantized = torch.clamp(torch.round(ow * QB), -32768, 32767).to(torch.int16)
        f.write(ow_quantized.cpu().numpy().tobytes())

        ob = model.output_layer.bias.detach()
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
    dataset = ChessDataset("wajpassant_training_data.txt")

    # HIGH-PERFORMANCE UPDATES: Multi-threaded staging & Unified Pinned Memory
    dataloader = DataLoader(
        dataset,
        batch_size=8192,
        shuffle=True,
        num_workers=8,  # Parallelizes string parsing across 4 CPU cores
        pin_memory=False,
        persistent_workers=True,  # Prevents recreating CPU threads at every epoch step
    )

    criterion = nn.MSELoss()

    # Regularized AdamW to smoothly control hidden layer parameter spaces
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    EPOCHS = 50
    # Cosine Annealing Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for active, inactive, target in progress_bar:
            # non_blocking=True allows streaming transfers to overlap with GPU compute steps
            active, inactive, target = (
                active.to(device, non_blocking=True),
                inactive.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )

            optimizer.zero_grad()
            output = model(active, inactive)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Move the learning rate scheduler down one step per epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"--- Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(dataloader):.4f} | LR: {current_lr:.2e} ---\n"
        )

    export_to_bin(model, "wajpassant.bin")
