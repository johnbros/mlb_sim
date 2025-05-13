import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from pitch_type_model.cold_start.dataset import ColdStartDataset
from pitch_type_model.models import PitchTypeLSTM
import matplotlib.pyplot as plt
import numpy as np
from constants.id_map import pitch_id_to_name
import os
import json


# Welcome to the training script for the Pitch Type LSTM model.
# This file as of now is called train.py but may be renamed to trainlstm.py in the future if I try different models for this problem.

def make_prediction_log(batch, output, num_topk=2):
    # Record predictions for any potential future analysis
    true_pitch = batch["pitch_type_id"].item()
    probs = output.exp()[0].tolist()  
    topk = torch.topk(output, k=num_topk, dim=1).indices[0].tolist()

    return {
        "game_id": batch["game_id"].item(),
        "pitch_num": batch["pitch_num"].item(),
        "pitcher_id": batch["pitcher_id"].item(),
        "true_pitch_id": true_pitch,
        "top1": topk[0],
        "top2": topk[:2], 
        "top1_correct": topk[0] == true_pitch,
        "top2_correct": true_pitch in topk[:2],
        "probs": probs,
        "valid_pitch_types": batch["valid_pitch_types"].tolist(),
        "prior_pitch_type_id": batch["prior_pitch_type_id"].item(),
        "prior_outcome_id": batch["prior_pitch_outcome_id"].item(),
        "prior_zone": batch["prior_zone"].item(),
        "zone": batch["zone"].item() if "zone" in batch else None,
        "score": batch["score"].item()
    }

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PitchTypeLSTM(config).to(device)

    criterion = torch.nn.NLLLoss()  # For log_softmax output
    optimizer = Adam(model.parameters(), lr=config["lr"])

    # Plots for analysis of the model
    model_dir = os.path.abspath(config.get("model_dir", "unknown_model"))
    model_name = config.get("model_name", "unknown_model")
    val_scores = []
    train_losses = []
    loss_steps = []
    total_loss_steps = []
    num_pitch_types = 21 # Bin for each pitch type we filter out which pitches never get predicted later
    num_bins = 20
    step_counter = 0

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "plots", "loss"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "plots", "validation"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "future_analysis"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "checkpoints"), exist_ok=True)

    total_bins = np.zeros((num_pitch_types, num_bins, 2), dtype=np.int32)

    for i, train_year in enumerate(config["train_years"]):
        print(f"\n=== Training on {train_year} ===")
        train_dataset = ColdStartDataset(year=train_year, split="train", config=config)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        bin_counts = np.zeros((num_pitch_types, num_bins, 2), dtype=np.int32)

        model.train()
        epoch_loss = 0.0
        cur_pitcher = None

        for batch in tqdm(train_loader):
            # Move all tensors to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Reset hidden state if pitcher changes
            if cur_pitcher is None or batch["pitcher_id"].item() != cur_pitcher:
                model.reset_state()
                cur_pitcher = batch["pitcher_id"].item()

            optimizer.zero_grad()
            output = model(batch)

            loss = criterion(output, batch["pitch_type_id"])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            step_counter += 1
            if step_counter % 500 == 0:
                print(f"[Step {step_counter}] Loss: {loss.item():.4f}")
                loss_steps.append((step_counter, loss.item()))
                total_counter = step_counter * (train_year - 2014)
                total_loss_steps.append((total_counter, loss.item()))

        print(f"Year {train_year} avg loss: {epoch_loss / len(train_loader):.4f}")
        train_losses.append((train_year, epoch_loss / len(train_loader)))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'year': train_year
        }, os.path.join(model_dir, "checkpoints", f"{model_name}_{train_year}.pt"))
        print(f"Saved model after {train_year}")

        #Plot loss step graph for a year
        if loss_steps:
            steps, step_losses = zip(*loss_steps)
            plt.figure()
            plt.plot(steps, step_losses, label="Training Loss (per step)", color="red")
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.title(f"Training Loss During Year {train_year}")
            plt.grid(True)
            plt.savefig(os.path.join(model_dir,"plots", "loss", f"{model_name}_{train_year}.png"))
            plt.close()

        
        prediction_log = []
        if config.get("validate_shift"):
            val_year = train_year
            print(f"Validating on {val_year}...")
            val_dataset = ColdStartDataset(year=val_year, split="val", config=config)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            model.eval()
            correct, total = 0, 0
            cur_pitcher = None
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if cur_pitcher is None or batch["pitcher_id"].item() != cur_pitcher:
                        model.reset_state()
                        cur_pitcher = batch["pitcher_id"].item()


                    output = model(batch)
                    # Top 1 accuracy based validation
                    pred = output.argmax(dim=1)
                    correct += (pred == batch["pitch_type_id"]).sum().item()
                    total += 1

                    # Probability bin based validation
                    true_pitch = batch["pitch_type_id"].item()
                    probs = output.exp()[0]
                    for pitch_id in range(len(probs)):
                        prob = probs[pitch_id].item()
                        bin_idx = 0
                        if prob > 0.0:
                            bin_idx = min(int(prob * num_bins), num_bins - 1)
                            bin_counts[pitch_id, bin_idx, 0] += 1  
                            total_bins[pitch_id, bin_idx, 0] += 1

                        if pitch_id == true_pitch:
                            bin_counts[pitch_id, bin_idx, 1] += 1  
                            total_bins[pitch_id, bin_idx, 1] += 1

                    log_entry = make_prediction_log(batch, output)
                    prediction_log.append(log_entry)

                    

            acc = correct / total
            print(f"Validation accuracy on {val_year}: {acc:.4f}") 
            val_scores.append((val_year, acc))
            with open(os.path.join(model_dir, "future_analysis", f"{model_name}_{val_year}_predictions.json"), "w") as f:
                json.dump(prediction_log, f)

            # Plot each pitches bins
            for pitch_id in range(num_pitch_types):
                totals = bin_counts[pitch_id, :, 0]
                corrects = bin_counts[pitch_id, :, 1]

                with np.errstate(divide='ignore', invalid='ignore'):
                    empirical_acc = np.divide(corrects, totals, where=totals > 0)
                    bin_centers = np.linspace(0.05, 0.95, num_bins)

                # Filter out pitches that were never predicted
                if np.sum(totals) == 0:
                    continue  
                pitch_name = pitch_id_to_name.get(pitch_id, "Unknown")
                plt.figure()
                plt.plot(bin_centers, empirical_acc, marker='o', label=f"Pitch {pitch_id}")
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
                plt.title(f"Calibration Curve - Pitch Type {pitch_name} ({train_year})")
                plt.xlabel("Predicted Probability")
                plt.ylabel("Empirical Accuracy")
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(model_dir,"plots", "validation", f"{model_name}_{pitch_name}_{train_year}.png"))
                plt.close()
        
        loss_steps = []
    
    # Save model
    torch.save(model.state_dict(), os.path.join(model_dir, "final_state.pt"))
    print(f"Model saved to {os.path.join(model_dir, 'final_state.pt')}")


    # Plot loss step graph for all years
    if total_loss_steps:
        steps, step_losses = zip(*total_loss_steps)
        plt.figure()
        plt.plot(steps, step_losses, label="Training Loss (per step)", color="red")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(f"Training Loss During All Years")
        plt.grid(True)
        plt.savefig(os.path.join(model_dir, "plots", f"{model_name}_all_years_loss_steps.png"))
        plt.close()

    # Plot combined loss and validation scores across all years
    if train_losses and val_scores:
        years, losses = zip(*train_losses)
        _, val_accs = zip(*val_scores)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(years, losses, color='red', label='Train Loss')
        ax2.plot(years, val_accs, color='blue', label='Validation Accuracy')

        ax1.set_xlabel('Year')
        ax1.set_ylabel('Loss', color='red')
        ax2.set_ylabel('Accuracy', color='blue')
        plt.title('Yearly Training Loss & Validation Accuracy')
        plt.savefig(os.path.join(model_dir, "plots", "combined_metrics.png"))

    # Plot each pitches bins
    for pitch_id in range(num_pitch_types):
        totals = total_bins[pitch_id, :, 0]
        corrects = total_bins[pitch_id, :, 1]

        with np.errstate(divide='ignore', invalid='ignore'):
            empirical_acc = np.divide(corrects, totals, where=totals > 0)
            bin_centers = np.linspace(0.05, 0.95, num_bins)

        # Filter out pitches that were never predicted
        if np.sum(totals) == 0:
            continue  
        pitch_name = pitch_id_to_name.get(pitch_id, "Unknown")
        plt.figure()
        plt.plot(bin_centers, empirical_acc, marker='o', label=f"Pitch {pitch_id}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
        plt.title(f"Calibration Curve - Pitch Type {pitch_name} ({train_year})")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Empirical Accuracy")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(model_dir,"plots", f"{model_name}_{pitch_name}_all_year_bins.png"))
        plt.close()

