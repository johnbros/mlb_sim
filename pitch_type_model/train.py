import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from pitch_type_model.cold_start.dataset import ColdStartDataset
from pitch_type_model.warm_start.dataset import WarmStartDataset
from pitch_type_model.models import PitchTypeLSTM
import matplotlib.pyplot as plt
import numpy as np
from constants.id_map import pitch_id_to_name
import os
import json
import yaml
import argparse

# Welcome to the training script for the Pitch Type LSTM model.
# This file as of now is called train.py but may be renamed to trainlstm.py in the future if I try different models for this problem.

## TODO: Refactor this file to be a bit more compact it's a bit long and has a lot of repeated code.

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
        "score": batch["score"].item(),
        "balls": batch["balls"].item(),
        "strikes": batch["strikes"].item(),
        "outs": batch["outs"].item(),
    }

def data_switcher(train_year, split, config):
    model_name = config.get("model_name", "unknown_model")
    match model_name:
        case "cold_start":
            dataset = ColdStartDataset(year=train_year, split=split, config=config)
        case "warm_start":
            dataset = WarmStartDataset(year=train_year, split=split, config=config)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    return dataset

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PitchTypeLSTM(config).to(device)

    criterion = torch.nn.NLLLoss()  # For log_softmax output
    optimizer = Adam(model.parameters(), lr=config["lr"])

    # Plots for analysis of the model
    model_name = config.get("model_name", "unknown_model")
    val_scores = []
    train_losses = []
    train_entropies = []
    loss_steps = []
    total_loss_steps = []
    
    num_pitch_types = 21 # Bin for each pitch type we filter out which pitches never get predicted later
    num_bins = 40
    step_counter = 0
    model_type = "pitch_type_model"
    model_root = os.path.join(model_type, model_name)
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(os.path.join(model_root, "plots", "loss"), exist_ok=True)
    os.makedirs(os.path.join(model_root, "plots", "validation"), exist_ok=True)
    os.makedirs(os.path.join(model_root, "future_analysis"), exist_ok=True)
    os.makedirs(os.path.join(model_root, "checkpoints"), exist_ok=True)

    total_bins = np.zeros((num_pitch_types, num_bins, 2), dtype=np.int32)
    bin_conf_sums = np.zeros((num_pitch_types, num_bins), dtype=np.float32)
    resume_year = 2015
    if config.get("resume_checkpoint", False):
        checkpoint_path = config["checkpoint_path"]
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resume_year = config.get("resume_year", 2016)
    else:
        resume_year = config["train_years"][0]
    
    if config.get("validate_only_year") is not None:
        val_year = config["validate_only_year"] + config["validate_shift"]
        print(f"\n=== Running validation-only on {val_year} ===")

        checkpoint_path = os.path.join(model_root, "checkpoints", f"{model_name}_{val_year-1}.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Skipping validation.")
            return

        val_dataset = data_switcher(val_year, "val", config)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        bin_counts = np.zeros((num_pitch_types, num_bins, 2), dtype=np.int32)
        prediction_log = []
        correct, total = 0, 0
        cur_pitcher = None
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                if cur_pitcher is None or batch["pitcher_id"].item() != cur_pitcher:
                    model.reset_state()
                    cur_pitcher = batch["pitcher_id"].item()

                output = model(batch)
                pred = output.argmax(dim=1)
                correct += (pred == batch["pitch_type_id"]).sum().item()
                total += 1

                # Probability bin validation
                true_pitch = batch["pitch_type_id"].item()
                probs = output.exp()[0]
                for pitch_id in range(len(probs)):
                    prob = probs[pitch_id].item()
                    bin_idx = min(int(prob * num_bins), num_bins - 1)
                    bin_counts[pitch_id, bin_idx, 0] += 1
                    if pitch_id == true_pitch:
                        bin_counts[pitch_id, bin_idx, 1] += 1

                prediction_log.append(make_prediction_log(batch, output))

        acc = correct / total
        print(f"Validation-only accuracy on {val_year}: {acc:.4f}")

       
        chunk_size = 15000
        num_chunks = (len(prediction_log) + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(prediction_log))
            chunk = prediction_log[start:end]
            file_path = os.path.join(model_root, "future_analysis", f"{model_name}_{val_year}_predictions_part{i+1}.json")
            with open(file_path, "w") as f:
                json.dump(chunk, f, indent=4)

        
        for pitch_id in range(num_pitch_types):
            totals = bin_counts[pitch_id, :, 0]
            corrects = bin_counts[pitch_id, :, 1]
            if np.sum(totals) == 0:
                continue
            empirical_acc = np.divide(corrects, totals, where=totals > 0)
            bin_centers = np.linspace(0.0125, 0.9875, num_bins)
            pitch_name = pitch_id_to_name.get(pitch_id, "Unknown")
            plt.figure()
            plt.plot(bin_centers, empirical_acc, marker='o')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.title(f"Calibration Curve - Pitch {pitch_name} ({val_year+1})")
            plt.xlabel("Predicted Probability")
            plt.ylabel("Empirical Accuracy")
            plt.grid(True)
            os.makedirs(os.path.join(model_root, "plots", "testing"), exist_ok=True)
            plt.savefig(os.path.join(model_root, "plots", "testing", f"{model_name}_{val_year+1}.png"))
            plt.close()

        print("Finished validation-only mode.")



    for i, train_year in enumerate(config["train_years"]):
        if train_year < resume_year:
            continue
        print(f"\n=== Training on {train_year} ===")
        train_dataset = data_switcher(train_year, "train", config)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        bin_counts = np.zeros((num_pitch_types, num_bins, 2), dtype=np.int32)

        model.train()
        epoch_loss = 0.0
        cur_pitcher = None
        loss_accum = 0.0
        step_accum = 0
        entropy_accum = 0.0
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Reset hidden state on pitcher substitutions
            if cur_pitcher is None or batch["pitcher_id"].item() != cur_pitcher:
                if step_accum > 0:
                    loss_accum.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_accum = 0.0
                    step_accum = 0
                model.reset_state()
                cur_pitcher = batch["pitcher_id"].item()

            

            output = model(batch)
            log_probs = output
            nll_loss = criterion(log_probs, batch["pitch_type_id"])
            probs = log_probs.exp()
            entropy = -torch.sum(probs * log_probs, dim=1).mean()
            loss = nll_loss - config.get("lambda_entropy", 0.01) * entropy
            loss_accum += loss
            step_accum += 1

            if step_accum == 30:
                loss_accum.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if model.hidden_state is not None:
                    model.hidden_state = tuple(h.detach() for h in model.hidden_state)
                optimizer.step()
                optimizer.zero_grad()
                loss_accum = 0.0
                step_accum = 0

            epoch_loss += loss.item()
            entropy_accum += entropy.item()

            step_counter += 1
            if step_counter % 1000 == 0:
                print(f"[Step {step_counter}] Loss: {loss.item():.4f} | Entropy: {entropy.item():.4f}")

                loss_steps.append((step_counter, loss.item()))
                total_counter = step_counter * (train_year - 2014)
                total_loss_steps.append((total_counter, loss.item()))

        print(f"Year {train_year} avg loss: {epoch_loss / len(train_loader):.4f}")
        print(f"Year {train_year} avg entropy: {entropy_accum / len(train_loader):.4f}")
        train_losses.append((train_year, epoch_loss / len(train_loader)))
        train_entropies.append((train_year, entropy_accum / len(train_loader)))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'year': train_year
        }, os.path.join(model_root, "checkpoints", f"{model_name}_{train_year}.pt"))
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
            plt.savefig(os.path.join(model_root,"plots", "loss", f"{model_name}_{train_year}.png"))
            plt.close()

        
        prediction_log = []
        if config.get("validate_shift"):
            if train_year == 2023:
                val_year = train_year + config["validate_shift"]
                print(f"Validating on {val_year}...")
                val_dataset = data_switcher(val_year, "val", config)
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
                            if prob > 0.0025:
                                bin_idx = min(int(prob * num_bins), num_bins - 1)
                                bin_counts[pitch_id, bin_idx, 0] += 1  
                                total_bins[pitch_id, bin_idx, 0] += 1
                                bin_conf_sums[pitch_id, bin_idx] += prob

                            if pitch_id == true_pitch:
                                bin_counts[pitch_id, bin_idx, 1] += 1  
                                total_bins[pitch_id, bin_idx, 1] += 1

                        log_entry = make_prediction_log(batch, output)
                        prediction_log.append(log_entry)

                        

                acc = correct / total
                print(f"Validation accuracy on {val_year}: {acc:.4f}") 
                val_scores.append((val_year, acc))

                chunk_size = 15000
                num_chunks = (len(prediction_log) + chunk_size - 1) // chunk_size
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min((i + 1) * chunk_size, len(prediction_log))
                    chunk = prediction_log[start:end]

                    file_path = os.path.join(model_root, "future_analysis", f"{model_name}_{val_year+1}_predictions_part{i+1}.json")
                    with open(file_path, "w") as f:
                        json.dump(chunk, f, indent=4)

                # Plot each pitches bins
                ECE_per_pitch = np.zeros(num_pitch_types, dtype=np.float32)
                samples_per_pitch = np.zeros(num_pitch_types, dtype=np.int32)
                for pitch_id in range(num_pitch_types):
                    totals = bin_counts[pitch_id, :, 0]
                    corrects = bin_counts[pitch_id, :, 1]

                    with np.errstate(divide='ignore', invalid='ignore'):
                        empirical_acc = np.divide(corrects, totals, where=totals > 0)
                        bin_centers = np.linspace(0.0125, 0.9875, num_bins)

                    # Filter out pitches that were never predicted
                    if np.sum(totals) == 0:
                        continue  
                    pitch_name = pitch_id_to_name.get(pitch_id, "Unknown")


                    fig, ax1 = plt.subplots()
                    # Line plot: calibration curve
                    ax1.plot(bin_centers, empirical_acc, marker='o', color='blue', label='Empirical Accuracy')
                    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
                    ax1.set_xlabel("Predicted Probability")
                    ax1.set_ylabel("Empirical Accuracy", color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')
                    ax1.set_ylim(0, 1)
                    ax1.grid(True)

                    # Bar plot: bin density
                    ax2 = ax1.twinx()
                    ax2.bar(bin_centers, totals, width=0.02, alpha=0.3, color='orange', label='Bin Density')
                    ax2.set_ylabel("Number of Predictions", color='orange')
                    ax2.tick_params(axis='y', labelcolor='orange')

                    # Title and legend
                    plt.title(f"Calibration Curve & Bin Density - Pitch {pitch_name} ({train_year+1})")
                    fig.tight_layout()
                    os.makedirs(os.path.join(model_root, "plots", "validation", pitch_name), exist_ok=True)
                    plt.savefig(os.path.join(model_root, "plots", "validation", pitch_name, f"{model_name}_{train_year+1}.png"))
                    plt.close()

                    total_samples = np.sum(totals)
                    bin_conf_sums_cur_pitch = bin_conf_sums[pitch_id]
                    avg_conf = np.divide(bin_conf_sums_cur_pitch, totals, where=totals > 0)
                    ECE = 0.0
                    for i in range(num_bins):
                        confidence = avg_conf[i]
                        accuracy = corrects[i] / totals[i] if totals[i] > 0 else 0
                        samples_in_bin = totals[i]
                        
                        ECE += (samples_in_bin / total_samples) * abs(accuracy - confidence)

                    print(f"ECE for {pitch_name}: {ECE:.4f}")
                    ECE_per_pitch[pitch_id] = ECE
                    samples_per_pitch[pitch_id] = total_samples


                weighted_ECE = sum(ECE_per_pitch[p] * samples_per_pitch[p] for p in range(num_pitch_types)) / sum(samples_per_pitch[p] for p in range(num_pitch_types))
                print(f"\nClass-Weighted Average ECE: {weighted_ECE:.4f}")

        loss_steps = []
    
    # Save model
    torch.save(model.state_dict(), os.path.join(model_root, "final_state.pt"))
    print(f"Model saved to {os.path.join(model_root, 'final_state.pt')}")


    # Plot loss step graph for all years
    if total_loss_steps:
        steps, step_losses = zip(*total_loss_steps)
        plt.figure()
        plt.plot(steps, step_losses, label="Training Loss (per step)", color="red")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(f"Training Loss During All Years")
        plt.grid(True)
        plt.savefig(os.path.join(model_root, "plots", f"{model_name}_all_years_loss_steps.png"))
        plt.close()

    # Plot combined loss and validation scores across all years
    # if train_losses and val_scores and train_entropies:
    #     years, losses = zip(*train_losses)
    #     _, val_accs = zip(*val_scores)
    #     _, entropies = zip(*train_entropies)

    #     fig, ax1 = plt.subplots()
    #     ax2 = ax1.twinx()
    #     ax3 = ax1.twinx()
    #     ax3.spines["right"].set_position(("axes", 1.1))  # Push entropy axis further right

    #     ax1.plot(years, losses, color='red', label='Train Loss')
    #     ax2.plot(years, val_accs, color='blue', label='Validation Accuracy')
    #     ax3.plot(years, entropies, color='green', label='Entropy')

    #     ax1.set_xlabel('Year')
    #     ax1.set_ylabel('Loss', color='red')
    #     ax2.set_ylabel('Accuracy', color='blue')
    #     ax3.set_ylabel('Entropy', color='green')
    #     lines, labels = [], []
    #     for ax in [ax1, ax2, ax3]:
    #         l, lab = ax.get_legend_handles_labels()
    #         lines += l
    #         labels += lab
    #     fig.legend(lines, labels, loc='upper center')

    #     plt.title('Yearly Training Loss, Entropy, & Validation Accuracy')
    #     plt.savefig(os.path.join(model_root, "plots", "combined_metrics.png"))

    

    # Plot each pitches bins
    # for pitch_id in range(num_pitch_types):
    #     totals = total_bins[pitch_id, :, 0]
    #     corrects = total_bins[pitch_id, :, 1]

    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         empirical_acc = np.divide(corrects, totals, where=totals > 0)
    #         bin_centers = np.linspace(0.05, 0.95, num_bins)

    #     # Filter out pitches that were never predicted
    #     if np.sum(totals) == 0:
    #         continue  
    #     pitch_name = pitch_id_to_name.get(pitch_id, "Unknown")
    #     plt.figure()
    #     plt.plot(bin_centers, empirical_acc, marker='o', label=f"Pitch {pitch_id}")
    #     plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
    #     plt.title(f"Calibration Curve - Pitch Type {pitch_name} (all years)")
    #     plt.xlabel("Predicted Probability")
    #     plt.ylabel("Empirical Accuracy")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(os.path.join(model_root,"plots", f"{model_name}_{pitch_name}_all_year_bins.png"))
    #     plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Pitch Type LSTM model.")

    
    parser.add_argument("--config", type=str, default="pitch_type_model/cold_start/config.yaml", help="Path to the config file.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_model(config)