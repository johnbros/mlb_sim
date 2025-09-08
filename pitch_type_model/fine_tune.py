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

# This script is for fine tuning the LSTM model for pitch type classification within a season.


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

def tune_model(config):

    num_pitch_types = 21 # Bin for each pitch type we filter out which pitches never get predicted later
    num_bins = 40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PitchTypeLSTM(config).to(device)

    criterion = torch.nn.NLLLoss() 
    optimizer = Adam(model.parameters(), lr=config["lr"])

    weight_path = config["weight_path"]

    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    tune_year = config["year"]
    model_name = config.get("model_name", "unknown_model")

    model_type = "pitch_type_model"
    model_root = os.path.join(model_type, model_name)
    tuning = os.path.join(model_root, "tuning")
    os.makedirs(tuning, exist_ok=True)
    os.makedirs(os.path.join(tuning, "plots"), exist_ok=True)
    os.makedirs(os.path.join(tuning, "plots", "calibration"), exist_ok=True)

    start_week = config["start_week"]
    end_week = config["end_week"]

    ECE_per_week = np.zeros((end_week - start_week + 1, num_pitch_types), dtype=np.float32)
    pitch_samples_per_week = np.zeros((end_week - start_week + 1, num_pitch_types), dtype=np.int32)
    weighted_ECE_weekly = np.zeros((end_week - start_week + 1), dtype=np.float32)
    accuracy_per_week = np.zeros((end_week - start_week + 1), dtype=np.float32)
    totals_per_week = np.zeros((end_week - start_week + 1), dtype=np.int32)

    for target_week in range(start_week, end_week + 1):

        print(f"Testing on {target_week}...")
        val_dataset = ColdStartDataset(year=tune_year, split='test', config=config, mode="test", target_week=target_week)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        model.eval()


        cur_pitcher = None
        prediction_log = []
        correct, total = 0, 0
        bin_counts = np.zeros((num_pitch_types, num_bins, 2), dtype=np.int32)
        total_bins = np.zeros((num_pitch_types, num_bins, 2), dtype=np.int32)
        bin_conf_sums = np.zeros((num_pitch_types, num_bins), dtype=np.float32)
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
        accuracy = correct / total
        print(f"Validation accuracy for week {target_week}: {accuracy:.4f}")
        accuracy_per_week[target_week - start_week] = accuracy
        totals_per_week[target_week - start_week] = total

        with open(os.path.join(tuning, f"predictions_week_{target_week}.json"), "w") as f:
            json.dump(prediction_log, f)
        
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
            plt.title(f"Calibration Curve & Bin Density - Pitch {pitch_name} ({target_week}")
            fig.tight_layout()
            os.makedirs(os.path.join(tuning, "plots", "calibration", pitch_name), exist_ok=True)
            plt.savefig(os.path.join(tuning, "plots", "calibration", pitch_name, f"{model_name}_{target_week}.png"))
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
            ECE_per_pitch[pitch_id] = ECE
            ECE_per_week[target_week - start_week, pitch_id] = ECE
            pitch_samples_per_week[target_week - start_week, pitch_id] = total_samples
            samples_per_pitch[pitch_id] = total_samples


        weighted_ECE = sum(ECE_per_pitch[p] * samples_per_pitch[p] for p in range(num_pitch_types)) / sum(samples_per_pitch[p] for p in range(num_pitch_types))
        weighted_ECE_weekly[target_week - start_week] = weighted_ECE


        
        print(f"Training on {target_week}...")
        train_dataset = ColdStartDataset(year=tune_year, split='train', config=config, mode="train", target_week=target_week)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
        loss_accum = 0.0
        step_accum = 0
        epoch_loss = 0.0
        entropy_accum = 0.0
        cur_pitcher = None
        model.train()
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
        
        print(f"Week {target_week} avg loss: {epoch_loss / len(train_loader):.4f}")
        print(f"Week {target_week} avg entropy: {entropy_accum / len(train_loader):.4f}")

    #Plot weekly ECE for each pitch
    for pitch_id in range(num_pitch_types):
        weeks, ECE = zip(*enumerate(ECE_per_week[:, pitch_id], start=start_week))
        _, samples = zip(*enumerate(pitch_samples_per_week[:, pitch_id], start=start_week))
        fig, ax1 = plt.subplots()
        ax1.plot(weeks, ECE, marker='o', color='blue', label='ECE')
        ax1.set_xlabel("Week")
        ax1.set_ylabel("ECE", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(0, 1)
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.bar(weeks, samples, width=0.6, alpha=0.3, color='orange', label='Number of Predictions')
        ax2.set_ylabel("Number of Predictions", color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        plt.title(f"Pitch ECE over time w/ Weekly density - Pitch {pitch_name}")
        fig.tight_layout()
        os.makedirs(os.path.join(tuning, "plots", "calibration", pitch_name), exist_ok=True)
        plt.savefig(os.path.join(tuning, "plots", "calibration", pitch_name, f"{model_name}_ECE_all_weeks.png"))
        plt.close()

    #Plot class weighted ECE for all pitches
    weeks, weighted_ECE = zip(*enumerate(weighted_ECE_weekly, start=start_week))
    _, totals = zip(*enumerate(totals_per_week, start=start_week))
    fig, ax1 = plt.subplots()
    ax1.plot(weeks, weighted_ECE, marker='o', color='blue', label='ECE')
    ax1.set_xlabel("Week")
    ax1.set_ylabel("ECE", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 0.5)
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.bar(weeks, totals, width=0.6, alpha=0.3, color='orange', label='Number of Predictions')
    ax2.set_ylabel("Number of Predictions", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    plt.title(f"Class Weighted ECE over time w/ Weekly Density - {model_name} (All Pitches)")
    fig.tight_layout()
    os.makedirs(os.path.join(tuning, "plots", "calibration"), exist_ok=True)
    plt.savefig(os.path.join(tuning, "plots", "calibration", f"{model_name}_ECE_all.png"))
    plt.close()

    #Plot accuracy over all weeks
    weeks, accuracy = zip(*enumerate(accuracy_per_week, start=start_week))
    fig, ax1 = plt.subplots()
    ax1.plot(weeks, accuracy, marker='o', color='blue', label='Accuracy')
    ax1.set_xlabel("Week")
    ax1.set_ylabel("Accuracy", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.bar(weeks, totals, width=0.6, alpha=0.3, color='orange', label='Number of Predictions')
    ax2.set_ylabel("Number of Predictions", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    plt.title(f"Accuracy over time w/ Weekly Density - {model_name}")
    fig.tight_layout()
    os.makedirs(os.path.join(tuning, "plots"), exist_ok=True)
    plt.savefig(os.path.join(tuning, "plots", f"{model_name}_accuracy_all_weeks.png"))
    plt.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Pitch Type LSTM model.")

    #Start by defaulting to cold_start config since theres no other model right now.
    parser.add_argument("--config", type=str, default="pitch_type_model/cold_start/tune.yaml", help="Path to the tuning file.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    tune_model(config)