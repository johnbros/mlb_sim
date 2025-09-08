import torch
import torch.nn as nn
import torch.nn.functional as F

class PitchTypeLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_players = config["num_players"]
        num_pitch_types = config["num_pitch_types"]
        num_zones = config["num_zones"]
        num_prior_outcomes = config["num_prior_outcomes"]
        hidden_size = config.get("hidden_size", 128)
        num_layers = config.get("num_layers", 1)
        dropout = config.get("lstm_dropout", 0.0)
        self.temperature = config.get("temperature", 1.0)

        pitcher_embed_dim = config["pitcher_embed_dim"]
        batter_embed_dim = config["batter_embed_dim"]
        catcher_embed_dim = config["catcher_embed_dim"]
        runner_embed_dim = config["runner_embed_dim"]
        zones_embed_dim = config["zones_embed_dim"]
        remaining_dims = config["remaining_dims"] # Number of non embedded features
        prior_pitch_type_embed_dim = config["prior_pitch_type_embed_dim"]
        prior_outcome_embed_dim = config["prior_outcome_embed_dim"]

        # Embeddings
        self.pitcher_embed = nn.Embedding(num_players + 1, pitcher_embed_dim)
        self.batter_embed = nn.Embedding(num_players + 1, batter_embed_dim)
        self.catcher_embed = nn.Embedding(num_players + 1, catcher_embed_dim)
        #i'm padding runners with 0s because index 0 of this embedding means nobody is on that base
        self.runner_embed = nn.Embedding(num_players + 1, runner_embed_dim, padding_idx=0)
        self.prior_zone_embed = nn.Embedding(num_zones + 1, zones_embed_dim, padding_idx=0)
        self.prior_pitch_type_embed = nn.Embedding(num_pitch_types + 1, prior_pitch_type_embed_dim, padding_idx=0)
        self.prior_outcome_embed = nn.Embedding(num_prior_outcomes + 1, prior_outcome_embed_dim, padding_idx=0)  # Assuming binary outcome (e.g., ball/strike)

        self.hidden_state = None

        # LSTM input dimensions, lets set these in the yaml file so it's adaptive to difference datasets
        input_dim = pitcher_embed_dim + batter_embed_dim + catcher_embed_dim + 3 * runner_embed_dim + zones_embed_dim + prior_pitch_type_embed_dim+ prior_outcome_embed_dim + remaining_dims
        self.lstm = nn.LSTM(input_dim, hidden_size=hidden_size, num_layers=num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        # Output projection
        self.output_layer = nn.Linear(hidden_size, num_pitch_types)

    def reset_state(self):
        self.hidden_state = None

    def forward(self, batch):
        # Embed the embedded features
        pitcher_vec = self.pitcher_embed(batch["pitcher_id"])  
        batter_vec = self.batter_embed(batch["batter_id"])    
        catcher_vec = self.catcher_embed(batch["catcher_id"]) 
        runner_vecs = self.runner_embed(batch["runners"])
        prior_zone_vec = self.prior_zone_embed(batch["prior_zone"])
        prior_pitch_type_vec = self.prior_pitch_type_embed(batch["prior_pitch_type_id"])
        prior_outcome_vec = self.prior_outcome_embed(batch["prior_pitch_outcome_id"])
        runner1, runner2, runner3 = torch.unbind(runner_vecs, dim=1)

        #Assemble the non-embedded features
        count_feats = torch.stack([
            batch["balls"].float(),
            batch["strikes"].float(),
            batch["outs"].float()
        ], dim=1)  

        handedness = torch.stack([
            batch["pitcher_hand"].float(),
            batch["batter_hand"].float()
        ], dim=1)

        time_feats = torch.stack([
            batch["inning_num"].float(),
            batch["inning_half"].float(),
            batch["at_bat_num"].float(),
            batch["pitch_num"].float()
        ], dim=1)

        score_feat = batch["score"].float().unsqueeze(1) 

        x = torch.cat([
            pitcher_vec, batter_vec, catcher_vec, prior_zone_vec,
            prior_pitch_type_vec, prior_outcome_vec,
            runner1, runner2, runner3,
            handedness, time_feats, 
            count_feats, score_feat
        ], dim=1).unsqueeze(1)  

        lstm_out, self.hidden_state = self.lstm(x, self.hidden_state) 
        output = self.output_layer(lstm_out.squeeze(1))  

        # Softmax the output if we are masking
        if "valid_pitch_types" in batch:
            logits = output.clone()
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for i, valid_types in enumerate(batch["valid_pitch_types"]):
                mask[i, valid_types] = True

            # Changing the masking strategy to try and solve the infinite loss issue
            if batch["split"].item() == 0:
                logits[~mask] -= self.config["mask_value"]
            else:
                epsilon = 1e-6
                logits[~mask] = -1e9
                logits[mask] = logits[mask] * (1 + epsilon)
                logits = logits / self.temperature
            return F.log_softmax(logits, dim=1)

        return output 
