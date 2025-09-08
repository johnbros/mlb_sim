import psycopg2
from database.passwords import POSTGRES
from torch.utils.data import Dataset
import torch
import json
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, '../..', 'constants', 'valid_pitch_map.json')
with open(JSON_PATH, 'r') as f:
    pitch_masks = json.load(f)

def encode_id(player_id):
    return player_id + 1 if player_id != -1 else 0

class ColdStartDataset(Dataset):
    def __init__(self, year=None, split='train', config=None, mode="train", target_week=None):
        self.year = year
        self.split = split
        self.config = config
        self.target_week = target_week
        self.mode = mode
        self.conn = psycopg2.connect(
            user="postgres",
            password=POSTGRES,
            host="localhost",
            port="5432",
            database="mlb_data"
        )
        self._load_data()
        self.conn.close()
        self.pitch_masks = pitch_masks
        split_map = {"train": 0, "val": 1, "test": 2}
        self.split_id = split_map[self.split]

    def _load_data(self):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT game_id, game_date_time, inning_num, inning_half, at_bat_num, pitch_num,
                pitcher_id, batter_id, catcher_id,
                runner_first, runner_second, runner_third,
                balls, strikes, outs, score, pitch_outcome_id,
                pitch_type_id, zone, pitcher_hand, batter_hand
            FROM cold_start
            WHERE EXTRACT(YEAR FROM game_date_time) = %s
            ORDER BY game_date_time, game_id, inning_half, inning_num, at_bat_num, pitch_num
        """, (self.year,))
        r = cur.fetchall()
        cur_pitcher = None
        prior_pitch_outcome_id = -1
        prior_pitch_type_id = -1
        prior_zone = 0
        accumulated_rows = []
        for row in r:
            (
            game_id, game_date_time, inning_num, inning_half, at_bat_num, pitch_num,
            pitcher_id, batter_id, catcher_id,
            runner_first, runner_second, runner_third,
            balls, strikes, outs, score, pitch_outcome_id,
            pitch_type_id, zone, pitcher_hand, batter_hand
        ) = row
            game_week = game_date_time.isocalendar()[1] 
            if self.target_week is not None:
                if self.mode == "test" and game_week != self.target_week:
                    continue
                if self.mode == "train" and game_week != self.target_week:
                    continue
            if pitcher_id != cur_pitcher:
                prior_pitch_outcome_id = prior_pitch_type_id = -1
                prior_zone = 0
            
            #Shift zone at time t to time t+1
            accumulated_rows.append((game_id, game_date_time, inning_num, inning_half, at_bat_num, pitch_num, 
                                     pitcher_id, batter_id, catcher_id,
                                     runner_first, runner_second, runner_third,
                                     balls, strikes, outs, score, prior_pitch_outcome_id,
                                     prior_pitch_type_id, prior_zone, pitcher_hand, batter_hand, pitch_type_id, zone))
            prior_pitch_outcome_id = pitch_outcome_id
            prior_pitch_type_id = pitch_type_id
            prior_zone = zone
            cur_pitcher = pitcher_id
        
        self.rows = accumulated_rows


        cur.close()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        (
            game_id, game_date_time, inning_num, inning_half, at_bat_num, pitch_num, 
            pitcher_id, batter_id, catcher_id,
            runner_first, runner_second, runner_third,
            balls, strikes, outs, score, prior_pitch_outcome_id,
            prior_pitch_type_id, prior_zone, pitcher_hand, batter_hand, pitch_type_id, zone
        ) = row
        valid_types = self.pitch_masks[str(self.year)].get(str(pitcher_id), [])
        

        # Assembling all the feature tensors for a pitch
        return {
            "pitcher_id": torch.tensor(encode_id(pitcher_id), dtype=torch.long),
            "batter_id": torch.tensor(encode_id(batter_id), dtype=torch.long),
            "catcher_id": torch.tensor(encode_id(catcher_id), dtype=torch.long),
            "runners": torch.tensor([
                encode_id(runner_first),
                encode_id(runner_second),
                encode_id(runner_third)
            ], dtype=torch.long),
            "inning_num": torch.tensor(inning_num / 10.0, dtype=torch.float),
            "inning_half": torch.tensor(inning_half, dtype=torch.long),
            "at_bat_num": torch.tensor(at_bat_num / 10.0, dtype=torch.float),
            "pitch_num": torch.tensor(pitch_num / 10.0, dtype=torch.float),
            "balls": torch.tensor(balls, dtype=torch.long),
            "strikes": torch.tensor(strikes, dtype=torch.long),
            "outs": torch.tensor(outs, dtype=torch.long),
            "score": torch.tensor(score / 10.0, dtype=torch.float),
            "prior_zone": torch.tensor(prior_zone, dtype=torch.long),
            "pitcher_hand": torch.tensor(pitcher_hand, dtype=torch.long),
            "batter_hand": torch.tensor(batter_hand, dtype=torch.long),
            "prior_pitch_outcome_id": torch.tensor(encode_id(prior_pitch_outcome_id), dtype=torch.long),
            "prior_pitch_type_id": torch.tensor(encode_id(prior_pitch_type_id), dtype=torch.long),
            "valid_pitch_types": torch.tensor(valid_types, dtype=torch.long),
            "pitch_type_id": torch.tensor(pitch_type_id, dtype=torch.long),
            "game_id": torch.tensor(game_id, dtype=torch.long),
            "zone": torch.tensor(zone, dtype=torch.long),
            "split": torch.tensor(self.split_id, dtype=torch.long)
        }