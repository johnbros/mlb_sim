import psycopg2
from database.passwords import POSTGRES
from constants.id_map import valid_pitch_ids
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_dir, '..', 'constants', 'valid_pitch_map.json')

conn = psycopg2.connect(
    user="postgres",
    password=POSTGRES,
    host="localhost",
    port="5432",
    database="mlb_data"
)
years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]



valid_pitch_map = {}

with conn.cursor() as cur:
    try:
        
        for current_year in years:
            valid_pitch_map[current_year] = {}
            cur.execute("""
        SELECT distinct pitcher_id FROM cold_start where EXTRACT(YEAR FROM game_date_time) = %s"""
                        , (current_year,))
    
            pitcher_ids = cur.fetchall()
            if pitcher_ids:
                pitcher_ids = [pitcher[0] for pitcher in pitcher_ids]
                print(f"Found {len(pitcher_ids)} pitchers for year {current_year}")
                for pitcher_id in pitcher_ids:
                    cur.execute("""
                            SELECT distinct pitch_type_id FROM cold_start
                            WHERE pitcher_id = %s and EXTRACT(YEAR FROM game_date_time) <= %s
                            """, (pitcher_id, current_year))
                    pitch_types = cur.fetchall()
                    if pitch_types:
                        pitch_types = [pitch_type[0] for pitch_type in pitch_types]
                        pitch_types = [pitch_type for pitch_type in pitch_types if pitch_type in valid_pitch_ids]
                        valid_pitch_map[current_year][pitcher_id] = pitch_types
                    else:
                        print(f"No pitch types found for pitcher {pitcher_id} in year {current_year}")

        
    except psycopg2.Error as e:
        print(f"Error executing query: {e}")

with open(output_path, "w") as f:
    json.dump(valid_pitch_map, f, indent=4)