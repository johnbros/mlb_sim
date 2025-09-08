from psycopg2 import pool
from database.passwords import POSTGRES
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
conn_pool = pool.SimpleConnectionPool(5, 30, dbname="mlb_data", user="postgres", password=f"{POSTGRES}", host="127.0.0.1", port="5432")



#  p_outcome_id |   p_description   | p_result_code |    p_call_name    | p_call
# --------------+-------------------+---------------+-------------------+--------
#             2 | Ball              | B             | Ball              | B
#             3 | In play, no out   | D             | In Play           | X
#            25 | Swinging Strike   | S             | Strike            | S
#             1 | In play, out(s)   | X             | In Play           | X
#            13 | Called Strike     | C             | Strike            | S
#            16 | Foul              | F             | Strike            | S
#           823 | Intent Ball       | I             | Intent Ball       | Z
#         99230 | Pitchout          | P             | Pitchout          | Z
#       4045673 | Swinging Strike   | W             | Strike            | S
#          2971 | Foul Bunt         | L             | In Play           | X
#            27 | Hit By Pitch      | H             | In Play           | X
#        520201 | Foul Tip          | T             | Strike            | S
#       1249504 | Ball In Dirt      | *B            | Ball              | B
#            17 | In play, run(s)   | E             | In Play           | X
#        319370 | Unknown Strike    | K             | Unknown Strike    | Z
#         70759 | Foul Pitchout     | R             | Foul Pitchout     | Z
#            14 | Missed Bunt       | M             | Missed Bunt       | Z
#       1744521 | Foul Tip          | O             | Foul Tip          | Z
#      18744593 | In play, run(s)   | Z             | In play, run(s)   | Z
#      18785882 | In play, no out   | J             | In play, no out   | Z
#       1824123 | In play, out(s)   | Y             | In play, out(s)   | Z
#        457354 | Swinging Pitchout | Q             | Swinging Pitchout | Z

# 0 = No Swing
# 1 = Swing and Miss
# 2 = Swing and Foul
# 3 = Swing and In Play
no_swing_ids = [2, 13, 823, 99230, 1249504, 319370, 27]
swing_miss_ids = [25, 4045673, 457354, 14]
swing_foul_ids = [16, 2971, 520201, 70759, 1744521]
swing_inplay_ids = [1, 3, 17, 1824123, 18744593, 18785882]

# Map pitch outcomes to encoded simple states for pitcher use
pitch_outcome_id_map = {}

for pid in no_swing_ids:
    pitch_outcome_id_map[pid] = 0

for pid in swing_miss_ids:
    pitch_outcome_id_map[pid] = 1

for pid in swing_foul_ids:
    pitch_outcome_id_map[pid] = 2

for pid in swing_inplay_ids:
    pitch_outcome_id_map[pid] = 3



#  pitch_type_id | pitch_type_code | pitch_type_name
# ---------------+-----------------+------------------
#           2472 | CU              | Curveball
#            162 | SI              | Sinker
#            391 | CH              | Changeup
#            155 | FC              | Cutter
#              9 | FF              | 4-Seam Fastball
#          25599 | ST              | Sweeper
#             69 | SL              | Slider
#           1155 | Unknown         | Unknown
#            505 | KC              | Knuckle Curve
#          25020 | FS              | Splitter
#          68933 | SV              | Slurve
#        7696650 | FT              | 2-Seam Fastball
#         713266 | CS              | Slow Curve
#          47691 | KN              | Knuckle Ball
#          14118 | FA              | Fastball
#          36023 | EP              | Eephus
#          43092 | FO              | Forkball
#         153740 | AB              | Automatic Ball
#          81408 | SC              | Screwball
#          24698 | PO              | Pitchout
#          26023 | IN              | Intentional Ball


# Due to db structure I'm renormalizing pitch type ids to a 0 indexed range
pitch_type_id_map = {
    2472: 0,
    162: 1,
    391: 2,
    155: 3,
    9: 4,
    25599: 5,
    69: 6,
    1155: 7,
    505: 8,
    25020: 9,
    68933: 10,
    7696650: 11,
    713266: 12,
    47691: 13,
    14118: 4,
    36023: 15,
    43092: 16,
    153740: 17,
    81408: 18,
    24698: 19,
    26023: 20,
}

#Quick coordinate mapping for the zones
ZONE_LOOKUP = {
        (0,0): 11, (0,1): 11, (0,2): 11, (0,3): 12, (0,4): 12, (0,5): 12,
        (1,0): 11, (1,1): 1, (1,2): 2, (1,3): 2, (1,4): 3, (1,5): 12,
        (2,0): 11, (2,1): 4, (2,2): 5, (2,3): 5, (2,4): 6, (2,5): 12,
        (3,0): 13, (3,1): 4, (3,2): 5, (3,3): 5, (3,4): 6, (3,5): 14,
        (4,0): 13, (4,1): 7, (4,2): 8, (4,3): 8, (4,4): 9, (4,5): 14,
        (5,0): 13, (5,1): 13, (5,2): 13, (5,3): 14, (5,4): 14,(5,5): 14,
    }

def map_zone(px, pz, sz_top=None, sz_bot=None):
    """
    Maps px, pz (Statcast plate coordinates) to one of the 14 MLB pitch zones.
    Using batter specific strike zone when available... see note below for more info

    Parameters:
        px: Location of pitch where it crossed the plate horizontally (0 is center of plate negative is left, positive is right)
        pz: Location of pitch where it crossed the plate vertically (0 is ground,sz_bot is bottom of strike zone, sz_top is top of strike zone)
        sz_top : top of batter's strike zone
        sz_bot : bottom of batter's strike zone

    Returns:
        zone_id (int or None): Zone ID 1-14, or None if pitch is outside mappable region
    """

    if px is None or pz is None:
        return 10

    # Default SZ values(this will never happen we have every SZ)
    sz_top = sz_top if sz_top is not None else 3.5
    sz_bot = sz_bot if sz_bot is not None else 1.5

    if px < -0.83:
        col = 0  
    elif px < -0.28:
        col = 1
    elif px < 0:
        col = 2
    elif px < 0.28:
        col = 3
    elif px < 0.83:
        col = 4
    else:
        col = 5 

    zone_height = sz_top - sz_bot
    upper = sz_bot + zone_height * 2/3
    midpoint = sz_bot + zone_height * 1/2
    lower = sz_bot + zone_height * 1/3

    if pz < sz_bot:
        row = 5  
    elif pz < lower:
        row = 4
    elif pz < midpoint:
        row = 3
    elif pz < upper:
        row = 2
    elif pz <= sz_top:
        row = 1
    else:
        row = 0  
    

    return ZONE_LOOKUP.get((row, col), 10)

# One hot for batter and pitcher hand
hand_map = {
    'R': 0,
    'L': 1,
}


#Map player ids to embedding ids
player_embedding_map = {}
pitching_embedding_map = {}

c1 = conn_pool.getconn()  
with c1.cursor() as cur:
    cur.execute("SELECT player_id, embedding_id, pitching_id FROM players")
    rows = cur.fetchall()
    if rows:
        for row in rows:
            player_id, embedding_id, pitching_id = row
            player_embedding_map[player_id] = embedding_id
            pitching_embedding_map[player_id] = pitching_id

conn_pool.putconn(c1) 


def populate_cold_start(game_id):
    conn = conn_pool.getconn() 
    with conn.cursor() as cursor:
        try:
            cursor.execute("""
                SELECT p.inning_num, p.inning_half_order, p.at_bat_num, p.pitch_num, g.datetime, ab.pitcher_id, ab.batter_id, bs.first, bs.second, bs.third, p.balls, p.strikes, p.outs, p.home_score, p.away_score, p.pitch_outcome_id, pd.pitch_type_id, pd.px, pd.pz, pd.sz_top, pd.sz_bot, ab.batter_hand, ab.pitcher_hand FROM pitches p 
                           JOIN at_bats ab ON p.game_id = ab.game_id AND p.inning_num = ab.inning_num AND p.inning_half = ab.inning_half AND p.at_bat_num = ab.at_bat_num 
                           JOIN games g ON p.game_id = g.game_id 
                           JOIN pitch_data pd ON p.pitch_data_id = pd.pitch_data_id 
                           JOIN base_states bs on p.game_id = bs.game_id AND p.inning_num = bs.inning_num AND p.inning_half = bs.inning_half AND p.at_bat_num = bs.at_bat_num AND p.pitch_num = bs.pitch_num WHERE p.game_id = %s""",(game_id,))
            rows = cursor.fetchall()
            if not rows:
                print(f"No data found for game_id {game_id}")
                return
            i = 0
            for row in rows:
                i += 1
                catcher_id = -1
                inning_num, inning_half, at_bat_num, pitch_num, game_date_time, pitcher_id, batter_id, runner_first, runner_second, runner_third, balls, strikes, outs, score_home, score_away, pitch_outcome_id, pitch_type_id, px, pz, sz_top, sz_bot, batter_hand, pitcher_hand = row
                b_hand = hand_map[batter_hand]  # no need for .get()
                p_hand = hand_map[pitcher_hand]
                pitch_outcome = pitch_outcome_id_map.get(pitch_outcome_id, 0)
                pitch_type = pitch_type_id_map.get(pitch_type_id, 0)
                zone = map_zone(px, pz, sz_top, sz_bot)

                pitcher_id = pitching_embedding_map.get(pitcher_id, -1)
                batter_id = player_embedding_map.get(batter_id, -1)
                catcher_id = player_embedding_map.get(catcher_id, -1)
                runner_first = player_embedding_map.get(runner_first, -1)
                runner_second = player_embedding_map.get(runner_second, -1)
                runner_third = player_embedding_map.get(runner_third, -1)

                if inning_half == 1:
                    score_diff = score_away - score_home
                    cursor.execute("""
                        SELECT pos_2 FROM defense_states where game_id = %s AND inning_num = %s AND inning_half = 'B' AND at_bat_num = %s AND pitch_num = %s AND home""",(game_id, inning_num, at_bat_num, pitch_num))
                    result = cursor.fetchone()
                    catcher_id = result[0] if result else catcher_id
                    catcher_id = player_embedding_map.get(catcher_id, -1)


                else:
                    score_diff = score_home - score_away
                    cursor.execute("""
                        SELECT pos_2 FROM defense_states where game_id = %s AND inning_num = %s AND inning_half = 'T' AND at_bat_num = %s AND pitch_num = %s AND not home""",(game_id, inning_num, at_bat_num, pitch_num))
                    result = cursor.fetchone()
                    catcher_id = result[0] if result else catcher_id
                    catcher_id = player_embedding_map.get(catcher_id, -1)

                try:
                    cursor.execute("""INSERT INTO cold_start (game_id, inning_num, inning_half, at_bat_num, pitch_num, game_date_time, pitcher_id, 
                                batter_id, catcher_id, runner_first, runner_second, runner_third, balls, strikes, outs, score, pitch_outcome_id, pitch_type_id, zone, pitcher_hand, batter_hand) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (game_id, inning_num, inning_half, at_bat_num, pitch_num) DO 
                                UPDATE SET pitcher_id = EXCLUDED.pitcher_id, batter_id = EXCLUDED.batter_id, catcher_id = EXCLUDED.catcher_id, runner_first = EXCLUDED.runner_first, runner_second = EXCLUDED.runner_second, runner_third = EXCLUDED.runner_third""",
                                (game_id, inning_num, inning_half, at_bat_num, pitch_num, game_date_time, pitcher_id, batter_id, catcher_id, runner_first, runner_second, runner_third, balls, strikes, outs, score_diff, pitch_outcome, pitch_type, zone, p_hand, b_hand))
                    conn.commit()
                except Exception as e:
                    print(f"[SKIP] Failed insert: game {game_id}, pitch {pitch_num} — {e}", file=sys.stderr)
                # if i % 50 == 0:
                #     conn.commit()

            print(f"Successfully populated cold_start for game_id {game_id}")
        except Exception as e:
            print(f"Error fetching data for game_id {game_id}: {e}", file=sys.stderr)
            conn.rollback()
            
        
    conn_pool.putconn(conn)  # Return the connection to the pool
# Reference so i can remember the table structure  
#cold_start(game_id integer, inning_num integer, inning_half integer, at_bat_num integer, pitch_num integer, game_date_time timestamp, pitcher_id integer, batter_id integer, catcher_id integer, runner_first integer, runner_second integer, runner_third integer, balls integer, strikes integer, outs integer, score integer, pitch_outcome_id integer, pitch_type_id integer, zone integer, pitcher_hand integer, batter_hand integer, primary key (game_id));





def get_game_ids():
    conn = conn_pool.getconn()  # Get a connection from the pool
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT game_id FROM games WHERE season_year BETWEEN 2015 and 2024 AND type_id != 1 AND level_id = 1")
        game_ids = cursor.fetchall()
        return [game[0] for game in game_ids]
    except Exception as e:
        print(f"Error fetching game IDs: {e}")
        return []
    finally:
        cursor.close()
        conn_pool.putconn(conn)  # Return the connection to the pool



game_ids = get_game_ids()

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = []
    for game_id in game_ids:
        futures.append(executor.submit(populate_cold_start, game_id))
    for future in as_completed(futures):
        try:
            future.result()  # This will raise an exception if the function raised one
        except Exception as e:
            print(f"Error in thread: {e}")


662872