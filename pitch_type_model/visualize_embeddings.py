import torch
import numpy as np
import psycopg2
from database.passwords import POSTGRES
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import json

def mask_to_rgb(mask):
    r, g, b = 0, 0, 0
    for pid in mask:
        if 0 <= pid <= 6:
            r += 1
        elif 7 <= pid <= 13:
            g += 1
        elif 14 <= pid <= 20:
            b += 1

    max_val = max(r, g, b, 1)  # Avoid divide-by-zero
    return (r / max_val, g / max_val, b / max_val)

checkpoint = torch.load("pitch_type_model/cold_start/checkpoints/cold_start_2023.pt")

pitcher_embeddings = checkpoint['model_state_dict']['pitcher_embed.weight'].cpu().numpy() 
valid_pitchers_2024 = []
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
end_path = os.path.join(BASE_DIR, 'embedding_analysis')
os.makedirs(end_path, exist_ok=True)
JSON_PATH = os.path.join(BASE_DIR, '..', 'constants', 'valid_pitch_map.json')
with open(JSON_PATH, 'r') as f:
    pitch_masks = json.load(f)
masks24 = pitch_masks["2024"]



conn = psycopg2.connect(
            user="postgres",
            password=POSTGRES,
            host="localhost",
            port="5432",
            database="mlb_data"
        )
with conn.cursor() as cur:
    cur.execute("""
              SELECT DISTINCT pitcher_id FROM cold_start WHERE game_date_time BETWEEN '2024-01-01' AND '2025-01-01';
        """)
    rows = cur.fetchall()
    if rows:
        valid_pitchers_2024 = [row[0]+1 for row in rows]
    else:
        valid_pitchers_2024 = []

conn.close()


    

filtered_ids = valid_pitchers_2024
colors = []
for pid in filtered_ids:
    str_id = str(pid-1)
    if str_id in masks24:
        rgb = mask_to_rgb(masks24[str_id])
    else:
        rgb = (0.7, 0.7, 0.7)  # fallback light gray
    colors.append(rgb)

filtered_embeddings = pitcher_embeddings[filtered_ids]
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced = tsne.fit_transform(filtered_embeddings) 

annotate_ids = {
    33607: "Skubal",
    22452: "Sale",
    41308: "Yamamoto",
    16750: 'Scherzer',
    26983: 'Snell',
    0: 'Position Players',
    33516: 'Burnes',
    27700: 'Fried',
    30801: 'Houck',
}

plt.figure(figsize=(12, 10))
plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.7, edgecolors='k', s=20)




for pid, name in annotate_ids.items():
    if pid in filtered_ids:
        idx = filtered_ids.index(pid)
        plt.text(reduced[idx, 0], reduced[idx, 1], name, fontsize=8, color='red')

plt.title("Pitcher Embedding Space (TSNE of 2023 Weights, 2024 Pitchers)")
plt.text(0.99, 0.01, "Color = Pitch Arsenal Composition", transform=plt.gca().transAxes,
         fontsize=8, ha='right', va='bottom', color='gray')
plt.grid(True)
plt.savefig(os.path.join(end_path, "2024pitcher_embeddings.png"))
plt.show()