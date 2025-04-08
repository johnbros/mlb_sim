# Baseline Embedding Models

This directory contains player embedding models trained on pre-Statcast era data (2005–2014). These embeddings are frozen and used as initialization for players in downstream simulation models (2015 onward).

## Models Included

- **hitters_model/**: Embeddings based on pitch and at-bat outcomes
- **pitchers_model/**: Embeddings based on pitch outcomes and performance
- **baserunning_model/**: Embeddings from basepath outcomes and stolen base attempts
- **fielding_model/**: Embeddings based on defensive plays and fielding stats

Each subfolder contains:

- train.py: Script to train the embedding model
- config.yaml: Config file with hyperparameters and training options

## Notes

These models are not part of the real-time simulation pipeline but are crucial for initializing players with realistic profiles.
