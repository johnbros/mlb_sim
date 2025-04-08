# Pitch Type Prediction Model

## Overview

Predicts the next pitch type based on:

- Pitch history
- Count, inning, game state
- Pitcher and batter profiles

## Inputs

- Encoded game state features
- Historical pitch context
- Pitcher/batter identities

## Outputs

- Probability distribution over pitch types
- Sampled pitch type

## Key Files

- train.py
- inference.py
- config.yaml

## Notes

- Training data excluded.
- Outputs feed into the pitch characteristics model.
