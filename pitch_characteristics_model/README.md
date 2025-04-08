# Pitch Characteristics Model

## Overview

Predicts physical pitch attributes based on:

- Pitch type (sampled from previous model)
- Game context
- Pitcher profile

## Possible Outputs

- Velocity
- Spin rate
- Horizontal/vertical break
- Release extension

## Key Files

- train.py
- inference.py
- config.yaml

## Notes

- Outputs are fed into the pitch outcome model.
