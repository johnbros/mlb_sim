# Batted Ball Model

## Overview

Predicts the result of contact with the ball:

- Launch angle
- Exit velocity
- Hit location
- Barrel quality

## Inputs

- Contact event data from pitch outcome model
- Pitch and batter info

## Outputs

- Hit type: grounder, line drive, fly ball
- Initial ball-in-play characteristics

## Key Files

- train.py
- inference.py
- config.yaml
