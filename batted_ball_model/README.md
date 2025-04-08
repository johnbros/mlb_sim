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
- 	rain.py
- inference.py
- config.yaml
"@

    "base_running_model" = @"
# Base Running Model

## Overview

Predicts base state transitions:
- Runner advances
- Outs on basepaths
- Stolen base attempts

## Inputs
- Batted ball outcome
- Current base state
- Batter & runner profiles

## Outputs
- Updated base state
- Scoring / out events

## Key Files
- model.py
- logic.py
- config.yaml

## Notes
- Integrates with the simulation to maintain game state.
