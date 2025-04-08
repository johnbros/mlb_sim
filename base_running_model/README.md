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
