# Pitch Type Prediction Models

## Overview

Predicts the next pitch type based on a variety of factors

- Pitch history
- Count, inning, game state
- Pitcher and batter profiles
- Weather

## Inputs

- Depends on the iteration see each sub model for specifics.

## Outputs

- Probability distribution over pitch types

## Notes and Disclaimers

- Training and test data is derived from reconstructed game states using Retrosheet, Statcast, and stadium-proximate weather station data.
  
- Total volume: 6,818,628 pitches across 23,155 MLB games (2015–2024).
  
- While 100% fidelity cannot be guaranteed for every individual game-state moment, I performed extensive validation on my state:
  
  - The scores produced from game reconstruction using retrosheet event logs line up with statcasts final scores.

  - Manual verification of ~250 games confirmed base state accuracy, correct out counts, and precise score state before each pitch.
  
- Weather mapping methodology:

  - For each of the 55 stadiums used in MLB games between 2015 and 2024, I manually retrieved the latitude, longitude, and field angle relative to true north (angle between due north and the line from home plate to the pitcher's mound).

  - Due to gaps in historical weather granularity, weather at the time of pitch was mapped to the nearest quarter-hour report (e.g., a pitch at 8:10 PM uses the 8:15 PM weather). This was considered acceptable noise.

- Pitch timestamp interpolation:

  - No game in the Statcast era is missing timestamps for every pitch.

  - When timestamps were missing for a subset of pitches, I interpolated based on:

    - Known pitch timing patterns across innings

    - Distribution of known timestamps within the game

    - Resume times for suspended games

- Suspended games:

  - 4 games involved mid-game relocation. These represent <1,000 pitches. After evaluation I decided that it was acceptable to treat these as if they had remainined in the original location for simplicity. I may revisit this later but I believe the impact to be minimal so only if I have time.

  - For other suspended games, timestamp gaps were filled using the same interpolation strategy. In every case I validated, resumed timestamps aligned with the actual resume date and time.
