# MLB Simulation Project

This repository contains a modular, end-to-end simulation pipeline that models MLB pitch sequences, outcomes, and game progression using machine learning.

## Modules

- **pitch_type_model**: Predicts the next pitch type.
- **pitch_characteristics_model**: Predicts pitch movement/speed given type.
- **pitch_outcome_model**: Determines swing/no-swing, contact type.
- **at_bat_outcome_model**: Aggregates pitch results into at-bat outcome.
- **batted_ball_model**: Determines hit type from ball-in-play data.
- **base_running_model**: Updates base state from hits and runners.

## Project Status

Currently in development. Data and model weights excluded. Will be public once finalized.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

You may remix, adapt, and build upon this work **non-commercially**, as long as you credit the original author and license your new creations under identical terms.
