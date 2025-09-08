# Cold Start Pitch Type Model V1

## Input Features

- Embedding of state information
  - Players impacting pitcher decision(Pitcher, Catcher, Batter, Base Runners)
  - Zone, Pitch type, and Pitch outcome of previous pitch
  - Count, Outs, and Score
  - Game situation information(Inning number, Bottom/Top, At bat number within inning, Pitch number within at bat)

## Model Description

- First iteration tackling this problem is an LSTM
  - This was chosen for a variety of reasons but mainly because of the inherently short sequences built into pitcher appearances. Starters throw about 100 pitches max a game and represent about 66% of the dataset given that on average teams throw around 150 pitches a game. There is concern about vanishing gradient on long starter sequences but I am going to analyze the impact and make a decision based on that.
  - I may consider a transformer in the future but from my reading LSTM performance seems acceptable as a baseline and offers good potential.
- The model relies on many embedded features to model who and what is impacting the decision making at that given moment.
  - Omitted features that I may attempt to implement in future iterations include but may not be limited to weather, pitching coach/manager, home/away, and fatigue.
- As my first iteration I will try to not judge the model too harshly and look at it from a perspective to where I can improve performance.
- The methodology used when designing the data was that I would only use data that was available at time T as input features. That means where a pitch goes, what the outcome of the pitch is(regarding the batter interaction), and obviously what type of pitch it was is only available up to time T-1, with those T-1 values being used directly at input features and the rest getting carried forward in the hidden state as learned by the model.

## Run Analysis

### Run 1

- First time through I masked out pitch types that a pitcher didn't throw, the idea was to not punish the model for predicting a pitch that wasn't actually possible to predict. This worked ok in most situations but the main issue was that an infinite loss occasionallly came from this mask say for example position player A comes in for the first time ever and pitches in a blowout and only throws eephus pitches. We've never seen this player before the embedding is randomly initialized and hasn't been tuned at all so he may accidentally be close in the embedding space to a typical pitcher the model comes up with an output that says Fastball/Sinker/Slider with 0 probability of Eephus but Eephus is the only valid pitch accordig to the mask I was applying, this made it so that our log softmax output actually predicted all pitches with 0% chance giving us infinite loss because there was a pitch thrown.
  - **Potential Solutions**
  - I am considering is no masking - model will converge slower but can start to explore novel situations and then we can mask in actual usage to get a understanding of valid outcomes.
  - Another option is partial masking where instead of completely invalidating known impossible outcomes we just lower their probabilities by some factor so that the true actual outcomes come through at higher probabilities. This offers a middleground between not invalidating known invalid solutions and completely invalidating them.
  - Another potential solution is to converge known position players to a single embedding in the pitcher space that way due to similarities across position players pitching will be learned and our sample size for these rare occurrences becomes cumulative allowing the model to learn their tendencies in a more densely sampled space.
- I'm writing this during the run due to the length so i'm jotting down things I notice as I notice them and potential issue number 2 would be that we got a large validation accuracy drop after training on the 2016 data... this potentially suggests overfitting to training data but it could also be a side effect of the mask that we applied because the infinite loss would also show up in validation as us predicting no pitches meaning a guaranteed miss. I think that may actually be due to the fact that statcast started to give a broader understanding of sweeper and slurve in 2017 but in 2016 the lower frequency meant we weren't able to generalize well to 2017's new information.
  - **Potential Solutions**
  - Same as earlier potentially change the way we construct the mask.
  - Maybe in season validation some sort of split like train first 90% validate on 10% at the end or maybe intermittently validate like train on April validate first week of May train rest of May validate first week of June etc.
- Calling this run quits after 2019, it was "performing" decently initially but started to drop in performance since it started to overfit to fastballs i'm looking at some solutions and may run this a second time before moving towards a pre trained embedding method.

### Run 2

- Adjustments I made since run 1 were simple, during training I eased the harshness of the mask so that invalid pitch types were still allowed exploration in training and trusted that the model would be able to figure out distributions and it solved the infinite loss problem and the model did become better calibrated as seen in the all time analysis plots. We were still overconfident on fastballs in the higher probability bins but plotting the bin sizes overlaid with the calibration curve helped me better see that we are tapering off across all pitches and not often getting over confident and even in high confidence spots actually not too far miscalibrated(peak difference is about 20% off)
- After analysis of the ECE(Expected Calibration Error) and calibration plots I experimented with softmax temperature scaling I was able to achieve a ECE average across all classes of 0.0265 or 2.65% this showcases a very promising result for future iterations this will help me decide best ways to adress issues.
- The model achieved a top 1 accuracy of 42.4% in my rolling test approach. The rolling test included training the model up to year Y-1 then testing it on year Y before then continuing to train the model. For more clarity the method was train on 2015 test on 2016 then continue training on 2016 and so on. This made it so information was never leaked to the model in training or testing since we did not modify the model parameters, weights, or embeddings during testing splits. This gave us a 6 million pitch test space and a 6 million pitch training space(no point in training on 2024 or testing on 2015). So accross the test set the average accuracy was 42.4% wich would beat a naive approach of predicting fastballs only by around 8-9% on average across those same years. Top 2 accuracy is where we saw a significant jump with an average of accuracy of 69.28% beating naive approach of predicting the top 2 most frequent pitches by nearly 20%.

## Additional Notes and Next Steps

- This first model is a culmination of about 70 days of data collection, data validation, and work combined with the time I have spent in my classes. It was difficult, I struggled, I took days off, and maybe theres someone out there who could've done this faster or more effectively. To them I say do it because right now typing this readme I feel incredibly rewarded in knowing that I built something from the ground up with the skills i've learned throughout my time as an undergrad.
- Regarding what's next, I think probably location... the dataset will be similar and I have collected the weather data at the sub quarter hourly level to line up with the pitches so location could definitely be a valuable asset. Maybe something with pose and fatigue analysis but we will have to see what I decide and have time for.
