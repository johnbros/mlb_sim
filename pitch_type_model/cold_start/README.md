# Cold Start Pitch Type Model

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

## Additional Notes and Next Steps

- This first model is a culmination of about 70 days of data collection, data validation, and work combined with the time I have spent in my classes. It was difficult, I struggled, I took days off, and maybe theres someone out there who could've done this faster or more effectively. To them I say do it because right now typing this readme I feel incredibly rewarded in knowing that I built something from the ground up with the skills i've learned throughout my time as an undergrad.
- Regarding what's next, I think probably location... the dataset will be similar and I have collected the weather data at the sub quarter hourly level to line up with the pitches so location could definitely be a valuable asset. Maybe something with pose and fatigue analysis but we will have to see what I decide.
- To anyone reading this i'm excited for this journey and I hope that the people after me will be able to do things like this better than I ever had the ability or knowledge to attempt.
