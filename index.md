# FIFAI

### By Eshaan Lumba, Kenneth Ochieng, Jett Bronstein and Aidan Garton

## Introduction Draft 1

The English Premier League is considered one of the most exciting soccer leagues in all of the world making Premier League games some of the most watched cable events in the world. Like other mainstream sporting leagues, the Premier League attracts significant attention from the gaming industry. Currently, predicting the victor of a Premier League match is impossible...as of now. We, as a group, will construct a comprehensive neural network to successfully predict the result of Premier League matches on a consistent basis.

Soccer is regarded as one of the most unpredictable sports globally, so building a Neural Network model to accurately predict the results of English Premier League soccer matches is a challenging, yet exciting task. 

<!--There has been some success with predicting the results of soccer matches in the past using recurrent Neural Networks, but we aim to improve on this accuracy and take it a step further by predicting scorelines.-->

To improve upon the previous [results](https://link.springer.com/chapter/10.1007/978-981-15-9509-7_57), we will try to use a more robust dataset with more pre-game and post-game datapoints such as expected goals for and against along with a more optimized algorithm. Another key modification is that their results are outdated and do not continue to learn, whereas we want to keep our model as up to date as possible. This might mean reading data in from the last 3-4 years instead of the last 7-8 years. 

Ideally, we would like to have an algorithm that can accurately predict the result upwards of 70% of the time. A key challenge is understanding exactly what datapoints matter the most and what matter the least. This is difficult to predict beforehand and trial and error with different sets of datapoints might lead us to improved algorithms. 

We acknowledge that predicting the exact scoreline of a Premier League match is a very difficult problem, but we hope to succeed and improve on previous results. Ultimately, we would like to test our data on the current set of ongoing premier league matches to determine the success rate of our algorithm. A lot of factors can change as the form of teams changes throughout the season, but we are confident that we can build a strong model. 

If we are able to build a successful model, we realise that it might lead to an increase in the amount of bets made on Premier League matches, leading to those consequences faced by excessive gambling. For one, the number of bets might drastically increase with people taking higher risks. Clearly, if people are aware of the result of the game from beforehand for about 70% of the time, there is a lot of money to be made. Another issue that might surface is a lack of enjoyment in watching football. If people know the result beforehand, would they still enjoy watching the game as much? These are some questions and consequences we need to consider if we end up building a successful model. 

## Related Works Draft 1

Due to the high demand and market for sports betting and the surrounding fan anticipation/engagement, there have been several previous studies on the prediction of sporting events using neural networks. While much of the work produced to predict soccer outcomes has been adapted from the prediction of other sporting events, such as those of NBA and NFL games, the following papers have provided essential and extensive background knowledge on our topic for soccer matches, some of which focus entirely on the English Premier League, just as we do in our study. Here, we outline the main contributions of these works and explain the nuances and points of expansion that our study focuses on. 

The primary source and motivation for our project comes from a study done by Sarika, et al. titled [“Soccer Result Prediction Using Deep Learning and Neural Networks”](https://link.springer.com/chapter/10.1007/978-981-15-9509-7_57).  This study compares several different prior sport outcome predictors and constructs their own in the form of an RNN (recurrent neural network). Using data from the 2011-2017 seasons of the EPL, Sarika and his colleagues trained their model to accurately predict EPL match outcomes up to 80% of the time. Sarika and his team used LSTM (long, short-term memory) cells composed of three gates: forget gates, input gates, and output gates. We use this architecture as our main source of motivation to produce an even higher accuracy model for EPL game outcomes.

Another key piece of research for our project comes from Wagenaar et al. in their paper [“Using Deep Convolutional Neural Networks to Predict Goal-scoring Opportunities in Soccer”](https://www.ai.rug.nl/~mrolarik/Publications/ICPRAM_2017_67.pdf). This paper uses two-dimensional image data to predict the chance of a goal-scoring opportunity based on the location of the ball and players relative to each other and each team’s goal. By using a CNN architecture, Wagenaar and his team were able to accurately predict the chance of a goal being scored given a snapshot of the field 67.1% of the time. Notably, this paper uses training data from the Bundesliga, but it still provides pertinent and helpful information on how snapshots of game positions might aid the prediction of overall soccer match score-lines and outcomes as we aim to improve upon with our study.

## Literature Review

### 1. Soccer Result Prediction Using Deep Learning and Neural Networks
This paper compares several different score/outcome prediction strategies for various sports and leagues (including the NBA and soccer matches) and proposes an RNN (recuring neural network) architecture to predict match outcomes in the English Premier League. Taking as parameters a team's statistics from past seasons (and updating data by adding current games of a season) the RNN constructed in this paper ended up with around an 80% accuracy prediction of match outcomes. Some parameters included were points gained by the home vs. away team (1 for draw, 3 for win, 0 for loss), goals scored by the home vs. away team, winning and losing streak of the home vs. away team (a streak of >3 and >5 were tracked). This paper concludes that RNNs provide a better architecture for the problem of score prediction over other architectures and provide measurable evidence for such in soccer matches in the EPL. 

### 2. Using Deep Convolutional Neural Networks to Predict Goal-scoring Opportunities in Soccer
This paper trained a NN on Bundesliga game data to predict the chance of a goal scoring opportunity based on a given position of the game's players and the game ball. Using 2-dimensional data gathered from the Amisco multi-camera system, this network showed as a high as a 67% accuracy rate for predicting goals. This paper coul be helpful if we wanted to analyze past scoring chances, positional data, and etc... from past games played by a team in order to provide a prediction of current and future game outcomes.

### 3. Predict soccer matches with 50% accuracy
In the article, the author used the Poisson Distribution and Poisson Regression through Python to predict the results of the last 10 matches of the 2018-2019 Premier League season. It was quite clear to see that for that season at least, the Poisson distribution very clearly matched the results of the whole season. The key data points he used were, for a certain team, the likelihood that they would score, the likelihood that they would concede and the likelihoods that they would score if they were a home or away team. It resulted in a success rate of almost 50%. The author also suggested using more data points such as form, more matches and correctness of under/over underestimating results.

### 4. Football Match Statistics Prediction using Artificial Neural Networks
This paper discusses a soccer match statistics prediction NN framwork. The NN is trained with data from two Bundesliga teams (Bayern Munich and Borussia Dortmound). They analyze various input factors selected to help the Neural Net detect patterns and predict match statisitics such as the winning team, goals scored, and bookings. Their data shows that the Neural Net is able to predict the match statistics with a higher accuracy than an existing if-else case framework (Football Result Expert System (FRES) used to predict American soccer. According to their analysis, there are several key factors which affect the outcome of matches and a NN can be trained to learn from previous history to predict future outcomes. These factors include: transfer money spent by a team in a season, a team's UEFA co-effecient (for each team participating in the champions league), year of match, league rank, goals scored and conceded,  wins and losses , home advantage etc. They advice that each of these factors have certain limitations in their application and some teams generally have more data than others because of their length of participation in competitions. A few of these factors can also only be applied within a league in a single country. This paper could be useful to help guide our choices when considering the type of data we need for teams and also the labels we need to train a NN. In addition, we can learn from their backpropagation algorithm and normalization of data points for further improvement. 

## Project Update 1

### Software
We will use a combination of [PyTorch](https://pytorch.org/) and [FastAI](https://docs.fast.ai/). 

### Our Dataset
We will use a combination of ["English Premier League stats 2019-2020"](https://www.kaggle.com/idoyo92/epl-stats-20192020?select=epl2020.csv), ["2021-2022 Premier League Stats"](https://fbref.com/en/comps/9/Premier-League-Stats) and ["English Premier League (football)"](https://datahub.io/sports-data/english-premier-league#readme).

### Overview of project
We want to use a recurrent neural network along with long short-term memory since we want to keep track of patterns and form. Our inputs will be a vector of floating point values (expected goals for and expected goals against for the home team for the particular game). We will be performing classification. Our algorithm will predict the most probable scoreline out of a multitude of options. Hence, our output will be a vector of floating point values that hold the probablility of the match ending in a given scoreline. 




