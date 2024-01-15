The dataset is from Association of Tennis Professionals (ATP), the men’s professional tennis league.
The data is for the top 1500 ranked players in the ATP over the span of 2009 to 2017.
The statistics recorded for each player in each year include service game (offensive) statistics, return game (defensive) statistics and outcomes.

Goals:
- create a linear regression model that predicts the outcome for a tennis player based on their playing habits 
- determine what it takes to be one of the best tennis players in the world
- split data into training and test datasets
- evaluate your model on the test set

The ATP men’s tennis dataset includes a wide array of tennis statistics, which are described below:

Identifying Data
- Player: name of the tennis player
- Year: year data was recorded

Service Game Columns (Offensive)
- Aces: number of serves by the player where the receiver does not touch the ball
- DoubleFaults: number of times player missed both first and second serve attempts
- FirstServe: % of first-serve attempts made
- FirstServePointsWon: % of first-serve attempt points won by the player
- SecondServePointsWon: % of second-serve attempt points won by the player
- BreakPointsFaced: number of times where the receiver could have won service game of the player
- BreakPointsSaved: % of the time the player was able to stop the receiver from winning service game when they had the chance
- ServiceGamesPlayed: total number of games where the player served
- ServiceGamesWon: total number of games where the player served and won
- TotalServicePointsWon: % of points in games where the player served that they won

Return Game Columns (Defensive)
- FirstServeReturnPointsWon: % of opponents first-serve points the player was able to win
- SecondServeReturnPointsWon: % of opponents second-serve points the player was able to win
- BreakPointsOpportunities: number of times where the player could have won the service game of the opponent
- BreakPointsConverted: % of the time the player was able to win their opponent’s service game when they had the chance
- ReturnGamesPlayed: total number of games where the player’s opponent served
- ReturnGamesWon: total number of games where the player’s opponent served and the player won
- ReturnPointsWon: total number of points where the player’s opponent served and the player won
- TotalPointsWon: % of points won by the player

Outcomes
- Wins: number of matches won in a year
- Losses: number of matches lost in a year
- Winnings: total winnings in USD($) in a year
- Ranking: ranking at the end of year