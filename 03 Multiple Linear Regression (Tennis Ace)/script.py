import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data:
tennis_stats = pd.read_csv('tennis_stats.csv')
print(tennis_stats.columns)
print(tennis_stats.info())

# perform exploratory analysis:
features = tennis_stats["BreakPointsOpportunities"].values.reshape(-1, 1)
outcome = tennis_stats["Winnings"]

plt.plot(features, outcome, 'o', alpha=0.1)
# plt.show()

## perform single feature linear regressions:
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

regr = LinearRegression()
regr.fit(features_train, outcome_train)
outcome_predict = regr.predict(features_train)

plt.plot(features_train, outcome_predict)
plt.show()
plt.close()

# evaluate model
model_score = regr.score(features_test, outcome_test)
print(model_score)

outcome_test_predict = regr.predict(features_test)

plt.scatter(outcome_test_predict, outcome_test, alpha=0.4)
plt.show()

# perform another single feature linear regressions:
features = tennis_stats["Aces"].values.reshape(-1, 1)
outcome = tennis_stats["Winnings"]

plt.plot(features, outcome, 'o', alpha=0.1)
# plt.show()

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

regr.fit(features_train, outcome_train)
outcome_predict = regr.predict(features_train)

plt.plot(features_train, outcome_predict)
plt.show()
plt.close()

model_score = regr.score(features_test, outcome_test)
print(model_score)

## perform two feature linear regressions:
features = tennis_stats[['BreakPointsOpportunities',
                         'Aces']]
outcome = tennis_stats[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

regr.fit(features_train, outcome_train)
outcome_predict = regr.predict(features_train)

model_score = regr.score(features_test, outcome_test)
print(model_score)

## perform multiple feature linear regressions:
features = tennis_stats[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
                         'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
                         'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
                         'BreakPointsSaved','ReturnGamesPlayed','ReturnGamesWon',
                         'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
                         'TotalServicePointsWon']]
outcome = tennis_stats[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

regr.fit(features_train, outcome_train)
outcome_predict = regr.predict(features_train)

model_score = regr.score(features_test, outcome_test)
print(model_score)
