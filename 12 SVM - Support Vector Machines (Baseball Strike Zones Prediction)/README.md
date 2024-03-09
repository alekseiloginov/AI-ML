SVM - Support Vector Machines (Baseball Strike Zones Prediction)

A baseball dataset is compiled using the pybaseball Python package: https://github.com/jldbc/pybaseball.
It's related to baseball stars: Aaron Judge - one of the tallest players in the league, Jose Altuve - one of the shortest, and David Ortiz. 
Their strike zones should be pretty different. Each row corresponds to a single pitch that the batter saw in the 2017 season.

A picture attached shows a strict definition of the strike zone.

Goals:
- Use an SVM to find the decision boundary of the strike zone that determines whether or not a pitch is a strike or a ball.
- Use our knowledge of SVMs to find the real strike zone of several baseball players.