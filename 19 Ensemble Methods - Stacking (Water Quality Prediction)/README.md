Ensemble Methods - Stacking (Water Quality Prediction)

The water quality dataset is from Kaggle: https://www.kaggle.com/datasets/adityakadiwal/water-potability.
The features of this dataset are ones that can be used to determine how safe a supply of water is to drink or 
its “potability” represented by a 1 for drinkable and a 0 for not drinkable.

Goals:
- Use Logistic Regression model and a Random Forest model as our base estimators.
- Expand training dataset with new features provided by the predictions of the trained base estimators.
- Use Random Forest as the final estimator.
- Use Stratified K-Folds cross-validator as a splitting strategy used to train the final estimator.
- Use Stacking classifier to combine all above and make the final predictions.