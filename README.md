This workflow covers the essential steps of a supervised machine learning process: data preparation, model training, and prediction.


Import Libraries: Loads pandas for data handling and scikit-learn's LogisticRegression for modeling.
Create Dataset: Defines a small sample dataset with iris flower measurements and species, then converts it to a pandas DataFrame.
Save and Display Data: Saves the dataset to a CSV file and displays it in the notebook.
Prepare Features and Labels: Splits the data into features (SepalWidth, PetalLength, PetalWidth) and labels (Species).
View Features: Shows the first few rows of the features for inspection.
Create Model: Initializes a logistic regression model.
Train Model: Fits the model to the features and labels.
Predict: Uses the trained model to predict the species of new iris samples based on their measurements.
Show Predictions: Prints the predicted species.

