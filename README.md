# titanicML
A logistic regression machine learning classification model created based off SoloLearn's Machine Learning curriculum and the Titanic data sets sourced from Kaggle.
The Python packages used in this project include NumPy, pandas, Matplotlib, and Scikit-learn.
The NumPy package is used to create arrays of data values and features a whole array of arithmetic operations and functions to perform on such data.
The pandas package allows for the cleaning and organization of data into DataFrame structures and series.
Matplotlib is a package that allows for visualization of the data sorted in pandas through a variety of different plot and graph structures.
Scikit-learn is a package that allows for the performance of machine learning models on the data.
In this project, the model chosen was the Logistic Regression model.
Learning was supervised learning based off Kaggle's titanic_train.csv data set.
The same project was replicated on Google Colaboratory, with the code attached.
Link to the Titanic data set on Kaggle: https://www.kaggle.com/competitions/titanic
Link to SoloLearn's Machine Learning curriculum: https://www.sololearn.com/learning/1094

Some things done in this project included calculating model metric scores, such as precision, recall, accuracy, and F1 scores.
Accuracy is equal to the sum of True Positives and True Negatives divided over the total size of the dataset.
Precision is equal to the number of True Positives over the total size of data points predicted to be positive (True Positives and False Positives) in the dataset.
Recall is equal to the number of True Positives over the total size of data points that are actually positive (True Positives and False Negatives) in the dataset.
F1 is equal to 2 times the precision times the recall all divided by the sum of precision and recall.

Before introducing the data to the Logistic Regression model however, the data set must be split into a training and a test data set.
Generally, the percentage allocated for the size of such data set is 80% for training and 20% for test.
Additionally, the data set can be split multiple times to create randomly different training and test data sets in a process called k-fold cross validation.
Such a process generates multiple models with varying accuracy scores for comparison.

A ROC curve can also be created to show a continuous line of all possible models created that compares each model's 1-specificity against its sensitivity.
Generally, the model is considered to be of good use when the ROC curve extends towards the top left corner and is well above the diagonal line starting from the origin and extending towards the upper right corner.
