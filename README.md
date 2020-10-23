# MedTourEasy_blood_prediction<br>
## ProjectDescription <br>
Task1: <br>
Inspecting the file that contains the dataset. <br>
•Print out the first 5 lines from datasets/transfusion.data <br>
Task2: <br>
Load the dataset. <br>
•Import the pandas library.<br>
•Load the transfusion.data file from datasets/transfusion.data and assign it tothe transfusion variable.<br>
•Display the first rows of the DataFrame with the head() method to verify the file was loadedcorrectly.<br>
If you print the first few rows of data, you should see a table with only 5 columns.<br>
Task3:<br>
Inspect the DataFrame's structure.<br>
•Print a concise summary of the transfusion DataFrame with the info() method.<br>
DataFrame's info() method prints some useful information about a DataFrame:•index type•column types•non-null values•memory usage<br>
including the index dtype and column dtypes, non-null values and memoryusage.
Task4:<br>
Rename a column.<br>
•Rename whether he/she donated blood in March 2007 to target for brevity.<br>
•Print the first 2 rows of the DataFrame with the head() method to verify the change was done correctly.<br>
By setting the inplace parameter of the rename() method to True, the transfusion DataFrame is changed in-place,<br>
i.e., the transfusion variable will now point to the updated DataFrame asyou'll verify by printing the first 2 rows.<br>
Task5:<br>
Print target incidence.<br>
•Use value_counts() method on transfusion.target column to print target incidence proportions, setting normalize=True and rounding the output to 3 decimal places.<br>
By default, value_counts() method returns counts of unique values. Bysetting normalize=True, the value_counts() will return the relative frequencies of the uniquevalues instead.<br>
Task6:<br>
Split the transfusion DataFrame into train and test datasets.<br>
•Import train_test_split from sklearn.model_selection module.<br>
•Split transfusion into X_train, X_test, y_train and y_test datasets, stratifying onthe target column.<br>
•Print the first 2 rows of the X_train DataFrame with the head() method.<br>
Writing the code to split the data into the 4 datasets needed would require a lot of work. <br>
Instead, use the train_test_split() method in the scikit-learn library.<br>
Task7:<br>
Use the TPOT library to find the best machine learning pipeline.<br>
•Import TPOTClassifier from tpot and roc_auc_score from sklearn.metrics.<br>
•Create an instance of TPOTClassifier and assign it to tpot variable.<br>
•Print tpot_auc_score, rounding it to 4 decimal places.<br>
•Print idx and transform in the for-loop to display the pipeline steps.<br>
Adapt the classification example from the TPOT's documentation. <br>
In particular, specify scoring='roc_auc' because this is the metric that you want to optimize for andadd random_state=42 for reproducibility. <br>
Use TPOT lightconfiguration with only fastmodels and preprocessors.<br>
The nice thing about TPOT is that it has the same API as scikit-learn, i.e., you first instantiate a model and then train it, using the fit method.<br>
Data pre-processing affects the model's performance, and tpot's fitted_pipeline_ attribute willallow to see what pre-processing (if any) was done in the best pipeline.<br>
Task8:<br>
Check the variance.<br>
•Print X_train's variance using var() method and round it to 3 decimal places.<br>
pandas.DataFrame.var() method returns column-wise variance of a DataFrame, which makescomparing the variance across the features in X_train simple and straightforward.<br>
Task9:<br>
Correct for high variance.<br>
•Copy X_train and X_test into X_train_normed and X_test_normed respectively.<br>
•Assign the column name (a string) that has the highest varianceto col_to_normalize variable.<br>
•For X_train and X_test DataFrames:<br>
•Log normalize col_to_normalize to add it to the DataFrame.<br>
•Drop col_to_normalize.<br>
•Print X_train_normed variance using var() method and round it to 3 decimal places.<br>
X_train and X_test must have the same structure. <br>
To keep your code "DRY" (Don't RepeatYourself), you are using a for-loop to apply the same set of transformations to each of theDataFrames.<br>
Normally, you'll do pre-processing before you split the data (it could be one of the steps in machinelearning pipeline). <br>
Here, you are testing various ideas with the goal to improve model performance,and therefore this approach is fine.<br>
Task 10: <br>
Train the logistic regression model.<br>
•Import linear_model from sklearn.<br>
•Create an instance of linear_model.<br>
LogisticRegression and assign it to logreg variable.<br>
•Train logreg model using the fit() method.<br>
•Print logreg_auc_score.<br>
The scikit-learn library has a consistent API when it comes to fitting a model:<br>
1.Create an instance of a model you want to train.<br>
2.Train it on your train datasets using the fit method.<br>
You may recognise this pattern from when you trained TPOT model. <br>
This is the beauty ofthe scikit-learn library: you can quickly try out different models with only a few code changes.<br>
Task 11: <br>
Sort your models based on their AUC score from highest to lowest.<br>
•Import itemgetter from operator module.<br>
•Sort the list of (model_name, model_score) pairs from highest to lowestusing reverse=True parameter.
