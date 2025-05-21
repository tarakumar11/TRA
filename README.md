# TRA
Ticket Routing Automation development 
 ---------------------------------------------------

# Data Preprocessing
The raw dataset (***aa_dataset-tickets-multi-lang-5-2-50-version.csv***) is preprocessed using ***preprocessing.py*** script

Save the cleaned dataset as ***preprocessed_data.csv*** in the ***Data/*** directory.

# First Approach (***idependent_model.py***)
Two separate models:
Model 1: reads the email and guesses Category
Model 2: reads the same email and guesses Sub Category
But: Model 2 doesn’t know what Model 1 guessed

# second approach (***combined_label_model.py***)
Combine Category and Sub Category into one label and train one single model to predict that joint label.
 
# third approach (***heirarchical_model.py***)
Step 1: Predict Category.
Step 2: Based on predicted Category, pass the text to a Category-specific Sub Category classifier.

# Conclusion

The first approach (***independent_model.py***) gave the best results overall.
Among all models, ***Random Forest*** performed the best for both Category and Sub-Category prediction.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Model Ensembles (***independent_model_ensemble_train.py***) 

Using VotingClassifier for both Category and Sub-Category predictions improved robustness and performance over individual models.


# Class weighted training (***idependent_model_class_weighted.py***)
To avoid classes imbalance