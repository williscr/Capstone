# Capstone

MEDIUM DELIVERABLE : https://medium.com/@clarewillis.79/binary-classification-learning-by-doing-9c53c7c32ac2

**Table of Contents :**

Installation

Project Motivation

File Descriptions

Results

Licensing, Authors, and Acknowledgements

**Installation :**

The neccessary libraries required to run the code are :

numpy, pandas , matplotlib, seaborn, dython, sklearn, lightgbm, xgboost, optuna

The code should run with no issues using Python versions 3.*.

**Project Motivation :**

For this project, I was interestested in using an insurance dataset to preict customers interest in a new policy. 

Study Aim : To predict a binary classifier utilising Vehicle insurance to predict if existing customers will be interested in the health insurance offering.
What neighbourhood in seattle is the cheapest to rent a listing ?

**File Descriptions :**

EDA : EDA.ipynb 
- This files contains all the Exploratory Data analysis of the study. Including graphs, descriptive statistics and correlation calculations. 

Pre_Processing : Preprocessing.ipynb 
- Preprocesses the data using a sklearn pipeline object so that 

Models: lgbm.ipynb , xgboost.ipynb, LogisticRegression.ipynb 
- Files that contain the code for training the models using the optuna hyper-parameter framework. 

main.ipynb 
- File that runs the preprocessing and model notebooks. Outputs the results of the models, with confusion matricies and the F1 score. 

**Run** 
To Run the EDA and produce graphics - run EDA.ipynb 
To Run the models and produce model results - run main.ipynb

**Results**

The main findings of the code can be found at the post available at :https://medium.com/@clarewillis.79/binary-classification-learning-by-doing-9c53c7c32ac2

The results are also found by running main.ipynb.

All models had high performance on the test set with f1 scores of 0.82 or 0.83 for all models. 

Licensing, Authors, Acknowledgements

Must give credit to Kaggle for the data. You can find the Licensing for the data and other descriptive information at : Data Link : https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

**REFERENCES**

Articles : 
https://medium.com/analytics-vidhya/linear-regression-and-random-forest-33d4297a186a
https://medium.com/optuna/an-introduction-to-the-implementation-of-optuna-a-hyperparameter-optimization-framework-33995d9ec354
https://towardsdatascience.com/how-to-make-your-model-awesome-with-optuna-b56d490368af
https://towardsdatascience.com/exploring-optuna-a-hyper-parameter-framework-using-logistic-regression-84bd622cd3a5
https://towardsdatascience.com/optuna-a-flexible-efficient-and-scalable-hyperparameter-optimization-framework-d26bc7a23fff
https://towardsdatascience.com/cross-validation-430d9a5fee22
https://towardsdatascience.com/the-f1-score-bec2bbc38aa6
https://sebastianraschka.com/faq/docs/computing-the-f1-score.html
https://towardsdatascience.com/metrics-for-imbalanced-classification-41c71549bbb5
https://medium.com/flutter-community/dealing-with-categorical-features-with-high-cardinality-feature-hashing-7c406ff867cb
https://medium.com/jungletronics/pandas-one-hot-encoding-ohe-eb7467dc92e8
https://medium.com/mlearning-ai/neat-data-preprocessing-with-pipeline-and-columntransformer-2a0468865b6b
https://towardsdatascience.com/how-to-use-sklearn-pipelines-for-ridiculously-neat-code-a61ab66ca90d
https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
https://towardsdatascience.com/normalization-standardization-and-normal-distribution-bfbe14e12df0

Kaggle Notebooks : 
https://www.kaggle.com/code/yashvi/vehicle-insurance-eda-and-boosting-models
https://www.kaggle.com/code/songulerdem/health-insurance-cross-sell-prediction-xgboost
https://www.kaggle.com/code/anmolkumar/vehicle-insurance-eda-lgbm-vs-catboost-85-83
https://github.com/nihalhabeeb/Health_Insurance_Cross_Sell_Prediction/blob/main/Health_Insurance_Cross_Sell_Prediction_Capstone_Project.ipynb
