# Titanic Survival Prediction

This is the popular [Kaggle project](https://www.kaggle.com/competitions/titanic/overview) that aims to predict the survival of passengers on the Titanic using machine learning techniques. The dataset includes various features such as passenger demographics and ticket information, which are processed and analyzed to create predictive models.


<img src="titanic.jpg" alt="A ship nearing an iceberg" width="400"/>

Greatful to the image from [Unsplash](https://unsplash.com).

## Table of Contents

1. [Project Overview](#titanic-survival-prediction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Methodology](#methodology)
5. [Example Code](#example-code)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Acknowledgements](#acknowledgements)
9. [License](#license)


## Contents

- Data: Description of the dataset used.
- Preprocessing: Steps taken to clean and prepare the data.
- Modeling: Details about the models used and their evaluation.
- Results: Summary of the findings and model performance metrics.

## Dataset

The dataset used in this project is the Titanic dataset, available from Kaggle. The dataset contains information about the passengers, including features like:

| Feature   | Description                                 |
|-----------|---------------------------------------------|
| Pclass    | Ticket class                                |
| Sex       | Gender of the passenger                     |
| Age       | Age of the passenger                        |
| SibSp     | Number of siblings or spouses aboard        |
| Parch     | Number of parents or children aboard        |
| Fare      | Ticket fare                                 |
| Cabin     | Cabin number                                |
| Embarked  | Port of embarkation                         |
| Survived  | Survival (1 = Yes, 0 = No)                 |


## Installation

To run this project, install the necessary packages.

    ```
    pip install numpy pandas scikit-learn seaborn matplotlib xgboost


## Methodology

- Load the dataset and explore its contents.
- Perform data preprocessing, including encoding categorical variables, filling missing values, and feature scaling.
- Train various machine learning models, including:
  -  Logistic Regression
  -  Stochastic Gradient Descent (SGD)
  -  Decision Trees
  -  Random Forests
  - Support Vector Machines (SVM)
  - XGBoost

- Use GridSearchCV to optimize model hyperparameters.
- Evaluate the models using metrics such as accuracy, precision, recall, and F1 score.
- Print classification reports and compare predictions with actual values.

## Example Code

Here’s a snippet to demonstrate how to fit a model using a pipeline:

    ```
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV
    
    # Define the SGD Classifier
    model_reg = SGDClassifier(random_state=0)
    
    # Create a pipeline for preprocessing and modeling
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Your preprocessing step
        ('model', model_reg)
    ])
    
    # Fit the pipeline on the training data
    my_pipeline.fit(train_X, train_y)
    
    # Make predictions
    preds = my_pipeline.predict(val_X)

## Results

After training and evaluating the models, key metrics are printed for both training and validation datasets, including accuracy, precision, recall, and F1 score.
The XGBoost model demonstrated strong performance on the Titanic survival prediction task. Below are the evaluation metrics for the model:

### XGBoost Model Performance
- Train Accuracy: 0.9165
- Train Precision: 0.9178
- Train Recall: 0.9165
- Train F1 Score: 0.9154
- Validation Accuracy: 0.7985
- Validation Precision: 0.7975
- Validation Recall: 0.7985
- Validation F1 Score: 0.7976
  
In addition to the XGBoost model, other models also performed very well, contributing to a comprehensive understanding of the data and enhancing prediction capabilities.

When the test results were submitted to Kaggle, the score was 0.7679. 

## Conclusion

This project provides a comprehensive approach to predicting Titanic survival using various machine learning techniques. Further improvements can be made by experimenting with feature engineering and additional models.

## Acknowledgements

Dataset source: Kaggle Titanic Dataset
Libraries used: Numpy, Pandas, Scikit-learn, Seaborn, Matplotlib, XGBoost

## License

This project is licensed under the MIT License.

