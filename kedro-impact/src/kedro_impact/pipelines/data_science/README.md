# Data Science pipeline


## Overview

This File contains five main pipelines:

1. baseline_model
2. baseline_model_extra
3. feature_selection
4. NN_model
5. threshold_pipeline

Additionally, this File contains one combined pipeline:  
data_science_pipeline: combines feature_selection and baseline_model_extra  

## How to run the code

If you run the code for the first time, you will want to run the data science pipeline. This will run the feature selection pipeline based on the previously by us trained and tuned XGBoost model and the baseline model for the extra sheets i1,i2,c1 and g1. You can run this pipeline using this command:     

`kedro run --pipeline=ds`  

If you want to run just one single pipeline, please to the general rule below.    

In general:   
-> you can run each pipeline individually by running this command: `run kedro --pipeline=selected_pipeline`  
-> you can check the name of each pipeline at /src/kedro_impact/hooks.py  


# 1. baseline_model

## Pipeline inputs

This modular pipeline:
1. trains a tuned xgboost model (`train_xgboost` node)
2. makes predictions given a trained model from (1) and a test set (`prediction` node)
3. reports the model accuracy on a test set (`report_accuracy` node)

Note that the hyper parameters for the model were selected based on GridSearchCV from XGBoost using the training set.
Also other machine learning models were tested including Random Forest, SVM and Catboost. The best performance was achieved
by XGBoost.

run this pipeline : `kedro run --pipeline=baseline`

### `train_x`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing train set features |

### `train_y`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing train set one-hot encoded target variable |

### `test_x`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing test set features |

### `test_y`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing test set one-hot encoded target variable |


## Pipeline outputs

### `xgb_baseline`

|      |                    |
| ---- | ------------------ |
| Type | `pickle` |
| Description | xgboost model saved on data/06_models/|

# 2. baseline_model_extra

Same structure as baseline_model designed for extra sheets i1, i2, c1 and g1. These extrasheets were not integrated in the main dataset v4, because the interviews were not conducted for all uuids and some uuids have multiple rows. Therefore, we faced problems with aggregation. We tried to model the falsification of these data sheets with various classifiers as well, including Naive Bayes and an unsupervised algorithm, however we did not achieve good peformance. Even with the best performing algorithm XGBoost, the prdiction performances for the sheets are relatively bad. In the future it could be advised to do some feature engineering on these extra sheets and to use some domain knowledge to integrate them smartly into the main data sheet v4 in some way. Alternatively, it could be tried to enhance performance of falsification classification on each of these sheets separately.   

 run this pipeline : `kedro run --pipeline=base_extra`


# 3. feature_selection

## Pipeline inputs

This modular pipeline:
1. selects features. The selection model could be chosen between xgboost, random forest, catboost, pca and power
transformers (`feature_selector` node)
2. Applies feature selection model on training and test set (`feature_transform` node)
3. trains a tuned xgboost model (`train_xgboost` node)
4. makes predictions given a trained model from (1) and a test set (`prediction` node)
5. reports the model accuracy on a test set (`report_accuracy` node)

 run this pipeline : `kedro run --pipeline=fs`

### `train_x`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing train set features |

### `train_y`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing train set one-hot encoded target variable |

### `test_x`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing test set features |

### `test_y`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing test set one-hot encoded target variable |


## Pipeline outputs

### `xgb_baseline`

|      |                    |
| ---- | ------------------ |
| Type | `pickle` |
| Description | xgboost model saved on data/06_models/|



# 4. NN_model

## Pipeline inputs


This modular pipeline:
1. trains a tuned convolutional neural network model (`train_NN` node)
2. makes predictions given a trained model from (1) and a test set (`prediction_NN` node)
3. reports the model accuracy on a test set (`report_accuracy` node)

The function from this pipeline are a bit different than for the other pipelines because the model requires that we reshape the data.

 run this pipeline : `kedro run --pipeline=NN`

### `train_x`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing train set features |

### `train_y`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing train set one-hot encoded target variable |

### `test_x`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing test set features |

### `test_y`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing test set one-hot encoded target variable |


## Pipeline outputs

### `train_NN`

|      |                    |
| ---- | ------------------ |
| Type | `keras model` |
| Description | The model is temporary saved while running the pipeline|

# 5. threshold_pipeline

This modular pipeline:
1. trains xgboost model via 5 fold cross-validation, computes the precision-recall curve using different probability threshold values.
2. plots and saves the precision-recall curve in 'data/08_reporting' directory.
3. reports the model results for test dataset using the specified threshold value.

run this pipeline : `kedro run --pipeline=thresholds`

The threshold here is the probability from which on an interview is classified as falsified. If the threshold is 0.5, then all interviews, where the algorithm finds they have >50% probability of being falsified, are classified as falsified.   
If the threshold is set lower, more interviews are classified as alsified and vice versa.

Here it depends what your goal is: If you want that the number of missed falsified interviews to be as close to zero as possible, you can set the threshold lower. If you set the threshold to 0, then the model will always classify the interview as "falsified", so you won-t miss any, which is great - but the model will also classify all the clean interviews as "falsified" and then you will again have to check the whole survey by hand, which is not better than without the model.    
So there is a clear trade-off between missing as few falsified interviews as possible and not having to (double-)check all interviews by hand.    

# Additional information: Notebooks

We put some additional code and visualization results in jupyter notebook for reference.

# 1. LightGBM_model_LIME.ipynb

In this notebook, a LightGBM model is trained using customized objective function: focal loss. Then LIME is used to explain individual predictions.

### Focal loss
Focal loss is a loss function designed to handle extreme imbalanced dataset. It's similar to binary log loss, but has lower loss value for correctly classified data points, and higher loss value for wrongly classified data points. It allows the model to take risks on those data points that are correctly classified but with low confidence (e.g. probability score 0.6 for label 1). This helps to prevent overfitting and enhances the generalization of the model. It also pays more attention to wrongly classified data points to achieve better accuracy. Focal loss is helpful when there's a ratio of [1:1000] or more for the imbalanced classes. There are two hyper-parameters in focal loss function: alpha and gamma. In this notebook the hyper-paremeters are tuned by cross validation to maximize the balanced f1 score. For more details about focal loss please refer to the following paper. https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf

### LIME
LIME(Local Interpretable Model-agnostic Explanations) is a novel explanation technique that tries to solve for model interpretability by producing locally faithful explanations. It learns an interpretable model locally around the prediction. For example, it can learn a hyper-plane that seperates the prediction from neighboring data points having different label to that predictions. Thus, by looking at the coefficients of this hyper-plane, we can tell the importance of features and gain insights about the logic behind the individual prediction.
