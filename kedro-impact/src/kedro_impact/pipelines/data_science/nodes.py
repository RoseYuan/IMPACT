import os
import logging
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Import Scikit-Learn Data Science Packages
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as confusion_matrix_sk
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix, f1_score, balanced_accuracy_score, \
    recall_score, precision_score
from sklearn.utils import shuffle, compute_class_weight
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# import sampling methods
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Import XGBoost and CatBoost packages for boosted decision trees
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold

# Import Tensorflow and Keras for Neural Networks
from tensorflow.python.keras.saving.save import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten, BatchNormalization
from keras.metrics import AUC


def balance_sampling(train_x: pd.DataFrame, train_y: pd.DataFrame, mode='rus') -> [pd.DataFrame, pd.DataFrame]:
    """
    :param train_x: train dataset as data frame
    :param train_y: one column with the labels
    :param mode: sampling mode: random oversampling is ros, random undersampling is rus
    :return: balanced train dataset
    """
    if mode == 'rus':
        sampler = RandomUnderSampler()
    else:
        sampler = RandomOverSampler()

    X_balance, y_balance = sampler.fit_sample(train_x, train_y)
    train_x, train_y = shuffle(X_balance, y_balance, random_state=42)
    return train_x, train_y


def prediction(model: pickle, test_x: pd.DataFrame, pred_type="normal") -> np.ndarray:
    """
    :param pred_type: use normal if the model predict function returns labels not probability
    :param model: model saved as pickle file
    :param test_x: test dataset, data frame
    :return: prediction of model on test_x: one column with the labels
    """
    if pred_type == "normal":
        y_pred = model.predict(test_x)
        return y_pred
    else:
        dtest = xgb.DMatrix(test_x)
        y_pred = model.predict(dtest)
        labels = np.array([0 if pred < 0.5 else 1 for pred in y_pred])
        return labels


def prediction_NN(model: pickle, test_x: pd.DataFrame) -> np.ndarray:
    """
    :param model: model saved as pickle file
    :param test_x: test dataset, data frame
    :return: prediction of model on test_x: one column with the labels
    """
    test_x = test_x.values.reshape((test_x.shape[0], test_x.shape[1], 1))
    # loaded_model = load_model(model)
    loaded_model = model
    y_pred = loaded_model.predict(test_x)
    y_pred = (y_pred > 0.4)
    return y_pred


def report_accuracy(predictions: np.ndarray, test_y: pd.DataFrame, model_name: str, sheet_name: str) -> None:
    """
    Node for reporting the accuracy of the predictions performed by the
    previous node. It saves the output at log file.
    :param predictions: predictions from model
    :param test_y: labels from test dataset
    :param model_name: string contains model name for logging file
    :param sheet_name: one of these options: v4,i1,i2,c1,g1
    :return:
    """
    logger = logging.getLogger('report')
    if sheet_name == 'v4':
        hdlr = logging.FileHandler(
            os.getcwd() + '/data/08_reporting/report_evaluation_{}_{}.log'.format(model_name, sheet_name))
    else:
        hdlr = logging.FileHandler(
            os.getcwd() + '/data/08_reporting/report_evaluation_{}_extra_sheet.log'.format(model_name))

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    acc_score = accuracy_score(test_y, predictions)
    bacc_score = balanced_accuracy_score(test_y, predictions)
    f1 = f1_score(test_y, predictions, average='weighted')
    recall = recall_score(test_y, predictions)
    precision = precision_score(test_y, predictions)
    logger.info("Report:")
    logger.info("Report for sheet {}:".format(sheet_name))
    logger.info("Number of falsified data in the test set : " + str(
        round((test_y.sum() / len(test_y)) * 100, 2)) + "%")
    logger.info("Number of true data in the test set : " + str(
        round(((len(test_y) - test_y.sum()) / len(test_y)) * 100, 2)) + "%")
    logger.info("Model F1 on test set: %0.2f%%", f1 * 100)
    logger.info("Model Recall on test set: %0.2f%%", recall * 100)
    logger.info("Model Precision on test set: %0.2f%%", precision * 100)
    logger.info("Model accuracy on test set: %0.2f%%", acc_score * 100)
    logger.info("Model balanced accuracy on test set: %0.2f%%", bacc_score * 100)


def train_xgboost(train_x: pd.DataFrame, train_y: pd.DataFrame) -> pickle:
    """
    This method trains xgboost with already tuned parameters. parameters where selected based on GridSearchCV
    :param train_x: train dataset as data frame
    :param train_y: one column with the labels
    :return: xgboost model as pickle
    """

    train_x, train_y = balance_sampling(train_x, train_y, 'rus')
    X_train, X_val, y_train, y_val = train_test_split(
        train_x, train_y, test_size=0.1, random_state=42, shuffle=True,
        stratify=train_y)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {'max_depth': 10, 'min_child_weight': 6, 'eta': .1, 'subsample': 1, 'colsample_bytree': 0.9,
              'objective': 'binary:logistic', 'eval_metric': "logloss"}
    num_boost_round = 999
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "Test")],
        early_stopping_rounds=10
    )

    return model


def feature_selector(train_x: pd.DataFrame, train_y: pd.DataFrame, model: str = "xgboost_model"):
    """
    :param train_x:  train dataset as data frame
    :param train_y: one column with the labels
    :param model: it can be selected between different models since the xgboost is the best we just kept that
    :return: feature selector object
    """
    train_x, train_y = balance_sampling(train_x, train_y, 'rus')
    if model == "xgboost_model":
        clf = XGBClassifier()
    elif model == "rf_model":
        clf = RandomForestClassifier(random_state=42, class_weight="balanced", max_depth=10, n_estimators=200)
    elif model == "catboost_model":
        clf = CatBoostClassifier(iterations=10, learning_rate=1, depth=8)
    elif model == "pca":
        pipe = sklearn.pipeline.Pipeline([('scaler', preprocessing.StandardScaler()), ('pca', PCA(n_components=100))])
        pipe.fit(train_x)
        return pipe
    elif model == "p_tranformer":
        clf = PowerTransformer(method='yeo-johnson')
        clf.fit(train_x)
        return clf

    selector = SelectFromModel(estimator=clf, max_features=50).fit(train_x, train_y)
    return selector


def feature_transform(df: pd.DataFrame, selector: pickle) -> pd.DataFrame:
    """
    :param df: input dataframe
    :param selector: feature selector object
    :return: dataframe with just selected features
    """
    df_selected = selector.transform(df)
    return df_selected


def train_NN(train_x: pd.DataFrame, train_y: pd.DataFrame) -> str:
    """Neural Network model: convolutional neural network that need to be tuned to give better results. We tried
    several architectures and layer types but we never came close to the results from xgboost with this model"""

    # Validation set creation
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=0)
    train_x = train_x.values.reshape((train_x.shape[0], train_x.shape[1], 1))
    val_x = val_x.values.reshape((val_x.shape[0], val_x.shape[1], 1))

    # class weight for imbalanced dataset
    class_weight = compute_class_weight('balanced', [0, 1], train_y)
    class_weight = {0: class_weight[0], 1: class_weight[1]}

    model = Sequential()
    model.add(Conv1D(filters=25, kernel_size=2, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=10, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.build()
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=[AUC()])
    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=10, batch_size=100, verbose=1,
              class_weight=class_weight)

    # save model
    # filepath = "/data/06_models/NN_model"
    # model.save(filepath)
    return model  # filepath


def train_baseline_xgboost(train_x: pd.DataFrame, train_y: pd.DataFrame) -> pickle:
    """
    This is baseline xgboost without tuning
    :param train_x: train dataset as data frame
    :param train_y: one column with the labels
    :return: xgboost model as pickle
    """
    boost_model = XGBClassifier()
    boost_model.fit(train_x, train_y)

    return boost_model


def confusion_matrix_NN(test_y: pd.DataFrame, pred_y: pd.DataFrame) -> None:
    print(confusion_matrix(test_y, pred_y))


def confusion_matrix(test_x: pd.DataFrame, test_y: pd.DataFrame, model, model_name: str) -> None:
    disp = plot_confusion_matrix(model, test_x, test_y, cmap='YlOrRd')
    disp.ax_.set_title(model_name)
    plt.savefig('data/08_reporting/' + model_name + '_confusion_matrix.png')

'''

THRESHOLD TUNING
In the following, you can play with different threshold values and see the corresponding result evaluated 
via cross-validation. Here we implement it using the xgboost model. Other models can also be applied. 

'''
def train_xgbm(fit_x, fit_y, eval_x, eval_y, params_train=None):
    """
    Train a XGBoost model.
    :param fit_x: train dataset as data frame.
    :param fit_y: label for the train dataset, one column.
    :param eval_x: evaluation dataset as data frame.
    :param eval_y: label for the evaluation dataset, one column.
    :param params_train: parameters for the model.
    :return: xgbm model object.
    """
    n_train, _ = fit_x.shape
    n_positive = sum(fit_y)
    print("{} samples in training set, {}({:.2f}%) positive, {}({:.2f}%) negative."
          .format(n_train,
                  n_positive,
                  n_positive * 100 / n_train,
                  n_train - n_positive, 100 - (n_positive * 100 / n_train)))

    dtrain = xgb.DMatrix(fit_x, label=fit_y)
    dval = xgb.DMatrix(eval_x, label=eval_y)

    num_boost_round = 999
    xgb_model = xgb.train(
        params_train,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "Test")],
        early_stopping_rounds=10
    )
    return xgb_model

def predict_xgbm(xgb_model,eval_x,eval_y):
    """
    Predict using a trained XGBoost model.

    :param xgb_model: xgbm model object.
    :param eval_x: evaluation dataset as data frame.
    :param eval_y: label for the evaluation dataset, one column.
    :return: predicted probabilities for the evaluation dataset.
    """
    # Performance on evaluation set
    dval = xgb.DMatrix(eval_x, label=eval_y)
    y_pred_p = xgb_model.predict(dval)
    return y_pred_p

def to_labels(pos_probs, threshold):
    """
    Convert the probability scores into 0,1 labels with specified score threshold.
    """
    return (pos_probs >= threshold).astype('int')

def cv_train(CV_NUMBER, train_x, train_y, train_model, predict_model, params,imb_sampling=False,mode=None):
    """
    Train a model via cross-validation.

    :param train_model: function name to train the model.
    :param predict_model: function name to predict using the model.
    :param params: model parameter.
    :param imb_sampling: Boolean, do subsampling for the training dataset or not.
    :param mode: string, the method for subsampling.
    :return: predicted probability score and corresponding true
    """
    cv_y_predict_p = []
    cv_eval_y = []
    k_fold = KFold(CV_NUMBER,shuffle=True)
    for k, (train, test) in enumerate(k_fold.split(train_x, train_y)):
        fit_x = train_x.iloc[train,:]
        fit_y = train_y.iloc[train]
        eval_x = train_x.iloc[test,:]
        eval_y = train_y.iloc[test]

        if imb_sampling:
            fit_x, fit_y = balance_sampling(fit_x, fit_y, mode=mode)

        model = train_model(fit_x, fit_y, eval_x,eval_y,params)
        y_pred_p = predict_model(model,eval_x,eval_y)

        cv_y_predict_p.append(y_pred_p)
        cv_eval_y.append(eval_y)

    return cv_y_predict_p,cv_eval_y

def presicion_recall_curve_cv(CV_NUMBER, train_x, train_y, params=None,train_model=train_xgbm,
                              predict_model=predict_xgbm,imb_sampling=True,mode='ros'):
    """
    Train a model via cross-validation and plot the precision-recall curve.

    :param CV_NUMBER: number of folds for cross validation.
    :param train_model: function name to train the model.
    :param predict_model: function name to predict using the model.
    :param params: model parameter.
    :param imb_sampling: Boolean, do subsampling for the training dataset or not.
    :param mode: string, the method for subsampling.
    :return: dataframe, precision recall values when using different threshold.
    """
    cv_y_predict_p,cv_eval_y = cv_train(CV_NUMBER, train_x, train_y, train_model,
                                        predict_model,params,imb_sampling,mode)
    prc = pd.DataFrame(columns=['precision','recall'])
    for threshold in np.arange(0,1,0.01):
        precisions = []
        recalls = []
        for i in range(CV_NUMBER):
            y_predict_p = cv_y_predict_p[i]
            eval_y = cv_eval_y[i]
            y_pred = to_labels(y_predict_p, threshold)
            precisions.append(precision_score(eval_y, y_pred))
            recalls.append(recall_score(eval_y, y_pred))
            prc.loc[threshold]=[np.mean(precisions),np.mean(recalls)]
    return prc

def plot_presicion_recall_curve_cv(prc,model_name='XGBoost'):
    """
    Plot the precision-recall curve and save to the file.

    :param prc:dataframe, precision recall values when using different threshold.
    """
    plt.figure()
    plt.step(prc['recall'], prc['precision'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.savefig('data/08_reporting/' + model_name + '_precision_recall_curve.png')


def report_results(test_y, y_pred):
    """
    Report the results using different metrics.
    """
    # metrics
    f1 = f1_score(test_y, y_pred, average='weighted')
    acc_score = accuracy_score(test_y, y_pred)
    bacc_score = balanced_accuracy_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred)
    tn, fp, fn, tp = confusion_matrix_sk(test_y, y_pred).ravel()

    print("Report:")
    print("----")
    print("Percentage of falsified data in the test set : " + str(
        round((test_y.sum() / len(test_y)) * 100, 2)) + "%")
    print("Percentage of true data in the test set : " + str(
        round(((len(test_y) - test_y.sum()) / len(test_y)) * 100, 2)) + "%")

    print("METRICS:")
    print("Model F1 on test set: {}%".format(round(f1 * 100, 2)))
    print("Model Recall on test set: {}%".format(round(recall * 100, 2)))
    print("Model Precision on test set: {}%".format(round(precision * 100, 2)))
    print("Model accuracy on test set: {}%".format(round(acc_score * 100, 2)))
    print("Model balanced accuracy on test set: {}%".format(round(bacc_score * 100, 2)))

    print("DATA:")
    print("Number of total data in the test set : " + str(len(test_y)))
    print("Number of falsified data in the test set : " + str(test_y.sum()))
    print("Number of true data in the test set : " + str(len(test_y) - test_y.sum()))

    print("EVALUATION BASED ON FN:")
    print("TN: {} \nFP: {} \nFN: {} \nTP: {} ".format(tn, fp, fn, tp))
    print("Goal: low number of FN (here {}), even if we get more FP (here {})".format(fn, fp))
    print("Now, the client " + "has to look at {} data points detected as potentially falsified".format(
        y_pred.sum()) +  ", \ninstead of checking all {} ".format(len(test_y)))
    print("Now, the client "  + "has to check {} less data ponts resp. {}% less data points.".format(
        len(test_y) - y_pred.sum(), 100 * round(1 - y_pred.sum() / len(test_y), 2)) )
    print("But: there are {} falsified instances left undetected.".format(fn))
    print("Of these predictions, {} have been falsely classified as falsified".format(fp))
    print("----------------------------------------------------------------------------------------------")

def report_results_threshold(threshold,y_pred_p,test_y):
    """
    Report the results using different metrics with the specified threshold.
    """
    print('Selected threshold: '+ str(threshold))
    y_pred = to_labels(y_pred_p, threshold)
    report_results(test_y, y_pred)

