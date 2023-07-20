import os
import pickle
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


log = logging.getLogger(__name__)

def sentiment_analysis_model_caliberation(x_train, y_train, x_test, y_test):
    """Sentiment Analysis Model Calibration using Isotonic Scaling"""
    try:
        log.info("Loading training and testing data")
        x_train = pickle.load(open('./data/05_model_input/x_train.pkl', 'rb'))
        y_train = pickle.load(open('./data/05_model_input/y_train.pkl', 'rb'))
        x_test = pickle.load(open('./data/05_model_input/x_test.pkl', 'rb'))
        y_test = pickle.load(open('./data/05_model_input/y_test.pkl', 'rb'))
    except Exception as err:
        log.info("Error occured while loading data: ", err)
    finally:
        log.info("Succesfully loded the data")

    """Bagging Classifer"""
    clf_bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(criterion='gini',
      max_depth=50,
      min_samples_split=3
      ),
    n_estimators=300,
    bootstrap_features=True,
    oob_score=True,
    max_features=1.0,
    n_jobs = -1
    )
    caliberation_clf_bag = CalibratedClassifierCV(
        estimator=clf_bag,
        cv=3,
        method='isotonic'
    )

    caliberation_clf_bag.fit(x_train , y_train)

    y_pred_bag = caliberation_clf_bag.predict_proba(x_test)[:, 1]
    acc = caliberation_clf_bag.score(x_test, y_test)
    log.info("Bagging Classifer: %s", acc)

    log.info("Bagging Classifer: %s", brier_score_loss(y_test, y_pred_bag))


    plt.rcParams.update({'font.size': 10})
    fraction_postives_bag, predict_prob_bag = calibration_curve(y_test, y_pred_bag, n_bins=10)
    X = np.linspace(0, 1, 10)
    Y = X
    sns.lineplot(x=X, y=Y, linestyle='dotted')
    sns.lineplot(x=predict_prob_bag, y=fraction_postives_bag)
    plt.grid(linestyle='-', linewidth=0.2)
    plt.title("Probability vs Fraction Postives")
    xlabel = plt.xlabel("Probability of positive")
    ylabel = plt.ylabel("Fraction of positives")
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    xticks = plt.xticks(ticks)
    yticks = plt.yticks(ticks)
    os.makedirs('./data/08_reporting/Bagging_Classifer/model_caliberation/', exist_ok=True)
    plt.savefig('./data/08_reporting/Bagging_Classifer/model_caliberation/bagging_caliberation.png')

    """Gradient Boosting"""

    clf_gradient_boost = GradientBoostingClassifier(
        n_estimators=500,
        loss='log_loss',
        max_features=0.5
    )
    
    caliberation_clf_gradient_boost = CalibratedClassifierCV(
        estimator=clf_gradient_boost,
        cv=3,
        method='isotonic'
    )
    caliberation_clf_gradient_boost.fit(x_train , y_train)

    y_pred_gradient_boost = caliberation_clf_gradient_boost.predict_proba(x_test)[:, 1]
    acc = caliberation_clf_bag.score(x_test, y_test)
    log.info("Gradient Boost Classifer %s: ", acc)

    log.info("Gradient Boost Classifer: %s", brier_score_loss(y_test, y_pred_gradient_boost))


    plt.rcParams.update({'font.size': 10})
    fraction_postives_gradient_boost, predict_prob_gradient_boost = calibration_curve(y_test, y_pred_gradient_boost, n_bins=10)
    X = np.linspace(0, 1, 10)
    Y = X
    sns.lineplot(x=X, y=Y, linestyle='dotted')
    sns.lineplot(x=predict_prob_gradient_boost, y=fraction_postives_gradient_boost)
    plt.grid(linestyle='-', linewidth=0.2)
    plt.title("Probability vs Fraction Postives")
    xlabel = plt.xlabel("Probability of positive")
    ylabel = plt.ylabel("Fraction of positives")
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    xticks = plt.xticks(ticks)
    yticks = plt.yticks(ticks)
    os.makedirs('./data/08_reporting/Gradient_Boosting_Classifer/model_caliberation/', exist_ok=True)
    plt.savefig('./data/08_reporting/Gradient_Boosting_Classifer/model_caliberation/gradient_boost_caliberation.png')