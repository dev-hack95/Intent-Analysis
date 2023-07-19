import yaml
import pickle
import logging
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from kedro.io import DataCatalog
from kedro.extras.datasets.pickle import PickleDataSet


log = logging.getLogger(__name__)



def split_test_train(data, split=0.7, random_state=42):
    train = data.sample(frac = split, random_state=random_state)
    test = data.drop(train.index)

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for index, train_data in train.iterrows():
        x_train.append(str(train_data['review_comment_message']))
        y_train.append(train_data['review_score'])

    for index, test_data in test.iterrows():
        x_test.append(str(test_data['review_comment_message']))
        y_test.append(test_data['review_score'])

    return x_train, y_train, x_test, y_test




def preprocess(orders_reviews: pd.DataFrame) -> pd.DataFrame:
    """Create a data set for training and testing the model."""

    try:
        log.info('Creating data set')
        orders_reviews.loc[orders_reviews['review_score'] <= 3, 'review_score'] = 0
        orders_reviews.loc[orders_reviews['review_score'] > 3, 'review_score'] = 1
        data = orders_reviews[['review_comment_message', 'review_score']]
        # Drop rows with missing values (NaNs).
        data.dropna(inplace=True)
    except Exception as err:
        log.error("Error: ", str(err))
    finally:
        log.info(data.head())


    log.info("Resample the data and balcing the classes")
    minority = data[data['review_score'] == 0]
    majority = data[data['review_score'] == 1]

    under_sampling = resample(
            majority,
            replace=False,
            n_samples=len(minority),
            random_state=42
            )

    balanced_data = pd.concat([minority, under_sampling])
    data = balanced_data.sample(frac=1, random_state=42)

    log.info(data['review_score'].value_counts())

    try:
        log.info("Splitting data into training and testing")
        x_train, y_train, x_test, y_test = split_test_train(data, split=0.7, random_state=42)

    except ValueError as err:
        log.error("Error while splitting data", err)
    finally:
        log.info("Splitting of data is done")

    # Tokenization

    try:
        log.info("Tokenizing data")
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(x_train)
    except Exception as err:
        log.error("Error while tokenizing data: ", err)
    finally:
        log.info("Tokenizing of data is done")

    # padding
    def padding(train, test, max_length, vocab_size, trunc_type='post'):
        train_sequence = tokenizer.texts_to_sequences(train)
        padded_train = tf.keras.preprocessing.sequence.pad_sequences(train_sequence, maxlen=max_length, truncating=trunc_type)

        test_sequence = tokenizer.texts_to_sequences(test)
        padded_test = tf.keras.preprocessing.sequence.pad_sequences(test_sequence, maxlen = max_length, truncating=trunc_type)

        return padded_train, padded_test
    
    try:
        log.info("Padding of the training data")
        x_train, x_test = padding(x_train, x_test, 120, 10000)
    except Exception as err:
        log.error("Error occured while padding: ", err)
    finally:
        log.info("Training and testing dataset are now ready for model building.")

    try:
        log.info("Stats pickeling the data")
        catlog = DataCatalog(
            data_sets={
                'x_train': PickleDataSet(filepath='./data/05_model_input/x_train.pkl'),
                'y_train': PickleDataSet(filepath='./data/05_model_input/y_train.pkl'),
                'x_test': PickleDataSet(filepath='./data/05_model_input/x_test.pkl'),
                'y_test': PickleDataSet(filepath='./data/05_model_input/y_test.pkl')
            }

        )

        catlog.save('x_train', x_train)
        catlog.save('y_train', y_train)
        catlog.save('x_test', x_test)
        catlog.save('y_test', y_test)

    except Exception as err:
        log.error("Error occured while pickling the data: ", err)

    finally:
        log.info("Finished pickling of the data")

    return x_train, y_train, x_test, y_test




def train_model(x_train, y_train, x_test, y_test):
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

    models = {
        'LogisticRegression': LogisticRegression(C=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['LogisticRegression']['params']['C'],
                                                penalty=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['LogisticRegression']['params']['penalty'],
                                                max_iter=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['LogisticRegression']['params']['max_iter'],
                                                n_jobs=-1),
        'RandomForest': RandomForestClassifier(n_estimators=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['RandomForestClassifer']['params']['n_estimators'],
                                               criterion=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['RandomForestClassifer']['params']['criterion'],
                                               n_jobs = -1),
        'AdaBoost': AdaBoostClassifier(n_estimators=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['AdaBoost']['params']['n_estimators']),
        'GradientBoost': GradientBoostingClassifier(n_estimators=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['GradientBoost']['params']['n_estimators'],
                                                    loss=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['GradientBoost']['params']['loss'],
                                                    max_features=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['GradientBoost']['params']['max_features']),
        'BaggingClassifer': BaggingClassifier(n_estimators=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['BaggingClassifer']['params']['n_estimators'],
                                              max_features=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['BaggingClassifer']['params']['max_features'],
                                              max_samples=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['BaggingClassifer']['params']['max_samples']),
        'DecisionTree': DecisionTreeClassifier(),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['ExtraTrees']['params']['n_estimators'],
                                           criterion=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['ExtraTrees']['params']['criterion']),
        'SVC': SVC(kernel=yaml.safe_load(open('conf/base/parameters/sentiment_analysis.yml'))['SVC']['params']['kernel'])
    }




    def train_model(model, model_name, x_train, y_train, x_test, y_test):
        log.info("#"*50)
        model = model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        log.info("%s : %f", model_name, accuracy_score(y_test, y_pred))
        log.info("-"*50)
        matirx = confusion_matrix(y_test, y_pred)
        log.info(matirx)
        log.info("-"*50)
        report = classification_report(y_test, y_pred)
        log.info(report)

    for model_name, model in models.items():
        train_model(model, model_name, x_train, y_train, x_test, y_test)
            
