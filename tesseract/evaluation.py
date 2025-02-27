# -*- coding: utf-8 -*-

"""
evaluation.py
~~~~~~~~~~~~~



"""
import multiprocessing as mp


import scipy.sparse
from tqdm import tqdm

from tesseract import utils as utils, metrics as metrics
from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight
import numpy as np
import dalex as dx
import pandas as pd
from keras import callbacks
from keras.utils import np_utils
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import confusion_matrix
import Utils
import time


class Stage:
    """Parent class representing stage of the time-aware evaluation cycle.

    The time-aware evaluation cycle is divided into stages, offering the
    ability for the system designer to interact with the classification
    process. The stages can generally be thought of as the following:

        * Rebalancing: Alterations can be made to the training set composition.
        * Training: The classifier is fit to the training data.
        * Prediction: Labels are predicted by the classifier.
        * Rejection: Low-quality predictions can be discarded/quarantined.
        * Selection: Test objects can be selected and added to the training.

    The rebalancing, prediction and selection stages can all be implemented by
    subclassing Stage or its children.

    Subclasses of Stage can be coupled together with Stages of the same type,
    for example, tesseract.evaluation.fit_predict_update accepts lists of
    Rejectors which will be activated in order during the rejection 'stage' of
    the evaluation cycle. To determine whether a Stage is activated during that
    cycle, it contains a schedule.

    A schedule is simply a list of booleans, the length of the total periods
    expected during that cycle; the Stage is active if the index of the
    schedule for that period is True. Some special values exist which will be
    resolved to valid schedules:

        * 'first': Activate on the first cycle only.
        * 'last': Activate on the last cycle only.
        * 1: Activate every cycle.
        * 0: Never activate.

    These settings don't require the total number of test periods to be known
    in advance, the schedule will be resolved once fit_predict_update has been
    called, by checking the X_tests parameter.

    Attributes:
        schedule (list): A list of booleans indicating when the Stage should be
            active during the evaluation cycle.

    """

    def __init__(self, schedule=1):
        self.schedule = schedule

    def resolve_schedule(self, total_periods):
        """Produces a valid schedule for the total periods specified.

        A schedule is a list of booleans, the length of the total periods
        expected during that cycle; the Stage is active if the index of the
        schedule for that period is True.

        Some special values exist which will be resolved to valid schedules:

            * 'first': Activate on the first cycle only.
            * 'last': Activate on the last cycle only.
            * 1: Activate every cycle.
            * 0: Never activate.

        """
        if self.schedule == 'first':
            self.schedule = [True] + [False] * (total_periods - 1)
        elif self.schedule == 'last':
            self.schedule = [False] * (total_periods - 1) + [True]
        elif self.schedule in (1, '1'):
            self.schedule = [True] * total_periods
        elif self.schedule in (0, '0'):
            self.schedule = [False] * total_periods
        elif hasattr(self.schedule, '__iter__'):
            self.schedule = [int(x) == 0 for x in self.schedule]
        else:
            raise ValueError('Schedule `{}` cannot be understood.'.format(
                self.schedule))


class TrackingStage(Stage):
    """

    """

    def __init__(self, schedule=1, tracking=True, interaction='intersection'):
        super().__init__(schedule=schedule)

        self._interactions = ('intersection', 'union', 'sym_diff', 'ignore')

        self.tracking = tracking
        self.interaction = interaction

        if interaction not in self._interactions:
            raise ValueError('Interaction mode must be one of {}'.format(
                self._interactions))

    def merge_results(self, past, present):
        # Case for first test period in a cycle
        # (distinct from when past is an empty array)
        if past is None:
            return present

        if self.interaction == 'union':
            return np.union1d(past, present)
        elif self.interaction == 'intersection':
            return np.intersect1d(past, present)
        elif self.interaction == 'sym_diff':
            return np.setxor1d(past, present)


def fit_predict_update(clf, X_train, X_tests,
                       y_train, y_tests, t_train, t_tests,
                       fit_function=None, predict_function=None,
                       rebalancers=(), rejectors=(), selectors=()):
    """Sliding window classification of a timestamp partitioned dataset.

    This function assumes that the dataset has been partitioned into
    historically coherent training and testing sets such that all objects in
    the training set are historically anterior to all objects in the testing
    sets, and in each testing set i, all objects in the set are historically
    anterior to all objects in testing set i + 1.

    The set of testing objects X_tests is split into a series of rolling
    testing windows (as are the corresponding y_tests). Each round of
    prediction is performed on the next test partition in the series.

    This arrangement is depicted here with the parameters:

        * Training dataset size: 6 months
        * Testing dataset size: 2 months
        * Date range of the dataset: 12 months (Jan - Dec)

    Months tagged ■ are included in the training dataset.
    Months tagged □ are included in the testing dataset.
    Months tagged ▣ are included in training dataset but the results from the
       previous round of testing are concatenated to the latest results.

    Rolling testing
    ---------------

       Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
    1    ■   ■   ■   ■   ■   ■   □   □
    2    ■   ■   ■   ■   ■   ■           □   □
    3    ■   ■   ■   ■   ■   ■                   □   □

    Example:
        >>> from sklearn.svm import LinearSVC
        >>> from tesseract import mock, temporal, evaluation
        >>>
        >>> X, y, t = mock.generate_binary_test_data(10000, '2000')
        >>>
        >>> splits = temporal.time_aware_train_test_split(
        >>>     X, y, t, train_size=6, test_size=2, granularity='month')
        >>>
        >>> clf = LinearSVC()
        >>>
        >>> results = evaluation.fit_predict_update(clf, *splits)

    For comparison, here's the same set of parameters combined with
    a FullRetrainingSelector to achieve incremental retraining at each
    testing period:

    Rolling testing, incremental retraining
    ---------------------------------------

       Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
    1    ■   ■   ■   ■   ■   ■   □   □
    2    ■   ■   ■   ■   ■   ■   ■   ■   □   □
    3    ■   ■   ■   ■   ■   ■   ■   ■   ■   ■   □   □

    Example:
        >>> from tesseract.selection import FullRetrainingSelector
        >>>
        >>> results = evaluation.fit_predict_update(
        >>>     clf, *splits, selectors=[FullRetrainingSelector()])

    The time-aware evaluation cycle is divided into stages, offering the
    ability for the system designer to interact with the classification
    process. The stages can generally be thought of as the following:

        * Rebalancing: Alterations can be made to the training set composition.
        * Training: The classifier is fit to the training data.
        * Prediction: Labels are predicted by the classifier.
        * Rejection: Low-quality predictions can be discarded/quarantined.
        * Selection: Test objects can be selected and added to the training.

    This cycle repeats for each testing period. The rebalancing, prediction
    and selection stages are each triggered by passing in lists of Rebalancer,
    Rejector or Selector objects respectively. These are then invoked
    (in order) at the appropriate stages in the training phase. Stages can be
    switched on and off for certain testing periods by passing them a
    schedule and the way they interact with previous stages of the same type
    can also be controlled.

    Fitting will use the fit() method of the classifier while prediction will
    try to resolve the most appropriate one for the classifier (either to
    produce output labels or raw scores). This behaviour can be overridden by
    passing a function to fit_function or predict_function.

    The form of these functions must maintain the following contract:

        * fit_function(X_train, y_train)
        * y_pred = predict_function(X_test)

    Note, there are plans to improve the rudimentary predict-function-detection
    and to perhaps replace the fit_function and predict_function parameters
    with Fitter and Predictor objects which would allow for greater control.

    Args:
        clf: A scikit-learn or Keras classifier with fit and predict methods.
        X_train (np.ndarray): Training partition of predictors X.
        X_tests (list): List of testing partitions of predictors X.
        y_train (np.ndarray): Training partition of output variables y.
        y_tests (list): List of testing partitions of predictors y.
        t_train (np.ndarray): Training partition of datetimes for X.
        t_tests (list): List of testing partitions of datetimes for X.
        fit_function (function): The function to use to fit clf.
        predict_function (function): The function to predict with.
        rebalancers (list): A list of rebalancers to alter the training set.
        rejectors (list): A list of rejectors to reject poor predictions.
        selectors (list): A list of selectors to pick test items to train with.

    Returns:
        dict: Performance metrics for each round of predictions, including
            precision, recall, F1 score, AUC ROC, TPR, TNR, FPR, FNR, TP, FP,
            TN, FN, actual positive and actual negative counts.

    See Also:
        tesseract.temporal.time_aware_train_test_split
        tesseract.evaluation.Stage
        tesseract.selection.Selector
        tesseract.rejection.Rejector
        tesseract.rebalancing.Rebalancer

    """
    fit_function = clf.fit if fit_function is None else fit_function
    predict_function = (utils.select_prediction_function(clf, labels_only=True)
                        if predict_function is None else predict_function)

    for stage in tuple(rebalancers) + tuple(rejectors) + tuple(selectors):
        stage.resolve_schedule(len(X_tests))

    results = {}
    for i, (X_test, y_test, t_test) in tqdm(enumerate(
            zip(X_tests, y_tests, t_tests))):
        #print("i" + str(i))

        # --------------------------------------------------------------- #
        # Make alterations to the dataset before testing (optional)       #
        # --------------------------------------------------------------- #

        for rebalancer in rebalancers:
            if not rebalancer.schedule[i]:
                continue

            X_train, y_train, t_train = rebalancer.alter(
                clf, X_train, y_train, t_train, X_test, y_test, t_test)

        # --------------------------------------------------------------- #
        # (Re)fit and predict                                             #
        # --------------------------------------------------------------- #

        results = metrics.get_train_info(
            X_train, y_train, t_train, existing=results)

        fit_function(X_train, y_train)
        y_pred = predict_function(X_test)



        # --------------------------------------------------------------- #
        # Discard/quarantine observations (optional)                      #
        # --------------------------------------------------------------- #

        kept_indexes, rejected_indexes = None, None
        for rejector in rejectors:
            if not rejector.schedule[i]:
                continue

            kept_indexes, rejected_indexes = rejector.reject_wrapper(
                clf, X_train, y_train, t_train,
                X_test, y_test, t_test,
                kept_indexes, rejected_indexes)

        if kept_indexes is not None:
            y_test = y_test[kept_indexes]
            y_pred = y_pred[kept_indexes]

            results['rejected'].append(rejected_indexes.size)
        else:
            results['rejected'].append(0)

        # --------------------------------------------------------------- #
        # Calculate performance                                           #
        # --------------------------------------------------------------- #


        results = metrics.calculate_metrics(
            y_test, y_pred, existing=results)


        #print(results['p'])

        # --------------------------------------------------------------- #
        # Select test observations for retraining (optional)              #
        # --------------------------------------------------------------- #

        selected_indexes = None
        for selector in selectors:
            if not selector.schedule[i]:
                continue

            selected_indexes = selector.query_wrapper(
                clf, X_train, y_train, t_train,
                X_test, y_test, t_test, selected_indexes)

        if selected_indexes is not None:
            # Select observations for training using chosen indices
            X_selected = X_test[selected_indexes]
            y_selected = y_test[selected_indexes]
            t_selected = t_test[selected_indexes]

            # Update training model with N selected points
            X_train = scipy.sparse.vstack((X_train, X_selected))
            y_train = np.hstack((y_train, y_selected))
            t_train = np.hstack((t_train, t_selected))

            results['selected'].append(selected_indexes.size)
        else:
            results['selected'].append(0)

    return results


def predict_update_INSOMNIA(clf, X_train, X_tests,
                       y_train, y_tests, t_train, t_tests,
                       fit_function=None, predict_function=None,
                       rebalancers=(), rejectors=(), selectors=(), oracle=1, positive_label=0, pathModels=None):
    """Sliding window classification of a window partitioned dataset.

    This function assumes that the dataset has been partitioned into
    historically coherent training and testing sets such that all objects in
    the training set are historically anterior to all objects in the testing
    sets, and in each testing set i, all objects in the set are historically
    anterior to all objects in testing set i + 1.

    The set of testing objects X_tests is split into a series of rolling
    testing windows (as are the corresponding y_tests). Each round of
    prediction is performed on the next test partition in the series.

    Args:
        clf: A scikit-learn or Keras classifier with fit and predict methods.
        X_train (np.ndarray): Training partition of predictors X.
        X_tests (list): List of testing partitions of predictors X.
        y_train (np.ndarray): Training partition of output variables y.
        y_tests (list): List of testing partitions of predictors y.
        t_train (np.ndarray): Training partition of datetimes for X.
        t_tests (list): List of testing partitions of datetimes for X.
        fit_function (function): The function to use to fit clf.
        predict_function (function): The function to predict with.
        rebalancers (list): A list of rebalancers to alter the training set.
        rejectors (list): A list of rejectors to reject poor predictions.
        selectors (list): A list of selectors to pick test items to train with.
        oracle: if the parameter is setting to 1 the active learning is performed with the human oracle otherwise with a pseudo oracle


    Returns:
        dict: Performance metrics for each round of predictions, including
            precision, recall, F1 score, AUC ROC, TPR, TNR, FPR, FNR, TP, FP,
            TN, FN, actual positive and actual negative counts.

    See Also:
        tesseract.temporal.time_aware_train_test_split
        tesseract.evaluation.Stage
        tesseract.selection.Selector
        tesseract.rejection.Rejector
        tesseract.rebalancing.Rebalancer

    """
    columns = ['Type', 'TP', 'FN', 'FP', 'TN', 'OA', 'AA', 'P', 'R', 'F1', 'FAR(FPR)', 'TPR']
    columnsTime=['TimeDalex', 'T_Label','T_NN', 'Tot']
    nm=0



    predict_function = (utils.select_prediction_function(clf, labels_only=True)
                        if predict_function is None else predict_function)

    for stage in tuple(rebalancers) + tuple(rejectors) + tuple(selectors):
        stage.resolve_schedule(len(X_tests))

    results = {}
    resultsM = pd.DataFrame(columns=columns)
    timeDf = pd.DataFrame(columns=columnsTime)


    for i, (X_test, y_test, t_test) in tqdm(enumerate(
            zip(X_tests, y_tests, t_tests))):

        #print("i" + str(i))

        # --------------------------------------------------------------- #
        # Make alterations to the dataset before testing (optional)       #
        # --------------------------------------------------------------- #

        for rebalancer in rebalancers:
            if not rebalancer.schedule[i]:
                continue

            X_train, y_train, t_train = rebalancer.alter(
                clf, X_train, y_train, t_train, X_test, y_test, t_test)


        # --------------------------------------------------------------- #
        # (Re)fit and predict                                             #
        # --------------------------------------------------------------- #

        results = metrics.get_train_info(
            X_train, y_train, t_train, existing=results, positive_label=0)


        #windows without samples
        if len(X_test) > 1:
            y_pred = predict_function(X_test)

        else:
            y_pred=[-1]

        # --------------------------------------------------------------- #
        # XAI module                                           #
        # --------------------------------------------------------------- #
        feature, tDalex =XAI_explainer(clf, X_train, y_train)
        fs = feature.result
        #print(fs)
        fs.to_csv('Explainable/XAI'+str(nm)+'.csv')

        # --------------------------------------------------------------- #
        # Discard/quarantine observations (optional)                      #
        # --------------------------------------------------------------- #

        kept_indexes, rejected_indexes = None, None
        for rejector in rejectors:
            if not rejector.schedule[i]:
                continue

            kept_indexes, rejected_indexes = rejector.reject_wrapper(
                clf, X_train, y_train, t_train,
                X_test, y_test, t_test,
                kept_indexes, rejected_indexes)

        if rejected_indexes is not None:
            # Select observations for training using rejected indices
            X_selected = X_test[rejected_indexes]
            y_selected = y_test[rejected_indexes]
            t_selected = t_test[rejected_indexes]


            X_train = scipy.sparse.vstack((X_train, X_selected))
            y_train = np.hstack((y_train, y_selected))
            t_train = np.hstack((t_train, t_selected))

            results['rejected'].append(rejected_indexes.size)
        else:
            results['rejected'].append(0)


        # --------------------------------------------------------------- #
        # Calculate performance                                           #
        # --------------------------------------------------------------- #


        results = metrics.calculate_metrics(
            y_test, y_pred, existing=results, positive_label=positive_label)
        print(results)

        # --------------------------------------------------------------- #
        # Select test observations for retraining (optional)              #
        # --------------------------------------------------------------- #


        if len(X_test) > 1:
            selected_indexes = None
            for selector in selectors:
                if not selector.schedule[i]:
                    continue

                selected_indexes = selector.query_wrapper(
                    clf, X_train, y_train, t_train,
                    X_test, y_test, t_test, selected_indexes)
            if type(X_train) != np.ndarray:
                X_train = X_train.toarray()


            if selected_indexes is not None:
                # Select observations for training using chosen indices
                X_selected = X_test[selected_indexes]
                t_selected = t_test[selected_indexes]

                #the labelling phase is performed with he Oracle
                if oracle==1:
                    print(' Using Oracle')
                    y_selected = y_test[selected_indexes]
                    tNN = 0
                else:
                    print(' Using Pseudo Label')

                    #This is only if you want to compute the results of DNN on the selected samples- start
                    y_test = y_test[selected_indexes]
                    r=getResultsDNN(clf, X_selected,y_test, positive_label=positive_label)
                    r.insert(0, 'Classifier'+str(i))
                    dfResults = pd.DataFrame([r], columns=columns)
                    resultsM = resultsM.append(dfResults, ignore_index=True)
                    #print('DNN')
                    #print(dfResults)
                    # This is only if you want to compute the results of DNN on the selected samples- end


                    y_selected, tNN=pseudoLabel(X_selected, X_train, y_train)

                    #Only to compute the accuracy of the NC on the selected samples- start
                    cm = confusion_matrix(y_test, y_selected)
                    r = Utils.getResult(cm, 2, positive_label=positive_label)
                    r.insert(0, 'NC' + str(i))
                    dfResults = pd.DataFrame([r], columns=columns)
                    resultsM = resultsM.append(dfResults, ignore_index=True)
                    #print('NC')
                    #print(dfResults)
                    resultsM.to_csv(str(1)+'_results.csv', index=False)
                    # Only to compute the accuracy of the NC on the selected samples -end


                X_train = scipy.sparse.vstack((X_train, X_selected))
                y_train = np.hstack((y_train, y_selected))
                t_train = np.hstack((t_train, t_selected))


                results['selected'].append(selected_indexes.size)

            #fine tuning phase
            clf, timeClassifier = finetuning(clf, X_train, y_train)

            #write time to computation
            totTime=tNN+timeClassifier
            dfTime= pd.DataFrame([[tDalex, tNN, timeClassifier, totTime]], columns=columnsTime)
            timeDf = timeDf.append(dfTime, ignore_index=True)
            timeDf.to_csv(str(1)+'_time.csv', index=False)



            nm+=1
            clf.save(pathModels+'NN'+str(nm)+'.h5')
        else:
            results['selected'].append(0)


    return results





def predict(clf, X_tests, decision_threshold=None,
            labels_only=False, predict_function=None, nproc=1, positive_label=0):
    
    """Standalone prediction of a set of test periods.

    Takes a set of historically aware test periods and performs prediction
    across them. This can be useful when there is no need for the interactive
    stages of a prediction as in that case the process can be performed in
    parallel.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from tesseract import mock, temporal, evaluation, metrics
        >>>
        >>> X, y, t = mock.generate_binary_test_data(10000, '2000')
        >>>
        >>> splits = temporal.time_aware_train_test_split(
        >>>     X, y, t, train_size=6, test_size=2, granularity='month')
        >>>
        >>> X_train, X_tests, y_train, y_tests, t_train, t_tests = splits
        >>>
        >>> clf = RandomForestClassifier(n_estimators=101, max_depth=64)
        >>> clf.fit(X_train, y_train)
        >>>
        >>> y_preds = evaluation.predict(clf, X_tests, nproc=4)
        >>> results = metrics.calculate_metrics(y_tests, y_preds, periods=-1, positive_label=positive_label)
        >>> metrics.print_metrics(results)

    Args:
        clf: A scikit-learn or Keras classifier with fit and predict methods.
        X_tests (list): List of testing partitions of predictors X.
        decision_threshold (float): Calibrate prediction function by
            supplying a threshold over which scores are labelled positive.
            This is intended for classifiers that output probabilities only.
        labels_only (bool): Prefer a labelling prediction function over one
            that outputs raw scores.
        predict_function (function): A custom function to predict with.
        nproc (int): The number of processors to use.

    Returns:
        list: A list of np.array objects containing the classification results
            for each test period in X_tests.

    """
    predict_function = (
        utils.select_prediction_function(clf, labels_only=labels_only) if
        predict_function is None else predict_function)

    # `nproc = -1` becomes `nproc = mp.cpu_count() + (- 1)`, etc
    nproc = mp.cpu_count() + nproc if nproc < 0 else nproc

    # Predictions have no dependencies in this context, we can parallelize them
    if nproc > 1:
        with mp.Pool(nproc) as p:
            y_preds = list(tqdm(
                p.imap(predict_function, X_tests), total=len(X_tests)))

    # Avoid invoking parallelism and associated overhead for a single CPU
    else:
        y_preds = []
        for X_test in tqdm(X_tests):
            if len(X_test) >1 :
                y_pred = predict_function(X_test)
                y_preds.append(y_pred)
            else:
                y_preds.append([[-1]])

    # TODO | Move to an "apply_decision_threshold" function to better test
    # TODO | and include the option in fit_predict_update (probas only).
    if decision_threshold:
        for i, y_pred in enumerate(y_preds):
            if y_pred.ndim > 1:
                y_scores = np.array([np.max(v) for v in y_pred])
            else:
                y_scores = y_pred
            y_preds[i] = np.array(y_scores > decision_threshold, dtype=int)

    return y_preds






def pseudoLabel(X_selected, X, y):
    '''
    Create the pseudo label for the selected samples
    :param X_selected: the selected samples
    :param X:  the  X training set for the learning of the pseudo Oracle
    :param y:  the y training for the learning of the pseudo Oracle
    :return:  y pseudo label for the selected samples, time to computation
    '''
    if type(X) != np.ndarray:
        X= X.toarray()
    print('Create pseudo label')
    tic = time.time()
    clf = NearestCentroid()
    clf.fit(X, y)
    pred=clf.predict(X_selected)
    toc = time.time()
    time_NC= toc - tic
    return pred, time_NC

def XAI_explainer(clf, X_train,y_train):
    '''
    Perform the explainable phase at each updating step
    :param clf: the classifier
    :param X_train: x train
    :param y_train: y train
    :return: feature importance and time to compute the explainable phase
    '''
    if type(X_train) != np.ndarray:
        X_train = X_train.toarray()
    tic=time.time()
    explainer = dx.Explainer(clf, X_train, y_train, model_type='classification')
    seed = 123
    feature = explainer.model_parts(random_state=seed)
    toc = time.time()
    timeXAI = toc - tic
    return feature, timeXAI


def finetuning(clf, X_train,y_train):
    '''
    Perform the fine tuning of the DNN model
    :param clf: the DNN to update
    :param X_train: X samples
    :param y_train: y samples
    :return: update DNN and time to computation
    '''
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10,
                                restore_best_weights=True),
    ]
    y_train2 = np_utils.to_categorical(y_train, 2)
    class_weights = dict(zip(np.unique(y_train), class_weight.compute_class_weight('balanced', np.unique(y_train),
                                                                                   y_train)))
    print(class_weights)
    XTraining, XValidation, YTraining, YValidation = train_test_split(X_train, y_train2, stratify=y_train2,
                                                                      test_size=0.2)
    tic = time.time()
    clf.fit(XTraining, YTraining, validation_data=(XValidation, YValidation),
            batch_size=512, epochs=150, callbacks=callbacks_list, class_weight=class_weights, shuffle=True)
    toc = time.time()
    timeClassifier = toc - tic

    return clf, timeClassifier


def getResultsDNN(clf, X_selected, y_test, positive_label=0):
    pred = clf.predict(X_selected)
    y_pred = np.argmax(pred, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    r = Utils.getResult(cm, 2, positive_label=positive_label)
    return r
