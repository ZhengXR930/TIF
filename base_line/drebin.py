

from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,confusion_matrix
from sklearn import svm
import os
import utils

import numpy as np
import tesseract.metrics as tm



def drebin_svm_train(X_train, y_train):
    print(f"train Linear SVM")
    clf = svm.LinearSVC(C=1.0, max_iter=10000)
    clf.fit(X_train, y_train)

    return clf

def drebin_svm_pred(clf, X_test, y_test):
    print(f"eval Linear SVM on overall test data")
    pred_labels = clf.predict(X_test)

    precision = precision_score(y_test, pred_labels,average='macro')
    recall = recall_score(y_test, pred_labels,average='macro')
    f1 = f1_score(y_test, pred_labels,average='macro')

    print(f"precision: {precision}, recall: {recall}, f1: {f1}")

    return precision, recall, f1

def drebin_svm_monthly(clf, results_f1_list,result_folder,data_folder,month_list, file_name):
    print(f"train Linear SVM and eval in overall test set")

    number = file_name.split('_')[-1]

    monthly_results_path = os.path.join(result_folder, f'{file_name}.csv')
    with open(monthly_results_path, 'w') as f:
        f.write("month,precision,recall,f1,aut\n")  

    
    for month in month_list:
        print(f"test month: {month}")
        file_path = os.path.join(data_folder, f"test_features_round_{number}" ,f"{month}.pkl")
        x_test, y_test = utils.load_single_month_data(file_path)
        precision, recall, f1 = drebin_svm_pred(clf, x_test, y_test)
        print(f"test month: {month}, precision: {precision}, recall: {recall}, f1: {f1}")

        results_f1_list.append(f1)
        m_aut = tm.aut(results_f1_list)

        with open(monthly_results_path, 'a') as f:
            f.write(f"{month},{precision},{recall},{f1},{m_aut}\n")