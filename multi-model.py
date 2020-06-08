import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from seaborn import heatmap
import scikitplot as skplt
from sklearn import model_selection
# import cv2

def cal_acc(result, test):
    count = 0
    for i in range(len(result)):
        if result[i] == test_y[i]:
            count += 1
    acc = round((count/len(result))*100, 2)
    print(f'Correct Predictions : {count}\nIncorrect Predictions : {len(result) - count}\nAccuracy = {acc}%')
    return acc

def evsp_graph(expected, predicted):
    expected_count = [0]*26
    predicted_count = [0]*26
    
    for i in range(len(expected)): expected_count[expected[i]] += 1
    for i in range(len(predicted)): predicted_count[predicted[i]] += 1
    
    df = pd.DataFrame({"Expected" : expected_count, "Predicted" : predicted_count})
    # print(df)
    graph = df.plot.bar(title='Expected vs Predicted')
    graph.set_xlabel('Categories')
    graph.set_ylabel('Occurence Count')
    # graph.bar()

def metrics(confusion_matrix):
    x = [i for i in range(25) if i != 9]
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    print('False Positive', FP)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    print('False Negative', FN)
    TP = np.diag(confusion_matrix)
    print('True Positive', TP)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    print('True Negative', TN)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print('True Positive Rate:', TPR)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    print('True Negative Rate:', TNR)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print('Positive Predictive Value:', PPV)
    # Negative predictive value
    NPV = TN/(TN+FN)
    print('Negative Predictive Value:', NPV)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print('False Positive Rate:', FPR)
    # False negative rate
    FNR = FN/(TP+FN)
    print('False Negative Rate:', FNR)
    # False discovery rate
    FDR = FP/(TP+FP)
    print('False Discovery Rate:', FDR)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print('Overall Accuracy:', ACC)
    
    plt.subplot(221)
    plt.bar(x, FP)
    plt.title('False Posititve')
    plt.subplot(222)
    plt.bar(x, FN)
    plt.title('False Negative')
    plt.subplot(223)
    plt.bar(x, TP)
    plt.title('True Positive')
    plt.subplot(224)
    plt.bar(x, TN)
    plt.title('True Negative')
    plt.show()

data = pd.read_csv('sign_mnist_train.csv')
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
print(X.shape)

test = pd.read_csv('sign_mnist_test.csv')
test_X = test.iloc[:, 1:].values
test_y = test.iloc[:, 0].values
print(test_X.shape)

# Loading the models
#Random forest
rfc = joblib.load('rfc.pkl')
print('rfc loaded')
# Predict Results
rfc_predicted = rfc.predict(test_X)
# Calculate Accuracy
rfc_acc = cal_acc(rfc_predicted, test_y)
# Comparison graph for each category
evsp_graph(test_y, rfc_predicted)
# Classification report
result = classification_report(test_y, rfc_predicted)
print(result)
# confusion matrix
rfc_cm = confusion_matrix(test_y, rfc_predicted)
rfc_cm = pd.DataFrame(rfc_cm, columns=[x for x in range(25) if x != 9])
heatmap(rfc_cm, annot=True, fmt='d', cmap='Pastel1')
# Various Metrics
metrics(confusion_matrix(test_y, rfc_predicted))
# Plotting ROC
skplt.metrics.plot_roc(test_y, rfc.predict_proba(test_X))
# Plotting Precision Recall
skplt.metrics.plot_precision_recall(test_y, rfc.predict_proba(test_X))


#ADA Boost
abc = joblib.load('abc.pkl')
print('abc loaded')
# Predict Results
abc_predicted = abc.predict(test_X)
# Calculate Accuracy
abc_acc = cal_acc(abc_predicted, test_y)
# Comparison graph for each category
evsp_graph(test_y, abc_predicted)
# Classification report
result = classification_report(test_y, abc_predicted)
print(result)
# confusion matrix
abc_cm = confusion_matrix(test_y, abc_predicted)
abc_cm = pd.DataFrame(abc_cm, columns=[x for x in range(25) if x != 9])
heatmap(abc_cm, annot=True, fmt='d', cmap='Pastel1')
# Various Metrics
metrics(confusion_matrix(test_y, abc_predicted))
# Plotting ROC
skplt.metrics.plot_roc(test_y, abc.predict_proba(test_X))
# Plotting Precision Recall
skplt.metrics.plot_precision_recall(test_y, abc.predict_proba(test_X))


# KNN
knn = joblib.load('knn.pkl')
print('knn loaded')
# Predict Results
knn_predicted = knn.predict(test_X)
# Calculate Accuracy
knn_acc = cal_acc(knn_predicted, test_y)
# Comparison graph for each category
evsp_graph(test_y, knn_predicted)
# Classification report
result = classification_report(test_y, knn_predicted)
print(result)
# confusion matrix
knn_cm = confusion_matrix(test_y, knn_predicted)
knn_cm = pd.DataFrame(knn_cm, columns=[x for x in range(25) if x != 9])
heatmap(knn_cm, annot=True, fmt='d', cmap='Pastel1')
# Various Metrics
metrics(confusion_matrix(test_y, knn_predicted))
# Plotting ROC
skplt.metrics.plot_roc(test_y, knn.predict_proba(test_X))
# Plotting Precision Recall
skplt.metrics.plot_precision_recall(test_y, knn.predict_proba(test_X))


# XG Boost
xgb = joblib.load('xgb.pkl')
print('xgb loaded')
# Predict Results
xgb_predicted = xgb.predict(test_X)
# Calculate Accuracy
xgb_acc = cal_acc(xgb_predicted, test_y)
# Comparison graph for each category
evsp_graph(test_y, xgb_predicted)
# Classification report
result = classification_report(test_y, xgb_predicted)
print(result)
# confusion matrix
xgb_cm = confusion_matrix(test_y, xgb_predicted)
xgb_cm = pd.DataFrame(xgb_cm, columns=[x for x in range(25) if x != 9])
heatmap(xgb_cm, annot=True, fmt='d', cmap='Pastel1')
# Various Metrics
metrics(confusion_matrix(test_y, xgb_predicted))
# Plotting ROC
skplt.metrics.plot_roc(test_y, xgb.predict_proba(test_X))
# Plotting Precision Recall
skplt.metrics.plot_precision_recall(test_y, xgb.predict_proba(test_X))


# Support Vector
svc = joblib.load('svc.pkl')
print('svc loaded')
# Predict Results
svc_predicted = svc.predict(test_X)
# Calculate Accuracy
svc_acc = cal_acc(svc_predicted, test_y)
# Comparison graph for each category
evsp_graph(test_y, svc_predicted)
# Classification report
result = classification_report(test_y, svc_predicted)
print(result)
# confusion matrix
svc_cm = confusion_matrix(test_y, svc_predicted)
svc_cm = pd.DataFrame(svc_cm, columns=[x for x in range(25) if x != 9])
heatmap(svc_cm, annot=True, fmt='d', cmap='Pastel1')
# Various Metrics
metrics(confusion_matrix(test_y, svc_predicted))
# Plotting ROC
skplt.metrics.plot_roc(test_y, svc.predict_proba(test_X))
# Plotting Precision Recall
skplt.metrics.plot_precision_recall(test_y, svc.predict_proba(test_X))


# Logistic Regression
lr = joblib.load('lr.pkl')
print('Logistic Regression loaded')
# Predict Results
lr_predicted = lr.predict(test_X)
# Calculate Accuracy
lr_acc = cal_acc(lr_predicted, test_y)
# Comparison graph for each category
evsp_graph(test_y, lr_predicted)
# Classification report
result = classification_report(test_y, lr_predicted)
print(result)
# confusion matrix
lr_cm = confusion_matrix(test_y, lr_predicted)
lr_cm = pd.DataFrame(lr_cm, columns=[x for x in range(25) if x != 9])
heatmap(lr_cm, annot=True, fmt='d', cmap='Pastel1')
# Various Metrics
metrics(confusion_matrix(test_y, lr_predicted))
# Plotting ROC
skplt.metrics.plot_roc(test_y, lr.predict_proba(test_X))
# Plotting Precision Recall
skplt.metrics.plot_precision_recall(test_y, lr.predict_proba(test_X))


# Naive Bayes
gnb = joblib.load('nb.pkl')
print('Naive Bayes loaded')
# Predict Results
gnb_predicted = gnb.predict(test_X)
# Calculate Accuracy
gnb_acc = cal_acc(gnb_predicted, test_y)
# Comparison graph for each category
evsp_graph(test_y, gnb_predicted)
# Classification report
result = classification_report(test_y, gnb_predicted)
print(result)
# confusion matrix
gnb_cm = confusion_matrix(test_y, gnb_predicted)
gnb_cm = pd.DataFrame(gnb_cm, columns=[x for x in range(25) if x != 9])
heatmap(gnb_cm, annot=True, fmt='d', cmap='Pastel1')
# Various Metrics
metrics(confusion_matrix(test_y, rfc_predicted))
# Various Metrics
metrics(confusion_matrix(test_y, gnb_predicted))
# Plotting ROC
skplt.metrics.plot_roc(test_y, gnb.predict_proba(test_X))
# Plotting Precision Recall
skplt.metrics.plot_precision_recall(test_y, gnb.predict_proba(test_X))


'''
# Random Forest
print('RANDOM FOREST')
rfc = RandomForestClassifier(n_estimators=500, random_state=21)
rfc.fit(X, y)
rfc_predicted = rfc.predict(test_X)
cal_acc(rfc_predicted, test_y)

# AdaBoost
print('ADA BOOST')
abc = AdaBoostClassifier(n_estimators=500, random_state=0)
abc.fit(X, y)
abc_predicted = abc.predict(test_X)
cal_acc(abc_predicted, test_y)

# KNN
print('KNN')
knn = KNeighborsClassifier(n_neighbors=5, p=1)
knn.fit(X, y)
knn_predicted = knn.predict(test_X)
cal_acc(knn_predicted, test_y)

# XGB Classifier
print('XGB')
xgb = XGBClassifier()
xgb.fit(X, y)
xgb_predicted = xgb.predict(test_X)
cal_acc(xgb_predicted, test_y)

# Support Vector
print('SVC')
svc = SVC(random_state=0, probability=True)
svc.fit(X, y)
svc_predicted = svc.predict(test_X)
cal_acc(svc_predicted, test_y)

# Logistic Regression
print('Logistic Regression')
lr = LogisticRegression(max_iter=5000, random_state=21)
lr.fit(X, y)
lr_predicted = lr.predict(test_X)
cal_acc(lr_predicted, test_y)

# Naive Bayes
print('Naive Bayes')
gnb = MultinomialNB()
gnb.fit(X, y)
gnb_predicted = gnb.predict(test_X)
cal_acc(gnb_predicted, test_y)

# Saving the model
joblib.dump(rfc, 'rfc.pkl')
print('rfc Saved')
joblib.dump(abc, 'abc.pkl')
print('abc Saved')
joblib.dump(knn, 'knn.pkl')
print('knn Saved')
joblib.dump(xgb, 'xgb.pkl')
print('xgb Saved')
joblib.dump(svc, 'svc.pkl')
print('svc Saved')
joblib.dump(lr, 'lr.pkl')
print('logistic Regression Saved')
joblib.dump(gnb, 'nb.pkl')
print('Naive Bayes Saved')
'''

# Accuracy Comparison Graph
acc_list = [rfc_acc, abc_acc, knn_acc, xgb_acc, svc_acc, lr_acc, gnb_acc]
models = ['Random Forest', 'ADA Boost', 'KNN', 'XGB', 'Support Vector', 'Logistic Regression', 'Naive Bayes']

plt.ylim(0, 100)
plt.bar(models, acc_list, width=0.4, color='red')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.show()

# Box Plot
# =============================================================================
# # prepare models
# models = []
# models.append(('RFC', RandomForestClassifier(random_state=21)))
# models.append(('ABC', AdaBoostClassifier(random_state=0)))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('XGB', XGBClassifier()))
# models.append(('SVM', SVC(random_state=0)))
# models.append(('LR', LogisticRegression(random_state=21)))
# models.append(('NB', MultinomialNB()))
# 
# 
# # evaluate each model in turn
# results = []
# names = []
# scoring = 'accuracy'
# for name, model in models:
# # =============================================================================
# #     if name == 'ABC' or name == 'SVM':
# #         seed = 0
# #     else:
# #         seed = 21
# # =============================================================================
#     
#     kfold = model_selection.KFold(n_splits=10)
#     cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
# # boxplot algorithm comparison
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()
# 
# =============================================================================
