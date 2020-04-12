import pandas as pd
# =============================================================================
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
# =============================================================================
# import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from seaborn import heatmap
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
    graph = df.plot()
    graph.set_xlabel('Categories')
    graph.set_ylabel('Occurence Count')
    graph.bar()

# =============================================================================
# data = pd.read_csv('sign_mnist_train.csv')
# X = data.iloc[:, 1:].values
# y = data.iloc[:, 0].values
# print(X.shape)
# =============================================================================

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
svc = SVC(random_state=0)
svc.fit(X, y)
svc_predicted = svc.predict(test_X)
cal_acc(svc_predicted, test_y)

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
'''

# Accuracy Comparison Graph
acc_list = [rfc_acc, abc_acc, knn_acc, xgb_acc, svc_acc]
models = ['Random Forest', 'ADA Boost', 'KNN', 'XGB', 'Support Vector']

plt.ylim(0, 100)
plt.bar(models, acc_list, width=0.4, color='red')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.show()

