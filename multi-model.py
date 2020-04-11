import pandas as pd
# =============================================================================
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
# =============================================================================
# import numpy as np
import joblib
# import cv2

def cal_acc(result, test):
    count = 0
    for i in range(len(result)):
        if result[i] == test_y[i]:
            count += 1
    print('Correct Predictions : {}\nIncorrect Predictions : {}\nAccuracy = {}%'.format(count, len(result) - count, ((round(count/len(result), 2))*100)))


data = pd.read_csv('sign_mnist_train.csv')
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

print(X.shape)

test = pd.read_csv('sign_mnist_test.csv')
test_X = test.iloc[:, 1:].values
test_y = test.iloc[:, 0].values

print(test_X.shape)

# Loading the model
rfc = joblib.load('rfc.pkl')
print('rfc loaded')
rfc_predicted = rfc.predict(test_X)
cal_acc(rfc_predicted, test_y)

abc = joblib.load('abc.pkl')
print('abc loaded')
abc_predicted = abc.predict(test_X)
cal_acc(abc_predicted, test_y)

knn = joblib.load('knn.pkl')
print('knn loaded')
knn_predicted = knn.predict(test_X)
cal_acc(knn_predicted, test_y)

xgb = joblib.load('xgb.pkl')
print('xgb loaded')
xgb_predicted = xgb.predict(test_X)
cal_acc(xgb_predicted, test_y)

svc = joblib.load('svc.pkl')
print('svc loaded')
svc_predicted = svc.predict(test_X)
cal_acc(svc_predicted, test_y)


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
