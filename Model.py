import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
# import cv2

data = pd.read_csv('sign_mnist_train.csv')
X = data.iloc[:,1:].values
y = data.iloc[:, 0].values

test = pd.read_csv('sign_mnist_test.csv')
test_X = test.iloc[:, 1:].values
test_y = test.iloc[:, 0].values

clf = RandomForestClassifier(n_estimators = 300, random_state = 0)
clf.fit(X,y)

Saving the model
joblib.dump(clf, 'model.pkl')

# Loading the model
# clf = joblib.load('model.pkl')

result = clf.predict(test_X)
count = 0
for i in range(len(result)):
    if result[i] == test_y[i]:
        count += 1
print('Correct Predictions : {}\nIncorrect Predictons : {}\nAccuracy = {}'.format(count, len(result) - count, ((round(count/len(result), 2))*100)))