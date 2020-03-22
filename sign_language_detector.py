import cv2
import joblib

# palm_cascade = cv2.CascadeClassifier('aGest.xml')

clf = joblib.load('model.pkl')
cap = cv2.VideoCapture('test1.mp4')
x = 375
y = 130
h = 448 # 600 - 175
w = 500 # 800 - 352
# count = 1
while 1:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# palms = palm_cascade.detectMultiScale(gray, 1.3, 5)

	# for (x, y, w, h) in palms:
	cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
	gray_palm = gray[y:y+h, x:x+w]
	# # cv2.imshow('grey', gray_palm)
	# gray_palm = cv2.bilateralFilter(gray_palm, 5, 2, 2)
	gray_palm = cv2.resize(gray_palm, (28, 28), interpolation=cv2.INTER_CUBIC)
	result = clf.predict(gray_palm.reshape(1, -1))
	cv2.putText(img, chr(result + 65), (320, 125), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
	# print(chr(result + 65))

	# Display an image in a window
	img = cv2.resize(img, (1024, 720), interpolation=cv2.INTER_AREA)
	cv2.imshow('Hand', img)

	# Wait for Esc key to stop
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
