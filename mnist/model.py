import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix,precision_score

def main():

	#fetching the data
	mnist = fetch_openml('mnist_784')
	X,Y = mnist["data"],mnist["target"]
	
	#70000 images each of 28*28 pixels
	print(X.shape)
	print(Y.shape)
	
	
	#looking at image
	'''
	some_digit = X[36000]
	some_digit_ans = Y[36000]
	some_digit_image = some_digit.reshape(28,28)
	plt.imshow(some_digit_image,	cmap	=	matplotlib.cm.binary,
											interpolation="nearest")
	plt.axis("off")
	plt.show()
	print(some_digit_ans)
	'''

	x_train,x_test,y_train,y_test = X[:60000],X[60000:],Y[:60000],Y[60000:]

	#sometimes training does not works good if same answer comes too near
	shuffle_index = np.random.permutation(60000)
	X_train,Y_train = x_train[shuffle_index],y_train[shuffle_index]

	#Trying binary classifier for checking number 5
	y_train_5 = (y_train == '5')
	y_test_5 = (y_test == '5')
	print(y_train_5)

	#using sgdc classifier
	sgd_clf = SGDClassifier(random_state = 42)
	sgd_clf.fit(x_train,y_train_5)

	#predicting the value
	#print(sgd_clf.predict([x_train[0]]))
	#print(y_train[0])

	#checking cross validation score
	print(cross_val_score(sgd_clf,x_train,y_train_5,cv=3,scoring= "accuracy"))

	#accuracy is 95% but the testing data is skewed because only 10% data has 5 value so accuracy is already 90%

	#confusion matrix
	y_train_pred  = cross_val_predict(sgd_clf,x_train,y_train_5,cv = 3)
	print(confusion_matrix(y_train_5,y_train_pred))
	print(precision_score(y_train_5,y_train_pred))
	#precision



if __name__ == '__main__':
	main()