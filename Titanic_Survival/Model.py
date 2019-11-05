import os 
import pandas as pd
from DataFrameTransform import DataFrameSelector,MostFrequentImputer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def drop_column(df,columns):
	''' drops columns from dataframe '''

	for col in columns:
		df =  df.drop(col,axis=1)
	return df


def clean_data(train_data):
	train_data = drop_column(train_data,["Name","PassengerId","Ticket"])
	num_pipeline = Pipeline([
		("select_numeric",DataFrameSelector(["Age","SibSp","Parch","Fare"])),
		(("imputer"),SimpleImputer(strategy = "median")),
		])
	cat_pipeline = Pipeline([
		("select_cat",DataFrameSelector(["Pclass","Sex","Embarked"])),
		("imputer",MostFrequentImputer()),
		("cat_encoder",OneHotEncoder(sparse=False)),
		])
	
	preprocess_pipeline = FeatureUnion(transformer_list = [
		("num_pipeline",num_pipeline),
		("cat_pipeline",cat_pipeline),
		])
	return preprocess_pipeline.fit_transform(train_data)
	

def main():
	train_data = pd.read_csv("./Dataset/train.csv")
	test_data = pd.read_csv("./Dataset/test.csv")
	x_train = clean_data(train_data)
	y_train = train_data["Survived"]
	svm_clf = SVC(gamma = "auto" )
	svm_clf.fit(x_train,y_train)
	x_test = clean_data(test_data)
	y_pred = svm_clf.predict(x_test)
	#y_test = test_data["Survived"]
	svm_score = cross_val_score(svm_clf,x_train,y_train,cv = 10)
	print(svm_score.mean())

if __name__ == '__main__':
	main()