import matplotlib.pyplot as plt
import os
import pandas as pd
import tarfile
from six.moves import urllib
from sklearn.model_selection import train_test_split
from clean_data import drop_missing,drop_column,fill_median
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


DOWNLOAD_ROOT="https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH=os.path.join("datasets","housing")
HOUSING_URL=DOWNLOAD_ROOT+"datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
	
	''' Fetches Data and extract it '''

	if not os.path.isdir(housing_path):
		os.makedirs(housing_path)
	tgz_path=os.path.join(housing_path,"housing.tgz")
	urllib.request.urlretrieve(housing_url,tgz_path)
	housing_tgz=tarfile.open(tgz_path)
	housing_tgz.extractall(path=housing_path)
	housing_tgz.close()
	print(tgz_path)

def load_housing_data(housing_path=HOUSING_PATH):
	csv_path=os.path.join(housing_path,"housing.csv")
	return pd.read_csv(csv_path)

def main():
	#fetch_housing_data()
	housing=load_housing_data()
	'''
	print("Head data is \n {}".format(housing.head()))
	print("info data is \n {}".format(housing.info()))
	print("\n {} \n".format(housing["ocean_proximity"].value_counts()))
	print("Describe Method \n {} \n".format(housing.describe()))x
	housing.hist(bins=50,figsize=(20,15))
	plt.show()
	'''
	train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
	
	#train_set_copy = train_set.copy()
	#housing.plot(kind="scatter",x="longitude",y="latitude",alpha="0.4",s=housing["population"]/100,label="population",figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
	#plt.show()
	
	#cor_matrix = housing.corr()
	#print(cor_matrix["median_house_value"].sort_values(ascending=False))

	#housing  = drop_missing(housing,["total_rooms"])
	#housing = drop_column(housing,["total_rooms","total_bedrooms"])
	#housing = fill_median(housing,["total_rooms","total_bedrooms"])

	#one hot encoding
	encoder = LabelEncoder()
	housing_cat = housing["ocean_proximity"]
	housing_cat_encoded = encoder.fit_transform(housing_cat)
	print(housing_cat_encoded)
	print(encoder.classes_)

	encoder = OneHotEncoder()
	housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
	print(housing_cat_1hot.toarray()) 
	#print(housing.info())



if __name__ == '__main__':
        	main()        