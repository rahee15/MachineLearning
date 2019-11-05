from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,Imputer
from Housing import load_housing_data
num_pipeline = Pipeline([
	('imputer',Imputer(strategy="median")),
	('std_scaler',StandardScaler()),
	])


def main():
	df = load_housing_data()
	df = df.drop("ocean_proximity",axis=1)
	housing_num_tr = num_pipeline.fit_transform(df)


if __name__ == '__main__':
	main()