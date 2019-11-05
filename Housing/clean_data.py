import os
import pandas as pd

def drop_missing(df,columns):
	''' removes rows for which the columns contains null values '''

	return df.dropna(subset = columns)
		

def drop_column(df,columns):
	''' drops columns from dataframe '''

	for col in columns:
		df =  df.drop(col,axis=1)
	return df

def fill_median(df,columns):
	''' for each column null values will be replcaed by their median values '''

	for col in columns:
		if col in df:
			median = df[col].median()
			df[col].fillna(median,inplace=True)
	return df	



