import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import jupyter 
import pandas 
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model as lm
from sklearn import metrics as mc
from numpy import loadtxt


def predictNew(lasso_regressor, X):
	print(lasso_regressor.predict(X))


def runRegression(X, Y, cvX, cvY):

	
	lasso = lm.Lasso(max_iter = 2000)

	parameters = {'alpha': [0.0001, 0.0003, 0.001, 0.003, 0.03, 0.1, 0.3, 1, 3]}

	lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)

	lasso_regressor.fit(X, Y)


	print(lasso_regressor.best_params_)
	print(lasso_regressor.best_score_)

	return lasso_regressor




def splitSets(data, label, testPercent, crossValPercent, trainPercent):
	if(trainPercent + crossValPercent + testPercent > 100):
		print("ERROR: Sum of percentages greater than 100")
		exit()

	totalRows = len(data)
	
	#rounding all down, I may lose one record if testing 100% of data.
	testSize = int(totalRows * (testPercent/100))
	crossValSize = int(totalRows * (crossValPercent/100))
	trainSize = int(totalRows * (trainPercent/100))

	print("Size of test set: ", testSize)
	print("Size of cross validation set: ", crossValSize)
	print("Size of train set: ", trainSize)


	crossValRowStart = totalRows - testSize - crossValSize
	crossValRowEnd = totalRows - testSize

	trainRowStart = totalRows - testSize - crossValSize - trainSize
	trainRowEnd = totalRows - testSize - crossValSize


	testX = data[totalRows - testSize:totalRows,:]
	testY = label[totalRows - testSize:totalRows]


	crossValX = data[crossValRowStart : crossValRowEnd , :]
	crossValY = label[crossValRowStart : crossValRowEnd]

	trainX = data[trainRowStart : trainRowEnd , :]
	trainY = label[trainRowStart : trainRowEnd]

	return (trainX, trainY, crossValX, crossValY, testX, testY)

def structureFrementYear(tmpbrew_year, tmpdrink_date):
	tmpbrew_year[tmpbrew_year == '\\N'] = 0
	tmpbrew_year[tmpbrew_year == ''] = 0
	tmpbrew_year = list(map(int, tmpbrew_year)) 

	tmpdrink_date[tmpdrink_date == ''] = '0000'
	tmpdrink_date = list(map(lambda x: x[0:4], tmpdrink_date))
	tmpdrink_date = list(map(int, tmpdrink_date)) 

	vferment_years = np.zeros(len(tmpbrew_year))

	for idx in range(len(tmpbrew_year)):
		if tmpbrew_year[idx] > 0:
			vferment_years[idx] = tmpdrink_date[idx] - tmpbrew_year[idx]

	vferment_years = np.c_[vferment_years]

	return vferment_years


def structureData(filename):

	lines = loadtxt(filename, dtype=str, comments="`", delimiter="|", unpack=False)

	print("rows: ", len(lines))
	print("columns: ", len(lines[0]))

	# Sorting array by drink date
	print("Sorting array based on drink-date...")
	dataRaw = lines[np.argsort(lines[:, 10])]

	#Extract labels
	print("Splitting labels from input...")
	labels = np.c_[dataRaw[:,5].astype(np.float)]	

	#Structure data columns
	vbrewery = pandas.get_dummies(dataRaw[:,2],dummy_na=True)
	vabv = np.c_[dataRaw[:,4].astype(np.float)]		# Need to np.c_[] to define as a column vector for hstack / Need to convert to float because otherwise lasso will convert strings to float representation
	vstyle = pandas.get_dummies(dataRaw[:,6],dummy_na=True)
	vcountry = pandas.get_dummies(dataRaw[:,7],dummy_na=True)
	vbrew_with = np.c_[np.where(dataRaw[:,15] != '', 1, 0)]
	vferment_years = structureFrementYear(dataRaw[:,16], dataRaw[:,10])


	print("Stracking structured vectors...")
	data = np.hstack((vabv, vbrewery, vstyle, vcountry, vbrew_with, vferment_years))

	print("Final Transform Size: ", data.shape)

	return (data, labels)


