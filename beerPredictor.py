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


def predictNew(lassoRegressor, X):
	print(lassoRegressor.predict(np.reshape(X, (1, len(X)))))

def rebuildDummies(columns, value):
	if(value in columns):
		columns[columns != value] = 0
		columns[columns == value] = 1
	else:
		columns[columns != 'nan'] = 0
		columns[columns == 'nan'] = 1

	columns = columns.astype(int)

	return columns


def structureNewData(filename, vbreweryCol, vstyleCol, vcountryCol):
	print("Loading new data...")
	dataRaw = loadtxt(filename, dtype=str, comments="`", delimiter="|", unpack=False)

	#Structure data columns
	vabv = np.array(dataRaw[4].astype(np.float))
	vbrew_with = np.array(np.where(dataRaw[15] != '', 1, 0))
	
	tmpdrink_date = int(dataRaw[10][0:4])

	if(dataRaw[16] == '\\N'):
		tmpbrew_year = tmpdrink_date
	else:
		tmpbrew_year = int(dataRaw[16][0:4])

	vferment_years = np.array(tmpdrink_date - tmpbrew_year)

	print(vferment_years)

	vbrewery = rebuildDummies(vbreweryCol, dataRaw[2])
	vstyle = rebuildDummies(vstyleCol, dataRaw[6])
	vcountry = rebuildDummies(vcountryCol, dataRaw[7])

	data = np.hstack((vabv, vbrewery, vstyle, vcountry, vbrew_with, vferment_years))

	return data



def testRegression(lassoRegressor, trainX, trainY, tolerance):
	trainPredict = lassoRegressor.predict(trainX)

	count = 0
	idx = 0

	np.savetxt("trainY.txt", trainY, comments='`')
	np.savetxt("predictY.txt", trainPredict, comments='`')

	while idx < len(trainPredict):
		tp = float(trainPredict[idx])
		ty = float(trainY[idx])
		if(tp > ty - tolerance and tp < ty + tolerance):
			count += 1
		idx += 1
	
	accuracy = (count / len(trainPredict)) * 100
	print("Accuracy of model: ", accuracy)


def trainRegression(trainX, trainY, cvX, cvY):

	
	lasso = lm.Lasso(max_iter = 2000)

	alphas = [0.0001, 0.0003, 0.001, 0.003, 0.03, 0.1, 0.3, 1, 3]

	oldError = -1

	for alpha in alphas:
		lasso = lm.Lasso(alpha=alpha, max_iter = 2000)
		lasso.fit(trainX, trainY)
		cvPredict = lasso.predict(cvX)
		newError = mc.mean_squared_error(cvPredict, cvY)

		
		if(newError < oldError or oldError == -1):
			bestLasso = lasso
			oldError = newError
	
	return bestLasso



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

def structureTrainFrementYear(tmpbrew_year, tmpdrink_date):
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


def structureTrainData(filename):

	dataRaw = loadtxt(filename, dtype=str, comments="`", delimiter="|", unpack=False)

	# Sorting array by drink date
	print("Sorting array based on drink-date...")
	dataRaw = dataRaw[np.argsort(dataRaw[:, 10])]

	#Extract labels
	print("Splitting labels from input...")
	labels = np.c_[dataRaw[:,5].astype(np.float)]	

	#Structure data columns
	vbrewery = pandas.get_dummies(dataRaw[:,2],dummy_na=True)
	vabv = np.c_[dataRaw[:,4].astype(np.float)]		# Need to np.c_[] to define as a column vector for hstack / Need to convert to float because otherwise lasso will convert strings to float representation
	vstyle = pandas.get_dummies(dataRaw[:,6],dummy_na=True)
	vcountry = pandas.get_dummies(dataRaw[:,7],dummy_na=True)
	vbrew_with = np.c_[np.where(dataRaw[:,15] != '', 1, 0)]
	vferment_years = structureTrainFrementYear(dataRaw[:,16], dataRaw[:,10])

	print("Stracking structured vectors...")
	data = np.hstack((vabv, vbrewery, vstyle, vcountry, vbrew_with, vferment_years))

	print("Final transform size: ", data.shape)

	vbreweryCol = np.asarray(vbrewery.columns.tolist())
	vstyleCol = np.asarray(vstyle.columns.tolist())
	vcountryCol = np.asarray(vcountry.columns.tolist())

	print("BrewCol: ", len(vbreweryCol))
	print("StyleCol: ", len(vstyleCol))
	print("CountryCol: ", len(vcountryCol))


	return (data, labels, vbreweryCol, vstyleCol, vcountryCol)


