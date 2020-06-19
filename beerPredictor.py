import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import jupyter 
import pandas 
from sklearn import linear_model as lm
from numpy import loadtxt

def runRegression(X, Y, cvX, cvY, alpha, normalize):
	clf = lm.Lasso(alpha=alpha, normalize=normalize)
	clf.fit(X, Y)
	print(clf.coef_)
	print(clf.intercept_)

	cvX = cvX.astype(np.float64)

	p = clf.predict(cvX)

	accuracy = np.mean(np.astype(p == cvY)) * 100
	print("Accuracy: ", accuracy)



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

	print("size testY : ", testY.shape)

	crossValX = data[crossValRowStart : crossValRowEnd , :]
	crossValY = label[crossValRowStart : crossValRowEnd]

	trainX = data[trainRowStart : trainRowEnd , :]
	trainY = label[trainRowStart : trainRowEnd]

	return (trainX, trainY, crossValX, crossValY, testX, testY)


def structureData(filename):

	lines = loadtxt(filename, dtype=str, comments="`", delimiter="|", unpack=False)

	print("rows: ", len(lines))
	print("columns: ", len(lines[0]))

	#Delete column headings
	dataRaw = np.delete(lines, 0, 0)


	# Sorting array by drink date
	print("Sorting array based on drink-date...")
	dataRaw = dataRaw[np.argsort(dataRaw[:, 10])]

	#Extract labels

	print("Splitting labels from input...")
	labels = dataRaw[:,5]

	vbrewery = pandas.get_dummies(dataRaw[:,2])
	vabv = dataRaw[:,4]
	vstyle = pandas.get_dummies(dataRaw[:,6])
	vcountry = pandas.get_dummies(dataRaw[:,7])
	vbrew_with = np.where(dataRaw[:,15] != '', 1, 0)

	tmpbrew_year = dataRaw[:,16]
	tmpbrew_year[tmpbrew_year == '\\N'] = 0
	tmpbrew_year[tmpbrew_year == ''] = 0
	tmpbrew_year = list(map(int, tmpbrew_year)) 

	tmpdrink_date = dataRaw[:,10]
	tmpdrink_date[tmpdrink_date == ''] = '0000'
	tmpdrink_date = list(map(lambda x: x[0:4], tmpdrink_date))
	tmpdrink_date = list(map(int, tmpdrink_date)) 

	vferment_years = np.zeros(len(tmpbrew_year))

	for idx in range(len(tmpbrew_year)):
		if tmpbrew_year[idx] > 0:
			vferment_years[idx] = tmpdrink_date[idx] - tmpbrew_year[idx]


	print("Stracking structured vectors...")
	data = np.column_stack([vabv, vbrewery])
	data = np.column_stack([data, vstyle])
	data = np.column_stack([data, vcountry])
	data = np.column_stack([data, vbrew_with])
	data = np.column_stack([data, vferment_years])
	print("Final Transform Size: ", data.shape)

	np.savetxt("transformed.csv", data, delimiter=",")

	return (data, labels)


