import beerPredictor as bp 

(data, labels) = bp.structureData("beers.csv")

(trainX, trainY, crossValX, crossValY, testX, testY) = bp.splitSets(data, labels, 10, 10, 80)

bp.runRegression(trainX, trainY, crossValX, crossValY, 1.0, False)

#Lasso