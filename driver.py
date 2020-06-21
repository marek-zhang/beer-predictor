import beerPredictor as bp 

(data, labels) = bp.structureData("beers.csv")

(trainX, trainY, crossValX, crossValY, testX, testY) = bp.splitSets(data, labels, 10, 10, 80)

#lasso_regressor = bp.runRegression(trainX, trainY, crossValX, crossValY)
lasso_regressor = bp.runRegression(data, labels, crossValX, crossValY)

#(newData, newLabels) = bp.structureData("newBeer.csv")

#bp.predictNew(lasso_regressor, newData)




