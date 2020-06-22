import beerPredictor as bp 

(X, Y, vbreweryCol, vstyleCol, vcountryCol) = bp.structureTrainData("beers.csv")

(trainX, trainY, crossValX, crossValY, testX, testY) = bp.splitSets(data, labels, 10, 10, 80)

lassoRegressor = bp.runRegression(trainX, trainY, crossValX, crossValY)

newX = bp.structureNewData("newBeer.csv", vbreweryCol, vstyleCol, vcountryCol)

#print("Predict data size: ", predictData.shape)
#bp.predictNew(lassoRegressor, predictData)




