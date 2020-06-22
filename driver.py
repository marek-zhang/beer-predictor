import beerPredictor as bp 

(X, Y, vbreweryCol, vstyleCol, vcountryCol) = bp.structureTrainData("beers.csv")

(trainX, trainY, crossValX, crossValY, testX, testY) = bp.splitSets(X, Y, 10, 10, 80)

lassoRegressor = bp.trainRegression(trainX, trainY, crossValX, crossValY)

bp.testRegression(lassoRegressor, trainX, trainY, 0.5)


#newX = bp.structureNewData("newBeer.csv", vbreweryCol, vstyleCol, vcountryCol)
#bp.predictNew(lassoRegressor, newX)




