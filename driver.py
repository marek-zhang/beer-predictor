import beerPredictor as bp 

(X, Y, vbreweryCol, vstyleCol, vcountryCol, totalCols) = bp.structureTrainData("beers4999.csv")

(trainX, trainY, crossValX, crossValY, testX, testY) = bp.splitSets(X, Y, 10, 10, 80)
lassoRegressor = bp.trainRegression(trainX, trainY, crossValX, crossValY)
bp.testRegression(lassoRegressor, trainX, trainY, 0.5)


#newX = bp.structureNewDataSingle("newBeer.csv", vbreweryCol, vstyleCol, vcountryCol)
#bp.predictNew(lassoRegressor, newX)


newLarge = bp.structureNewDataMulti("beersLargePredit.csv", vbreweryCol, vstyleCol, vcountryCol, totalCols)
bp.predictNew(lassoRegressor, newLarge)


