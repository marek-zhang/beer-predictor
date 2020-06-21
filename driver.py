import beerPredictor as bp 

(data, labels, vbreweryCol, vstyleCol, vcountryCol) = bp.structureTrainData("beers.csv")

(trainX, trainY, crossValX, crossValY, testX, testY) = bp.splitSets(data, labels, 10, 10, 80)

#lasso_regressor = bp.runRegression(trainX, trainY, crossValX, crossValY)
lasso_regressor = bp.runRegression(data, labels, crossValX, crossValY)

predictData = bp.structureNewData("newBeer.csv", vbreweryCol, vstyleCol, vcountryCol)

print("Predict data size: ", predictData.shape)
bp.predictNew(lasso_regressor, predictData)




