import pandas as pd

def predict(predictor,testSet,activites, actualValues, resultFile, crossValidation = False):
    predictedResult = predictor.predict(testSet)
    if crossValidation:
        result = sum([predictedResult[i] == actualValues[i] for i in range(len(actualValues))]) * 100.0 / len(actualValues)
        return result;
    else:
        resultDataFrame =pd.DataFrame();
        resultDataFrame["activity_id"] = activites;
        resultDataFrame["outcome"] = predictedResult;
        resultDataFrame.to_csv(resultFile, index=False)
        print "Predict result saved to %s "%resultFile
    return 0
