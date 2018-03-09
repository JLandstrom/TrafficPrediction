from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SarimaTrafficModeller:

    def __init__(self):
        pass

    def SarimaPrediction(self, train, test, nonSeasonalArgument, seasonalArguments, predictionsSteps=1):
        train = np.array(train).flatten()
        test = np.array(test).flatten()
        history = [x for x in train]
        predictions = list()

        for t in range(0, len(test), predictionsSteps):
            model = SARIMAX(history, order=nonSeasonalArgument, seasonal_order=seasonalArguments)
            model_fit = model.fit()
            print(model_fit.summary())
            output = model_fit.forecast(predictionsSteps)
            predictions.extend(output)
            for x in range(t, t + predictionsSteps):
                history.append(test[x])

        return predictions

    def NnHistoricalAveragePrediction(self,train,test):
        predictions = list()
        for date in test.index:
            history = train[train.index.time == date.time()]
            predictions.append(history['NoVehicles'].mean())
        return predictions

    def Mean_absolute_percentage_error(self,yTrue, yPrediction):
        yTrue, yPrediction = np.array(yTrue), np.array(yPrediction)
        return np.mean(np.abs((yTrue - yPrediction) / yTrue)) * 100

    def Evaluate(self, title, yTrue, yPrediction):
        rmse = math.sqrt(mean_squared_error(yTrue, yPrediction))
        mape = self.Mean_absolute_percentage_error(yTrue, yPrediction)
        mae = mean_absolute_error(yTrue, yPrediction)

        print("Evaluation results - " + title)
        print("-----------------------------------------------------")
        print("RMSE: %.3f" % rmse)
        print("MAPE: %.3f" % mape)
        print("MAE: %.3f" % mae)
        print("-----------------------------------------------------")

    def PlotPredictions(self, title, yTrue, sarimaPredictions, historicalPredictions=None):
        plt.plot(yTrue, color='blue')
        yTruePatch = patch.Patch(color='blue', label='Original data')
        plt.plot(sarimaPredictions, color='red')
        sarimaPredictionsPatch = patch.Patch(color='red', label='SARIMA')
        if historicalPredictions == None:
            plt.legend(handles=[yTruePatch, sarimaPredictionsPatch])
        else:
            plt.plot(historicalPredictions, color='green')
            historicalPredictionsPatch = patch.Patch(color='green', label='Historical NN Avarage')
            plt.legend(handles=[yTruePatch, sarimaPredictionsPatch, historicalPredictionsPatch])

        plt.title(title)
        plt.gca().yaxis.grid(True, which='major', ls='dotted')
        plt.show()

class PredictionResultContainer:
    def __init__(self, predictions, plotColor, plotLabel):
        self.predictions = predictions
        self.plotColor = plotColor
        self.plotLabel = plotLabel