import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patch

import FileHandler
import TrafficPreprocessor
import TrafficAnalysisHelper
import TrafficModeller

#region Application Settings
sys._enablelegacywindowsfsencoding()

"""Global variables"""
"""Spefic to the format of the file to read"""
allColumns = ["DetectorID", "VehicleClassID", "Timestamp", "Flow", "Speed", "Occupancy", "Confidence", "Tdiff",
              "TimeCycle", "NoVehicles", "Headway", "MeasuresIncluded"]
selectedColumns = ["DetectorID", "VehicleClassID", "Timestamp", "Flow", "NoVehicles"]
dayType = ['weekday', 'weekend']

"""Script parameters - maybe make input variables for them"""
"""Set these up before running script"""



#Import settings
shouldImport = False                            #if data should be imported from original files
detectorIds = [1064,1065]                       #detector ids to extract from original files
inputDirectory = "TrafficDataFiles"             #directory path from which to import files

#Preprocessing settings
aggregationinterval = 30                        #Aggregetation interval(minutes) in time series
Threshold = 0.9                                 #percentage of non-nans to keep day in time series
ClusterMethodDictionary = None                  #dictionary of 'clustername':func for clustring parts of day
ClusterDayType = True                           #cluster data into weekdays and weekends

#General settings
SeasonLength = int(1440/aggregationinterval)    #number of lags in a day
TestDays = 1                                    #No of test days
TrainingDays = 5 + 1 + TestDays                #No of training days
CustomTrainingSet = True                        #only use same day in training set
Scenario = 'Scenario 1'
PlotConsolidatedPredictionResults = True       # Print saved result to combine in single plot
ScenarioTitle = "Results Scenario 1-3"

#Analysis settings
RunDataAnalyze = False                           #specifies whether to print plots for analysis
EvaluateModels = False                          #specifies whether to create AIC/BIC matrix for model fitness evaluation
Seasonal = 1                                    #seasonal differencing for analysis
NonSeasonal = 0                                 #nonseasonal differencing for analysis
Nlags = SeasonLength*2                                    #No of lags to print i acf and pacf
MaxNonSeasonal = (2,1,2)                        #Model evaluation: max values for (p,d,q)
MaxSeasonal = (0,1,1)                           #Model evaluation: max values for (P,D,Q)

#Forecast settings
RunForecastProcess = False                      #Specifies if forecast should be done
TimeStepsToForecast = 1                         #No of step-ahead forecasts
NonSeasonalArgument = (1,0,0)                   #Nonseasonal orders (p,d,q)
SeasonalArgument = (0,1,0,SeasonLength)         #Seasonal orders (P,D,Q,S)
RunNaivePrediction = True                      #If comparative prediction should be done
PredictionSteps = SeasonLength                  #How many steps to forecast
SaveResult = True                               #Saves prediction results


#endregion

def ForecastManager(train,test):
    yTrue = np.array(test).flatten()
    trafficModeller = TrafficModeller.SarimaTrafficModeller()
    resultsToSave = pd.DataFrame()
    resultsToSave['yTrue' + Scenario] = yTrue
    sarimaResults = trafficModeller.SarimaPrediction(train, test,NonSeasonalArgument, SeasonalArgument,PredictionSteps)
    resultsToSave['SARIMA ' + Scenario] = sarimaResults
    trafficModeller.Evaluate("Sarima Forecast\nSettings:\ntraindays: " +
                             repr(TrainingDays) + "\nseason:" + repr(SeasonLength) +
                             "\nagg:" + repr(aggregationinterval) + "\nhorizon:" + repr(PredictionSteps),
                             yTrue,
                             sarimaResults)
    if RunNaivePrediction:
        nnResults = trafficModeller.NnHistoricalAveragePrediction(train, test)
        resultsToSave['NNHistorical ' + Scenario] = nnResults
        trafficModeller.Evaluate("Sarima Forecast\nSettings:\ntraindays: " +
                                 repr(TrainingDays) + "\nseason:" + repr(SeasonLength) +
                                 "\nagg:" + repr(aggregationinterval) + "\nhorizon:" + repr(PredictionSteps),
                                 yTrue,
                                 nnResults)
        trafficModeller.PlotPredictions("Traffic Forecast", yTrue, sarimaResults, nnResults)

    else:
        trafficModeller.PlotPredictions("Traffic Forecast", yTrue, sarimaResults)
    if SaveResult:
        resultsToSave.index = test.index
        resultsToSave.to_csv(Scenario + '.csv', sep=';')

def AnalysisManager(dataset):
    trafficAnalyzer = TrafficAnalysisHelper.TrafficAnalyzer()
    tempDataSet = trafficAnalyzer.DifferenceDataset(dataset, NonSeasonal, Seasonal, SeasonLength)
    trafficAnalyzer.AdfTest(tempDataSet)
    trafficAnalyzer.PlotDataset(tempDataSet, "Traffic data diff:" + repr(NonSeasonal) + " sdiff:" + repr(Seasonal) + "|" + repr(SeasonLength))
    trafficAnalyzer.PlotDataset(tempDataSet, "Traffic data diff:" + repr(NonSeasonal) + " sdiff:" + repr(Seasonal) + "|" + repr(SeasonLength))
    trafficAnalyzer.PlotAcf(tempDataSet, "Traffic data diff:" + repr(NonSeasonal) + " sdiff:" + repr(Seasonal) + "|" + repr(SeasonLength), nlags=Nlags)
    trafficAnalyzer.PlotPacf(tempDataSet, "Traffic data diff:" + repr(NonSeasonal) + " sdiff:" + repr(Seasonal) + "|" + repr(SeasonLength), nlags=Nlags)

    if EvaluateModels:
        trafficAnalyzer.FindBestModelFitness(trainSet, MaxNonSeasonal, MaxSeasonal, SeasonLength)

def PlotPredictionsFromFile():
    colors = ['blue','green', 'red', 'yellow', ' pink', 'orange']
    aggregatedResults = pd.read_csv('aggregated_results.csv', sep=";", decimal=",", encoding="utf-8", index_col=0, header=0, low_memory=False)
    colorindex = 0
    patches = list()
    for columnName in aggregatedResults.columns.values:
        plt.plot(np.array(aggregatedResults[columnName].values).astype(float), color=colors[colorindex])
        patches.append(patch.Patch(color=colors[colorindex], label=columnName))
        colorindex += 1
    plt.title(ScenarioTitle)
    plt.legend(handles=patches)
    plt.gca().yaxis.grid(True, which='major', ls='dotted')
    plt.show()

if __name__ == "__main__":
    fileHandler = FileHandler.CsvTrafficDataHandler(inputDirectory, allColumns, detectorIds)
    preprocessor = TrafficPreprocessor.TrafficPreprocessor(mdHandler='linear')
    if shouldImport:
        result = fileHandler.ReadFiles()
        dataset = fileHandler.dataset
        if result:
            dataset = preprocessor.ExtractDataFromColumns(dataset, selectedColumns)
            fileHandler.WriteFile(dataset)
        else:
            raise IOError('Error importing from: ' + inputDirectory)
    else:
        dataset = pd.read_csv(fileHandler.outputFilePath, sep=";", decimal=",", encoding="utf-8", header=0, low_memory=False)

    dataset = preprocessor.StandardizeTimeSeries(dataset, intervalInMinutes=aggregationinterval,threshold=Threshold)
    dataset, levels = preprocessor.Cluster(dataset, methodDictionaries=ClusterMethodDictionary, sortDayType=ClusterDayType)
    if ClusterDayType == True:
        dataset = preprocessor.Filter(dataset, 'weekday', levels)
    if CustomTrainingSet:
        trainSet = dataset[:int(-TestDays * SeasonLength)]
        testSet = dataset[int(-TestDays * SeasonLength):]
        trainSet = trainSet[trainSet.index.dayofweek.values == testSet.index[0].dayofweek]
    else:
        trainSet = dataset[int(-TrainingDays * SeasonLength):int(-TestDays * SeasonLength)]
        testSet = dataset[int(-TestDays * SeasonLength):]
    if RunDataAnalyze:
        AnalysisManager(trainSet)
    if RunForecastProcess:
        ForecastManager(trainSet, testSet)
    if PlotConsolidatedPredictionResults:
        PlotPredictionsFromFile()


#region Obsolete
# def premorningrush(time):
#     return (time < pd.datetime(1, 1, 1, 6).time())
# def morningrush(time):
#     return ((time >= pd.datetime(1,1,1,6,).time()) & (time < pd.datetime(1,1,1,9).time()))
# def middleoftheday(time):
#     return ((time >= pd.datetime(1, 1, 1, 9, ).time()) & (time < pd.datetime(1, 1, 1, 15).time()))
# def afternoonrush(time):
#     return ((time >= pd.datetime(1,1,1,15,).time()) & (time < pd.datetime(1,1,1,18).time()))
# def postafternoonrush(time):
#     return (time >= pd.datetime(1,1,1,18,).time())
# def default(time):
#     return True
#
# def GetClusterDict():
#     return {
#         'premorningrush' : premorningrush,
#         'morningrush' : morningrush,
#         'middleoftheday' : middleoftheday,
#         'afternoonrush' : afternoonrush,
#         'postafternoonrush' : postafternoonrush
#     }, ['premorningrush','morningrush','middleoftheday','afternoonrush', 'postafternoonrush']
#
# def PlotDetectorData(dataFrame):
#     dataFrame['Timestamp'] = pd.to_datetime(dataFrame['Timestamp'])
#     dataFrame = dataFrame[dataFrame['VehicleClassID'] == 0].groupby(['Timestamp'])['NoVehicles'].sum()
#     # dataFrame = dataFrame.groupby(dataFrame.index.map(lambda t: t.day)).sum()
#     # dataFrame = dataFrame.groupby(dataFrame.index.map(lambda t: t.day)).sum()
#     dataFrame.plot()
#     plt.show()
#
#
# def plotClusters(dataset, clusters, date=None):
#     if (date == None):
#         filteredFrame = dataset
#     else:
#         filteredFrame = dataset.xs(date, level=0, drop_level=False)
#
#     for cluster in clusters:
#         clusterFrame = filteredFrame.xs(cluster, level=1, drop_level=False)
#         with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
#             print(clusterFrame)
#         clusterFrame.plot(y='NoVehicles')
#         plt.title(cluster)
#         plt.show()
#         acf(clusterFrame.values)
#         plt.title(cluster + " acf")
#         plt.show()
#         pacf(clusterFrame.values)
#         plt.title(cluster + " pacf")
#         plt.show()
#
#
# def TrainArimaWholeSet(dataset):
#     morningData = dataset.xs('morning', level=1, drop_level=True)
#     print(morningData)
#     model = ARIMA(morningData, order=(2, 1, 1))
#     model_fit = model.fit(disp=0)
#     print(model_fit.summary())
#     # plot residual errors
#     residuals = pd.DataFrame(model_fit.resid)
#     residuals.plot()
#     plt.show()
#     residuals.plot(kind='kde')
#     plt.show()
#     print(residuals.describe())
#endregion