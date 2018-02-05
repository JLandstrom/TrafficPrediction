
import getopt
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.dates as dates


import FileHandler
import TrafficPreprocessor
import CorrelationExperimenter

sys._enablelegacywindowsfsencoding()

"""Global variables"""
"""Spefic to the format of the file to read"""
allColumns = ["DetectorID", "VehicleClassID", "Timestamp", "Flow", "Speed", "Occupancy", "Confidence", "Tdiff",
              "TimeCycle", "NoVehicles", "Headway", "MeasuresIncluded"]
selectedColumns = ["DetectorID", "VehicleClassID", "Timestamp", "Flow", "NoVehicles"]
clusters = ['dawn', 'morning', 'lunch','afternoon','dusk']
dayType = ['weekday', 'weekend']

"""General variables"""
detectorIds = []
inputfile = ""
shouldPlot = False
shouldRead = False

"""Methods"""
"""Handling program arguments"""

def toBool(arg):
    if arg.lower() == 'true':
        return True
    if arg.lower() == 'false':
        return False
    raise ValueError("Input argument -r (read) must be 'true' or 'false'")

def main(argv):
    global inputfile, detectorIds, shouldPlot, shouldRead, shouldWrite
    try:
        opts, args = getopt.getopt(argv, "i:d:p:r:", ["inputfile=", "detectors=", "plot=", "read="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            sys.exit()
        elif opt in ("-i", "--inputfile"):
            inputfile = arg
        elif opt in ("-d", "--detectors"):
            detectorIds = arg.split(',')
            detectorIds = list(map(int, detectorIds))
        elif opt in ("-p", "--plot"):
            shouldPlot = toBool(arg)
        elif opt in ("-r", "--read"):
            shouldRead = toBool(arg)


# Shall be moved to its own class
def PlotDetectorData(dataFrame):
    dataFrame['Timestamp'] = pd.to_datetime(dataFrame['Timestamp'])
    dataFrame = dataFrame[dataFrame['VehicleClassID'] == 0].groupby(['Timestamp'])['NoVehicles'].sum()
    #dataFrame = dataFrame.groupby(dataFrame.index.map(lambda t: t.day)).sum()
    #dataFrame = dataFrame.groupby(dataFrame.index.map(lambda t: t.day)).sum()
    dataFrame.plot()
    plt.show()

def plotClusters(dataset, clusters, date=None):
    if(date==None):
        filteredFrame = dataset
    else:
        filteredFrame = dataset.xs(date,level=0,drop_level=False)

    for cluster in clusters:
        clusterFrame = filteredFrame.xs(cluster,level=1,drop_level=False)
        with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
            print(clusterFrame)
        clusterFrame.plot(y='NoVehicles')
        plt.title(cluster)
        plt.show()
        acf(clusterFrame.values)
        plt.title(cluster + " acf")
        plt.show()
        pacf(clusterFrame.values)
        plt.title(cluster + " pacf")
        plt.show()


def TrainArimaWholeSet(dataset):
    morningData = dataset.xs('morning', level=1, drop_level=True)
    print(morningData)
    model = ARIMA(morningData, order=(2, 1, 1))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def ArimaForecasting(clusterFrame, cluster, predictionLags=1):
    train = clusterFrame[clusterFrame.index.date <= pd.datetime(2014,1,29).date()].values
    test = clusterFrame[clusterFrame.index.date > pd.datetime(2014,1,29).date()].values
    train = np.array(train).flatten()
    test = np.array(test).flatten()
    history = [x for x in train]
    history2 = [x for x in train]
    predictions = list()
    predictions2 = list()
    for t in range(0,len(test),predictionLags):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(predictionLags)
        yhat = output[0]
        predictions.extend(yhat)
        #predictions2.append(yhat)
        obs = test[t]
        history.append(obs)
        #history2.append(obs)
        # print(predictions)
        # print(predictions2)
        # print(history)
        # print(history2)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    error = math.sqrt(mean_squared_error(test, predictions))
    mape = mean_absolute_percentage_error(test, predictions)
    print(cluster)
    print('Test RMSE: %.3f' % error)
    print('Test MAPE: %.3f ' % mape)
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.title(cluster)
    plt.show()
        #
        # with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        #     print(train)
        #     print(test)
        #     print('klar')


"""Creates filename for output file"""

def CreateCsvFilePath(fileName, additions=[]):
    outputFile = fileName
    for addition in additions:
        outputFile += "_" + str(addition)
    return outputFile + ".csv"

def GetLagParam(cluster):
    if cluster == 'dawn':
        return 90;
    if cluster == 'morning':
        return 30;
    if cluster == 'lunch':
        return 48;
    if cluster == 'afternoon':
        return 36;
    if cluster == 'dusk':
        return 84;

def morningrush(time):
    return ((time >= pd.datetime(1,1,1,6,).time()) & (time < pd.datetime(1,1,1,9).time()))
def afternoonrush(time):
    return ((time >= pd.datetime(1,1,1,15,).time()) & (time < pd.datetime(1,1,1,18).time()))
def default(time):
    return True
def GetClusterDict():
    return {
        'morningrush' : morningrush,
        'afternoonrush' : afternoonrush,
        'nonrush' : default
    }, ['morningrush', 'afternoonrush', 'nonrush']

def GetWeekDay(date):
    if date.dayofweek == 0:
        return 'Mon '
    if date.dayofweek== 1:
        return 'Tue '
    if date.dayofweek == 2:
        return 'Wed '
    if date.dayofweek == 3:
        return 'Thu '
    if date.dayofweek == 4:
        return 'Fri '
    if date.dayofweek == 5:
        return 'Sat '
    if date.dayofweek == 6:
        return 'Sun '
    raise ValueError('Invalid day of week')

def ParseDate(date):
    return (GetWeekDay(date) + repr(date.day) + '-' + repr(date.hour) + ':' + repr(date.minute))


def VisualizePreprocessedDataset(dataset):
    indices = [i for i in range(len(dataset.values))]
    plt.plot(indices, dataset['NoVehicles'])
    plt.title('January')
    plt.show()

    week20to26 = dataset[((dataset.index.date >= pd.datetime(2014,1,20).date())
        & (dataset.index.date < pd.datetime(2014,1,27).date()))]

    indices = [i for i in range(len(week20to26.values))]
    plt.plot(indices, week20to26)
    plt.title('January 20 to 26')
    plt.show()

    jan22 = dataset[dataset.index.date == pd.datetime(2014,1,22).date()]
    indices = [i for i in range(len(jan22.values))]
    plt.plot(indices, jan22)
    plt.title('January 22')
    plt.show()


def AnalyzeDatasetNN(dataset, date):
    preProcessor = TrafficPreprocessor.TrafficPreprocessor()
    datasetWoWeekend = preProcessor.Cluster(dataset)
    datasetWoWeekend = preprocessor.Filter(datasetWoWeekend,'weekday',1)
    history = datasetWoWeekend[datasetWoWeekend.index.time == date.time()]

    corrExp = CorrelationExperimenter.CorrelationExperimenter(history)
    indices = [date.strftime('%d/%m %H:%M') for date in history.index]
    corrExp.PlotCorrelations(history, 'Whole set',indices)
    print('Dickey Fuller whole set')
    corrExp.DickeyFullerTest(history)

    diffData = corrExp.DifferenceData()
    indices = [date.strftime('%d/%m %H:%M') for date in diffData.index]
    corrExp.PlotCorrelations(diffData, 'Whole set diff 1 ', indices)
    print('Dickey Fuller whole set diff 1')
    corrExp.DickeyFullerTest(diffData)

def AnalyzeDataset(dataset):
    #Without Differencing
    preProcessor = TrafficPreprocessor.TrafficPreprocessor()
    dict, clusterNames = GetClusterDict()
    clusteredDataset = preProcessor.Cluster(dataset, methodDictionaries=dict)
    clusteredDatasetWoWeekends = preProcessor.Filter(clusteredDataset, 'weekday',2)
    corrExp = CorrelationExperimenter.CorrelationExperimenter(clusteredDatasetWoWeekends)

    corrExp.PlotCorrelations(clusteredDatasetWoWeekends, 'Whole set')
    print('Dickey Fuller whole set')
    corrExp.DickeyFullerTest(clusteredDatasetWoWeekends)

    for cluster in clusterNames:
        clusterFrame = preProcessor.Filter(clusteredDatasetWoWeekends, cluster,1)
        corrExp.PlotCorrelations(clusterFrame, cluster)
        print('Dickey Fuller ' + cluster)
        corrExp.DickeyFullerTest(clusterFrame)

    #With order 1 differencing

    preProcessor = TrafficPreprocessor.TrafficPreprocessor()
    dict, clusterNames = GetClusterDict()
    clusteredDataset = preProcessor.Cluster(dataset, methodDictionaries=dict)
    clusteredDatasetWoWeekends = preProcessor.Filter(clusteredDataset, 'weekday', 2)
    corrExp = CorrelationExperimenter.CorrelationExperimenter(clusteredDatasetWoWeekends)

    diffFrame = corrExp.DifferenceData()
    corrExp.PlotCorrelations(diffFrame, 'Whole set with diff order 1')
    print('Dickey Fuller with diff order 1 whole set')
    corrExp.DickeyFullerTest(diffFrame)

    for cluster in clusterNames:
        clusterFrame = preProcessor.Filter(diffFrame, cluster, 1)
        corrExp.PlotCorrelations(clusterFrame, cluster + ' with diff order 1')
        print('Dickey Fuller with diff order 1 ' + cluster)
        corrExp.DickeyFullerTest(clusterFrame)
def GetOrder(clusterName):
    if clusterName == 'morningrush':
        return (3,0,0)
    if clusterName == 'afternoonrush':
        return (3,1,0)
    if clusterName == 'nonrush':
        return (1,1,0)
    raise ValueError('No Arima-parameters for given cluster')

def GetXaxis(clusterName):
    if clusterName == 'nonrush':
        return True
    else:
        return False

def ForecastManager(dataset):
    preProcessor = TrafficPreprocessor.TrafficPreprocessor()
    dict, clusterNames = GetClusterDict();
    datasetWoWeekend = preProcessor.Cluster(dataset, methodDictionaries=dict)
    datasetWoWeekend = preprocessor.Filter(datasetWoWeekend, 'weekday', 2)
    train = datasetWoWeekend[:len(datasetWoWeekend) - 288]
    test = datasetWoWeekend[-288:]
    for cluster in clusterNames:
        clusterTrain = preprocessor.Filter(train,cluster,1)
        clusterTest = preprocessor.Filter(test,cluster,1)
        ArimaForeCastOnNearestNeighborsDataset(clusterTrain,clusterTest,GetOrder(cluster),GetXaxis(cluster))


def ArimaForeCastOnNearestNeighborsDataset(train, test, order, xaxis = False):
    nnAverage = list()
    Arima = list()
    indices = []
    for idx in test.index:
        indices.append(idx.strftime('%d/%m %H:%M'))
        Arimahistory = train[train.index.time == idx.time()]
        NNHistory = train[train.index.time == idx.time()]
        # Arimahistory.index = [for i in range(len(history.values))]

        model = ARIMA(Arimahistory, order=order)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()

        Arima.extend(output[0])
        nnAverage.append(NNHistory['NoVehicles'].mean())
    realValue = np.array(test.values).flatten()
    errorArima = math.sqrt(mean_squared_error(realValue, Arima))
    mapeArima = mean_absolute_percentage_error(realValue, Arima)
    errorNN = math.sqrt(mean_squared_error(realValue, nnAverage))
    mapeNN = mean_absolute_percentage_error(realValue, nnAverage)
    print('Test RMSE Arima: %.3f' % errorArima)
    print('Test MAPE Arima: %.3f ' % mapeArima)
    print('Test RMSE NN: %.3f' % errorNN)
    print('Test MAPE NN: %.3f ' % mapeNN)
    plt.plot(indices, realValue)
    plt.plot(Arima, color='red')
    plt.plot(nnAverage, color='green')
    plt.title("predictions 8-10 jan 30")
    plt.xticks(rotation=90)
    plt.gcf().subplots_adjust(bottom=0.25)
    if xaxis:
        plt.MaxNLocator(nbins=10)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
    outputFile = CreateCsvFilePath("data_detectorId", detectorIds)
    fileHandler = FileHandler.CsvTrafficDataHandler(inputfile, outputFile, allColumns)
    preprocessor = TrafficPreprocessor.TrafficPreprocessor(mdHandler='linear')
    if shouldRead:
        result = fileHandler.ReadFile(detectorIds)
        if result:
            fileHandler.WriteFile(selectedColumns)
        dataset = fileHandler.ExtractDataFromColumns(selectedColumns)
    else:
        dataset = dataFrame = pd.read_csv(outputFile, sep=";", decimal=",", encoding="utf-8", header=0, low_memory=False)

    dataset = preprocessor.PreProcess(dataset, filePath=CreateCsvFilePath("Aggregated5MinIntervals", detectorIds), threshold=1000)
    #ShowOriginalData
    #VisualizePreprocessedDataset(dataset)
    #Show Clusters and Autocorrelations
    #AnalyzeDataset(dataset)
    #AnalyzeDatasetNN(dataset, pd.datetime(2014,1,22,8,0,0))
    #MakeForecasts
    # ForecastTraffic(dataset)
    ForecastManager(dataset)
    # clusterSorters = GetClusterDict()
    # dict, clusterNames = GetClusterDict()
    # dataset = preprocessor.Cluster(dataset, methodDictionaries=dict)
    # dataset2 = preprocessor.Filter(dataset, 'weekday', 2, True)
    #
    # for cluster in clusterNames:
    #     clusterFrame = dataset2.xs(cluster, level=1, drop_level=True)
    #     ArimaForecasting(clusterFrame,cluster,predictionLags=6)






# Tutoring questions:
# - how specific should the model be? Working against a general dataset or can we make a specific implementation towards traffikverkets dataset?
# - Make assumption that we have enough data (always) for historical imputation?
# - How to handle single measurement. normal number of measuremen 1300-1436


# dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
# dataFrame = dataset[dataset['VehicleClassID'] == 0].groupby(['Timestamp'])['NoVehicles'].sum()
# print(dataFrame.index.date)
# print(dataFrame.groupby(dataFrame.index.date).count())


  # train = clusterFrame[clusterFrame.index.date <= pd.datetime(2014,1,29).date()].values
  #   test = clusterFrame[clusterFrame.index.date > pd.datetime(2014,1,29).date()].values
  #   history = [x[0] for x in train]
  #   predictions = list()
  #   for t in range(0,len(test), predictionLags):
  #       model = ARIMA(history, order=(2,1,1))
  #       model_fit = model.fit(disp=0)
  #       output = model_fit.forecast(predictionLags)
  #       yhat = output[predictionLags]
  #       predictions.extend(yhat)
  #       obs = [test[x] for x in range(t,t+predictionLags)]
  #       history.extend(obs)
  #       #for x in range(predictionLags):
  #           #print('predicted=%f, expected=%f' % (yhat[x], obs[x]))
  #   error = math.sqrt(mean_squared_error(test, predictions))
  #   mape = mean_absolute_percentage_error(test, predictions)
  #   print(cluster)
  #   print('Test RMSE: %.3f' % error)
  #   print('Test MAPE: %.3f ' % mape)
  #   plt.plot(test)
  # #   plt.plot(predictions, color='red')
  # #   plt.title(cluster)
  #   plt.show()
  #
  #   train = clusterFrame[clusterFrame.index.date <= pd.datetime(2014, 1, 29).date()].values
  #   test = clusterFrame[clusterFrame.index.date > pd.datetime(2014, 1, 29).date()].values
  #   history = [x[0] for x in train]
  #   predictions = list()
  #   for t in range(len(test)):
  #       model = ARIMA(history, order=(5, 1, 0))
  #       model_fit = model.fit(disp=0)
  #       output = model_fit.forecast()
  #       yhat = output[0]
  #       predictions.append(yhat)
  #       obs = test[t]
  #       history.append(obs)
  #       print('predicted=%f, expected=%f' % (yhat, obs))
  #   error = math.sqrt(mean_squared_error(test, predictions))
  #   mape = mean_absolute_percentage_error(test, predictions)
  #   print(cluster)
  #   print('Test RMSE: %.3f' % error)
  #   print('Test MAPE: %.3f ' % mape)
  #   plt.plot(test)
  #   plt.plot(predictions, color='red')
  #   plt.title(cluster)
  #   plt.show()