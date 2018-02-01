
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
def  PlotDetectorData(dataFrame):
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

def ArimaForecasting(dataset):

    for cluster in clusters:
        clusterFrame = dataset.xs(cluster, level=1, drop_level=True)
        train = clusterFrame[clusterFrame.index.date <= pd.datetime(2014,1,29).date()].values
        test = clusterFrame[clusterFrame.index.date > pd.datetime(2014,1,29).date()].values
        history = [x[0] for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))
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


    #dataset = preprocessor.RemoveDays(dataset, 1000)
    dataset = preprocessor.PreProcess(dataset, filePath=CreateCsvFilePath("Aggregated5MinIntervals", detectorIds), threshold=1000)
    dataset = preprocessor.Cluster(dataset)
    dataset2 = preprocessor.Filter(dataset, 'weekday', 2, True)
    #dataset2 = preprocessor.Filter(dataset2, 'dusk', 1, True)

    #acexp = CorrelationExperimenter.CorrelationExperimenter(dataset, clusters)
    #acexp.PlotCorrelations(dataset)
    #df = acexp.DifferenceData(nonSeasonal=1)
    #acexp.PlotCorrelations(df)
    ArimaForecasting(dataset2)
    #plotClusters(dataset, clusters)


    # morningData['Timestamp'] = [pd.to_datetime(x.year, x.month, x.day, y.hour, y.minute, y.second) for x,y,z in morningData.index]
    #         dataset.index <= pd.datetime(2014, 1, 22, 23, 59, 59)))]

    # flowOnADay.plot(y='NoVehicles')
    # plt.show()
    # autocorrelation_plot(flowOnADay)
    # plt.show()




# Tutoring questions:
# - how specific should the model be? Working against a general dataset or can we make a specific implementation towards traffikverkets dataset?
# - Make assumption that we have enough data (always) for historical imputation?
# - How to handle single measurement. normal number of measuremen 1300-1436


# dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
# dataFrame = dataset[dataset['VehicleClassID'] == 0].groupby(['Timestamp'])['NoVehicles'].sum()
# print(dataFrame.index.date)
# print(dataFrame.groupby(dataFrame.index.date).count())