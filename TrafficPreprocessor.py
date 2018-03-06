
import pandas as pd
import numpy as np
#Class for handling time series data in form of data provided by Traffikverket in Sweden

#Preprocessing steps:
#1. check if incorrect number of points.
#2. walk through points
#2.1 if consecutive points above threshhold

#Possible adds for a better preprocessing:
#add settings on random individual points or collection of points. Add threshold when to do what
import scipy.cluster
from sklearn.cluster import AgglomerativeClustering


class TrafficPreprocessor():
    """
    Class for preprocessing traffic time series data
    Class for preprocessing traffic time series data
    instance variables:
    mdHandler: Handler of missing data {'linear', 'knn'}
    """
    def __init__(self, mdHandler = 'knn'):
        self.mdHandler = mdHandler

    """"""
    def LinearImputation(self, dataframe):
        return dataframe.interpolate(method='linear')

    """"""
    def NearestNeighborsImputation(self, dataframe):
        return dataframe.interpolate(method='nearest')

    """
    Preprocesses dataset by accumulating into intervals and imputating missing values
    """
    def StandardizeTimeSeries(self, dataset, vehicleClass=0, intervalInMinutes=15, threshold=None):

        intervalString = repr(intervalInMinutes) +"T"
        dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
        dataframe = dataset[dataset['VehicleClassID'] == vehicleClass]
        dataframe = dataframe[['Timestamp', 'NoVehicles']].groupby(
            ['Timestamp']).sum()

        dataframe = dataframe.resample(intervalString).mean()
        datesToBeRemoved = []
        maxCount = int(1440 / intervalInMinutes)
        if threshold != None:
            datesToBeRemoved = self.RemoveDays(dataframe, threshold, maxCount)

        if datesToBeRemoved != []:
            booleans = [False if index.date() in datesToBeRemoved else True for index in dataframe.index]
            dataframe = dataframe[booleans]

        dataframe['NoVehicles'] = pd.to_numeric(dataframe['NoVehicles'] * intervalInMinutes, downcast='integer')
        dataframe['NoVehicles'] = dataframe['NoVehicles'].round()

        if(self.mdHandler == 'knn'):
            newDataframe = self.NearestNeighborsImputation(dataframe)
        elif(self.mdHandler == 'linear'):
            newDataframe = self.LinearImputation(dataframe)
        else:
            raise ValueError("mvHandler value not valid. Must be in {'knn', 'linear'}")
        return newDataframe

    """"""
    def Cluster(self, dataset, methodDictionaries = None, sortDayType=True):
        dataset['Date'] = pd.to_datetime(dataset.index)
        columnList = ['Date']
        levels = 0
        if methodDictionaries != None:
            categories = []
            for time in dataset['Date']:
                for cluster, method in methodDictionaries.items():
                    if method(time.time()):
                        categories.append(cluster)
                        break
            dataset['Category'] = categories
            if dataset.shape[0] != len(dataset['Category']):
                raise ValueError('All Time Series instances does not fit given categories')
            columnList.append('Category')
            levels += 1

        if(sortDayType):
            dataset['DayType'] = ['weekday' if day.dayofweek < 5
                               else 'weekend' for day in dataset['Date']]
            columnList.append('DayType')
            levels += 1

        return dataset.set_index(columnList), levels

    """"""
    def RemoveDays(self, dataFrame, threshold, maxCount):
        temp = dataFrame.groupby(dataFrame.index.date).count()
        temp.columns = ["NoMeasurements"]
        removeDates = [index for index,row in temp.iterrows() if row['NoMeasurements']/maxCount < threshold]
        return removeDates

    """"""
    def Filter(self, dataFrame, arg, level, dropLevel=True):
        return dataFrame.xs(arg,level=level,drop_level=dropLevel)

    """
    Extracts desired columns from the dataset
    parameters:
    selectedColumns: List of desired column names
    """
    def ExtractDataFromColumns(self, dataset,selectedColumns):
        return pd.DataFrame(dataset, columns=selectedColumns)
