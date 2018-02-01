
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

    """
    
    """
    def LinearImputation(self, dataframe):
        return dataframe.interpolate(method='linear')

    def NearestNeighborsImputation(self, dataframe):
        return dataframe.interpolate(method='nearest')

    """
    Preprocesses dataset by accumulating into intervals and imputating missing values
    
    Parameters:
    Required
    dataset: dataframe holding data
    
    Optional
    vehicleClasses: List of integers. Defines what vehicleclasses are to be handled. Empty list means handling all classes
    filePath: string. If given, accumulated and preprocessed dataset is printed.
    interval: string. Accumulation interval.
    """
    def PreProcess(self, dataset, vehicleClasses=[], filePath="", intervalInMinutes=5, threshold=None):
        if (vehicleClasses == []):
            pass

        intervalString = repr(intervalInMinutes) +"T"

        datesToBeRemoved = []
        if threshold != None:
            datesToBeRemoved = self.RemoveDays(dataset, threshold)

        dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])

        dataframe = dataset[dataset['VehicleClassID'] == 0]
        dataframe = dataframe[['Timestamp', 'NoVehicles']].groupby(
            ['Timestamp']).sum()

        dataframe = dataframe.resample(intervalString).sum()
        if datesToBeRemoved != []:
            booleans = [False if index.date() in datesToBeRemoved else True for index in dataframe.index]
            dataframe = dataframe[booleans]

        #dataframe = dataframe[dataframe.index.dayofweek < 5]

        dataframe['NoVehicles'] = pd.to_numeric(dataframe['NoVehicles'] * intervalInMinutes, downcast='integer')
        dataframe['NoVehicles'] = dataframe['NoVehicles'].round()

        if(self.mdHandler == 'knn'):
            newDataframe = self.NearestNeighborsImputation(dataframe)
        elif(self.mdHandler == 'linear'):
            newDataframe = self.LinearImputation(dataframe)
        else:
            raise ValueError("mvHandler value not valid. Must be in {'knn', 'linear'}")

        if (filePath != ""):
            newDataframe.to_csv(filePath, sep=";", index=True)

        #newDataframe['Timestamp'] = newDataframe.index.astype(np.int64)
        return newDataframe

    """
    Applies agglomerative clustering to dataset
    
    Parameters:
    Required
    dataset: dataframe holding data
    noOfClusters: Integer. How many clusters are to be created
    
    Optional
    type: String. Specifies if clustering is performed on time during day or days during week {'day', 'week'}.
    """
    def Cluster(self, dataset, sortCategory=True, sortDayType=True):
        # clusters based on visual inspection of jan 22
        # dawn < 07:30
        # morning >= 07:30 < 10:00
        # lunch >= 10:00 < 14:00
        # afternoon >= 14:00 < 17:00
        # dusk >= 17:00
        dataset['Date'] = pd.to_datetime(dataset.index)
        columnList = ['Date']
        if(sortCategory):
            dataset['Category'] = ['dawn' if time.time() < pd.datetime(2014,1,1,7,30).time()
                               else 'morning' if ((time.time() >= pd.datetime(2014,1,1,7,30).time()) & (time.time() < pd.datetime(2014,1,1,10,0).time()))
                               else 'lunch' if ((time.time() >= pd.datetime(2014,1,1,10,0).time()) & (time.time() < pd.datetime(2014,1,1,14,0).time()))
                               else 'afternoon' if((time.time() >= pd.datetime(2014,1,1,14,0).time()) & (time.time() < pd.datetime(2014,1,1,17,0).time()))
                               else 'dusk' for time in dataset['Date']]
            columnList.append('Category')

        if(sortDayType):
            dataset['DayType'] = ['weekday' if day.dayofweek < 5
                               else 'weekend' for day in dataset['Date']]
            columnList.append('DayType')

        return dataset.set_index(columnList)


    def RemoveDays(self, dataFrame, threshold):
        dataFrame['Timestamp'] = pd.to_datetime(dataFrame['Timestamp'])
        dataFrame = dataFrame[dataFrame["VehicleClassID"] == 0]
        dataFrame = dataFrame[["NoVehicles","Timestamp"]].groupby(["Timestamp"]).sum()
        #dataFrame = dataFrame[dataFrame['VehicleClassID'] == 0].groupby(['Timestamp'])['NoVehicles'].sum()
        temp = dataFrame.groupby(dataFrame.index.date).count()
        temp.columns = ["NoMeasurements"]

        removeDates = []
        # for index, row in temp.iterrows():
        #     if row['NoMeasurements'] < threshold:
        #         removeDates.append(index)

        removeDates2 = [index for index,row in temp.iterrows() if row['NoMeasurements'] < threshold]
        #booleans = [False if index.date() in removeDates2 else True for index in dataFrame.index]
        return removeDates2

    # dataframe = dataframe[dataframe.index.dayofweek < 5]
    def Filter(self, dataFrame, arg, level, dropLevel=True):
        return dataFrame.xs(arg,level=level,drop_level=dropLevel)