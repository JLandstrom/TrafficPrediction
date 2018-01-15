
import pandas as pd

#Class for handling time series data in form of data provided by Traffikverket in Sweden

#Preprocessing steps:
#1. check if incorrect number of points.
#2. walk through points
#2.1 if consecutive points above threshhold

#Possible adds for a better preprocessing:
#add settings on random individual points or collection of points. Add threshold when to do what

class TrafficPreprocessor():
    """
    Class for preprocessing traffic time series data
    instance variables:
    mdHandler: Handler of missing data {'Remove', 'KNN'}
    """
    def __init__(self, mdHandler = 'Remove'):
        self.mvHandler = mdHandler

    """
    
    """
    def RemoveDaysWithMissingData(self, dataset, requiredDailyMeasures):
        dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
        dataFrame = dataset[dataset['VehicleClassID'] == 0].groupby(['Timestamp'])['NoVehicles'].sum()
        measureFrequencies = dataFrame.groupby(dataFrame.index.date).count()
        dates = []
        print("Preprocessing starts")
        for index in measureFrequencies.index:
            if measureFrequencies[index] >= requiredDailyMeasures:
                dates.append(index)
        print("Preprocessing done")
        indices = [x.date() in dates for x in dataset['Timestamp']]
        return dataset[indices]




    def NearestNeighborsImputation(self, dataset, requiredDailyMeasures):
        pass

    """
    Method 
    """
    def PreProcess(self, dataset, vehicleClasses=[], filePath=""):
        dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])

        if(vehicleClasses == []):
            pass

        dataFrame = dataset[dataset['VehicleClassID'] == 0]
        dataFrame = dataFrame[['Timestamp', 'Flow', 'NoVehicles']].groupby(
            ['Timestamp']).sum()
        dataFrame = dataFrame.resample("5T").mean()

        if(filePath != ""):
            dataFrame.to_csv(filePath, sep=";", index=True)

        return dataFrame

        #start with validation of argument
        # if(isinstance(requiredDailyMeasures, int) == False or requiredDailyMeasures < 1):
        #     raise ValueError("TimeIntervalOfMeasure must be of type integer and above 0.")
        #
        # if(self.mvHandler.lower() == 'remove'):
        #     print("your reached removed. If this is written code is fine")
        #     return self.RemoveDaysWithMissingData(dataset, requiredDailyMeasures)
        # elif(self.mvHandler.lower() == 'knn'):
        #     print("your reached KNN. If this is written code is fine")
        #     return self.NearestNeighborsImputation(dataset, requiredDailyMeasures)
        # else:
        #     raise ValueError("mdHandler must be 'Remove' or 'KNN'")



