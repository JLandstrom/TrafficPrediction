from datetime import datetime

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import sys, getopt
import TrafficPreprocessor
import FileHandler
import scipy.stats as stats
from pandas.plotting import lag_plot


sys._enablelegacywindowsfsencoding()

"""Global variables"""
"""Spefic to the format of the file to read"""
allColumns = ["DetectorID", "VehicleClassID", "Timestamp", "Flow", "Speed", "Occupancy", "Confidence", "Tdiff",
              "TimeCycle", "NoVehicles", "Headway", "MeasuresIncluded"]
selectedColumns = ["DetectorID", "VehicleClassID", "Timestamp", "Flow", "NoVehicles"]

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

def PlotDistribution(dataset):
    dataset['Timestamp'] = pd.to_datetime(dataFrame['Timestamp'])
    dataset = dataFrame[dataFrame['VehicleClassID'] == 0].groupby(['Timestamp'])['NoVehicles'].sum()
    frequencies = dataset.groupby(dataset.index.date).count()
    frequencies = frequencies.sort_values()
    print(frequencies)
    mean = frequencies.mean()
    std = frequencies.std()
    print("std: " + repr(std))
    print("mean: " + repr(mean))

    fit = stats.norm.pdf(frequencies, mean, std)  # this is a fitting indeed
    plt.plot(frequencies, fit, '-o')
    plt.hist(frequencies, normed=True)  # use this to draw histogram of your data
    plt.show()

"""Creates filename for output file"""

def CreateFileName():
    outputfile = "data_detectorId"
    for detector in detectorIds:
        outputfile += "_" + str(detector)
    return outputfile + ".csv"

if __name__ == "__main__":
    main(sys.argv[1:])
    outputFile = CreateFileName()
    fileHandler = FileHandler.CsvTrafficDataHandler(inputfile, outputFile, allColumns)
    preprocessor = TrafficPreprocessor.TrafficPreprocessor()
    if shouldRead:
        result = fileHandler.ReadFile(detectorIds)
        if result:
            fileHandler.WriteFile(selectedColumns)
        dataset = fileHandler.ExtractDataFromColumns(selectedColumns)
    else:
        dataset = dataFrame = pd.read_csv(outputFile, sep=";", decimal=",", encoding="utf-8", header=0)

    dataset = preprocessor.PreProcess(dataset,1200)

    fileHandler.dataset = dataset
    fileHandler.WriteFile([])
    #CheckCorrectNumberOfMeasurements(dataset)
    PlotDetectorData(dataset)
    #if shouldPlot:
        #PlotDetectorData(dataset)



# Tutoring questions:
# - how specific should the model be? Working against a general dataset or can we make a specific implementation towards traffikverkets dataset?
# - Make assumption that we have enough data (always) for historical imputation?
# - How to handle single measurement. normal number of measuremen 1300-1436
