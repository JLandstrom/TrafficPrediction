import os
import pandas as pd
import traceback

class CsvTrafficDataHandler():
    """
    Class for reading, extracting, and writing
    data from file
    instance variables:
    inputFilePath: Full path of file to be read
    outputFilePath: Full path where to write file
    allColumns: List of column names in file being read
    separator: separator of segments in csv file
    """
    def __init__(self, inputDirectory, allColumns, detectorIds, separator=';' ):
        self.inputDirectory = inputDirectory
        self.allColumns = allColumns
        self.separator = separator
        self.dataset = pd.DataFrame()
        self.detectorIds = detectorIds
        self.outputFilePath = self.CreateCsvFilePath("data_detectorId", self.detectorIds)

    """
    reads data file and extracts data from chosen detectors
    parameters:
    detectorIds: List of integer ids to extract from data to read
    decimal: decimal sign in data to read
    encoding: encoding of data file to read
    returns:
    a dataframe with extracted data for selected ids
    """
    def ReadFiles(self, decimal=',', encoding='utf-8'):
        print("Starts reading input files...")
        booleans = []
        appendedFrames = []
        try:
            for file in os.listdir(self.inputDirectory):
                print('Reading file: ' + file)
                i = 1
                filename = self.CreateInputFilePath(file)
                if filename.endswith(".csv"):
                    for chunk in pd.read_csv(filename, sep=self.separator, decimal=decimal,
                                             encoding=encoding, chunksize=1000000, names=self.allColumns, index_col=False):
                        print("handling chunk: " + repr(i))
                        i = i + 1
                        booleans.clear()
                        for detId in chunk.DetectorID:
                            if detId in self.detectorIds:
                                booleans.append(True)
                            else:
                                booleans.append(False)
                        appendedFrames.append(chunk[booleans])
        except:
            print("Error while reading data")
            traceback.print_exc()
            return False
        print("Finished reading and extracting data.")
        self.dataset = pd.concat(appendedFrames)
        return True

    """
    Writing a dataframe to CSV-file
    parameters:
    dataset: dataframe to write
    extractedColumns: Columns in dataframe
    """
    def WriteFile(self, toPrint):
        try:
            print("Writing to file ", self.outputFilePath)
            toPrint.to_csv(self.outputFilePath, sep=self.separator, index=False)
        except:
            print("Error while writing data.")
            return False
        return True

    def CreateCsvFilePath(self, fileName, additions=[]):
        outputFile = fileName
        for addition in additions:
            outputFile += "_" + str(addition)
        return outputFile + ".csv"

    def CreateInputFilePath(self, file):
        filePath = self.inputDirectory
        if filePath.endswith('\\') == False:
            filePath += '\\'
        filePath += file
        return filePath
