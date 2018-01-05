from datetime import datetime

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import sys, getopt
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
    def __init__(self, inputFilePath, outputFilePath, allColumns, separator=';' ):
        self.inputFilePath = inputFilePath
        self.outputFilePath = outputFilePath
        self.allColumns = allColumns
        self.separator = separator
        self.dataset = pd.DataFrame()

    """
    reads data file and extracts data from chosen detectors
    parameters:
    detectorIds: List of integer ids to extract from data to read
    decimal: decimal sign in data to read
    encoding: encoding of data file to read
    returns:
    a dataframe with extracted data for selected ids
    """
    def ReadFile(self, detectorIds, decimal=',', encoding='utf-8'):
        print("Starts reading input file...")
        booleans = []
        appendedFrames = []
        i = 1
        try:
            for chunk in pd.read_csv(self.inputFilePath, sep=self.separator, decimal=decimal,
                                     encoding=encoding, chunksize=1000000, names=self.allColumns, index_col=False):
                print("handling chunk: " + repr(i))
                i = i + 1
                booleans.clear()
                for detId in chunk.DetectorID:
                    if detId in detectorIds:
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
    def WriteFile(self, selectedColumns):
        try:
            if not selectedColumns:
                toPrint = self.dataset
            else:
                toPrint = self.ExtractDataFromColumns(selectedColumns)
            print("Writing to file ", self.outputFilePath)
            toPrint.to_csv(self.outputFilePath, sep=self.separator, index=False)
        except:
            print("Error while writing data.")
            return False
        return True

    """
    Extracts desired columns from the dataset
    parameters:
    selectedColumns: List of desired column names
    """
    def ExtractDataFromColumns(self, selectedColumns):
        return pd.DataFrame(self.dataset, columns=selectedColumns)