from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.tools import diff
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

class CorrelationExperimenter():

    def __init__(self, dataset):
        self.dataset = dataset

    def DifferenceData(self, nonSeasonal=1, seasonal=None, seasonalPeriods=1):
        dataframe = diff(self.dataset,k_diff=nonSeasonal,k_seasonal_diff=seasonal, seasonal_periods=seasonalPeriods)
        return dataframe

    def DickeyFullerTest(self, dataset):
        testSet = np.array(dataset.values).flatten()
        result = adfuller(testSet)
        print('ADF: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical values: ')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key,value))

    def PlotCorrelations(self, data, title, indices=None):
        if indices == None:
            indices = [i for i in range(len(data.values))]

        plt.plot(indices, data.values)
        plt.title(title)
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.25)
        plt.show()

        acfFrame= acf(data.values)
        plot_acf(acfFrame, title=title+' ACF')
        plt.show()

        pacfFrame = pacf(data.values)
        plot_pacf(pacfFrame, title=title + 'PACF')
        plt.show()
