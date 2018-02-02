from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.tools import diff
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

class CorrelationExperimenter():

    def __init__(self, dataset, clusters):
        self.dataset = dataset
        self.clusters = clusters

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

    def PlotCorrelations(self, data, difference=False):
        data.plot(title="whole set")
        plt.show()

        acfFrame, qstat, pvalue = acf(data.values, qstat=True)
        print("Whole set B-J test:")
        print(qstat)
        print("Whole set p-value:")
        print(pvalue)
        plot_acf(acfFrame, title="whole set ACF")
        plt.show()

        pacfFrame = pacf(data.values)
        plot_pacf(pacfFrame, title="whole set PACF")

        for cluster in self.clusters:
            clusterFrame = data.xs(cluster, level=1, drop_level=True)
            clusterFrame.plot(title=cluster)
            plt.show()

            acfFrame, qstat, pvalue = acf(clusterFrame.values, qstat=True)
            print("B-J test for " +cluster + ":")
            print(qstat)
            print("p-values for " + cluster + ":")
            print(pvalue)
            plot_acf(acfFrame, title=cluster + " ACF")
            plt.show()

            pacfFrame = pacf(clusterFrame.values)
            plot_pacf(pacfFrame, title=cluster + " PACF")
            plt.show()

