from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.tools import diff
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class CorrelationExperimenter():

    def __init__(self, dataset, clusters):
        self.dataset = dataset
        self.clusters = clusters

    def DifferenceData(self, nonSeasonal=1, seasonal=None, seasonalPeriods=1):
        dataframe = diff(self.dataset,k_diff=nonSeasonal,k_seasonal_diff=seasonal, seasonal_periods=seasonalPeriods)
        return dataframe

    def PlotCorrelations(self, data):
        data.plot(title="whole set")
        plt.show()

        acfFrame, qstat, pvalue = acf(data.values, qstat=True)
        print("Whole set B-J test:")
        print(qstat)
        print("Whole set p-value:")
        print(pvalue)
        #acfFrame = pd.DataFrame([acfFrame]).T
        #acfFrame.columns = ['AutoCorrelation']
        #acfFrame.index += 1
        #acfFrame.plot(kind="bar", title="whole set ACF")
        #plt.show()
        plot_acf(acfFrame, title="whole set ACF")
        plt.show()


        pacfFrame = pacf(data.values)
        #pacfFrame = pd.DataFrame([pacfFrame]).T
        #pacfFrame.columns = ['Partial AutoCorrelation']
        #pacfFrame.index += 1
        #pacfFrame.plot(kind="bar", title="whole set PACF")
        #plt.show()
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
            # acfFrame = pd.DataFrame([acfFrame]).T
            # acfFrame.columns = ['AutoCorrelation']
            # acfFrame.index += 1
            # acfFrame.plot(kind="bar", title=cluster + " ACF")
            # plt.show()
            plot_acf(acfFrame, title=cluster + " ACF")



            pacfFrame = pacf(clusterFrame.values)
            # pacfFrame = pd.DataFrame([pacfFrame]).T
            # pacfFrame.columns = ['Partial AutoCorrelation']
            # pacfFrame.index += 1
            # pacfFrame.plot(kind="bar", title=cluster + " PACF")
            # plt.show()
            plot_pacf(pacfFrame, title=cluster + " PACF")

