from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.tools import diff
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
from matplotlib.ticker import Formatter

class TrafficAnalyzer():


    def PlotDataset(self, dataset,title):
        formatter = MyFormatter(dataset.index.date)
        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.grid(True, which='major', ls='dotted')
        ax.plot(np.arange(len(dataset.values)), dataset.values)
        fig.autofmt_xdate()
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.set_title(title)
        plt.show()

    def PlotAcf(self, dataset, title, indices=None, nlags=40):
        acfFrame = acf(dataset.values, nlags=nlags)
        plot_acf(acfFrame, title=title + ' ACF')
        plt.show()

    def PlotPacf(self, dataset, title, indices=None, nlags=40):
        pacfFrame = pacf(dataset.values, nlags=nlags)
        plot_pacf(pacfFrame, title=title + 'PACF')
        plt.show()

    def DifferenceDataset(self, dataset, nonSeasonal=1, seasonal=None, seasonalPeriods=1):
        dataframe = diff(dataset, k_diff=nonSeasonal, k_seasonal_diff=seasonal, seasonal_periods=seasonalPeriods)
        return dataframe

    def AdfTest(self, dataset):
        testSet = np.array(dataset.values).flatten()
        result = adfuller(testSet)
        print('ADF: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical values: ')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

    def FindBestModelFitness(self, dataset, maxNonSeasonal, maxSeasonal, lagsPerDay):
        max_p = maxNonSeasonal[0]
        max_d = maxNonSeasonal[1]
        max_q = maxNonSeasonal[2]
        max_P = maxSeasonal[0]
        max_D = maxSeasonal[1]
        max_Q = maxSeasonal[2]
        aic = list()
        bic = list()
        orders = list()

        for P, D, Q in itertools.product(range(0, max_P + 1),
                                         range(0, max_D + 1),
                                         range(0, max_Q + 1)):
            for p, d, q in itertools.product(range(0, max_p + 1),
                                             range(0, max_d + 1),
                                             range(0, max_q + 1)):
                if P == 0 and D == 0 and Q == 0 and p == 0 and d == 0 and q == 0:
                    continue

                try:
                    model = SARIMAX(dataset, order=(p, d, q), seasonal_order=(P, D, Q, int(lagsPerDay)))
                    result = model.fit()
                    order = 'Model: Nonseanonal (' + repr(p) + ',' + repr(d) + ',' + repr(q) + ') Seasonal: ' + \
                            '(' + repr(P) + ',' + repr(D) + ',' + repr(Q) + ')'
                    print(order)
                    print(result.summary())
                    orders.append(order)
                    bic.append(result.bic)
                    aic.append(result.aic)
                except Exception as e:
                    print(e)

        results = pd.DataFrame(index=orders, data={'aic': aic, 'bic': bic})
        print(results)

    def ParseDate(self, date):
        return date.strftime('%Y-%m-%d')

class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''

        return self.dates[ind].strftime(self.fmt)
