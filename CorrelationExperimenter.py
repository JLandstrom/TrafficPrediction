from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.tools import diff
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

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

    def PlotCorrelations(self, data, title, indices=None, nlags=40):
        if indices == None:
            indices = [i for i in range(len(data.values))]

        plt.plot(indices, data.values)
        plt.title(title)
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.25)
        plt.show()

        acfFrame= acf(data.values,nlags=nlags)
        plot_acf(acfFrame, title=title+' ACF')
        plt.show()

        pacfFrame = pacf(data.values,nlags=nlags)
        plot_pacf(pacfFrame, title=title + 'PACF')
        plt.show()

    def PrintParameterMatrix(self, dataset, maxNonSeasonal, maxSeasonal, lagsPerDay, noTrainingDays):
        max_p = maxNonSeasonal[0]
        max_d = maxNonSeasonal[1]
        max_q = maxNonSeasonal[2]
        max_P = maxSeasonal[0]
        max_D = maxSeasonal[1]
        max_Q = maxSeasonal[2]
        aic = list()
        bic = list()
        orders = list()

        train_set = dataset[-noTrainingDays*int(lagsPerDay):-int(lagsPerDay)]
        for P,D,Q in itertools.product(range(0,max_P+1),
                                       range(0,max_D+1),
                                       range(0,max_Q+1)):
            for p,d,q in itertools.product(range(0,max_p+1),
                                           range(0,max_d+1),
                                           range(0,max_q+1)):
                if P==0 and D==0 and Q==0 and p==0 and d==0 and q==0:
                    continue

                try:
                    model = SARIMAX(train_set, order=(p,d,q), seasonal_order=(P,D,Q,int(lagsPerDay)))
                    result = model.fit()
                    order = 'Model: Nonseanonal ('+ repr(p) +','+repr(d)+','+ repr(q) + ') Seasonal: '+\
                            '(' + repr(P) +',' + repr(D) +','+repr(Q) + ')'
                    print(order)
                    print(result.summary())
                    orders.append(order)
                    bic.append(result.bic)
                    aic.append(result.aic)
                except Exception as e:
                    print(e)

        results = pd.DataFrame(index = orders, data={'aic': aic, 'bic': bic})
        print(results)

