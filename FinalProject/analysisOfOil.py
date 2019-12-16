# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 21:08:54 2019

@author: TARUN
"""
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

def readData(csvFile):
    data = pd.read_csv(csvFile, index_col='Date', parse_dates=True)
    return data

def qqPlot(data, col):
    figure = plt.figure(figsize=(8,4))
    ax = figure.add_subplot(111)
    stats.probplot(data[col], dist='norm', plot=ax)
    
    plt.show()
    
def priceTrend(data):
    data.plot(figsize=(10,5), linewidth=1)
    plt.show()

def getVAR(data, col):
    return data[col].var()

def getMean(data, col):
    return data[col].mean()

def getSkewAndKurt(data, col):
    mean   = data[col].mean()
    median = data[col].median()
    mode   = data[col].mode()
    skewness = data[col].skew()
    kurtosis = data[col].kurt()
    
    print('Mean = {}'.format(mean))
    print('Median = {}'.format(median))
    print('Mode = ', mode[0])
    print('Skewness = ', skewness)
    print('Kurtosis = ', kurtosis)
    
    plt.figure(figsize=(10, 5))
    plt.hist(data[col], bins=10, color='grey')
    plt.axvline(mean,color='red',label='Mean')
    plt.axvline(median,color='yellow',label='Median')
    plt.axvline(mode[0],color='green',label='Mode')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    return skewness,kurtosis

def drawNormal(data, col):
    sns.distplot(data[col], hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = col)
    plt.ylabel('Density')
    plt.title('Density Plot')
    plt.show()

def drawAutoCorr(df, col):
    autocorrelation_plot(df[col])
    plt.show()
    
    
def arimaModelFitAnalysis(data, col):
    # fit model
    model = ARIMA(data[col], order=(5,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    
    # plot residual erros
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())

def arimaModelRollingForCast(data, col):
    X = data[col]
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = []
    
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat[0])
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    df = pd.DataFrame(columns={'Test'})
    df['Test'] = test
    df['Predictions'] = predictions 
    # plot
    print(df.head())
    df.plot()
    plt.show()

def olsModel(X, Y, y_col):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train)
    result = model.fit()
    yPredict = result.predict(X_test)
    df = pd.DataFrame()
    df[y_col] = y_test;
    df[y_col + '_Predict'] = yPredict
    print(result.summary())
    df.plot()
    plt.show()
    


## Read Data for WTI oil
def main():
    readFile = 'crudeOilPriceWTI.csv'
    ligthFuelOil = 'lightFuelOil.csv'
    gasoline = 'Gasoline.csv'
    jetFuel = 'usJetFuel.csv'
    kerosene = 'usKerosin.csv'
    
    dataCrudeOilWTI = readData(readFile)
    dataLigthFuelOil = readData(ligthFuelOil)
    dataGasoline = readData(gasoline)
    dataJetFuel = readData(jetFuel)
    keroseneOil = readData(kerosene)
    print(keroseneOil['Kerosene'])
    
    ##print(dataGasoline.head())
    qqPlot(dataCrudeOilWTI, 'WTI')
    # Merge two Dataframes on index of both the dataframes
    oilData = dataCrudeOilWTI.merge(dataLigthFuelOil, left_index=True, right_index=True)
    '''
    print(oilData.head())
    priceTrend(oilData)
    getSkewAndKurt(oilData, 'WTI')
    drawNormal(oilData, 'WTI')
    drawAutoCorr(oilData, 'WTI')
    
    arimaModelFitAnalysis(oilData, 'WTI')
    arimaModelRollingForCast(oilData, 'WTI')
    '''
    ##olsModel(oilData['WTI'], dataGasoline['Gasolin'], 'Gasoline')
    ##olsModel(oilData['WTI'], dataJetFuel['JetFuel'], 'US Jet Fuel')
    olsModel(oilData['WTI'], keroseneOil['Kerosene'], 'US Kerosene')
'''
    print(oilData.head())
    priceTrend(dataCrudeOilWTI, 'OilPriceWTI')
    qqPlot(dataCrudeOilWTI, 'OilPriceWTI')
    print(getVAR(dataCrudeOilWTI, 'OilPriceWTI'))
    print(getMean(dataCrudeOilWTI, 'OilPriceWTI'))
    print(getSkew(dataCrudeOilWTI, 'OilPriceWTI'))
'''   
    
if __name__=="__main__":
    main()


