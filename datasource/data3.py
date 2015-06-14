# -*- coding: utf-8 -*-
"""

@author: hyeonrok lee

"""


import numpy as np

import scipy.misc as misc
import scipy.optimize as opt

import matplotlib.pyplot as plt
import matplotlib.finance as mfinance
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

import panda as pd

from datetime import datetime

def generateRW(n=100,delta=0.001):
    series = []
    for i in range(n):
        p = 0        
        if i != 0:
            p = series[i-1]
        r = p + delta + np.random.normal(0.0,0.015)
        series.append(r)
    return series
    
def generateRandomWalk(n=100):
    series = []
    for i in range(n):
        r = np.random.normal(0.0,0.015)
        series.append(r)
    return series
    
def generateAR(n=100):
    pass
    
def generateMA(series):
    s = [0]
    for i in range(1,len(series)-1):
        s.append( (series[i-1] + series[i] + series[i+1]) * (1./3) )
    s.append(0)
    return s
    
def generate(n=100,m=4):
    allprices = []
    for i in range(m):
        prices = [1.0]    
        for j in range(n):
            direct = np.random.uniform()
            risk = np.random.normal(loc=0.0,scale=0.015)    
            if direct > 0.5:
                prices.append( prices[len(prices)-1] * (1+risk) )
            else:
                prices.append( prices[len(prices)-1] * (1-risk) )
        allprices.append(prices)
    r = np.asmatrix(allprices)
    return np.transpose(r)
    
def compute_mean_return(prices):
    daily_ret = (prices[1:prices.shape[0],:] - prices[0:prices.shape[0]-1,:]) / prices[0:prices.shape[0]-1,:]
    mean_ret = (daily_ret.T * np.asmatrix(np.ones( [daily_ret.shape[0],1] ))) / daily_ret.shape[0]
    diff = (daily_ret - mean_ret.T)
    cov_ret = (diff.T * diff) / daily_ret.shape[0]   
    return mean_ret,daily_ret,cov_ret

def compute_return_risk_pf(mean_ret,cov_ret,w):
    pf_expected_return = mean_ret.T * w
    pf_risk = w.T * cov_ret * w
    return pf_expected_return,pf_risk

if __name__ == '__main__':
    im = np.asmatrix(misc.lena())
    im = im - im.mean(0)
    (val,vec) = np.linalg.eig(np.cov(im.T))
    idx = np.argsort(val)
    idx = idx[::-1]
    vec = vec[:,idx]
    val = val[idx]

    coeff = vec[:,range(100)]
    re = np.dot(coeff.T,im.T)    
    rr = np.dot(coeff,re)
    a = np.asmatrix(np.array([ range(256) ]))
    a = a.reshape(16,16)
    a = a - a.mean(0)
    
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(im , cmap = plt.cm.gray, vmin=0, vmax=255 )
    plt.subplot(212)
    plt.imshow(coeff.astype(np.int64).T, cmap = plt.cm.gray)
    plt.show()
    
    '''    
    r = generateRandomWalk(250)
    s = generateMA(r)
    plt.figure(1)

    plt.subplot(211)
    plt.plot(r)
    plt.plot(s)
    
    plt.subplot(212)
    l = misc.lena()    
    a = np.array([ range(256) ])
    
    a = a.reshape(16,16)
    plt.imshow(a, cmap=plt.cm.gray )
    plt.show()
    '''
    
    '''
    w = np.matrix([1,0,0,0]).T    
    prices = generate()    
    mean_ret,daily_ret,cov_ret = compute_mean_return(prices)
    pf_exp_ret,pf_risk = compute_return_risk_pf(mean_ret,cov_ret,w)
    print(mean_ret)
    print(pf_exp_ret)
    print(pf_risk)    
    
    plt.figure(1)
    plt.subplot(221)
    plt.plot(prices[:,0].T.tolist()[0],'r-')
    plt.subplot(222)
    plt.plot(prices[:,1].T.tolist()[0],'r-')        
    plt.subplot(223)
    plt.plot(prices[:,2].T.tolist()[0],'r-')
    plt.subplot(224)
    plt.plot(prices[:,3].T.tolist()[0],'r-')

    plt.figure(2)
    plt.subplot(221)
    plt.plot(daily_ret[:,0].T.tolist()[0],'r-')    
    plt.subplot(222)
    plt.plot(daily_ret[:,1].T.tolist()[0],'r-')
    plt.subplot(223)
    plt.plot(daily_ret[:,2].T.tolist()[0],'r-')
    plt.subplot(224)
    plt.plot(daily_ret[:,3].T.tolist()[0],'r-')

    plt.show()
    '''
    


def percent_change(prices):
    return (prices[1:] - prices[:-1]) / prices[:-1]

def generate_stock_data(n=250):
    open_prices = np.ones(1)
    high_prices = np.ones(1)
    low_prices = np.ones(1)
    close_prices = np.ones(1)  
    
    for i in np.arange(1,n):
        prices = np.array([])
        prices = np.append(prices,close_prices[i-1] + close_prices[i-1] * np.random.normal(scale=0.02))
        prices = np.append(prices,prices[0] + prices[0] * np.random.normal(scale=0.03))
        prices = np.append(prices,prices[1] + prices[1] * np.random.normal(scale=0.03))
        prices = np.append(prices,prices[2] + prices[2] * np.random.normal(scale=0.02))

        open_prices = np.append(open_prices,prices[0])
        high_prices = np.append(high_prices,np.max(prices))
        low_prices = np.append(low_prices,np.min(prices))
        close_prices = np.append(close_prices,prices[3])
        
    return open_prices,high_prices,low_prices,close_prices

gen_stock_size = 5

historical_prices = []
historical_close_return = []
historical_close_expected_ret = []
historical_close_var = []

for i in range(gen_stock_size):
    open_prices,high_prices,low_prices,close_prices = generate_stock_data()
    close_return = percent_change(close_prices)
    historical_prices.append([ open_prices,high_prices,low_prices,close_prices ])
    historical_close_return.append(close_return)
    historical_close_var.append(np.var(close_return))
    historical_close_expected_ret.append(np.mean(close_return))

historical_close_cov = np.cov(np.matrix(historical_close_return))
portfolio_weight = np.matrix(np.ones(gen_stock_size) / gen_stock_size)
portfolio_ret = np.dot(weight,historical_close_expected_ret)
portfolio_var = portfolio_weight * historical_close_cov * portfolio_weight.T

fig, ax = plt.subplots()
ax.plot(historical_close_expected_ret,historical_close_var,'r+')
plt.show()

'''

d = []
changes = []
for i in range(gen_size):
    open_prices,high_prices,low_prices,close_prices = generate_stock_data()
    d.append((open_prices,high_prices,low_prices,close_prices))
    changes.append(percent_change(close_prices))

open_prices,high_prices,low_prices,close_prices = generate_stock_data()
'''
'''
mondays = mdates.WeekdayLocator(mdates.MONDAY)        # major ticks on the mondays
alldays    = mdates.DayLocator()              # minor ticks on the days
weekFormatter = mdates.DateFormatter('%b %d')  # e.g., Jan 12
dayFormatter = mdates.DateFormatter('%d')      # e.g., 12

#starting from dates expressed as strings...
Date1 = '01/01/2010'
Date2 = '02/01/2010'
#...you convert them in float numbers....
Date1 = mdates.date2num(datetime.strptime(Date1, "%d/%m/%Y"))
Date2 = mdates.date2num(datetime.strptime(Date2, "%d/%m/%Y"))
#so redefining the Prices list of tuples...
Prices = [(Date1, 1.123, 1.212, 1.463, 1.056), (Date2,1.121, 1.016, 1.498, 1.002)]
#and then following the official example. 

fig, (ax,ax1) = plt.subplots(2,1)
fig.subplots_adjust(bottom=0.2)
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
for i in range(gen_stock_size):
    mfinance.candlestick2_ohlc(ax, historical_prices[i][0],historical_prices[i][1], \
                                historical_prices[i][2], historical_prices[i][3] , width=0.6 , colordown=u'b', colorup=u'r' )
    ax.plot([0,250],[1,  1 + 1*historical_close_expected_ret[i]*250  ])
    ax1.plot(historical_close_return[i])    
    
ax.xaxis_date()
ax.autoscale_view()
plt.setp( plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
'''

'''
plt.subplot(2,1,1)
plt.plot(open_prices)
plt.plot(close_prices)
plt.subplot(2,1,2)
plt.plot(open_prices-close_prices)
plt.show()
'''