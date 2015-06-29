
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.finance as mfinance
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from datetime import datetime

def generate_random_walk(n=100,mu=0.0,sigma=0.015):
    series = [1.]
    for i in range(1,n):
        r = series[i-1] + series[i-1]*np.random.normal(mu,sigma)
        series.append(r)
    return series

def generate_stock_data(n=250,price=1,pos = 2):
    start_price = round(price,pos)
    open_prices = np.array([start_price])
    high_prices = np.array([start_price])
    low_prices = np.array([start_price])
    close_prices = np.array([start_price])
    
    for i in np.arange(1,n):
        prices = np.array([])
        prices = np.append(prices,round(close_prices[i-1] + close_prices[i-1] * np.random.normal(scale=0.02),pos) )
        prices = np.append(prices,round(prices[0] + prices[0] * np.random.normal(scale=0.03),pos) )
        prices = np.append(prices,round(prices[1] + prices[1] * np.random.normal(scale=0.03),pos) )
        prices = np.append(prices,round(prices[2] + prices[2] * np.random.normal(scale=0.02),pos) )

        open_prices = np.append(open_prices,prices[0])
        high_prices = np.append(high_prices,np.max(prices))
        low_prices = np.append(low_prices,np.min(prices))
        close_prices = np.append(close_prices,prices[3])
        
    return open_prices,high_prices,low_prices,close_prices

def read_mnist_digit_train():
    train = []
    test = []
    nrows = 0
    ncols = 0
    
    f = open('train-images.idx3-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))[0]
        numOfItems = struct.unpack('>I',f.read(4))[0]
        nrows = struct.unpack('>I',f.read(4))[0]
        ncols = struct.unpack('>I',f.read(4))[0]
        for i in range(numOfItems):
            data = struct.unpack('784B', f.read(nrows*ncols) )
            train.append([ data ])
    finally:
        f.close()
    
    f = open('train-labels.idx1-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))[0]
        numOfItems = struct.unpack('>I',f.read(4))[0]
        for i in range(numOfItems):
            label = struct.unpack('B',f.read(1))[0]
            train[i].append(label)
    finally:
        f.close()
        
    f = open('t10k-images.idx3-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))[0]
        numOfItems = struct.unpack('>I',f.read(4))[0]
        nrows = struct.unpack('>I',f.read(4))[0]
        ncols = struct.unpack('>I',f.read(4))[0]
        for i in range(numOfItems):
            data = struct.unpack('784B', f.read(nrows*ncols) )
            test.append([ data ])
    finally:
        f.close()
    
    f = open('t10k-labels.idx1-ubyte','rb')
    try:
        magic = struct.unpack('>I',f.read(4))[0]
        numOfItems = struct.unpack('>I',f.read(4))[0]
        for i in range(numOfItems):
            label = struct.unpack('B',f.read(1))[0]
            test[i].append(label)
    finally:
        f.close()        
        
    return train,test,nrows,ncols

if __name__ == '__main__':
    
    r = generate_random_walk()    
    fig, ax = plt.subplots(1,1)
    ax.plot(r)
    plt.show()
    
    '''
    # graphing chart
    open_prices,high_prices,low_prices,close_prices = generate_stock_data(n=20,price=132.12)
    mondays = mdates.WeekdayLocator(mdates.MONDAY)
    alldays = mdates.DayLocator()
    weekFormatter = mdates.DateFormatter('%b %d')
    dayFormatter = mdates.DateFormatter('%d')

    fig, ax = plt.subplots(1,1)
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)
    
    mfinance.candlestick2_ohlc(ax, open_prices,high_prices, \
        low_prices, close_prices, width=0.6 , colordown=u'b', colorup=u'r' )
        
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp( plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    
    plt.show()
    '''
    
    '''    
    train,test,nrows,ncols = read_mnist_digit_train()
    plt.imshow(np.reshape(np.array(train[0][0]),(nrows,ncols)),cmap = cm.Greys)
    plt.show()
    '''
