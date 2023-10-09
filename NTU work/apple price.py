# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:22:37 2023

@author: iru-ra2
"""

from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np


#initializing parameter
start_date  = datetime(2020,1,1)
end_date    = datetime(2023,8,23)

#get data - symbol to be in stock code
aapl_data = yf.download('AAPL', start=start_date, end=end_date)
AMD_data =  yf.download('AMD', start=start_date, end=end_date)
aaplMA = aapl_data['Open'].rolling(50).mean()
aaplMA2 = aapl_data['Open'].rolling(200).mean()


#display figsize is the x by y of the graph
plt.figure(figsize=(20,10)) 
plt.title('Opening Price from{} to {}'. format(start_date,end_date))

plt.plot(aapl_data['Open'], label = 'Apple Stock Price')
plt.plot(AMD_data['Open'], label = 'AMD Stock Price')
plt.plot(aaplMA, label = 'Apple moving average')
plt.plot(aaplMA2, label = 'Apple long moving average')

plt.grid(True)
plt.ylabel('Price')
plt.xlabel('Date')
plt.legend()
plt.show()

# ploting of Apple price completed
