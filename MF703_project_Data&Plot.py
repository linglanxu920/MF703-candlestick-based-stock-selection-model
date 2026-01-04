# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 00:19:29 2022

@author: 61012
"""

import pandas as pd
import yfinance as yf
import datetime
from mpl_finance import candlestick_ohlc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# Get Data
df = pd.read_csv('Stocks in the SP 500 Index.csv')
df['Symbol']
lst = []
for i in range(len(df['Symbol'])):
    lst.append(df['Symbol'][i])
lst
data = yf.download(lst,start="2015-01-01",end = "2019-12-31",group_by='ticker')
t1 = pd.DataFrame()
name = []
for i in range(503):
    for j in range(736):
        name.append(lst[i])
for i in range(503):
    t1 = t1.append(data[lst[i]])
t1['Stock'] = name
t1['Date'] = t1.index
t1.dropna()
del t1['Volume']
t1 = t1[['Date','Stock','Open','High','Low','Close','Adj Close']]
t1.to_csv('test5.csv',encoding ='utf-8',index=False)
spy = yf.download('SPY',start='2015-01-01',end='2019-12-31')
del spy['Volume']
spy['Date']=spy.index
spy = spy[['Date','Open','High','Low','Close','Adj Close']]
spy.to_csv('Benchmark.csv',encoding ='utf-8',index=False)

#Plot K-chart 
data = pd.read_csv('test5.csv')
dict_of_stocks = dict(iter(data.groupby('Stock')))
k1 =dict_of_stocks['ENPH'].loc[256281:256322]
k1.index=k1['Date']
k1['date'] = list(map(lambda x:mdates.date2num(datetime.datetime.strptime(x,'%Y-%m-%d')),k1.index.tolist()))
k1 = k1[['date','Open','High','Low','Close']]
ohlc = k1[['date','Open','High','Low','Close']]
f1, ax = plt.subplots(figsize = (21,7))                        # 创建图片
candlestick_ohlc(ax, ohlc.values.tolist(), width=.7
                 , colorup='green', colordown='red')           # 使用candlestick_ohlc绘图
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # 设置横轴日期格式
plt.xticks(rotation=30)                                        # 日期显示的旋转角度
plt.title('ENPH of June-July 2019',fontsize = 14)                            # 设置图片标题
plt.xlabel('Date',fontsize = 14)                               # 设置横轴标题
plt.ylabel('Price',fontsize = 14)                          # 设置纵轴标题
plt.savefig('ENPH of June-July 2019.jpg')
plt.show()

k2 =dict_of_stocks['EIX'].loc[364383:364424]
k2.index=k2['Date']
k2['date'] = list(map(lambda x:mdates.date2num(datetime.datetime.strptime(x,'%Y-%m-%d')),k2.index.tolist()))
k2 = k2[['date','Open','High','Low','Close']]
ohlc = k2[['date','Open','High','Low','Close']]
f1, ax = plt.subplots(figsize = (21,7))                        # 创建图片
candlestick_ohlc(ax, ohlc.values.tolist(), width=.7
                 , colorup='green', colordown='red')            # 使用candlestick_ohlc绘图
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # 设置横轴日期格式
plt.xticks(rotation=30)                                        # 日期显示的旋转角度
plt.title('EIX of June-July 2019',fontsize = 14)                            # 设置图片标题
plt.xlabel('Date',fontsize = 14)                               # 设置横轴标题
plt.ylabel('Price',fontsize = 14)                          # 设置纵轴标题
plt.savefig('EIX of June-July 2019.jpg')
plt.show()

k3 =dict_of_stocks['FISV'].loc[136866:136907]
k3.index=k3['Date']
k3['date'] = list(map(lambda x:mdates.date2num(datetime.datetime.strptime(x,'%Y-%m-%d')),k3.index.tolist()))
k3 = k3[['date','Open','High','Low','Close']]
ohlc = k3[['date','Open','High','Low','Close']]
f1, ax = plt.subplots(figsize = (21,7))                        # 创建图片
candlestick_ohlc(ax, ohlc.values.tolist(), width=.7
                 , colorup='green', colordown='red')            # 使用candlestick_ohlc绘图
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # 设置横轴日期格式
plt.xticks(rotation=30)                                        # 日期显示的旋转角度
plt.title('FISV of June-July 2019',fontsize = 14)                            # 设置图片标题
plt.xlabel('Date',fontsize = 14)                               # 设置横轴标题
plt.ylabel('Price',fontsize = 14)                          # 设置纵轴标题
plt.savefig('FISV of June-July 2019.jpg')
plt.show()

k4 =dict_of_stocks['AMD'].loc[92641:92684]
k4.index=k4['Date']
k4['date'] = list(map(lambda x:mdates.date2num(datetime.datetime.strptime(x,'%Y-%m-%d')),k4.index.tolist()))
k4 = k4[['date','Open','High','Low','Close']]
ohlc = k4[['date','Open','High','Low','Close']]
f1, ax = plt.subplots(figsize = (21,7))                        # 创建图片
candlestick_ohlc(ax, ohlc.values.tolist(), width=.7
                 , colorup='red', colordown='green')           # 使用candlestick_ohlc绘图
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # 设置横轴日期格式
plt.xticks(rotation=30)                                        # 日期显示的旋转角度
plt.title('AMD of July-August 2018',fontsize = 14)                            # 设置图片标题
plt.xlabel('Date',fontsize = 14)                               # 设置横轴标题
plt.ylabel('Price',fontsize = 14)                          # 设置纵轴标题
plt.savefig('AMD of July-August 2018.jpg')
plt.show()

k5 =dict_of_stocks['MTCH'].loc[493624:493667]
k5.index=k5['Date']
k5['date'] = list(map(lambda x:mdates.date2num(datetime.datetime.strptime(x,'%Y-%m-%d')),k5.index.tolist()))
k5 = k5[['date','Open','High','Low','Close']]
ohlc = k5[['date','Open','High','Low','Close']]
f1, ax = plt.subplots(figsize = (21,7))                        # 创建图片
candlestick_ohlc(ax, ohlc.values.tolist(), width=.7
                 , colorup='red', colordown='green')           # 使用candlestick_ohlc绘图
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # 设置横轴日期格式
plt.xticks(rotation=30)                                        # 日期显示的旋转角度
plt.title('MTCH of July-August 2018',fontsize = 14)                            # 设置图片标题
plt.xlabel('Date',fontsize = 14)                               # 设置横轴标题
plt.ylabel('Price',fontsize = 14)                          # 设置纵轴标题
plt.savefig('MTCH of July-August 2018.jpg')
plt.show()

k6 =dict_of_stocks['NTAP'].loc[479797:479840]
k6.index=k6['Date']
k6['date'] = list(map(lambda x:mdates.date2num(datetime.datetime.strptime(x,'%Y-%m-%d')),k6.index.tolist()))
k6 = k6[['date','Open','High','Low','Close']]
ohlc = k6[['date','Open','High','Low','Close']]
f1, ax = plt.subplots(figsize = (21,7))                        # 创建图片
candlestick_ohlc(ax, ohlc.values.tolist(), width=.7
                 , colorup='red', colordown='green')           # 使用candlestick_ohlc绘图
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # 设置横轴日期格式
plt.xticks(rotation=30)                                        # 日期显示的旋转角度
plt.title('NTAP of July-August 2018',fontsize = 14)                            # 设置图片标题
plt.xlabel('Date',fontsize = 14)                               # 设置横轴标题
plt.ylabel('Price',fontsize = 14)                          # 设置纵轴标题
plt.savefig('NTAP of July-August 2018.jpg')
plt.show()
