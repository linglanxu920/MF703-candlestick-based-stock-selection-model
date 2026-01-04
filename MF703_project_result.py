import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import empyrical as ep
import scipy.stats as st

data = pd.read_csv('test4.csv')  
benchmark=pd.read_csv('Benchmark.csv')
benchmark['Return']=(benchmark['Open']-benchmark['Close'])/benchmark['Open']
benchmark['month'] = pd.to_datetime(benchmark['Date']).dt.month
benchmark['year'] = pd.to_datetime(benchmark['Date']).dt.year  
bench1=benchmark[['year','month','Return']]
bench2=pd.DataFrame()
bench2['month_Return']=bench1.groupby(['year','month'])['Return'].sum()
bench2=bench2.reset_index()
bench2=bench2.drop(bench2.index[len(bench2)-1])

# factor
def Shadowline(data:pd.DataFrame) -> pd.DataFrame:
    high=data['High']
    low=data['Low']
    open_= data['Open']
    close= data['Close']
    data['upperShadow']=high - np.maximum(close,open_)
    data['lowerShadow']=np.minimum(close,open_)-low
    data['std_upperShadow']=data['upperShadow']/data['upperShadow'].rolling(5).mean().shift(1)
    data['std_lowerShadow']=data['lowerShadow']/data['lowerShadow'].rolling(5).mean().shift(1)
    data['ret']=(data['Close']-data['Open'])/data['Open']
    data['month']=pd.to_datetime(data['Date']).dt.month
    data['year']=pd.to_datetime(data['Date']).dt.year
    df = data[['std_upperShadow','std_lowerShadow','ret','year','month']]
    
    Factor=pd.DataFrame( )
    Factor['upper_mean']=df.groupby(['year','month'])['std_upperShadow'].mean()
    Factor['upper_std']=df.groupby(['year','month'])['std_upperShadow'].std()
    Factor['lower_mean']=df.groupby(['year','month'])['std_lowerShadow'].mean()
    Factor['lower_std']=df.groupby(['year','month'])['std_lowerShadow'].std()
    Factor['month_ret']=df.groupby(['year','month'])['ret'].sum()
    Factor=Factor.reset_index()
    Factor['Date']=Factor['year'].astype(str)+'-'+Factor['month'].astype(str)
    Factor=Factor.set_index(['Date'])
    return Factor

def William_factor(d:pd.DataFrame) -> pd.DataFrame:
    d['upperShadow']=d['High']-d['Close']
    d['std_upperShadow']=d['upperShadow']/d['upperShadow'].rolling(5).mean().shift(1)
    d['lowerShadow']=d['Close']-d['Low']
    d['std_lowerShadow']=d['lowerShadow']/d['lowerShadow'].rolling(5).mean().shift(1)
    d['ret']=(d['Close']-d['Open'])/d['Open']
    d['month'] = pd.to_datetime(d['Date']).dt.month
    d['year'] = pd.to_datetime(d['Date']).dt.year      
    d['day'] = pd.to_datetime(d['Date']).dt.day
    d1=d[['std_upperShadow','std_lowerShadow','ret','year','month','day']]
    
    Factor=pd.DataFrame()
    Factor['upper_mean']=d1.groupby(['year','month'])['std_upperShadow'].mean()
    Factor['upper_std']=d1.groupby(['year','month'])['std_upperShadow'].std()
    Factor['lower_mean']=d1.groupby(['year','month'])['std_lowerShadow'].mean()
    Factor['lower_std']=d1.groupby(['year','month'])['std_lowerShadow'].std()
    Factor['month_ret']=d1.groupby(['year','month'])['ret'].sum()
    Factor=Factor.reset_index()
    Factor['Date']=Factor['year'].astype(str)+'-'+Factor['month'].astype(str)
    Factor=Factor.set_index(['Date'])
    return Factor


# sort 5 groups
def sort_portfolio(m:pd.DataFrame,kind:str)->pd.DataFrame:
    m = m.dropna() 

    years = [2015, 2016, 2017, 2018, 2019]
    ##每个月的数据
    
    year = []
    month = [] 
    p1,p2,p3,p4,p5 = [[] for x in range(5)]
    for x in years:
        for j in range(1,13): 
           
            filted = m.loc[(m['year'] == x)& (m['month'] == j)].copy()#把每个月的数据弄出来
            filted['rank'] = filted[kind].rank(pct=True)#每个月排名
           
            if j == 12 & x==2019:
                break
            elif j == 12:
                next_month = 1
                next_month_filted = m.loc[(m['year'] == x+1)& (m['month'] == next_month)].copy()
            else:
                next_month_filted = m.loc[(m['year'] == x)& (m['month'] == j+1)].copy()
                
            filted20 = filted.loc[(filted['rank'] <= 0.2)]#20以下
            Stock20 = filted20['Stock'].values.tolist()#把股票名扔到list里
            filted20_next =  next_month_filted[(next_month_filted['Stock'].isin(Stock20))]#把下个月股票名字对应的拿出来一个df
            filted20_return = filted20_next['month_ret'].mean()#return
            filted40 = filted.loc[(filted['rank'] <= 0.4)& (filted['rank'] > 0.2)]
            Stock40 = filted40['Stock'].values.tolist()
            filted40_next =  next_month_filted[(next_month_filted['Stock'].isin(Stock40))]
            filted40_return = filted40_next['month_ret'].mean()
            filted60 = filted.loc[(filted['rank'] <= 0.6)& (filted['rank'] > 0.4)]
            Stock60 = filted60['Stock'].values.tolist()
            filted60_next =  next_month_filted[(next_month_filted['Stock'].isin(Stock60))]
            filted60_return = filted60_next['month_ret'].mean()
            filted80 = filted.loc[(filted['rank'] <= 0.8)& (filted['rank'] > 0.6)]
            Stock80 = filted80['Stock'].values.tolist()
            filted80_next =  next_month_filted[(next_month_filted['Stock'].isin(Stock80))]
            filted80_return = filted80_next['month_ret'].mean()
            filted100 = filted.loc[(filted['rank'] <= 1)& (filted['rank'] > 0.8)]
            Stock100 = filted100['Stock'].values.tolist()
            filted100_next =  next_month_filted[(next_month_filted['Stock'].isin(Stock100))]
            filted100_return = filted100_next['month_ret'].mean()   
            year.append(x)
            month.append(j)
            p1.append(filted20_return)
            p2.append(filted40_return)
            p3.append(filted60_return)
            p4.append(filted80_return)
            p5.append(filted100_return)
    
    result = pd.DataFrame({'year': year,'month':month,'porfolio 1':p1,'porfolio 2':p2
                                    ,'porfolio 3':p3,'porfolio 4':p4,'porfolio 5':p5} )
    result= result.iloc[:-1 , :]
    return result

def Strategy_performance(return_df: pd.DataFrame, periods='monthly') -> pd.DataFrame:
    

    ser: pd.DataFrame = pd.DataFrame()
    ser['Annual_Return'] = ep.annual_return(return_df, period=periods)
    ser['Annual_Volatility'] = return_df.apply(lambda x: ep.annual_volatility(x,period=periods))
    ser['sharpe_ratio'] = return_df.apply(ep.sharpe_ratio, period=periods)
    ser['Max Drawdown'] = return_df.apply(lambda x: ep.max_drawdown(x))

    if 'benchmark' in return_df.columns:

        select_col = [col for col in return_df.columns if col != 'benchmark']

        ser['IR'] = return_df[select_col].apply(lambda x: information_ratio(x, return_df['benchmark']))
        ser['Alpha'] = return_df[select_col].apply(lambda x: ep.alpha(x, return_df['benchmark'], period=periods))
        
    return ser.T

def information_ratio(returns, factor_returns):

    if len(returns) < 2:
        return np.nan

    active_return = _adjust_returns(returns, factor_returns)
    tracking_error = np.std(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    return np.mean(active_return) / tracking_error

def _adjust_returns(returns, adjustment_factor):

    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns.copy()
    return returns - adjustment_factor



def show_result(df: pd.DataFrame,name:str):
    table = Strategy_performance(df)
    print(table)
    df.plot()
    plt.title(name)
    
    
def best_stock(m:pd.DataFrame,kind:str,month,year)->pd.DataFrame:
    m = m.dropna() 

    filted = m.loc[(m['year'] == year)& (m['month'] == month-1)].copy()#把每个月的数据弄出来
    filted['rank'] = filted[kind].rank(pct=True)#每个月排名
       
    next_month_filted = m.loc[(m['year'] == year)& (m['month'] == month)].copy()
        
    best =  filted.loc[(filted['rank'] <= 1)& (filted['rank'] > 0.8)]
    Stock100 = best['Stock'].values.tolist()
    filted100_next =  next_month_filted[(next_month_filted['Stock'].isin(Stock100))]
    filted100_next['rank'] = filted100_next['month_ret'].rank(pct=True)  
    filted100_next = filted100_next.sort_values(by=['rank'], ascending=False)
    print(filted100_next)
    return filted100_next


def Strategy_performance(return_df: pd.DataFrame, periods='monthly') -> pd.DataFrame:
    

    ser: pd.DataFrame = pd.DataFrame()
    ser['Annual_Return'] = ep.annual_return(return_df, period=periods)
    ser['Annual_Volatility'] = return_df.apply(lambda x: ep.annual_volatility(x,period=periods))
    ser['sharpe_ratio'] = return_df.apply(ep.sharpe_ratio,period=periods)
    ser['Max Drawdown'] = return_df.apply(lambda x: ep.max_drawdown(x))
    if 'benchmark' in return_df.columns:

        select_col = [col for col in return_df.columns if col != 'benchmark']

        ser['IR'] = return_df[select_col].apply(lambda x: information_ratio(x, return_df['benchmark']))
        ser['Alpha'] = return_df[select_col].apply(lambda x: ep.alpha(x, return_df['benchmark'], period=periods))
    return ser.T

def max_dd(df,name:str,factor):
    Roll_Max = line_uppermean[name].rolling(60, min_periods=1).max()
    monthly_Drawdown = line_uppermean[name]/Roll_Max - 1.0

    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    Max_Daily_Drawdown = monthly_Drawdown.rolling(60, min_periods=1).min()

    # Plot the results
    monthly_Drawdown.plot()
    Max_Daily_Drawdown.plot()
    plt.title(factor+' '+name)
    plt.ylabel('Monthly_Drawdown')
    plt.show()

def show_result(df: pd.DataFrame,name:str):
    table = Strategy_performance(df)
    df.cumsum().plot()
    plt.title(name)
    plt.show()
    max_dd(table,'benchmark',name)
    max_dd(table,'excess_ret',name)
    table.to_csv(name+'.csv')
    return table
    
Group=list(data.groupby(['Stock']))
factor_shadowline=pd.DataFrame()
factor_william = pd.DataFrame()

for i in range(len(Group)):
    shadowline = Shadowline(Group[i][1])
    william_shadowline=William_factor(Group[i][1])
    shadowline['Stock']=Group[i][0]
    william_shadowline['Stock']=Group[i][0]
    factor_shadowline = factor_shadowline.append(shadowline)
    factor_william =factor_william.append(william_shadowline)

def rank_ic(m:pd.DataFrame,s:str) -> pd.DataFrame:
    years = [2015, 2016, 2017, 2018, 2019]
    ##每个月的数据
    
    year = []
    month = [] 
    
    ic20,ic40,ic60,ic80,ic100=[[] for x in range(5)]
    
    for x in years:
        for j in range(1,13): 
          
            filted = m.loc[(m['year'] == x)& (m['month'] == j)].copy()#把每个月的数据弄出来
            filted['rank'] = filted[s].rank(pct=True)#每个月排名
            if j ==12 & x==2019:
                break
            #elif j == 12:
                #next_month = 1
                #next_month_filted = m.loc[(m['year'] == x+1)& (m['month'] == next_month)].copy()
            #else:
                #next_month_filted = m.loc[(m['year'] == x)& (m['month'] == j+1)].copy()
                
            filted20 = filted.loc[(filted['rank'] <= 0.2)]#20以下
            filted20['IC value']=filted20['rank'].corr(filted20['month_ret'])
            filted40 = filted.loc[(filted['rank'] <= 0.4)& (filted['rank'] > 0.2)]
            filted40['IC value']=filted40['rank'].corr(filted40['month_ret'])
            filted60 = filted.loc[(filted['rank'] <= 0.6)& (filted['rank'] > 0.4)]
            filted60['IC value']=filted60['rank'].corr(filted60['month_ret'])
            filted80 = filted.loc[(filted['rank'] <= 0.8)& (filted['rank'] > 0.6)]
            filted80['IC value']=filted80['rank'].corr(filted80['month_ret'])
    
            filted100 = filted.loc[(filted['rank'] > 0.8)]
            filted100['IC value']=filted100['rank'].corr(filted100['month_ret'])
            year.append(x)
            month.append(j)
            ic20.append(np.mean(filted20['IC value']))
            ic40.append(np.mean(filted40['IC value']))
            ic60.append(np.mean(filted60['IC value']))
            ic80.append(np.mean(filted80['IC value']))
            ic100.append(np.mean(filted100['IC value']))
            
    
    #rankdf['porfolio_1'] = filted20['rank']
    #rankdf['portfolio_5'] = filted100['rank']
            
    result = pd.DataFrame({'year': year,'month':month,'ic20':ic20,'ic40':ic40
                                    ,'ic60':ic60,'ic80':ic80,'ic100':ic100} )
    
    result['ic'] = result['ic100']-result['ic20']
    result.drop(['ic20', 'ic40','ic60','ic80','ic100'], axis=1,inplace = True)
    return result


    
##Upper mean
line_uppermean = sort_portfolio(factor_shadowline,'upper_mean')
line_uppermean['benchmark']=bench2['month_Return']
line_uppermean['excess_ret']=line_uppermean['porfolio 5']-line_uppermean['porfolio 1']
line_uppermean=line_uppermean.set_index(['year','month'])
line_uppermean=line_uppermean.cumsum()
line_uppermean
line_uppermean.plot()
plt.title('Upper_shadow_mean')
show_result(line_uppermean,"Upper_shadow_mean")

line_uppermean_result = show_result(line_uppermean,"Upper_shadow_mean")
line_uppermean_best = best_stock(factor_shadowline,'upper_mean',7,2019)




##Upper std
line_upperstd = sort_portfolio(factor_shadowline,'upper_std')
line_upperstd['benchmark']=bench2['month_Return']
line_upperstd['excess_ret']=line_upperstd['porfolio 5']-line_upperstd['porfolio 1']
line_upperstd=line_upperstd.set_index(['year','month'])
line_upperstd=line_upperstd.cumsum()
line_upperstd.plot()
line_upperstd_result =show_result(line_upperstd,"Upper_shadow_std")
show_result(line_upperstd,"Upper_shadow_std")

##Lower mean
line_lowermean = sort_portfolio(factor_shadowline,'lower_mean')
line_lowermean['benchmark']=bench2['month_Return']
line_lowermean['excess_ret']=line_lowermean['porfolio 5']-line_lowermean['porfolio 1']
line_lowermean=line_lowermean.set_index(['year','month'])
line_lowermean=line_lowermean.cumsum()
show_result(line_lowermean,"Lower_shadow_mean")
##Lower std
line_lowerstd = sort_portfolio(factor_shadowline,'lower_std')
line_lowerstd['benchmark']=bench2['month_Return']
line_lowerstd['excess_ret']=line_lowerstd['porfolio 5']-line_lowerstd['porfolio 1']
line_lowerstd=line_lowerstd.set_index(['year','month'])
line_lowerstd=line_lowerstd.cumsum()
show_result(line_lowerstd,"Lower_shadow_std")
##
line_lowerstd['porfolio 5'].plot(color='red')
line_upperstd['porfolio 5'].plot(color='yellow')
line_lowermean['porfolio 5'].plot(color='blue')
line_uppermean['porfolio 5'].plot(color='green')

plt.plot(np.array(line_lowerstd['porfolio 5']), color='Navy')
##Upper mean
william_uppermean = sort_portfolio(factor_william,'upper_mean')
william_uppermean['benchmark']=bench2['month_Return']
william_uppermean['excess_ret']=william_uppermean['porfolio 5']-william_uppermean['porfolio 1']
william_uppermean=william_uppermean.set_index(['year','month'])
william_uppermean=william_uppermean.cumsum()
show_result(william_uppermean,"William_upper_mean")
##Upper std
william_upperstd = sort_portfolio(factor_william,'upper_std')
william_upperstd['benchmark']=bench2['month_Return']
william_upperstd['excess_ret']=william_upperstd['porfolio 5']-william_upperstd['porfolio 1']
william_upperstd=william_upperstd.set_index(['year','month'])
william_upperstd=william_upperstd.cumsum()
show_result(william_upperstd,"william_upper_std")
##Lower mean
william_lowermean = sort_portfolio(factor_william,'lower_mean')
william_lowermean['benchmark']=bench2['month_Return']
william_lowermean['excess_ret']=william_lowermean['porfolio 5']-william_lowermean['porfolio 1']
william_lowermean=william_lowermean.set_index(['year','month'])
william_lowermean=william_lowermean.cumsum()
show_result(william_lowermean,"william_lower_mean")
##Lower std
william_lowerstd = sort_portfolio(factor_william,'lower_std')
william_lowerstd['benchmark']=bench2['month_Return']
william_lowerstd['excess_ret']=william_lowerstd['porfolio 5']-william_lowerstd['porfolio 1']
william_lowerstd=william_lowerstd.set_index(['year','month'])
william_lowerstd=william_lowerstd.cumsum()
show_result(william_lowerstd,"william_lower_mean")
william_lowerstd_result = show_result(william_lowerstd,"william_lower_std")

william_lowerstd_best = best_stock(factor_william,'lower_std',8,2018)



# ic 
ic_uppermean = rank_ic(factor_shadowline,'upper_mean')
ic_upperstd = rank_ic(factor_shadowline,'upper_std')
ic_lowermean = rank_ic(factor_shadowline,'lower_mean')
ic_lowerstd = rank_ic(factor_shadowline,'lower_std')
ic_w_uppermean = rank_ic(factor_william,'upper_mean')
ic_w_upperstd = rank_ic(factor_william,'upper_std')
ic_w_lowermean = rank_ic(factor_william,'lower_mean')
ic_w_lowerstd = rank_ic(factor_william,'lower_std')

df_rankic = pd.DataFrame({'upper_mean':{'ic_mean':ic_uppermean['ic'].mean(),'ic_std':ic_uppermean['ic'].std()},
                          'upper_std':{'ic_mean':ic_upperstd['ic'].mean(),'ic_std':ic_upperstd['ic'].std()},
                          'lower_mean':{'ic_mean':ic_lowermean['ic'].mean(),'ic_std':ic_lowermean['ic'].std()},
                          'lower_std':{'ic_mean':ic_lowerstd['ic'].mean(),'ic_std':ic_lowerstd['ic'].std()},
                          'william_upper_mean':{'ic_mean':ic_w_uppermean['ic'].mean(),'ic_std':ic_w_uppermean['ic'].std()},
                          'william_upper_std':{'ic_mean':ic_w_upperstd['ic'].mean(),'ic_std':ic_w_upperstd['ic'].std()},
                          'william_lower_mean':{'ic_mean':ic_w_lowermean['ic'].mean(),'ic_std':ic_w_lowermean['ic'].std()},
                          'william_lower_std':{'ic_mean':ic_w_lowerstd['ic'].mean(),'ic_std':ic_w_lowerstd['ic'].std()}})
#df_rankic.to_csv('rankic_result.csv')
        

    