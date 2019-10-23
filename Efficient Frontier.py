# importing the required libraries
import numpy as np
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import pandas as pd

# grabbing stock data from yahoo finance

tickers = ['AAPL','KHC','PEP','NFLX']
dataSet = pd.DataFrame()
for t in tickers:
    dataSet[t] = wb.DataReader(t,data_source='yahoo',start='2018-10-22', end='2019-10-22')['Adj Close']

# calculating logarithmic stock returns and the covariance matrix, as well as the correlation between the stocks

logReturns = np.log(dataSet / dataSet.shift(1))
annual_returns = logReturns.mean() * 252
annual_covariance = logReturns.cov() *252
correlation = logReturns.corr()

# setting the number of portfolio combinations using different weights of the chosen stocks

stocks_num = len(tickers)
num_portfolios = 20000

# empty lists for the data which is to be generated in the next step

portfolio_returns = []
portfolio_volatilities = []
stocks_weights = []
sharpe_ratio = []

# calculating stock weights, portfolio returns and volatilties, as well as the Sharpe ratio

for single_portfolio in range(num_portfolios):
    weights = np.random.random(stocks_num)
    weights /= np.sum(weights)
    returns = np.dot(weights, annual_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(annual_covariance, weights)))
    sharpe = returns / volatility
    portfolio_returns.append(returns)
    portfolio_volatilities.append(volatility)
    stocks_weights.append(weights)
    sharpe_ratio.append(sharpe)

# setting a dictionary for the generated data

portfolio = {'Returns': portfolio_returns,
             'Volatility': portfolio_volatilities,
             'Sharpe Ratio' : sharpe_ratio}

# assigning weights to tickers and updating the dictionary

for counter, symbol in enumerate(tickers):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stocks_weights]

# arranging the data

df = pd.DataFrame(portfolio)
column_order = ['Returns', 'Volatility','Sharpe Ratio'] + [stock+' Weight' for stock in tickers]
df = df[column_order]

# calculating the minimum variance portfolio and the maximum Sharpe ratio portfolio

min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]
print(sharpe_portfolio.T)
print(min_variance_port.T)

# plotting the efficient frontier as well as the minimum variance portfolio and the maximum Sharpe ratio portfolio

plt.style.use('ggplot')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='viridis', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='blue', marker='*', s=400)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='red', marker='*', s=400 )
plt.xlabel('Volatility')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()
