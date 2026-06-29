import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import streamlit as st

# Define the function 
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    return data['Adj Close']

# Set up the test parameters
test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start = '2024-01-01'
end = '2026-01-01'

def calculate_daily_returns(price_data):
    # Calculate the % change from the previous day
    returns = price_data.pct_change()
    
    # Drop the empty row using .dropna() 
    return returns.dropna()

def get_portfolio_stats(daily_returns):
    # Calculate the average daily return for each stock and annualize it
    annual_returns = daily_returns.mean() * 252
    
    # Calculate the daily covariance matrix and annualize it
    annual_covariance = daily_returns.cov() * 252
    
    return annual_returns, annual_covariance

def calculate_portfolio_performance(weights, annual_returns, annual_covariance):
    # Calculate Portfolio Return (Sum of each asset's return multiplied by its weight)
    portfolio_return = np.sum(annual_returns * weights)
    
    # Calculate Portfolio Volatility (Risk)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(annual_covariance, weights)))
    
    # Calculate Sharpe Ratio (Assuming a 0% risk-free asset rate for simplicity)
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def maximize_sharpe_ratio(annual_returns, annual_covariance):
    num_assets = len(annual_returns)
    
    # Start with an equal guess (e.g., [0.33, 0.33, 0.33])
    init_weights = num_assets * [1.0 / num_assets]
    
    # Define the objective function (Scipy minimizes this)
    def objective(weights):
        _, _, sharpe_ratio = calculate_portfolio_performance(weights, annual_returns, annual_covariance)
        return -sharpe_ratio  # Negative so Scipy maximizes the actual Sharpe
    
    # Set constraints: The sum of all weights must equal 1 (100% of the money)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Set bounds: No short-selling (weights must be between 0.0 and 1.0)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Run the mathematical optimization algorithm (SLSQP)
    result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Return the optimal weights array
    return result.x

# Fetch prices
prices = get_stock_data(test_tickers, start, end)

# Calculate daily returns
daily_returns = calculate_daily_returns(prices)

# Get the annualized stats
annual_returns, annual_covariance = get_portfolio_stats(daily_returns)

print("--- ANNUALIZED RETURNS ---")
print(annual_returns)
print("\n--- ANNUALIZED COVARIANCE MATRIX ---")
print(annual_covariance)

annual_returns, annual_covariance = get_portfolio_stats(daily_returns)

# Define equal weights for 3 assets (0.3333, 0.3333, 0.3333)
test_weights = np.array([1/3, 1/3, 1/3])

# Calculate performance
p_return, p_volatility, p_sharpe = calculate_portfolio_performance(test_weights, annual_returns, annual_covariance)

print("\n--- EQUAL WEIGHT PORTFOLIO PERFORMANCE ---")
print(f"Expected Annual Return: {p_return:.2%}")
print(f"Annual Volatility (Risk): {p_volatility:.2%}")
print(f"Sharpe Ratio: {p_sharpe:.2f}")

# Run the optimizer
optimal_weights = maximize_sharpe_ratio(annual_returns, annual_covariance)

# Calculate the optimized portfolio's performance
opt_return, opt_volatility, opt_sharpe = calculate_portfolio_performance(optimal_weights, annual_returns, annual_covariance)

print("\n--- OPTIMIZED PORTFOLIO PERFORMANCE ---")
for ticker, weight in zip(test_tickers, optimal_weights):
    print(f"{ticker} Optimal Weight: {weight:.2%}")

print(f"\nExpected Annual Return: {opt_return:.2%}")
print(f"Annual Volatility (Risk): {opt_volatility:.2%}")
print(f"Sharpe Ratio: {opt_sharpe:.2f}")