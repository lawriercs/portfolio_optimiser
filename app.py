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

st.title("Portfolio Optimizer")
st.write("Enter your tickers to find the mathematically optimal asset allocation.")

# Format the sidebar for user input
st.sidebar.header("Portfolio Settings")
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "AAPL, MSFT, GOOGL")
ticker_list = [t.strip().upper() for t in tickers_input.split(",")]
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

if st.sidebar.button("Run Optimization"):
    
    # st.spinner shows a loading animation while the backend math runs
    with st.spinner("Fetching data and crunching numbers..."):
        
        # Fetch data and calculate stats using your backend functions
        formatted_start = start_date.strftime('%Y-%m-%d')
        formatted_end = end_date.strftime('%Y-%m-%d')
        prices = get_stock_data(ticker_list, start_date=formatted_start, end_date=formatted_end)
        daily_returns = calculate_daily_returns(prices)
        annual_returns, annual_covariance = get_portfolio_stats(daily_returns)
        
        # Run the Scipy Optimizer engine
        optimal_weights = maximize_sharpe_ratio(annual_returns, annual_covariance)
        
        # Calculate the final optimal performance metrics
        opt_return, opt_volatility, opt_sharpe = calculate_portfolio_performance(
            optimal_weights, annual_returns, annual_covariance
        )
        
        st.subheader("Optimal Asset Allocation")
        
        # Loop through each ticker and its calculated optimal weight
        for ticker, weight in zip(annual_returns.index, optimal_weights):
            # st.metric displays data beautifully with a label and value
            st.metric(label=f"{ticker} Weight", value=f"{weight:.2%}")
            
        st.subheader("Expected Portfolio Metrics")
        
        # Split into 3 coloumns with desired metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Annual Return", f"{opt_return:.2%}")
        col2.metric("Annual Volatility (Risk)", f"{opt_volatility:.2%}")
        col3.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")
