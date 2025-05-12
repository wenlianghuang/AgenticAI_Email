import requests
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import pandas as pd
import mplfinance as mpf
from pydantic import BaseModel
load_dotenv()

# Alpha Vantage API Key
API_KEY = os.getenv('Alpha_Vantage')
class StockReport(BaseModel):
    symbol: str
    choice: int
def get_financial_tool(symbol:str,choice: int)->str:
    """
    Get financial data and plot based on user choice.
    """
    if choice == "1":
        return get_quarterly_revenue_and_plot(symbol)
    elif choice == "2":
        return get_daily_data_and_plot_candle_chart(symbol)
    else:
        return "❌ Invalid choice. Please enter 1 or 2."
# Function to fetch quarterly revenue data
def get_quarterly_revenue(symbol):
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'INCOME_STATEMENT',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    if 'quarterlyReports' not in data:
        raise ValueError("Unable to fetch data. Check the symbol or API key.")
    
    quarterly_reports = data['quarterlyReports']
    revenues = []
    quarters = []

    for report in quarterly_reports:  # Get all available quarters
        quarters.append(report['fiscalDateEnding'])
        revenues.append(float(report['totalRevenue']))

    # Reverse for chronological order
    quarters = quarters[::-1]
    revenues = revenues[::-1]

    # Select the last 4 quarters
    last_4_quarters = quarters[-4:]
    last_4_revenues = revenues[-4:]

    # Calculate YOY changes for the last 4 quarters
    yoy_changes = []
    for i in range(-4, 0):  # Compare each of the last 4 quarters with the same quarter from the previous year
        if i - 4 >= -len(revenues):  # Ensure there is data for YOY comparison
            yoy_change = ((revenues[i] - revenues[i - 4]) / revenues[i - 4]) * 100
            yoy_changes.append(yoy_change)
        else:
            yoy_changes.append(None)  # Not enough data for YOY comparison

    return last_4_quarters, last_4_revenues, yoy_changes

# Plotting the revenue data
def plot_revenue(quarters, revenues, yoy_changes, symbol):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot revenue as a bar chart
    ax1.bar(quarters, revenues, color='skyblue', label='Quarterly Revenue')
    ax1.set_title(f'{symbol} Quarterly Revenue and YOY Comparison (Last 4 Quarters)', fontsize=14)
    ax1.set_xlabel('Quarter', fontsize=12)
    ax1.set_ylabel('Revenue (in USD)', fontsize=12)
    ax1.set_xticks(range(len(quarters)))
    ax1.set_xticklabels(quarters, rotation=45)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Create a secondary y-axis for YOY changes
    ax2 = ax1.twinx()
    ax2.plot(quarters, yoy_changes, color='red', marker='o', label='YOY Change (%)')
    ax2.set_ylabel('YOY Change (%)', fontsize=12)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Add a horizontal line at 0%

    # Add legends for both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{symbol}_quarterly_revenue_yoy.png")  # Save the plot before displaying
    plt.show()

# Combined function to fetch data and plot
def get_quarterly_revenue_and_plot(symbol):
    try:
        quarters, revenues, yoy_changes = get_quarterly_revenue(symbol)
        plot_revenue(quarters, revenues, yoy_changes, symbol)
        return f"✅ Successfully fetched and plotted quarterly revenue and YOY data for {symbol}."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Function to fetch daily stock data
def get_daily_stock_data(symbol):
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY,
        'outputsize': 'compact'  # Fetch last 100 days of data
    }
    response = requests.get(url, params=params)
    data = response.json()
    if 'Time Series (Daily)' not in data:
        raise ValueError("Unable to fetch data. Check the symbol or API key.")
    
    daily_data = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(daily_data, orient='index', dtype=float)
    df.index = pd.to_datetime(df.index)  # Convert index to datetime
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.sort_index()  # Sort by date in ascending order
    return df

# Function to plot candle chart
def plot_candle_chart(df, symbol):
    mpf.plot(
        df,
        type='candle',
        style='yahoo',
        title=f'{symbol} Daily Candle Chart',
        ylabel='Price (USD)',
        volume=True,
        savefig=f"{symbol}_candle_chart.png"  # Save the chart as an image
    )

    # Display the chart on the screen
    mpf.plot(
        df,
        type='candle',
        style='yahoo',
        title=f'{symbol} Daily Candle Chart',
        ylabel='Price (USD)',
        volume=True
    )

# Combined function to fetch and plot candle chart
def get_daily_data_and_plot_candle_chart(symbol):
    try:
        df = get_daily_stock_data(symbol)
        plot_candle_chart(df, symbol)
        return f"✅ Successfully fetched and plotted daily candle chart for {symbol}."
    except Exception as e:
        return f"❌ Error: {str(e)}"

'''
# Main interactive console
if __name__ == "__main__":
    company_symbol = input("Enter the company symbol (e.g., AAPL for Apple): ").strip()
    print("Choose an option:")
    print("1. Quarterly Revenue and YOY Comparison")
    print("2. Daily Candle Chart")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        print(get_quarterly_revenue_and_plot(company_symbol))
    elif choice == "2":
        print(get_daily_data_and_plot_candle_chart(company_symbol))
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")
'''