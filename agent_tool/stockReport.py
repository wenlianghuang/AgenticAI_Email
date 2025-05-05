import requests

import matplotlib.pyplot as plt
import os 
from dotenv import load_dotenv
load_dotenv()

# Alpha Vantage API Key
API_KEY = os.getenv('Alpha_Vantage')
#API_KEY='V0RVYSEBFEE4A0TA'
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
    #print(data)  # Debugging line to check the response
    if 'quarterlyReports' not in data:
        raise ValueError("Unable to fetch data. Check the symbol or API key.")
    
    quarterly_reports = data['quarterlyReports']
    revenues = []
    quarters = []

    for report in quarterly_reports[:4]:  # Get the latest 4 quarters
        quarters.append(report['fiscalDateEnding'])
        revenues.append(float(report['totalRevenue']))

    return quarters[::-1], revenues[::-1]  # Reverse for chronological order
    
# Plotting the revenue data
def plot_revenue(quarters, revenues, symbol):
    plt.figure(figsize=(10, 6))
    plt.bar(quarters, revenues, color='skyblue')
    plt.title(f'{symbol} Quarterly Revenue (Last 4 Quarters)', fontsize=14)
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Revenue (in USD)', fontsize=12)
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))  # Format y-axis as currency
    plt.tight_layout()
    plt.savefig(f"{symbol}_quarterly_revenue.png")  # Save the plot before displaying
    plt.show()  # Display the plot
def get_quarterly_revenue_and_plot(symbol):
    try:
        quarters, revenues = get_quarterly_revenue(symbol)
        plot_revenue(quarters, revenues, symbol)
        #symbol = symbol.strip()
        return f"✅ Successfully fetched and plotted quarterly revenue data for {symbol}."
    except Exception as e:
        print(f"Error:{symbol}")
        return f"❌ Error: {str(e)}"
'''
if __name__ == "__main__":
    company_symbol = input("Enter the company symbol (e.g., AAPL for Apple): ").strip()
    try:
        #get_quarterly_revenue(company_symbol)
        quarters, revenues = get_quarterly_revenue(company_symbol)
        plot_revenue(quarters, revenues, company_symbol)
    except Exception as e:
        print(f"Error: {e}")
'''