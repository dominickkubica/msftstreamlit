import streamlit as st
import pandas as pd
import requests
import datetime
import time
import json
import logging
from io import StringIO

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Stock Data Fetcher", layout="wide")
st.title("📈 Stock Data Fetcher with Alpha Vantage (2025)")

# Sidebar controls
ticker = st.sidebar.text_input("Ticker symbol", value="MSFT").upper()
start_date = st.sidebar.date_input("Start date", value=datetime.date.today() - datetime.timedelta(days=60))
end_date = st.sidebar.date_input("End date", value=datetime.date.today())
delay_sec = st.sidebar.slider("Delay between calls (s)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

# Alpha Vantage API with your personal key
def fetch_alpha_vantage(symbol, start_date, end_date):
    """Fetch stock data from Alpha Vantage API"""
    logger.info(f"Fetching data for {symbol} from Alpha Vantage")
    
    # Use your provided Alpha Vantage API key
    api_key = "KA66573DH1OPJRIP"
    
    try:
        # First try the TIME_SERIES_DAILY endpoint
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full"
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        
        # Check if we have valid data
        if "Time Series (Daily)" in data:
            # Convert JSON to DataFrame
            time_series = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Rename columns
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            
            # Convert strings to numeric values
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            # Filter by date range
            df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
            
            if df.empty:
                logger.warning(f"No data found in date range for {symbol}")
                return None
                
            return df
        # Check for error message
        elif "Error Message" in data:
            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            return None
        elif "Note" in data:
            logger.warning(f"Alpha Vantage API limit reached: {data['Note']}")
            return None
        else:
            logger.error(f"Unexpected response format from Alpha Vantage: {data}")
            return None
    except Exception as e:
        logger.error(f"Error fetching Alpha Vantage data: {e}")
        return None

# Alpha Vantage API - CSV format as backup
def fetch_alpha_vantage_csv(symbol, start_date, end_date):
    """Fetch stock data from Alpha Vantage API using CSV format"""
    logger.info(f"Fetching data for {symbol} from Alpha Vantage (CSV)")
    
    # Use your provided Alpha Vantage API key
    api_key = "KA66573DH1OPJRIP"
    
    try:
        # Using the TIME_SERIES_DAILY endpoint with CSV output
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full&datatype=csv"
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Check if we got CSV data back
        if response.text.startswith("timestamp"):
            # Convert to DataFrame
            df = pd.read_csv(StringIO(response.text))
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Filter by date range
            df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
            
            # Rename columns to match standard format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            if df.empty:
                logger.warning(f"No data found in date range for {symbol}")
                return None
                
            return df
        else:
            logger.error(f"Unexpected response format from Alpha Vantage CSV: {response.text[:100]}...")
            return None
    except Exception as e:
        logger.error(f"Error fetching Alpha Vantage CSV data: {e}")
        return None

# Alpha Vantage Intraday API
def fetch_alpha_vantage_intraday(symbol, start_date, end_date):
    """Fetch intraday stock data from Alpha Vantage API"""
    logger.info(f"Fetching intraday data for {symbol} from Alpha Vantage")
    
    # Use your provided Alpha Vantage API key
    api_key = "KA66573DH1OPJRIP"
    
    try:
        # Using the TIME_SERIES_INTRADAY endpoint
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=60min&apikey={api_key}&outputsize=full"
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        
        # Check if we have valid data
        meta_data_key = "Meta Data"
        time_series_key = "Time Series (60min)"
        
        if meta_data_key in data and time_series_key in data:
            # Convert JSON to DataFrame
            time_series = data[time_series_key]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Rename columns
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            
            # Convert strings to numeric values
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            # Filter by date range
            df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
            
            if df.empty:
                logger.warning(f"No intraday data found in date range for {symbol}")
                return None
                
            return df
        elif "Error Message" in data:
            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            return None
        elif "Note" in data:
            logger.warning(f"Alpha Vantage API limit reached: {data['Note']}")
            return None
        else:
            logger.error(f"Unexpected response format from Alpha Vantage intraday: {data}")
            return None
    except Exception as e:
        logger.error(f"Error fetching Alpha Vantage intraday data: {e}")
        return None

# Alpha Vantage Weekly API
def fetch_alpha_vantage_weekly(symbol, start_date, end_date):
    """Fetch weekly stock data from Alpha Vantage API"""
    logger.info(f"Fetching weekly data for {symbol} from Alpha Vantage")
    
    # Use your provided Alpha Vantage API key
    api_key = "KA66573DH1OPJRIP"
    
    try:
        # Using the TIME_SERIES_WEEKLY endpoint
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={api_key}"
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        
        # Check if we have valid data
        if "Weekly Time Series" in data:
            # Convert JSON to DataFrame
            time_series = data["Weekly Time Series"]
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Rename columns
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            
            # Convert strings to numeric values
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            # Filter by date range
            df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
            
            if df.empty:
                logger.warning(f"No weekly data found in date range for {symbol}")
                return None
                
            return df
        elif "Error Message" in data:
            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            return None
        elif "Note" in data:
            logger.warning(f"Alpha Vantage API limit reached: {data['Note']}")
            return None
        else:
            logger.error(f"Unexpected response format from Alpha Vantage weekly: {data}")
            return None
    except Exception as e:
        logger.error(f"Error fetching Alpha Vantage weekly data: {e}")
        return None

# Simple CSV data fetcher as fallback
def fetch_csv_fallback(symbol, start_date, end_date):
    """Fetch stock data from a public CSV repository as fallback"""
    logger.info(f"Fetching data for {symbol} from CSV fallback")
    
    try:
        # This is a public repo with some stock CSV files for common tickers
        url = f"https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
        
        # For demonstration, we'll use AAPL data for any ticker
        # In a real app, you would have different URLs for different tickers
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        # Convert CSV to DataFrame
        df = pd.read_csv(StringIO(response.text))
        
        # Convert date to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Filter by date range if possible
        if not df.empty:
            # Try to filter, but if dates are outside range, return all data
            filtered_df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
            if not filtered_df.empty:
                df = filtered_df
        
        # Check if we have data
        if df.empty:
            logger.warning(f"No data found in CSV fallback for {symbol}")
            return None
            
        # Rename columns to match standard format
        column_map = {}
        for col in df.columns:
            if 'open' in col.lower():
                column_map[col] = 'Open'
            elif 'high' in col.lower():
                column_map[col] = 'High'
            elif 'low' in col.lower():
                column_map[col] = 'Low'
            elif 'close' in col.lower():
                column_map[col] = 'Close'
            elif 'volume' in col.lower():
                column_map[col] = 'Volume'
        
        df = df.rename(columns=column_map)
        
        # Select only OHLCV columns if they exist
        available_columns = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df.columns]
        if available_columns:
            df = df[available_columns]
            return df
        else:
            logger.warning(f"No usable columns found in CSV fallback for {symbol}")
            return None
    except Exception as e:
        logger.error(f"Error fetching CSV fallback data: {e}")
        return None

# Define our data sources
data_sources = {
    "Alpha Vantage (Daily)": fetch_alpha_vantage,
    "Alpha Vantage (CSV)": fetch_alpha_vantage_csv,
    "Alpha Vantage (Intraday)": fetch_alpha_vantage_intraday,
    "Alpha Vantage (Weekly)": fetch_alpha_vantage_weekly,
    "CSV Fallback": fetch_csv_fallback
}

# Sidebar for source selection
selected_sources = st.sidebar.multiselect(
    "Select data sources", 
    options=list(data_sources.keys()),
    default=["Alpha Vantage (Daily)", "Alpha Vantage (CSV)"]
)

# If no sources selected, use the default ones
if not selected_sources:
    selected_sources = ["Alpha Vantage (Daily)", "Alpha Vantage (CSV)"]
    st.sidebar.warning("No sources selected, using default sources.")

# Add a run button
run_button = st.sidebar.button("Fetch Data")

# Info section
with st.expander("ℹ️ About this app"):
    st.markdown("""
    ### Stock Data Fetcher with Alpha Vantage
    
    This app fetches stock market data from multiple Alpha Vantage API endpoints:
    
    - **Alpha Vantage (Daily)**: Daily OHLCV data in JSON format
    - **Alpha Vantage (CSV)**: Daily OHLCV data in CSV format (more efficient)
    - **Alpha Vantage (Intraday)**: Hourly OHLCV data for more granular analysis
    - **Alpha Vantage (Weekly)**: Weekly OHLCV data for longer-term trends
    - **CSV Fallback**: Uses public GitHub data repository (very limited)
    
    **API Rate Limits**:
    Alpha Vantage free tier allows 25 API calls per day, so use them wisely!
    
    **Troubleshooting**:
    - If you see a "Note" about frequency of API calls, wait a minute and try again
    - Try different date ranges (shorter periods work better)
    - Common symbols like MSFT, AAPL, GOOGL have the best support
    """)

# Run the data fetching when requested
if run_button:
    # Create tabs for each data source
    tabs = st.tabs(selected_sources + ["Combined Results"])
    
    all_results = {}
    
    # Loop through each selected source
    for i, source_name in enumerate(selected_sources):
        with tabs[i]:
            st.subheader(f"{source_name} Data for {ticker}")
            
            # Get the appropriate function
            fetch_func = data_sources[source_name]
            
            # Fetch the data
            start_time = time.time()
            df = fetch_func(ticker, start_date, end_date)
            elapsed = time.time() - start_time
            
            # Store result for combined view
            all_results[source_name] = df
            
            # Display metrics
            st.metric("Time (s)", f"{elapsed:.2f}")
            
            # Display the data or error message
            if df is not None and not df.empty:
                st.success(f"✅ Successfully retrieved {len(df)} days of data")
                
                # Display chart
                if 'Close' in df.columns:
                    st.line_chart(df["Close"])
                else:
                    st.line_chart(df.iloc[:, 0])  # Just use the first column
                
                # Display dataframe
                st.dataframe(df)
                
                # Allow downloading as CSV
                csv = df.to_csv()
                st.download_button(
                    label=f"Download {source_name} data as CSV",
                    data=csv,
                    file_name=f"{ticker}_{source_name.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"❌ Failed to retrieve data from {source_name}")
            
            # Add some space
            st.markdown("---")
            
            # Pause to avoid API rate limits
            time.sleep(delay_sec)
    
    # Combined Results tab
    with tabs[-1]:
        st.subheader(f"Combined Results for {ticker}")
        
        # Check if we have any successful results
        successful_sources = [name for name, df in all_results.items() if df is not None and not df.empty]
        
        if successful_sources:
            st.success(f"✅ Successfully retrieved data from {len(successful_sources)} sources")
            
            # Create empty combined dataframe
            combined_df = pd.DataFrame()
            
            # Add close prices from each source
            for source_name, df in all_results.items():
                if df is not None and not df.empty:
                    # Check if 'Close' column exists
                    if 'Close' in df.columns:
                        # Add close price with source name
                        combined_df[f"{source_name} Close"] = df["Close"]
                    else:
                        # Use the first available column
                        col_name = df.columns[0]
                        combined_df[f"{source_name} {col_name}"] = df[col_name]
            
            # Display chart of all close prices
            st.line_chart(combined_df)
            
            # Display combined dataframe
            st.dataframe(combined_df)
            
            # Allow downloading as CSV
            csv = combined_df.to_csv()
            st.download_button(
                label="Download Combined Data as CSV",
                data=csv,
                file_name=f"{ticker}_combined_data.csv",
                mime="text/csv"
            )
            
            # Calculate correlation between sources
            if len(successful_sources) > 1:
                st.subheader("Correlation between sources")
                correlation = combined_df.corr()
                st.dataframe(correlation.style.background_gradient(cmap="coolwarm"))
        else:
            st.error("❌ Failed to retrieve data from any source")
else:
    st.info("👈 Select your parameters and click 'Fetch Data' to start")

# Footer
st.markdown("""
---
### 📝 Notes
- Alpha Vantage free tier allows 25 API calls per day
- After reaching the limit, you'll need to wait until the next day
- The CSV Fallback option uses sample data that may not match your requested ticker
- For best results, avoid requesting data too frequently
""")