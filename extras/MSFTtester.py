import streamlit as st

# 👇 MUST be before any other Streamlit call (including decorators)
st.set_page_config(page_title="Stock Data & Sentiment Analyzer", layout="wide")

# ---------- standard imports ----------
import pandas as pd
import numpy as np
import io
import datetime
import logging
import re
import unicodedata
from pathlib import Path
import requests
import time
import json
from io import StringIO

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download as nltk_dl
from nltk.tokenize import sent_tokenize            # ← sentence splitter

import PyPDF2
import docx
import yfinance as yf
import plotly.express as px

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, 
                    format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------- one-time setup ----------
@st.cache_resource
def _init_nltk():
    # download both VADER lexicon and sentence tokenizer
    nltk_dl("vader_lexicon")
    nltk_dl("punkt")
    return SentimentIntensityAnalyzer()

SIA = _init_nltk()

# ---------- helpers for text analysis ----------
def extract_text(file) -> str:
    ext = Path(file.name).suffix.lower()
    data = file.read()
    if ext == ".pdf":
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    if ext == ".docx":
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    return data.decode("utf-8", errors="ignore")

def clean(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    return re.sub(r"\s+", " ", t).strip()

def vader_sentiment(text: str) -> str:
    score = SIA.polarity_scores(text)["compound"]
    return "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"

# ---------- Alpha Vantage API Functions ----------
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

# ---------- Yfinance backup function ----------
def fetch_yfinance_backup(symbol, start_date, end_date):
    """Backup function using yfinance"""
    logger.info(f"Trying yfinance backup for {symbol}")
    
    try:
        # Convert to datetime for yfinance
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # Add a day to include end_date
        
        df = yf.download(symbol, start=start_dt, end=end_dt, progress=False)
        
        if df.empty:
            logger.warning(f"No data found for {symbol} using yfinance")
            return None
            
        return df
    except Exception as e:
        logger.error(f"Error fetching yfinance data: {e}")
        return None

# Define our data sources
data_sources = {
    "Alpha Vantage (JSON)": fetch_alpha_vantage,
    "Alpha Vantage (CSV)": fetch_alpha_vantage_csv,
    "Yahoo Finance (Backup)": fetch_yfinance_backup
}

# ---------- UI LAYOUT ----------
st.title("📈 Stock Data & Transcript Sentiment Analyzer")

# Create two tabs for the different functionalities
tab1, tab2 = st.tabs(["Stock Data", "Transcript Sentiment"])

with tab1:
    st.header("Stock Data Fetcher")
    
    # ---------- Sidebar inputs for stock data ----------
    st.sidebar.markdown("## Stock Data Settings")
    ticker = st.sidebar.text_input("Stock ticker", value="MSFT").upper()
    start_date = st.sidebar.date_input(
        "Start date", 
        value=datetime.date.today() - datetime.timedelta(days=60)
    )
    end_date = st.sidebar.date_input(
        "End date", 
        value=datetime.date.today()
    )
    delay_sec = st.sidebar.slider(
        "Delay between calls (s)", 
        min_value=0.0, 
        max_value=5.0, 
        value=1.0, 
        step=0.1
    )
    
    # Sidebar for source selection
    selected_sources = st.sidebar.multiselect(
        "Select data sources", 
        options=list(data_sources.keys()),
        default=["Alpha Vantage (JSON)", "Alpha Vantage (CSV)"]
    )
    
    # If no sources selected, use the default ones
    if not selected_sources:
        selected_sources = ["Alpha Vantage (JSON)", "Alpha Vantage (CSV)"]
        st.sidebar.warning("No sources selected, using default sources.")
    
    # Add a run button for stock data
    fetch_button = st.sidebar.button("Fetch Stock Data")
    
    # Info section for stock data
    with st.expander("ℹ️ About Stock Data Fetcher"):
        st.markdown("""
        ### Stock Data Fetcher with Alpha Vantage
        
        This section fetches stock market data from multiple sources:
        
        - **Alpha Vantage (JSON)**: Daily OHLCV data in JSON format
        - **Alpha Vantage (CSV)**: Daily OHLCV data in CSV format (more efficient)
        - **Yahoo Finance (Backup)**: Alternative source if Alpha Vantage fails
        
        **API Rate Limits**:
        Alpha Vantage free tier allows 25 API calls per day, so use them wisely!
        
        **Troubleshooting**:
        - If you see a "Note" about frequency of API calls, wait a minute and try again
        - Try different date ranges (shorter periods work better)
        - Common symbols like MSFT, AAPL, GOOGL have the best support
        """)
    
    # Run the data fetching when requested
    if fetch_button:
        # Create tabs for each data source
        source_tabs = st.tabs(selected_sources + ["Combined Results"])
        
        all_results = {}
        
        # Loop through each selected source
        for i, source_name in enumerate(selected_sources):
            with source_tabs[i]:
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
        with source_tabs[-1]:
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
        st.info("👈 Select your parameters and click 'Fetch Stock Data' to start")

with tab2:
    st.header("Transcript Sentiment Analysis")
    
    # ---------- sidebar inputs for transcript analysis ----------
    st.sidebar.markdown("## Transcript Analysis Settings")
    uploaded = st.sidebar.file_uploader(
        "Upload transcript (txt, pdf, docx)",
        type=["txt", "pdf", "docx"]
    )
    sentiment_ticker = st.sidebar.text_input("Stock ticker for comparison", value=ticker)
    
    # ---------- main logic for transcript analysis ----------
    if uploaded:
        with st.spinner("Extracting text…"):
            raw_text = extract_text(uploaded)

        if not raw_text:
            st.error("Could not read any text from this file.")
            st.stop()

        # sentence-level split → sentiment
        text = clean(raw_text)
        sentences = sent_tokenize(text)
        paragraphs = [s for s in sentences if s.strip()]

        df = pd.DataFrame({
            "sentence": paragraphs,
            "sentiment": [vader_sentiment(s) for s in paragraphs]
        })

        # Show summary statistics
        total_sentences = len(df)
        positive_count = len(df[df['sentiment'] == 'positive'])
        negative_count = len(df[df['sentiment'] == 'negative'])
        neutral_count = len(df[df['sentiment'] == 'neutral'])
        
        positive_pct = positive_count / total_sentences * 100
        negative_pct = negative_count / total_sentences * 100
        neutral_pct = neutral_count / total_sentences * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", f"{positive_count} ({positive_pct:.1f}%)")
        col2.metric("Negative", f"{negative_count} ({negative_pct:.1f}%)")
        col3.metric("Neutral", f"{neutral_count} ({neutral_pct:.1f}%)")
        
        # pie chart
        st.subheader("Sentiment Distribution")
        counts = df["sentiment"].value_counts()
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            hole=0.4,
            color=counts.index,
            color_discrete_map={'positive':'green', 'neutral':'gray', 'negative':'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sentiment table
        with st.expander("Show Sentiment Table"):
            st.dataframe(df)

        # ---------- optional stock overlay ----------
        if sentiment_ticker:
            st.subheader(f"{sentiment_ticker} Stock Price Comparison")
            
            # Use the same date range as in the stock data tab
            with st.spinner(f"Downloading {sentiment_ticker} prices…"):
                # Try first with Alpha Vantage
                prices = fetch_alpha_vantage(sentiment_ticker, start_date, end_date)
                
                # If Alpha Vantage fails, try with yfinance
                if prices is None or prices.empty:
                    st.warning("Alpha Vantage data not available. Trying Yahoo Finance...")
                    prices = fetch_yfinance_backup(sentiment_ticker, start_date, end_date)

            if prices is not None and not prices.empty:
                st.line_chart(prices["Close"])
                
                # Analysis hint
                st.info("""
                **Analysis Tip:** Compare sentiment distribution with stock price movements. 
                Look for correlations between sentiment shifts and price changes following earnings calls or major announcements.
                """)
            else:
                st.warning("No price data returned (check ticker or date range).")
    else:
        st.info("⬅️ Upload a transcript to analyze its sentiment.")

# Footer
st.markdown("""
---
### 📝 Notes
- Alpha Vantage free tier allows 25 API calls per day
- After reaching the limit, the app will fall back to Yahoo Finance
- For best results with sentiment analysis, use official earnings call transcripts
- Sentiment analysis uses VADER, which is optimized for social media but works well on financial text
""")