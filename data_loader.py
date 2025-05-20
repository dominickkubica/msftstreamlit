# data_loader.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import logging

# Setup simplified logging
logger = logging.getLogger("earnings_analyzer")

# Recent Microsoft earnings dates
MSFT_EARNINGS_DATES = [
    "2023-01-24", "2023-04-25", "2023-07-25", "2023-10-24",
    "2024-01-30", "2024-04-25", "2024-07-30", "2024-10-30",
    "2025-01-29", "2025-04-30"
]

# Function to load stock data from CSV
def load_stock_data(file_path):
    """Load stock data from CSV and perform minimal processing"""
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
        
    try:
        # Load raw data
        df = pd.read_csv(file_path)
        
        # Identify date and price columns
        date_col = None
        price_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'date' in col_lower:
                date_col = col
            elif 'close' in col_lower or 'last' in col_lower:
                price_col = col
        
        if not date_col or not price_col:
            st.error("Could not identify date and price columns")
            return None
        
        # Convert columns to proper types
        df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Find OHLC columns if they exist
        ohlc_cols = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                ohlc_cols['open'] = col
            elif 'high' in col_lower:
                ohlc_cols['high'] = col
            elif 'low' in col_lower:
                ohlc_cols['low'] = col
            elif 'volume' in col_lower:
                ohlc_cols['volume'] = col
        
        # Convert OHLC columns to numeric if they exist
        for key, col in ohlc_cols.items():
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
        
        # Drop any rows with invalid dates or prices
        df = df.dropna(subset=[date_col, price_col])
        
        # Sort by date (oldest first)
        df = df.sort_values(by=date_col)
        
        return df, date_col, price_col, ohlc_cols
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Function to find closest trading day to a given date
def find_closest_trading_day(df, date_col, target_date, direction='both'):
    """
    Find the closest trading day to the target date
    direction: 'both' (default), 'before', or 'after'
    """
    if direction == 'before':
        # Only consider dates before or on the target date
        valid_dates = [d for d in df[date_col].dt.date if d <= target_date]
        if not valid_dates:
            return None
        return max(valid_dates)
    
    elif direction == 'after':
        # Only consider dates after the target date
        valid_dates = [d for d in df[date_col].dt.date if d >= target_date]
        if not valid_dates:
            return None
        return min(valid_dates)
    
    else:  # both
        # Find the closest date in either direction
        min_diff = timedelta(days=365)
        closest_date = None
        
        for date in df[date_col].dt.date:
            diff = abs(date - target_date)
            if diff < min_diff:
                min_diff = diff
                closest_date = date
        
        return closest_date

# Generate sample stock data if actual data is not available
def generate_sample_stock_data(start_date, end_date):
    """Generate sample stock data if actual data is not available"""
    try:
        # Generate sample data
        date_range = pd.date_range(start=start_date, end=end_date)
        sample_data = pd.DataFrame(index=date_range)
        
        # Set column names
        sample_data['Date'] = date_range
        sample_data['Open'] = np.linspace(250, 280, len(date_range)) + np.random.normal(0, 5, len(date_range))
        sample_data['High'] = sample_data['Open'] + np.random.uniform(1, 10, len(date_range))
        sample_data['Low'] = sample_data['Open'] - np.random.uniform(1, 10, len(date_range))
        sample_data['Close'] = sample_data['Open'] + np.random.normal(0, 5, len(date_range))
        sample_data['Volume'] = np.random.uniform(20000000, 50000000, len(date_range))
        
        # Create the same structure as load_stock_data
        date_col = 'Date'
        price_col = 'Close'
        ohlc_cols = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume'
        }
        
        return sample_data, date_col, price_col, ohlc_cols
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return None

# Load pre-categorized sentences from CSV
@st.cache_data
def load_precategorized_sentences(csv_path):
    """Load sentences already categorized by business line from CSV file"""
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Rename columns if needed
            if 'Sentence' in df.columns and 'BusinessUnit' in df.columns:
                df = df.rename(columns={'Sentence': 'text', 'BusinessUnit': 'business_line'})
            
            # Add needed columns if they don't exist
            if 'sentiment' not in df.columns:
                df['sentiment'] = None
            
            if 'section' not in df.columns:
                # Determine if there's a Q&A section based on content
                df['section'] = 'main'
                qa_indicators = ['Q:', 'Question:', 'Analyst:', 'Q&A']
                
                for indicator in qa_indicators:
                    mask = df['text'].str.contains(indicator, case=False, na=False)
                    if mask.sum() > 3:  # If multiple sentences contain Q&A indicators
                        df.loc[mask, 'section'] = 'qa'
            
            if 'processed' not in df.columns:
                df['processed'] = False
                
            # Add paragraph info if not present
            if 'paragraph_id' not in df.columns:
                df['paragraph_id'] = range(len(df))
                df['paragraph'] = df['text'].apply(lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x))
            
            # Make sure we don't have NaN values in important columns
            df = df.fillna({'text': '', 'business_line': 'Other', 'section': 'main'})
            
            logger.info(f"Loaded {len(df)} pre-categorized sentences from CSV")
            return df
        else:
            st.warning(f"File not found: {csv_path}, using sample data instead")
            # Create sample sentences for demonstration
            sample_sentences = [
                {"text": "Our cloud revenue grew 28% year-over-year.", "business_line": "Intelligent Cloud", "sentiment": None, "section": "main"},
                {"text": "Office 365 continues to perform strongly with high adoption rates.", "business_line": "Productivity and Business Processes", "sentiment": None, "section": "main"},
                {"text": "Gaming revenue declined by 5% due to challenging comps.", "business_line": "More Personal Computing", "sentiment": None, "section": "main"},
                {"text": "Azure growth was 27%, which slightly underperformed our guidance.", "business_line": "Intelligent Cloud", "sentiment": None, "section": "main"},
                {"text": "Q: What are you seeing in terms of AI adoption among enterprise customers?", "business_line": "Q&A Section", "sentiment": None, "section": "qa"},
                {"text": "A: We're seeing tremendous interest in our AI solutions across all segments.", "business_line": "Q&A Section", "sentiment": None, "section": "qa"},
                {"text": "Our commercial cloud revenue was $29.6 billion, up 23% year-over-year.", "business_line": "Intelligent Cloud", "sentiment": None, "section": "main"},
                {"text": "LinkedIn revenue increased 10% with continued strength in Marketing Solutions.", "business_line": "Productivity and Business Processes", "sentiment": None, "section": "main"},
                {"text": "Windows OEM revenue decreased 2% as the PC market continues to stabilize.", "business_line": "More Personal Computing", "sentiment": None, "section": "main"},
                {"text": "Our Dynamics products and cloud services revenue increased 18%.", "business_line": "Productivity and Business Processes", "sentiment": None, "section": "main"},
                {"text": "Q: How do you see the competitive landscape evolving in AI?", "business_line": "Q&A Section", "sentiment": None, "section": "qa"},
                {"text": "A: We believe our comprehensive AI approach gives us a significant advantage.", "business_line": "Q&A Section", "sentiment": None, "section": "qa"},
                {"text": "Server products and cloud services revenue increased 25%, driven by Azure.", "business_line": "Intelligent Cloud", "sentiment": None, "section": "main"},
                {"text": "Operating expenses increased 11% reflecting investments in AI and cloud.", "business_line": "Other", "sentiment": None, "section": "main"},
                {"text": "Search and news advertising revenue increased 18% driven by higher search volume.", "business_line": "More Personal Computing", "sentiment": None, "section": "main"},
                {"text": "We returned $8.8 billion to shareholders through dividends and share repurchases.", "business_line": "Other", "sentiment": None, "section": "main"},
                {"text": "Q: Can you provide more color on your AI infrastructure investments?", "business_line": "Q&A Section", "sentiment": None, "section": "qa"},
                {"text": "A: We continue to expand our data center footprint to support AI workloads.", "business_line": "Q&A Section", "sentiment": None, "section": "qa"},
                {"text": "Surface revenue decreased 8% due to the challenging device market.", "business_line": "More Personal Computing", "sentiment": None, "section": "main"},
                {"text": "Office 365 Commercial revenue grew 15% driven by seat growth and ARPU increase.", "business_line": "Productivity and Business Processes", "sentiment": None, "section": "main"},
            ]
            df = pd.DataFrame(sample_sentences)
            df['processed'] = False
            df['paragraph_id'] = range(len(df))
            df['paragraph'] = df['text'].apply(lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x))
            return df
    except Exception as e:
        logger.error(f"Error loading pre-categorized sentences: {e}")
        st.error(f"Error loading pre-categorized sentences: {e}")
        return pd.DataFrame()

# Extract segment data from business lines
def extract_segment_data(sentences_df):
    """Extract segment info from business lines"""
    if sentences_df.empty or 'business_line' not in sentences_df.columns:
        return {
            "SegmentMap": {},
            "BusinessLines": [],
            "BusinessToSegment": {}
        }
    
    # Get unique business lines
    business_lines = sentences_df['business_line'].unique().tolist()
    
    # Microsoft's standard business segments
    segment_map = {
        "Productivity and Business Processes": ["Office Commercial", "Office Consumer", "LinkedIn", "Dynamics"],
        "Intelligent Cloud": ["Server products and cloud services", "Enterprise Services", "Azure"],
        "More Personal Computing": ["Windows", "Devices", "Gaming", "Search and news advertising"]
    }
    
    # Create business to segment mapping
    business_to_segment = {}
    for segment, businesses in segment_map.items():
        for business in businesses:
            business_to_segment[business] = segment
    
    # Map any business lines not in the standard mapping to "Other"
    for business in business_lines:
        if business not in business_to_segment and business not in ["Q&A Section", "Other"]:
            # Handle the case where the business line might be the segment itself
            if business in segment_map:
                business_to_segment[business] = business
            else:
                business_to_segment[business] = "Other"
    
    # Add segment column to DataFrame if not present
    if 'segment' not in sentences_df.columns:
        sentences_df['segment'] = sentences_df['business_line'].map(
            lambda x: business_to_segment.get(x, "Other")
        )
    
    # Make sure Q&A and Other are included
    if "Q&A Section" not in business_lines:
        business_lines.append("Q&A Section")
    
    if "Other" not in business_lines:
        business_lines.append("Other")
    
    return {
        "SegmentMap": segment_map,
        "BusinessLines": business_lines,
        "BusinessToSegment": business_to_segment
    }

# Helper function to check DataFrame
def safe_check_dataframe(df):
    """Safely check if a DataFrame exists and has data"""
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty