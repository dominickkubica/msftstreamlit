# chatbot.py
import json
import pandas as pd
import streamlit as st
import openai
from datetime import datetime, timedelta
import os.path
import re

def load_stock_data():
    """Load stock data from the CSV file with proper date handling"""
    file_path = "HistoricalData_1747025804532.csv"
    
    if not os.path.exists(file_path):
        st.error(f"Stock data file not found at: {file_path}")
        return None
    
    try:
        stock_data = pd.read_csv(file_path)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        # Convert price columns to numeric, removing '$' if present
        price_columns = ['Close/Last', 'Open', 'High', 'Low']
        for column in price_columns:
            if column in stock_data.columns:
                stock_data[column] = stock_data[column].str.replace('$', '').astype(float)
        
        return stock_data.sort_values('Date')
    
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return None

def find_closest_trading_date(stock_data, target_date, date_col='Date'):
    """Find the closest trading date to the target date in the stock data"""
    if stock_data is None or stock_data.empty:
        return None
    
    if not pd.api.types.is_datetime64_any_dtype(stock_data[date_col]):
        stock_data[date_col] = pd.to_datetime(stock_data[date_col])
    
    if not isinstance(target_date, pd.Timestamp):
        target_date = pd.to_datetime(target_date)
    
    all_dates = stock_data[date_col].dt.date.unique()
    
    if target_date.date() in all_dates:
        return target_date
    
    closest_date = None
    min_diff = float('inf')
    
    for date in all_dates:
        diff = abs((date - target_date.date()).days)
        if diff < min_diff:
            min_diff = diff
            closest_date = date
    
    if closest_date:
        return pd.Timestamp(closest_date)
    
    return None

def analyze_chat_query(query, sentiment_data=None, stock_data=None, date_col='Date', price_col='Close/Last'):
    """Process a chat query about financial data and return a response"""
    try:
        if not st.session_state.get('openai_api_key'):
            return "Please provide an OpenAI API key in the configuration section."
        
        openai.api_key = st.session_state.openai_api_key
        
        if stock_data is None or stock_data.empty:
            stock_data = load_stock_data()
            if stock_data is None or stock_data.empty:
                return "Unable to load stock data. Please check the file path."
        
        # Parse dates from the query with multiple formats
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[-]\d{1,2}[-]\d{4}\b'
        ]
        
        all_dates_mentioned = []
        for pattern in date_patterns:
            dates = re.findall(pattern, query, re.IGNORECASE)
            all_dates_mentioned.extend(dates)
        
        # Special handling for formats like "1-25-2025"
        special_format = re.findall(r'\b(\d{1,2})[-](\d{1,2})[-](\d{4})\b', query)
        for match in special_format:
            date_str1 = f"{match[0]}/{match[1]}/{match[2]}"
            date_str2 = f"{match[1]}/{match[0]}/{match[2]}"
            if date_str1 not in all_dates_mentioned:
                all_dates_mentioned.append(date_str1)
            if date_str2 not in all_dates_mentioned:
                all_dates_mentioned.append(date_str2)
        
        # Convert mentioned dates to datetime objects
        date_objects = []
        date_warnings = []
        
        if all_dates_mentioned and stock_data is not None and not stock_data.empty:
            min_date = stock_data[date_col].min().date()
            max_date = stock_data[date_col].max().date()
            
            for date_str in all_dates_mentioned:
                try:
                    parsed_date = pd.to_datetime(date_str)
                    date_objects.append(parsed_date.date())
                    
                    # Check if the date is in our dataset
                    date_in_data = False
                    for idx, row in stock_data.iterrows():
                        if row[date_col].date() == parsed_date.date():
                            date_in_data = True
                            break
                    
                    if not date_in_data:
                        if parsed_date.weekday() >= 5:  # Weekend
                            closest_date = find_closest_trading_date(stock_data, parsed_date)
                            if closest_date:
                                date_warnings.append(f"The date {date_str} falls on a weekend. The closest trading day is {closest_date.strftime('%Y-%m-%d')}.")
                            else:
                                date_warnings.append(f"The date {date_str} falls on a weekend and I couldn't find a close trading day.")
                        elif parsed_date.date() < min_date or parsed_date.date() > max_date:
                            date_warnings.append(f"The date {date_str} is outside of the available data range ({min_date} to {max_date}).")
                        else:
                            # Holiday or non-trading day
                            closest_date = find_closest_trading_date(stock_data, parsed_date)
                            if closest_date:
                                date_warnings.append(f"The date {date_str} appears to be a holiday or non-trading day. The closest trading day is {closest_date.strftime('%Y-%m-%d')}.")
                            else:
                                date_warnings.append(f"The date {date_str} doesn't have stock data available.")
                except Exception as e:
                    pass
        
        # Prepare context about the available data
        context = []
        
        # Add date warnings
        if date_warnings:
            context.append("DATE RANGE WARNINGS:\n" + "\n".join(date_warnings))
        
        if stock_data is not None and not stock_data.empty:
            # Data boundaries
            data_boundaries = f"STOCK DATA BOUNDARIES:\nStock data is available from {stock_data[date_col].min().strftime('%Y-%m-%d')} to {stock_data[date_col].max().strftime('%Y-%m-%d')}.\nTotal trading days: {len(stock_data)}"
            context.append(data_boundaries)
            
            # Stock data summary
            stock_summary = f"Stock data available from {stock_data[date_col].min().strftime('%Y-%m-%d')} to {stock_data[date_col].max().strftime('%Y-%m-%d')}.\n"
            
            recent_stock = stock_data.sort_values(by=date_col, ascending=False).head(30)
            oldest_price = stock_data.sort_values(by=date_col).iloc[0][price_col]
            newest_price = stock_data.sort_values(by=date_col, ascending=False).iloc[0][price_col]
            price_change = ((newest_price - oldest_price) / oldest_price) * 100
            
            stock_summary += f"Current price: ${newest_price:.2f}. "
            stock_summary += f"Price change over available period: {price_change:.2f}%.\n"
            
            # 30-day performance
            if len(recent_stock) >= 20:
                thirty_day_change = ((newest_price - recent_stock.iloc[19][price_col]) / recent_stock.iloc[19][price_col]) * 100
                stock_summary += f"30-day performance: {thirty_day_change:.2f}%.\n"
            
            # Specific date information
            date_specific_info = ""
            for date_obj in date_objects:
                date_data = stock_data[stock_data[date_col].dt.date == date_obj]
                if not date_data.empty:
                    row = date_data.iloc[0]
                    date_specific_info += f"\nStock data for {date_obj.strftime('%Y-%m-%d')}:\n"
                    date_specific_info += f"Open: ${row['Open']:.2f}, High: ${row['High']:.2f}, Low: ${row['Low']:.2f}, Close: ${row[price_col]:.2f}, Volume: {row['Volume']:,}\n"
            
            if date_specific_info:
                stock_summary += date_specific_info
                
            # Check for January 29, 2025 earnings date
            jan_29_2025 = pd.to_datetime('2025-01-29')
            earnings_date_data = stock_data[stock_data[date_col].dt.date == jan_29_2025.date()]
            
            if not earnings_date_data.empty:
                earnings_idx = earnings_date_data.index[0]
                next_days_indices = [i for i in range(earnings_idx, min(earnings_idx + 5, len(stock_data)))]
                next_days = stock_data.loc[next_days_indices]
                
                if len(next_days) >= 2:
                    earnings_price = next_days.iloc[0][price_col]
                    next_day_price = next_days.iloc[1][price_col]
                    price_change_pct = ((next_day_price - earnings_price) / earnings_price) * 100
                    stock_summary += f"\nEarnings announcement on 2025-01-29: ${earnings_price:.2f}.\n"
                    stock_summary += f"Next trading day: ${next_day_price:.2f} ({price_change_pct:.2f}% change).\n"
            
            # Recent stock data for context
            stock_data_sample = stock_data.sort_values(by=date_col, ascending=False).head(100).to_dict(orient='records')
            
            for record in stock_data_sample:
                for key, value in record.items():
                    if isinstance(value, pd.Timestamp):
                        record[key] = value.strftime('%Y-%m-%d')
            
            stock_data_str = json.dumps(stock_data_sample, default=str)
            
            context.append(f"STOCK DATA SUMMARY:\n{stock_summary}")
            context.append(f"RECENT STOCK DATA (most recent 100 days):\n{stock_data_str}")
        
        if sentiment_data is not None and not sentiment_data.empty:
            total_sentences = len(sentiment_data)
            context.append(f"SENTIMENT DATA SIZE: {total_sentences} total sentences from the earnings call.")
            
            sentiment_summary = f"Sentiment analysis results for Microsoft earnings call on January 29, 2025. Total sentences: {total_sentences}.\n"
            
            has_sentence_data = 'text' in sentiment_data.columns and 'sentiment' in sentiment_data.columns
            
            # Find negative sentences
            if has_sentence_data:
                negative_data = sentiment_data[sentiment_data['sentiment'] == 'negative']
                
                if not negative_data.empty:
                    negative_count = len(negative_data)
                    sentiment_summary += f"Found {negative_count} sentences with negative sentiment.\n"
                else:
                    sentiment_summary += "No sentences with negative sentiment found.\n"
            
            # Search for key terms
            key_terms = ["black ops", "call of duty", "gaming", "azure", "ai"]
            key_term_results = {}
            
            if has_sentence_data:
                for term in key_terms:
                    matches = sentiment_data[sentiment_data['text'].str.contains(term, case=False, na=False)]
                    if not matches.empty:
                        key_term_results[term] = []
                        for _, row in matches.iterrows():
                            key_term_results[term].append({
                                'text': row['text'],
                                'sentiment': row.get('sentiment', 'Unknown'),
                                'business_line': row.get('business_line', 'Unknown')
                            })
            
            # Segment-level sentiment
            if 'business_line' in sentiment_data.columns and 'sentiment' in sentiment_data.columns:
                segment_sentiment = sentiment_data.groupby('business_line')['sentiment'].value_counts().unstack().fillna(0)
                segment_sentiment['total'] = segment_sentiment.sum(axis=1)
                
                if 'positive' in segment_sentiment.columns and 'negative' in segment_sentiment.columns:
                    segment_sentiment['net_sentiment'] = (segment_sentiment['positive'] - segment_sentiment['negative']) / segment_sentiment['total']
                else:
                    segment_sentiment['net_sentiment'] = 0
                
                segments_sorted = segment_sentiment.sort_values('net_sentiment', ascending=False)
                
                sentiment_summary += "\nSegment sentiment (most positive to least):\n"
                for idx, row in segments_sorted.iterrows():
                    if 'net_sentiment' in row:
                        sentiment_label = "positive" if row['net_sentiment'] > 0.2 else "neutral" if row['net_sentiment'] > -0.2 else "negative"
                        sentiment_summary += f"- {idx}: {sentiment_label} (score: {row['net_sentiment']:.2f})\n"
            
            # Overall sentiment
            overall_sentiment_counts = sentiment_data['sentiment'].value_counts()
            
            positive_count = overall_sentiment_counts.get('positive', 0)
            neutral_count = overall_sentiment_counts.get('neutral', 0)
            negative_count = overall_sentiment_counts.get('negative', 0)
            
            positive_pct = positive_count / total_sentences * 100 if total_sentences > 0 else 0
            neutral_pct = neutral_count / total_sentences * 100 if total_sentences > 0 else 0
            negative_pct = negative_count / total_sentences * 100 if total_sentences > 0 else 0
            
            net_sentiment = (positive_count - negative_count) / total_sentences if total_sentences > 0 else 0
            sentiment_label = "positive" if net_sentiment > 0.2 else "neutral" if net_sentiment > -0.2 else "negative"
            
            sentiment_summary += f"\nOverall earnings call sentiment: {sentiment_label} (net score: {net_sentiment:.2f})\n"
            sentiment_summary += f"Positive: {positive_count} ({positive_pct:.1f}%)\n"
            sentiment_summary += f"Neutral: {neutral_count} ({neutral_pct:.1f}%)\n"
            sentiment_summary += f"Negative: {negative_count} ({negative_pct:.1f}%)\n"
            
            # Add key term results to context
            if key_term_results:
                key_terms_str = "KEY TERM SEARCH RESULTS:\n"
                for term, results in key_term_results.items():
                    key_terms_str += f"\nTerm '{term}' found in {len(results)} sentences:\n"
                    for i, result in enumerate(results[:5], 1):
                        key_terms_str += f"{i}. \"{result['text']}\" (sentiment: {result['sentiment']}, business line: {result['business_line']})\n"
                context.append(key_terms_str)
            
            # Sample sentiment data
            sentiment_data_sample = sentiment_data.head(200)
            sentiment_data_json = sentiment_data_sample.to_dict(orient='records')
            
            context.append(f"SENTIMENT DATA SUMMARY:\n{sentiment_summary}")
            context.append(f"DETAILED SENTIMENT DATA (sample of {len(sentiment_data_sample)} entries):\n{json.dumps(sentiment_data_json, default=str)}")
        
        # Compile context
        full_context = "\n\n".join(context)
        
        # Simple system message - let AI format naturally
        system_message = """You are a financial analysis assistant for Microsoft's January 29, 2025 earnings call. 
        Answer questions about Microsoft's stock performance and earnings call sentiment analysis using the provided data.
        Be accurate and only use the data provided. Format your response clearly with bullet points for key information."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Data available:\n\n{full_context}\n\nQuestion: {query}"}
        ]
        
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

def init_chat_interface(sentiment_data=None, stock_data=None, date_col='Date', price_col='Close/Last'):
    """Initialize and render the chat interface"""
    from data_loader import safe_check_dataframe
    
    st.header("Financial Analysis Chatbot")
    
    if stock_data is None or stock_data.empty:
        stock_data = load_stock_data()
    
    has_stock = safe_check_dataframe(stock_data)
    has_sentiment = safe_check_dataframe(sentiment_data)
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if has_sentiment or has_stock:
        st.markdown("""
        Ask questions about the earnings call sentiment analysis and stock data. Examples:
        - What is the overall sentiment in the earnings call?
        - How did the stock perform after the last earnings?
        - Which business segment had the most positive sentiment?
        - What was the stock price movement around earnings dates?
        """)
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    return has_sentiment, has_stock

def process_chat_input(chat_prompt, sentiment_data=None, stock_data=None, date_col='Date', price_col='Close/Last'):
    """Process a chat input and update the chat history"""
    if not chat_prompt:
        return
    
    st.session_state.messages.append({"role": "user", "content": chat_prompt})
    
    if not st.session_state.get('openai_api_key'):
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Please enter your OpenAI API key in the configuration section to use the chatbot."
        })
        return
    
    if stock_data is None or stock_data.empty:
        stock_data = load_stock_data()
    
    with st.spinner("Thinking..."):
        response = analyze_chat_query(chat_prompt, sentiment_data, stock_data, date_col, price_col)
        
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            error_msg = "I'm having trouble answering that. Please try again or rephrase your question."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.rerun()
