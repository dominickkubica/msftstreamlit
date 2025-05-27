# chatbot.py
import json
import pandas as pd
import streamlit as st
import logging
import openai
from datetime import datetime, timedelta
import os.path
import re
from pathlib import Path

# Setup logger
logger = logging.getLogger("earnings_analyzer")

def load_stock_data():
    """
    Load stock data from the CSV file with proper date handling
    """
    file_path = "HistoricalData_1747025804532.csv"
    
    if not os.path.exists(file_path):
        st.error(f"Stock data file not found at: {file_path}")
        return None
    
    try:
        # Load the CSV with proper date parsing
        stock_data = pd.read_csv(file_path)
        
        # Convert date column to datetime with proper format
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        # Convert price columns to numeric, removing '$' if present
        price_columns = ['Close/Last', 'Open', 'High', 'Low']
        for column in price_columns:
            if column in stock_data.columns:
                stock_data[column] = stock_data[column].str.replace('$', '').astype(float)
        
        # Sort by date
        stock_data = stock_data.sort_values('Date')
        
        return stock_data
    
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return None

def fix_response_formatting(text):
    """
    Apply targeted formatting fixes to model responses
    to ensure consistent and readable text output
    """
    # First, protect "stock" from being split
    text = text.replace('sto ck', 'stock')
    
    # Fix number+word combinations (no space between)
    text = re.sub(r'(\d+\.?\d*)([A-Za-z]+)', r'\1 \2', text)
    
    # Fix word+number combinations (no space between)
    text = re.sub(r'([A-Za-z]+)(\d+\.?\d*)', r'\1 \2', text)
    
    # Fix specific financial terms
    text = re.sub(r'(\d+\.?\d*)\s*billion', r'\1 billion', text)
    text = re.sub(r'(\d+\.?\d*)\s*million', r'\1 million', text)
    text = re.sub(r'(\d+\.?\d*)\s*trillion', r'\1 trillion', text)
    text = re.sub(r'(\d+\.?\d*)\s*percent', r'\1 percent', text)
    
    # Fix dollar sign spacing
    text = re.sub(r'\$\s*(\d)', r'$\1', text)
    
    # Fix percentage spacing
    text = re.sub(r'(\d+\.?\d*)\s*%', r'\1%', text)
    
    # Fix common problematic patterns
    text = text.replace('andincreased', 'and increased')
    text = text.replace('increasedto', 'increased to')
    text = text.replace('dayto', 'day to')
    text = text.replace('fromto', 'from to')
    text = text.replace('announcementdayto', 'announcement day to')
    text = text.replace('contributededed', 'contributed')
    text = text.replace('priceof', 'price of')
    
    # Ensure space before numbers that follow text
    text = re.sub(r'([A-Za-z])(\$\d)', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)
    
    # Fix sentence spacing (but only after punctuation)
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    text = re.sub(r',([A-Za-z])', r', \1', text)
    text = re.sub(r':([A-Za-z])', r': \1', text)
    text = re.sub(r';([A-Za-z])', r'; \1', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s{2,}', r' ', text)
    
    # Remove any markdown formatting
    text = text.replace("**", "").replace("*", "").replace("__", "").replace("_", "")
    text = text.replace("###", "").replace("##", "").replace("#", "")
    
    # Final pass to ensure "stock" stays intact
    text = text.replace('sto ck', 'stock')
    
    return text.strip()

def find_closest_trading_date(stock_data, target_date, date_col='Date'):
    """
    Find the closest trading date to the target date in the stock data
    
    Parameters:
    - stock_data: DataFrame containing stock data
    - target_date: The date to find the closest trading day for
    - date_col: The name of the date column in the stock data
    
    Returns:
    - The closest trading date or None if not found
    """
    if stock_data is None or stock_data.empty:
        return None
    
    # Convert all dates to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(stock_data[date_col]):
        stock_data[date_col] = pd.to_datetime(stock_data[date_col])
    
    # Convert target_date to datetime if it's not already
    if not isinstance(target_date, pd.Timestamp):
        target_date = pd.to_datetime(target_date)
    
    # Get all dates from the stock data
    all_dates = stock_data[date_col].dt.date.unique()
    
    # If target date is in the data, return it
    if target_date.date() in all_dates:
        return target_date
    
    # Otherwise, find the closest date
    closest_date = None
    min_diff = float('inf')
    
    for date in all_dates:
        diff = abs((date - target_date.date()).days)
        if diff < min_diff:
            min_diff = diff
            closest_date = date
    
    if closest_date:
        # Convert back to pandas Timestamp for consistency
        return pd.Timestamp(closest_date)
    
    return None

def analyze_chat_query(query, sentiment_data=None, stock_data=None, date_col='Date', price_col='Close/Last'):
    """
    Process a chat query about financial data and return a response.
    
    Parameters:
    - query: The user's question
    - sentiment_data: DataFrame containing sentiment analysis results
    - stock_data: DataFrame containing historical stock data
    - date_col: Name of the date column in stock_data
    - price_col: Name of the price column in stock_data
    
    Returns:
    - A string response to the user's query
    """
    try:
        # Check if API key is available
        if not st.session_state.get('openai_api_key'):
            return "Please provide an OpenAI API key in the configuration section."
        
        # Set OpenAI API key
        openai.api_key = st.session_state.openai_api_key
        
        # If stock_data not provided, try to load it
        if stock_data is None or stock_data.empty:
            stock_data = load_stock_data()
            if stock_data is None or stock_data.empty:
                return "Unable to load stock data. Please check the file path."
        
        # Parse dates from the query with multiple formats
        import re
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # MM/DD/YYYY, M/D/YYYY
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',    # YYYY/MM/DD, YYYY/M/D
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',  # Month D(st/nd/rd/th), YYYY
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Month D, YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',  # Mon D(st/nd/rd/th), YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b',  # Mon D, YYYY
            r'\b\d{1,2}[-]\d{1,2}[-]\d{4}\b'  # D-M-YYYY or M-D-YYYY
        ]
        
        all_dates_mentioned = []
        for pattern in date_patterns:
            dates = re.findall(pattern, query, re.IGNORECASE)
            all_dates_mentioned.extend(dates)
        
        # Special handling for formats like "1-25-2025"
        special_format = re.findall(r'\b(\d{1,2})[-](\d{1,2})[-](\d{4})\b', query)
        for match in special_format:
            # Try both MM-DD-YYYY and DD-MM-YYYY interpretations
            date_str1 = f"{match[0]}/{match[1]}/{match[2]}"  # MM/DD/YYYY
            date_str2 = f"{match[1]}/{match[0]}/{match[2]}"  # DD/MM/YYYY
            if date_str1 not in all_dates_mentioned:
                all_dates_mentioned.append(date_str1)
            if date_str2 not in all_dates_mentioned:
                all_dates_mentioned.append(date_str2)
        
        # Try to convert mentioned dates to actual datetime objects
        date_objects = []
        date_warnings = []
        
        if all_dates_mentioned and stock_data is not None and not stock_data.empty:
            min_date = stock_data[date_col].min().date()
            max_date = stock_data[date_col].max().date()
            
            for date_str in all_dates_mentioned:
                try:
                    # Convert various date formats to a standard format
                    parsed_date = pd.to_datetime(date_str)
                    date_objects.append(parsed_date.date())
                    
                    # Check if the date is in our dataset
                    date_in_data = False
                    for idx, row in stock_data.iterrows():
                        if row[date_col].date() == parsed_date.date():
                            date_in_data = True
                            break
                    
                    if not date_in_data:
                        # Check if it's a weekend or holiday
                        if parsed_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                            closest_date = find_closest_trading_date(stock_data, parsed_date)
                            if closest_date:
                                date_warnings.append(f"The date {date_str} falls on a weekend. The closest trading day is {closest_date.strftime('%Y-%m-%d')}.")
                            else:
                                date_warnings.append(f"The date {date_str} falls on a weekend and I couldn't find a close trading day.")
                        elif parsed_date.date() < min_date or parsed_date.date() > max_date:
                            date_warnings.append(f"The date {date_str} is outside of the available data range ({min_date} to {max_date}).")
                        else:
                            # Must be a holiday or non-trading day
                            closest_date = find_closest_trading_date(stock_data, parsed_date)
                            if closest_date:
                                date_warnings.append(f"The date {date_str} appears to be a holiday or non-trading day. The closest trading day is {closest_date.strftime('%Y-%m-%d')}.")
                            else:
                                date_warnings.append(f"The date {date_str} doesn't have stock data available.")
                except Exception as e:
                    # Skip unparseable dates
                    pass
        
        # Prepare context about the available data
        context = []
        
        # Add any date warnings to the beginning of the context
        if date_warnings:
            context.append("DATE RANGE WARNINGS:\n" + "\n".join(date_warnings))
        
        if stock_data is not None and not stock_data.empty:
            # Add clear data boundaries and verification data
            data_boundaries = f"IMPORTANT DATA BOUNDARIES:\nStock data is available ONLY from {stock_data[date_col].min().strftime('%Y-%m-%d')} to {stock_data[date_col].max().strftime('%Y-%m-%d')}.\nTotal trading days in dataset: {len(stock_data)}\nDO NOT provide specific stock prices for dates outside this range."
            context.append(data_boundaries)
            
            # Stock data summary
            stock_summary = f"Stock data available from {stock_data[date_col].min().strftime('%Y-%m-%d')} to {stock_data[date_col].max().strftime('%Y-%m-%d')}.\n"
            
            # Calculate some basic statistics on the stock data
            recent_stock = stock_data.sort_values(by=date_col, ascending=False).head(30)
            oldest_price = stock_data.sort_values(by=date_col).iloc[0][price_col]
            newest_price = stock_data.sort_values(by=date_col, ascending=False).iloc[0][price_col]
            price_change = ((newest_price - oldest_price) / oldest_price) * 100
            
            stock_summary += f"Current price: ${newest_price:.2f}. "
            stock_summary += f"Price change over available period: {price_change:.2f}%.\n"
            
            # Calculate 30-day performance
            if len(recent_stock) >= 20:
                thirty_day_change = ((newest_price - recent_stock.iloc[19][price_col]) / recent_stock.iloc[19][price_col]) * 100
                stock_summary += f"30-day performance: {thirty_day_change:.2f}%.\n"
            
            # Check if any of the dates in the query are in the data
            # If so, provide specific information for those dates
            date_specific_info = ""
            for date_obj in date_objects:
                date_data = stock_data[stock_data[date_col].dt.date == date_obj]
                if not date_data.empty:
                    row = date_data.iloc[0]
                    date_specific_info += f"\nStock data for {date_obj.strftime('%Y-%m-%d')}:\n"
                    date_specific_info += f"Open: ${row['Open']:.2f}, High: ${row['High']:.2f}, Low: ${row['Low']:.2f}, Close: ${row[price_col]:.2f}, Volume: {row['Volume']:,}\n"
            
            if date_specific_info:
                stock_summary += date_specific_info
                
            # Find January 29, 2025 in the data if it exists
            if date_col in stock_data.columns:
                # Look for earnings date data
                jan_29_2025 = pd.to_datetime('2025-01-29')
                earnings_date_data = stock_data[stock_data[date_col].dt.date == jan_29_2025.date()]
                
                if not earnings_date_data.empty:
                    # Get the next few trading days
                    earnings_idx = earnings_date_data.index[0]
                    next_days_indices = [i for i in range(earnings_idx, min(earnings_idx + 5, len(stock_data)))]
                    next_days = stock_data.loc[next_days_indices]
                    
                    if len(next_days) >= 2:
                        earnings_price = next_days.iloc[0][price_col]
                        next_day_price = next_days.iloc[1][price_col]
                        price_change_pct = ((next_day_price - earnings_price) / earnings_price) * 100
                        stock_summary += f"\nEarnings announcement on 2025-01-29: ${earnings_price:.2f}.\n"
                        stock_summary += f"Next trading day: ${next_day_price:.2f} ({price_change_pct:.2f}% change).\n"
            
            # Convert stock data to a string representation for the prompt
            # Include more days to ensure coverage
            stock_data_sample = stock_data.sort_values(by=date_col, ascending=False).head(100).to_dict(orient='records')
            
            # Format the stock data properly to avoid JSON formatting issues
            for record in stock_data_sample:
                for key, value in record.items():
                    if isinstance(value, pd.Timestamp):
                        record[key] = value.strftime('%Y-%m-%d')
            
            stock_data_str = json.dumps(stock_data_sample, default=str)
            
            context.append(f"STOCK DATA SUMMARY:\n{stock_summary}")
            context.append(f"RECENT STOCK DATA (most recent 100 days):\n{stock_data_str}")
        
        if sentiment_data is not None and not sentiment_data.empty:
            # Add sentiment data size information
            total_sentences = len(sentiment_data)
            context.append(f"SENTIMENT DATA SIZE: The sentiment dataset contains {total_sentences} total sentences from the earnings call.")
            
            # Sentiment data summary
            sentiment_summary = f"Sentiment analysis results available for the Microsoft earnings call on January 29, 2025. Total sentences analyzed: {total_sentences}.\n"
            
            # First, check if we have sentence-level sentiment data
            has_sentence_data = 'text' in sentiment_data.columns and 'sentiment' in sentiment_data.columns
            
            # Identify negative sentences if they exist
            negative_sentences = []
            if has_sentence_data:
                # Find sentences with negative sentiment
                negative_data = sentiment_data[sentiment_data['sentiment'] == 'negative']
                
                if not negative_data.empty:
                    negative_count = len(negative_data)
                    sentiment_summary += f"Found {negative_count} sentences with negative sentiment.\n"
                    
                    # Add the top 10 most negative sentences to the list
                    top_negative = negative_data.head(10)
                    for _, row in top_negative.iterrows():
                        negative_sentences.append({
                            'text': row['text'],
                            'sentiment': row['sentiment'],
                            'segment': row.get('segment', 'Unknown')
                        })
                else:
                    sentiment_summary += "No sentences with negative sentiment were found in the transcript.\n"
            
            # Search for specific key terms in the transcript
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
            
            # Segment-level sentiment if available
            if 'business_line' in sentiment_data.columns and 'sentiment' in sentiment_data.columns:
                # Calculate segment-level sentiment
                segment_sentiment = sentiment_data.groupby('business_line')['sentiment'].value_counts().unstack().fillna(0)
                
                # Calculate net sentiment
                segment_sentiment['total'] = segment_sentiment.sum(axis=1)
                if 'positive' in segment_sentiment.columns and 'negative' in segment_sentiment.columns:
                    segment_sentiment['net_sentiment'] = (segment_sentiment['positive'] - segment_sentiment['negative']) / segment_sentiment['total']
                else:
                    segment_sentiment['net_sentiment'] = 0
                
                segments_sorted = segment_sentiment.sort_values('net_sentiment', ascending=False)
                
                sentiment_summary += "\nSegment sentiment summary (from most positive to least positive):\n"
                for idx, row in segments_sorted.iterrows():
                    if 'net_sentiment' in row:
                        sentiment_label = "positive" if row['net_sentiment'] > 0.2 else "neutral" if row['net_sentiment'] > -0.2 else "negative"
                        sentiment_summary += f"- {idx}: {sentiment_label} (score: {row['net_sentiment']:.2f})\n"
                
                # Find most negative segment
                if len(segments_sorted) > 0 and 'net_sentiment' in segments_sorted.columns:
                    least_positive_segment = segments_sorted.index[-1]
                    least_positive_score = segments_sorted['net_sentiment'].iloc[-1]
                    sentiment_summary += f"\nSegment with least positive sentiment: {least_positive_segment} (score: {least_positive_score:.2f})\n"
                    
                    # Get sentences from the least positive segment
                    if has_sentence_data:
                        segment_sentences = sentiment_data[sentiment_data['business_line'] == least_positive_segment]
                        negative_segment_sentences = segment_sentences[segment_sentences['sentiment'] == 'negative']
                        segment_sample = negative_segment_sentences.head(5) if not negative_segment_sentences.empty else segment_sentences.head(5)
                        
                        sentiment_summary += f"\nSample sentences from {least_positive_segment} segment:\n"
                        for _, row in segment_sample.iterrows():
                            sentiment_summary += f"- \"{row['text']}\" (sentiment: {row['sentiment']})\n"
            
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
            
            # Add negative sentences to context
            if negative_sentences:
                neg_sentences_str = "TOP NEGATIVE SENTENCES FROM TRANSCRIPT:\n"
                for i, sent in enumerate(negative_sentences, 1):
                    neg_sentences_str += f"{i}. \"{sent['text']}\" (sentiment: {sent['sentiment']}, segment: {sent['segment']})\n"
                context.append(neg_sentences_str)
            
            # Add key term search results to context
            if key_term_results:
                key_terms_str = "KEY TERM SEARCH RESULTS:\n"
                for term, results in key_term_results.items():
                    key_terms_str += f"\nTerm '{term}' found in {len(results)} sentences:\n"
                    for i, result in enumerate(results[:5], 1):  # Limit to 5 examples per term
                        key_terms_str += f"{i}. \"{result['text']}\" (sentiment: {result['sentiment']}, business line: {result['business_line']})\n"
                context.append(key_terms_str)
            
            # Add sample sentiment data to context (limit to ensure accurate total count)
            sentiment_data_sample = sentiment_data.head(200)
            sentiment_data_json = "["
            for i, row in enumerate(sentiment_data_sample.to_dict(orient='records')):
                # Format each record properly to avoid JSON formatting issues
                formatted_row = {}
                for key, value in row.items():
                    if isinstance(value, pd.Timestamp):
                        formatted_row[key] = value.strftime('%Y-%m-%d')
                    else:
                        formatted_row[key] = value
                
                sentiment_data_json += json.dumps(formatted_row, default=str)
                if i < len(sentiment_data_sample) - 1:
                    sentiment_data_json += ",\n"
            sentiment_data_json += "]"
            
            context.append(f"SENTIMENT DATA SUMMARY:\n{sentiment_summary}")
            context.append(f"DETAILED SENTIMENT DATA (sample of {len(sentiment_data_sample)} entries):\n{sentiment_data_json}")
        
        # Compile the full context
        full_context = "\n\n".join(context)
        
        # Create the prompt for the model with improved instructions
        system_message = """You are a financial analysis assistant specifically focused on Microsoft's January 29, 2025 earnings call. 
        Use the provided data to answer questions about Microsoft's stock performance and earnings call sentiment analysis.
        
        ABSOLUTELY CRITICAL FORMATTING RULES:
        
        1. WORD SPACING - FOLLOW THESE RULES EXACTLY:
           - Put a space between EVERY word
           - Put a space after ALL punctuation marks
           - Put a space between numbers and words (e.g., "69.6 billion" not "69.6billion")
           - Put a space before numbers that follow words (e.g., "day 1" not "day1")
           - Keep common words intact (e.g., "stock" not "sto ck")
           
        2. CORRECT EXAMPLES:
           ✓ "The stock price was $442.33 on January 29, 2025"
           ✓ "Revenue was $69.6 billion, up 12% year over year"
           ✓ "From $442.33 to $447.20"
           ✓ "On the earnings announcement day to $447.20"
           
        3. INCORRECT EXAMPLES TO AVOID:
           ✗ "The stockprice was$442.33onJanuary29,2025"
           ✗ "Revenue was$69.6billion,up12%yearoveryear"
           ✗ "From$442.33to$447.20"
           ✗ "442.33ontheearningsannouncementdayto447.20"
        
        4. NUMBER FORMATTING RULES:
           - Currency: Always "$X.XX" with no space after $
           - With units: "69.6 billion" (space before "billion/million/trillion")
           - Percentages: "12%" (no space before %)
           - Dates: "January 29, 2025" (spaces between all parts)
        
        5. TEXT STRUCTURE:
           - Use short paragraphs (2-3 sentences)
           - Leave blank lines between paragraphs
           - Use numbered lists for multiple points:
             1. First point
             2. Second point
           - Start with a summary, end with a conclusion
        
        6. DATA ACCURACY:
           - Only use explicitly provided data
           - Never make up numbers
           - Say "I don't have that data" if information is missing
        
        7. NEVER USE:
           - Markdown formatting (no *, **, #)
           - Run-on text without spaces
           - Made-up data or numbers
        
        CRITICAL: Before responding, mentally check that there is proper spacing between ALL words and numbers. This is essential for readability."""
        
        # Create the messages for the chat model
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Here is the data available for analysis:\n\n{full_context}\n\nUser question: {query}"}
        ]
        
        # Get response from OpenAI
        response = openai.chat.completions.create(
            model="gpt-4-turbo",  # or use the appropriate model
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Extract and get the response
        result = response.choices[0].message.content
        
        # Apply our comprehensive formatting fixes
        result = fix_response_formatting(result)
        
        return result
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Error processing query: {str(e)}\n{error_details}")
        return f"I encountered an error while processing your question. Please try again or check the error message: {str(e)}"

def init_chat_interface(sentiment_data=None, stock_data=None, date_col='Date', price_col='Close/Last'):
    """Initialize and render the chat interface"""
    from data_loader import safe_check_dataframe
    
    st.header("Financial Analysis Chatbot")
    
    # If stock_data not provided, try to load it
    if stock_data is None or stock_data.empty:
        stock_data = load_stock_data()
    
    # Check if data is available
    has_stock = safe_check_dataframe(stock_data)
    has_sentiment = safe_check_dataframe(sentiment_data)
    
    # Initialize chat history if it doesn't exist
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
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    return has_sentiment, has_stock

def process_chat_input(chat_prompt, sentiment_data=None, stock_data=None, date_col='Date', price_col='Close/Last'):
    """Process a chat input and update the chat history"""
    if not chat_prompt:
        return
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": chat_prompt})
    
    # Check if OpenAI API key is available
    if not st.session_state.get('openai_api_key'):
        # Add assistant response about missing API key
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Please enter your OpenAI API key in the configuration section to use the chatbot."
        })
        return
    
    # Try to load stock data if not provided
    if stock_data is None or stock_data.empty:
        stock_data = load_stock_data()
    
    # Get message response from OpenAI
    with st.spinner("Thinking..."):
        response = analyze_chat_query(
            chat_prompt, 
            sentiment_data,
            stock_data,
            date_col,
            price_col
        )
        
        if response:
            # Apply our comprehensive formatting fixes once more to ensure clean output
            response = fix_response_formatting(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            error_msg = "I'm having trouble answering that. Please try again or rephrase your question."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Rerun to display the updated chat
    st.rerun()
