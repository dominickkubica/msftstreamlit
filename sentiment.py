# sentiment.py
import json
import re
import pandas as pd
import streamlit as st
import logging
import openai
from tenacity import retry, wait_exponential, stop_after_attempt

# Setup logger
logger = logging.getLogger("earnings_analyzer")

# OpenAI API call with retries
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_openai_api(system_prompt, user_prompt, model="gpt-4-turbo", temperature=0.2):
    """Call OpenAI API with retry logic for robustness"""
    api_key = st.session_state.get('openai_api_key')
    if not api_key:
        st.error("OpenAI API key not found")
        return None
    
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        st.error(f"OpenAI API error: {e}")
        return None

# Analyze sentiment for the pre-categorized sentences
def analyze_sentiment_for_sentences(sentences_df, progress_callback=None):
    """Analyze sentiment for pre-categorized sentences with batch processing"""
    if sentences_df.empty:
        return sentences_df
    
    # Create a copy to avoid warning about setting values on a slice
    sentences_df = sentences_df.copy()
    
    # Process in batches for efficiency
    batch_size = 10  # Smaller batch size to ensure proper analysis
    total_batches = (len(sentences_df) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        # Calculate batch start and end
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sentences_df))
        batch = sentences_df.iloc[start_idx:end_idx].copy()  # Create a copy to avoid warnings
        
        # Skip already processed sentences
        if 'processed' in batch.columns:
            batch_to_process = batch[~batch['processed']].copy()
            if batch_to_process.empty:
                if progress_callback:
                    progress_value = (batch_idx + 1) / total_batches
                    progress_callback(progress_value)
                continue
        else:
            batch_to_process = batch.copy()
        
        # Create batch text with context
        batch_texts = []
        for i, (idx, row) in enumerate(batch_to_process.iterrows()):
            batch_texts.append(f"Sentence {i+1}: {row['text']}")
        
        batch_text = "\n".join(batch_texts)
        
        # Create system prompt
        system_prompt = """
        You are an expert financial analyst specialized in Microsoft earnings call analysis.
        
        Determine the sentiment of each numbered sentence from an earnings call transcript.
        
        For sentiment, classify as:
        - "positive" (growth, success, exceeding expectations, improvements)
        - "neutral" (factual statements, on-target performance, stable metrics)
        - "negative" (declines, challenges, missed targets, concerns)
        
        Return ONLY a JSON array in this exact format:
        [
          {
            "sentence_num": 1,
            "sentiment": "positive/neutral/negative"
          },
          ... (entries for all sentences)
        ]
        """
        
        user_prompt = f"Analyze the sentiment of these earnings call statements:\n\n{batch_text}"
        
        # Call API
        response = call_openai_api(system_prompt, user_prompt)
        
        # Update progress
        if progress_callback:
            progress_value = (batch_idx + 1) / total_batches
            progress_callback(progress_value)
        
        if response:
            # Parse the response
            try:
                results = json.loads(response)
                
                # Update DataFrame with classifications
                for i, result in enumerate(results):
                    sentence_num = result.get('sentence_num')
                    if 1 <= sentence_num <= len(batch_to_process):
                        batch_idx_to_update = batch_to_process.index[sentence_num - 1]
                        sentences_df.at[batch_idx_to_update, 'sentiment'] = result.get('sentiment')
                        sentences_df.at[batch_idx_to_update, 'processed'] = True
                
                logger.info(f"Processed batch {batch_idx+1}/{total_batches} with {len(results)} classifications")
            
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response for batch {batch_idx+1}")
                # Try a simpler parsing approach if JSON fails
                try:
                    # Look for patterns like "1. Sentiment: positive"
                    lines = response.split('\n')
                    for i, line in enumerate(lines):
                        match = re.search(r'(\d+).*?sentiment:?\s*(\w+)', line.lower())
                        if match:
                            sentence_num = int(match.group(1))
                            sentiment = match.group(2).strip()
                            
                            if 1 <= sentence_num <= len(batch_to_process):
                                batch_idx_to_update = batch_to_process.index[sentence_num - 1]
                                
                                if sentiment in ['positive', 'neutral', 'negative']:
                                    sentences_df.at[batch_idx_to_update, 'sentiment'] = sentiment
                                else:
                                    sentences_df.at[batch_idx_to_update, 'sentiment'] = 'neutral'
                                
                                sentences_df.at[batch_idx_to_update, 'processed'] = True
                except Exception as e:
                    logger.error(f"Error in fallback parsing: {e}")
        else:
            logger.warning(f"No response for batch {batch_idx+1}")
            # Set default classifications for this batch
            for idx in batch_to_process.index:
                sentences_df.at[idx, 'sentiment'] = 'neutral'
                sentences_df.at[idx, 'processed'] = True
    
    # Fill in any remaining unprocessed sentences
    if 'processed' in sentences_df.columns and not sentences_df['processed'].all():
        unprocessed_mask = ~sentences_df['processed']
        sentences_df.loc[unprocessed_mask, 'sentiment'] = 'neutral'
        sentences_df.loc[unprocessed_mask, 'processed'] = True
    
    return sentences_df

# Aggregate sentiment by business line
def aggregate_sentiment_by_business_line(sentences_df):
    """
    Aggregate sentiment data by business line for visualization
    """
    if sentences_df.empty or 'business_line' not in sentences_df.columns:
        return pd.DataFrame()
    
    # Create a summary DataFrame
    aggregated = sentences_df.groupby('business_line')['sentiment'].value_counts().unstack().fillna(0)
    
    # Calculate additional metrics
    aggregated['total'] = aggregated.sum(axis=1)
    if 'positive' in aggregated.columns and 'negative' in aggregated.columns:
        aggregated['net_sentiment'] = (aggregated['positive'] - aggregated['negative']) / aggregated['total']
    else:
        aggregated['net_sentiment'] = 0
    
    return aggregated

# Create sentiment timeline data for stock correlation
def create_sentiment_timeline(sentences_df, stock_data, date_col):
    """
    Create timeline data for sentiment correlation with stock price
    """
    if sentences_df.empty or stock_data is None or stock_data.empty:
        return pd.DataFrame()
    
    # Get dates from stock data
    dates = stock_data[date_col].dt.date.sort_values().unique()
    if len(dates) < 2:
        return pd.DataFrame()
    
    # Calculate sentiment scores by business line
    bl_sentiment = aggregate_sentiment_by_business_line(sentences_df)
    
    # Distribute sentiment scores across timeline
    timeline_data = []
    
    # Get 3-5 key dates spread across the stock data
    num_points = min(5, len(dates))
    date_indices = [int(i * (len(dates) - 1) / (num_points - 1)) for i in range(num_points)]
    key_dates = [dates[i] for i in date_indices]
    
    # Get business lines sorted by net sentiment
    business_lines = []
    if not bl_sentiment.empty:
        business_lines = bl_sentiment.sort_values('net_sentiment', ascending=False).index.tolist()
    
    # Assign each business line to different dates for visualization
    for i, business_line in enumerate(business_lines):
        if i >= len(key_dates):
            break
            
        date = key_dates[i]
        sentiment_value = bl_sentiment.loc[business_line, 'net_sentiment'] if business_line in bl_sentiment.index else 0
        
        # Map net sentiment to category
        if sentiment_value > 0.2:
            sentiment_category = 'positive'
        elif sentiment_value < -0.2:
            sentiment_category = 'negative'
        else:
            sentiment_category = 'neutral'
            
        timeline_data.append({
            'date': date,
            'business_line': business_line,
            'sentiment': sentiment_category,
            'sentiment_score': sentiment_value
        })
    
    return pd.DataFrame(timeline_data)

# Process workflow for analysis
def process_analysis_workflow(ticker_symbol, start_date, end_date, stock_data_path, sentiment_data_path):
    """Process the complete workflow for analysis"""
    from data_loader import load_stock_data, generate_sample_stock_data, load_precategorized_sentences, extract_segment_data
    
    try:
        with st.spinner("Running analysis..."):
            # Progress tracking
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0.0)
            
            # Step 1: Load stock data
            status_placeholder.text("Loading stock data...")
            
            # Try to load from hardcoded path
            stock_data_result = load_stock_data(stock_data_path)
            
            # If loading failed, generate sample data
            if not stock_data_result:
                status_placeholder.text("Generating sample stock data...")
                stock_data_result = generate_sample_stock_data(start_date, end_date)
            
            # Update session state
            if stock_data_result:
                stock_data, date_col, price_col, ohlc_cols = stock_data_result
                st.session_state.stock_data = stock_data
                st.session_state.date_col = date_col
                st.session_state.price_col = price_col
                st.session_state.ohlc_cols = ohlc_cols
            else:
                st.error("Failed to load or generate stock data")
                return False
                
            progress_bar.progress(0.3)
            
            # Step 2: Load pre-categorized sentences
            status_placeholder.text("Loading pre-categorized sentences...")
            sentences_df = load_precategorized_sentences(sentiment_data_path)
            progress_bar.progress(0.4)
            
            # Step 3: Extract segment info from the pre-categorized data
            status_placeholder.text("Extracting business structure...")
            segment_data = extract_segment_data(sentences_df)
            
            # Store in session state
            st.session_state.segment_data = segment_data
            st.session_state.business_lines = segment_data["BusinessLines"]
            progress_bar.progress(0.5)
            
            # Step 4: Analyze sentiment for each sentence
            status_placeholder.text("Analyzing sentiment...")
            
            # Function to update progress from within analyze_sentiment
            def update_progress(value):
                # Scale the progress from 0.5 to 0.8
                scaled_progress = 0.5 + (value * 0.3)
                progress_bar.progress(scaled_progress)
                status_placeholder.text(f"Analyzing sentiment: {int(value * 100)}% complete")
            
            # Run the sentiment analysis with progress updates
            classified_df = analyze_sentiment_for_sentences(sentences_df, update_progress)
            
            # Store complete results
            st.session_state.sentiment_results = classified_df
            
            # Check for Q&A section
            q_a_mask = classified_df['section'] == 'qa' if 'section' in classified_df.columns else pd.Series([False] * len(classified_df))
            st.session_state.has_qa_section = q_a_mask.any()
            
            # Extract Q&A results if applicable
            if st.session_state.has_qa_section:
                st.session_state.qa_results = classified_df[q_a_mask]
            
            # Step 5: Create timeline data for visualization
            status_placeholder.text("Creating visualization data...")
            sentiment_timeline = create_sentiment_timeline(classified_df, stock_data, date_col)
            st.session_state.sentiment_timeline = sentiment_timeline
            progress_bar.progress(1.0)
            
            # Clear progress indicators
            status_placeholder.empty()
            progress_placeholder.empty()
            
            return True
    except Exception as e:
        logger.error(f"Error in analysis workflow: {e}")
        st.error(f"An error occurred during analysis: {e}")
        return False