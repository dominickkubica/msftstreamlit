# chatbot.py
import json
import pandas as pd
import streamlit as st
import openai
import os

def load_stock_data():
    """Load stock data from CSV"""
    file_path = "HistoricalData_1747025804532.csv"
    if not os.path.exists(file_path):
        return None
    
    stock_data = pd.read_csv(file_path)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
    price_columns = ['Close/Last', 'Open', 'High', 'Low']
    for column in price_columns:
        if column in stock_data.columns:
            stock_data[column] = stock_data[column].str.replace('$', '').astype(float)
    
    return stock_data.sort_values('Date')

def analyze_chat_query(query, sentiment_data=None, stock_data=None):
    """Process chat query and return response"""
    if not st.session_state.get('openai_api_key'):
        return "Please provide an OpenAI API key."
    
    openai.api_key = st.session_state.openai_api_key
    
    if stock_data is None:
        stock_data = load_stock_data()
    
    # Prepare context
    context = []
    
    if stock_data is not None and not stock_data.empty:
        recent_data = stock_data.tail(30).to_dict('records')
        context.append(f"Stock data: {json.dumps(recent_data, default=str)}")
    
    if sentiment_data is not None and not sentiment_data.empty:
        sample_data = sentiment_data.head(100).to_dict('records')
        context.append(f"Sentiment data: {json.dumps(sample_data, default=str)}")
    
    messages = [
        {"role": "system", "content": "You are a financial assistant. Answer questions about Microsoft's earnings and stock data. Use bullet points for key information."},
        {"role": "user", "content": f"Context: {' '.join(context)}\n\nQuestion: {query}"}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def init_chat_interface(sentiment_data=None, stock_data=None, date_col='Date', price_col='Close/Last'):
    """Initialize chat interface"""
    st.header("Financial Analysis Chatbot")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    return True, True

def process_chat_input(chat_prompt, sentiment_data=None, stock_data=None, date_col='Date', price_col='Close/Last'):
    """Process chat input"""
    if not chat_prompt:
        return
    
    st.session_state.messages.append({"role": "user", "content": chat_prompt})
    
    if not st.session_state.get('openai_api_key'):
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Please enter your OpenAI API key."
        })
        return
    
    response = analyze_chat_query(chat_prompt, sentiment_data, stock_data)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
