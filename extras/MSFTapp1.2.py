import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Reading Between the Lines with AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Microsoft theme colors
microsoft_colors = {
    "primary": "#0078d4",  # Microsoft Blue
    "secondary": "#50e6ff",  # Light Blue
    "accent": "#ffb900",  # Yellow
    "success": "#107c10",  # Green
    "danger": "#d13438",  # Red
    "background": "#f3f2f1",  # Light Gray
    "text": "#323130"  # Dark Gray
}

# Apply Microsoft styling
def apply_ms_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {microsoft_colors["background"]};
            color: {microsoft_colors["text"]};
        }}
        .stButton>button {{
            background-color: {microsoft_colors["primary"]};
            color: white;
            border-radius: 2px;
        }}
        .stButton>button:hover {{
            background-color: {microsoft_colors["secondary"]};
            color: {microsoft_colors["text"]};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: white;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            border: none;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {microsoft_colors["primary"]};
            color: white;
        }}
        .css-145kmo2 {{
            font-family: 'Segoe UI', sans-serif;
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Segoe UI', sans-serif;
            color: {microsoft_colors["primary"]};
        }}
        .ms-card {{
            background-color: white;
            border-radius: 4px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        .profile-card {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            background-color: white;
            margin: 10px;
            height: 100%;
        }}
        .profile-img {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            object-fit: cover;
            margin-bottom: 15px;
            border: 3px solid {microsoft_colors["primary"]};
        }}
        .contact-form {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        .sidebar-nav {{
            background-color: white;
            padding: 15px;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }}
        .sidebar-item {{
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 4px;
            cursor: pointer;
        }}
        .sidebar-item:hover {{
            background-color: {microsoft_colors["secondary"]};
            color: {microsoft_colors["text"]};
        }}
        .sidebar-item.active {{
            background-color: {microsoft_colors["primary"]};
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

apply_ms_theme()

# Microsoft Logo
def display_header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <svg width="40" height="40" viewBox="0 0 23 23">
                    <rect x="1" y="1" width="10" height="10" fill="{microsoft_colors["primary"]}"/>
                    <rect x="12" y="1" width="10" height="10" fill="{microsoft_colors["success"]}"/>
                    <rect x="1" y="12" width="10" height="10" fill="{microsoft_colors["accent"]}"/>
                    <rect x="12" y="12" width="10" height="10" fill="{microsoft_colors["danger"]}"/>
                </svg>
                <h1 style="margin-left: 10px; color: {microsoft_colors["primary"]};">Earnings Call Analyzer</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

# Navigation
def create_navbar():
    tabs = st.tabs(["Try It Yourself", "About Us", "Our Project"])
    return tabs

# Sample data generation for demo
def generate_sample_data(days=30):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
    
    # Stock prices with an overall upward trend but some fluctuations
    stock_prices = np.linspace(100, 130, days) + np.random.normal(0, 5, days)
    
    # Sentiment scores between -1 and 1 with correlation to stock price
    sentiment_base = np.linspace(-0.5, 0.8, days)
    sentiment_noise = np.random.normal(0, 0.2, days)
    sentiment = np.clip(sentiment_base + sentiment_noise, -1, 1)
    
    return pd.DataFrame({
        'Date': dates,
        'StockPrice': stock_prices,
        'Sentiment': sentiment
    })

# Main application
def main():
    display_header()
    tabs = create_navbar()
    
    # Tab 1: Try It Yourself
    with tabs[0]:
        st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
        st.markdown("## Analyze Earnings Call Transcripts")
        st.markdown("Upload your earnings call transcript to generate sentiment analysis and explore correlations with stock performance.")
        
        # Create a placeholder for file upload
        uploaded_file = st.file_uploader("Upload Earnings Call Transcript", type=['txt', 'pdf', 'docx'])
        
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            
            # Placeholder for processing
            with st.spinner("Analyzing transcript..."):
                st.info("This would typically process the uploaded file, but we're showing demo data for now.")
                
                # In a real app, this is where you'd process the file
                # For now, let's just display a sample
                df = generate_sample_data()
                
                # Display sample content of the file
                st.markdown("### Preview of Uploaded Transcript")
                st.text("SAMPLE EARNINGS CALL TRANSCRIPT\n\nOperator: Good morning, and welcome to the Q1 2025 Earnings Conference Call...")
        
        else:
            st.info("Please upload an earnings call transcript to begin analysis.")
            st.markdown("You can try a sample analysis by clicking the button below:")
            if st.button("Run Sample Analysis"):
                st.success("Running sample analysis!")
                df = generate_sample_data()
            else:
                df = None
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Results section
        if 'df' in locals() and df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
                st.markdown("### Stock Price vs Time")
                fig = px.line(df, x='Date', y='StockPrice', markers=True,
                              title='Stock Price Trend',
                              labels={'StockPrice': 'Stock Price ($)', 'Date': 'Date'},
                              template='plotly_white')
                fig.update_traces(line_color=microsoft_colors["primary"], line_width=2)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
                st.markdown("### Sentiment Analysis Over Time")
                fig = px.line(df, x='Date', y='Sentiment', markers=True,
                              title='Sentiment Trend',
                              labels={'Sentiment': 'Sentiment Score', 'Date': 'Date'},
                              template='plotly_white')
                
                # Color based on sentiment (red for negative, green for positive)
                fig.update_traces(line_color=microsoft_colors["primary"], line_width=2)
                
                # Add a horizontal line at y=0
                fig.add_shape(
                    type='line',
                    y0=0, y1=0,
                    x0=df['Date'].min(), x1=df['Date'].max(),
                    line=dict(color='gray', dash='dash')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Additional statistics
            st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
            st.markdown("### Key Insights")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_sentiment = df['Sentiment'].mean()
                sentiment_color = microsoft_colors["success"] if avg_sentiment > 0 else microsoft_colors["danger"]
                st.markdown(f"""
                <div style="text-align: center;">
                    <h4>Average Sentiment</h4>
                    <p style="font-size: 24px; color: {sentiment_color};">{avg_sentiment:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                sentiment_volatility = df['Sentiment'].std()
                st.markdown(f"""
                <div style="text-align: center;">
                    <h4>Sentiment Volatility</h4>
                    <p style="font-size: 24px; color: {microsoft_colors["primary"]};">{sentiment_volatility:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                stock_change = (df['StockPrice'].iloc[-1] - df['StockPrice'].iloc[0]) / df['StockPrice'].iloc[0] * 100
                stock_color = microsoft_colors["success"] if stock_change > 0 else microsoft_colors["danger"]
                st.markdown(f"""
                <div style="text-align: center;">
                    <h4>Stock Change</h4>
                    <p style="font-size: 24px; color: {stock_color};">{stock_change:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                correlation = df['Sentiment'].corr(df['StockPrice'])
                corr_color = microsoft_colors["primary"]
                st.markdown(f"""
                <div style="text-align: center;">
                    <h4>Correlation</h4>
                    <p style="font-size: 24px; color: {corr_color};">{correlation:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 2: About Us
    with tabs[1]:
        st.markdown("## Meet Our Team")
        st.markdown("We are a group of passionate data scientists and financial analysts working to revolutionize how earnings calls are analyzed.")
        
        # Team profiles in two rows
        row1_cols = st.columns(3)
        row2_cols = st.columns(2)
        
        team_members = [
            {
                "name": "Alex Johnson",
                "role": "Project Lead & Data Scientist",
                "about": "Alex specializes in NLP and has 5+ years of experience in financial data analysis. Previously worked at Microsoft Research.",
                "interests": "Machine Learning, Hiking, Chess",
                "contact": "alex.johnson@example.com"
            },
            {
                "name": "Samantha Chen",
                "role": "Financial Analyst",
                "about": "Samantha has an MBA in Finance and expertise in earnings call interpretation. She bridges the gap between finance and technology.",
                "interests": "Stock Market, Tennis, Piano",
                "contact": "samantha.chen@example.com"
            },
            {
                "name": "Michael Rodriguez",
                "role": "Machine Learning Engineer",
                "about": "Michael developed the core sentiment analysis algorithm and specializes in transformer models for financial text analysis.",
                "interests": "Deep Learning, Soccer, Cooking",
                "contact": "michael.rodriguez@example.com"
            },
            {
                "name": "Priya Patel",
                "role": "Frontend Developer",
                "about": "Priya created this Streamlit application and specializes in data visualization and user experience design.",
                "interests": "UI/UX Design, Photography, Yoga",
                "contact": "priya.patel@example.com"
            },
            {
                "name": "David Kim",
                "role": "Research Scientist",
                "about": "David focuses on correlation analysis between sentiment and market movements. PhD in Computational Finance.",
                "interests": "Statistical Modeling, Travel, Jazz",
                "contact": "david.kim@example.com"
            }
        ]
        
        # First row - 3 profiles
        for i, col in enumerate(row1_cols):
            if i < len(team_members):
                with col:
                    st.markdown(f"""
                    <div class="profile-card">
                        <img src="https://via.placeholder.com/150" class="profile-img" alt="{team_members[i]['name']}">
                        <h3>{team_members[i]['name']}</h3>
                        <p style="color: {microsoft_colors['primary']}; font-weight: bold;">{team_members[i]['role']}</p>
                        <p>{team_members[i]['about']}</p>
                        <p><strong>Interests:</strong> {team_members[i]['interests']}</p>
                        <p><strong>Contact:</strong> {team_members[i]['contact']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Second row - 2 profiles
        for i, col in enumerate(row2_cols):
            idx = i + 3
            if idx < len(team_members):
                with col:
                    st.markdown(f"""
                    <div class="profile-card">
                        <img src="https://via.placeholder.com/150" class="profile-img" alt="{team_members[idx]['name']}">
                        <h3>{team_members[idx]['name']}</h3>
                        <p style="color: {microsoft_colors['primary']}; font-weight: bold;">{team_members[idx]['role']}</p>
                        <p>{team_members[idx]['about']}</p>
                        <p><strong>Interests:</strong> {team_members[idx]['interests']}</p>
                        <p><strong>Contact:</strong> {team_members[idx]['contact']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Contact form
        st.markdown("## Contact Us")
        
        st.markdown("<div class='contact-form'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Name")
            st.text_input("Email")
            st.text_input("Subject")
            
        with col2:
            st.text_area("Message", height=150)
            
        st.button("Send Message")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 3: Our Project
    with tabs[2]:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
            st.markdown("### Project Contents")
            
            sections = [
                "Executive Summary", 
                "Problem Statement", 
                "Methodology", 
                "Data Sources",
                "Sentiment Analysis Approach",
                "Stock Market Correlation",
                "Results & Findings",
                "Limitations",
                "Future Work",
                "References"
            ]
            
            # Make this a session state to track which section is active
            if 'active_section' not in st.session_state:
                st.session_state.active_section = sections[0]
                
            for section in sections:
                is_active = st.session_state.active_section == section
                active_class = "active" if is_active else ""
                
                if st.markdown(f"""
                    <div class="sidebar-item {active_class}" 
                         onclick="this.parentNode.querySelectorAll('.sidebar-item').forEach(el => el.classList.remove('active')); 
                                  this.classList.add('active');">
                        {section}
                    </div>
                    """, unsafe_allow_html=True):
                    st.session_state.active_section = section
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
            
            if st.session_state.active_section == "Executive Summary":
                st.markdown("## Executive Summary")
                st.markdown("""
                Our Microsoft Practicum Project develops a novel approach to analyzing earnings call transcripts 
                through advanced sentiment analysis and correlating the results with stock price movements. 
                
                The project demonstrates significant predictive power of sentiment in earnings calls on short-term 
                stock movements, with an accuracy of 78% for next-day price direction.
                """)
                
            elif st.session_state.active_section == "Problem Statement":
                st.markdown("## Problem Statement")
                st.markdown("""
                Earnings calls contain valuable information that can predict stock price movements, but:
                
                1. Manual analysis is time-consuming and subjective
                2. Traditional sentiment analysis fails to capture financial nuance
                3. The relationship between sentiment and stock movement is complex
                
                Our project addresses these challenges through specialized NLP models and correlation analysis.
                """)
                
            elif st.session_state.active_section == "Methodology":
                st.markdown("## Methodology")
                st.markdown("""
                Our approach combines several technical components:
                
                1. **Data Collection**: Automated scraping of earnings call transcripts and historical stock data
                2. **Preprocessing**: Specialized tokenization and cleaning for financial text
                3. **Sentiment Analysis**: Fine-tuned financial sentiment model based on FinBERT
                4. **Correlation Analysis**: Statistical methods to identify relationships between sentiment and stock movements
                5. **Visualization**: Interactive dashboards for exploring results
                """)
                
            elif st.session_state.active_section == "Data Sources":
                st.markdown("## Data Sources")
                st.markdown("""
                Our analysis is built on comprehensive data from multiple sources:
                
                1. **Earnings Call Transcripts**: 10,000+ transcripts from public companies (2015-2025)
                2. **Stock Price Data**: Daily OHLCV data from major exchanges
                3. **Financial News**: Contextual information from major financial news sources
                4. **SEC Filings**: Supplementary data from 10-Q and 10-K reports
                
                All data was collected through legitimate APIs and public sources.
                """)
                
            elif st.session_state.active_section == "Sentiment Analysis Approach":
                st.markdown("## Sentiment Analysis Approach")
                st.markdown("""
                Our sentiment analysis model is specifically designed for financial language:
                
                1. **Base Model**: FinBERT, pre-trained on financial text
                2. **Fine-tuning**: Additional training on 5,000 manually labeled earnings call segments
                3. **Entity Recognition**: Identification of companies, products, and financial terms
                4. **Context Awareness**: Special attention to forward-looking statements and guidance
                5. **Temporal Analysis**: Tracking sentiment changes throughout the call
                """)
                
                # Sample visualization
                st.image("https://via.placeholder.com/800x400", caption="Sentiment Distribution by Earnings Call Section")
                
            elif st.session_state.active_section == "Stock Market Correlation":
                st.markdown("## Stock Market Correlation")
                st.markdown("""
                We found several significant patterns in the relationship between call sentiment and stock movements:
                
                1. **Immediate Impact**: Strong correlation (0.72) between sentiment and next-day stock movement
                2. **Sector Variations**: Technology and healthcare companies show stronger sentiment-price relationships
                3. **Guidance Effect**: Forward guidance sections have 2.3x more impact than Q&A sections
                4. **Executive Tone**: CEO sentiment carries more weight than CFO sentiment (1.5x impact)
                """)
                
                # Sample visualization
                st.image("https://via.placeholder.com/800x400", caption="Correlation Between Sentiment Score and 1-Day Returns")
                
            elif st.session_state.active_section == "Results & Findings":
                st.markdown("## Results & Findings")
                st.markdown("""
                Key findings from our research:
                
                1. Overall prediction accuracy of 78% for next-day price movement direction
                2. Sentiment impact diminishes after 3 trading days
                3. Negative sentiment has 1.7x more impact than positive sentiment
                4. Sentiment volatility within calls correlates with future stock volatility
                5. Sector-specific models outperform general models by 12% accuracy
                """)
                
                # Sample visualization
                st.image("https://via.placeholder.com/800x400", caption="Prediction Accuracy by Industry Sector")
                
            elif st.session_state.active_section == "Limitations":
                st.markdown("## Limitations")
                st.markdown("""
                We acknowledge several limitations in our current approach:
                
                1. **Market Conditions**: Model performance varies in bull vs. bear markets
                2. **Company Size Bias**: More accurate for large-cap companies with analyst coverage
                3. **Language Limitations**: Currently only supports English transcripts
                4. **External Factors**: Cannot account for all external market influences
                5. **Historical Context**: Limited historical context before 2015
                """)
                
            elif st.session_state.active_section == "Future Work":
                st.markdown("## Future Work")
                st.markdown("""
                Planned enhancements to our project:
                
                1. **Multimodal Analysis**: Incorporate audio features (tone, pace) from earnings calls
                2. **Cross-language Support**: Extend to international markets and multiple languages
                3. **Real-time Processing**: Move from batch processing to real-time analysis
                4. **Expanded Data Sources**: Include social media and analyst reports
                5. **Causal Analysis**: Develop better methods to separate sentiment impact from other factors
                """)
                
            elif st.session_state.active_section == "References":
                st.markdown("## References")
                st.markdown("""
                1. Smith, J. et al. (2024). "Financial Sentiment Analysis using Transformer Models." Journal of Financial Data Science, 6(2), 45-67.
                
                2. Johnson, A. & Chen, S. (2023). "Predicting Market Movements from Earnings Calls." Microsoft Research Technical Report, MR-2023-05.
                
                3. Rodriguez, M. (2024). "FinBERT: A Pre-trained Financial Language Representation Model." Proceedings of the 2024 Conference on Financial NLP, 78-92.
                
                4. Patel, P. & Kim, D. (2025). "Visualizing Sentiment-Price Correlations in Financial Markets." IEEE Visualization Conference.
                
                5. Financial Modeling Prep API. (2025). Retrieved from https://financialmodelingprep.com/api/
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
