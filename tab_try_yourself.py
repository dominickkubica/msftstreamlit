# tab_try_yourself.py
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import logging

# Import our custom modules
from data_loader import (
    load_stock_data, 
    generate_sample_stock_data, 
    load_precategorized_sentences, 
    safe_check_dataframe,
    MSFT_EARNINGS_DATES
)
from sentiment import (
    process_analysis_workflow, 
    aggregate_sentiment_by_business_line, 
    create_sentiment_timeline
)
from plotting import (
    colored_metric, 
    plot_sentiment_distribution, 
    plot_sentiment_by_business_line, 
    plot_sentiment_and_stock,
    create_earnings_impact_analysis,
    create_stock_chart
)
from chatbot import (
    analyze_chat_query, 
    init_chat_interface, 
    process_chat_input
)

# Setup logger
logger = logging.getLogger("earnings_analyzer")

# Hardcoded file paths
STOCK_DATA_PATH = "HistoricalData_1747025804532.csv"
SENTIMENT_DATA_PATH = "MSFTQ2_preload.csv.csv.csv"

# Microsoft colors for UI elements
MICROSOFT_COLORS = {
    'primary': '#0078d4',    # Microsoft blue
    'secondary': '#5c2d91',  # Microsoft purple
    'success': '#107c10',    # Microsoft green
    'danger': '#d13438',     # Microsoft red
    'warning': '#ffb900',    # Microsoft yellow
    'info': '#00b7c3',       # Microsoft teal
    'accent': '#ffb900',     # Microsoft yellow
}

def _init_state():
    """Initialize session state variables if they don't exist"""
    # Initialize chat messages in session state if not already present
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize other session state variables if needed
    for key, default in {
        'sentiment_results': None, 
        'business_lines': [],
        'stock_data': None,
        'date_col': None,
        'price_col': None,
        'ohlc_cols': None,
        'ticker_symbol': "MSFT", 
        'openai_api_key': "",
        'sentiment_timeline': None,
        'segment_data': {},
        'has_qa_section': False,
        'qa_results': None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

def render_dashboard_tab():
    """Render the dashboard tab"""
    if safe_check_dataframe(st.session_state.get('sentiment_results')) and safe_check_dataframe(st.session_state.get('stock_data')):
        st.subheader("Dashboard Overview")
        
        # Create a 2x2 dashboard layout
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            # Stock price summary card
            stock_data = st.session_state.stock_data
            price_col = st.session_state.price_col
            
            if safe_check_dataframe(stock_data) and price_col:
                st.markdown("### Stock Performance")
                
                # Calculate key metrics
                start_price = stock_data[price_col].iloc[0]
                end_price = stock_data[price_col].iloc[-1]
                percent_change = ((end_price - start_price) / start_price * 100)
                
                # Display metrics
                metric_row1_1, metric_row1_2, metric_row1_3 = st.columns(3)
                
                with metric_row1_1:
                    st.metric("Starting Price", f"${start_price:.2f}")
                
                with metric_row1_2:
                    st.metric("Current Price", f"${end_price:.2f}", f"{percent_change:.2f}%")
                
                with metric_row1_3:
                    st.metric("Price Range", f"${stock_data[price_col].min():.2f} - ${stock_data[price_col].max():.2f}")
                
                # Simple stock chart
                st.line_chart(stock_data.set_index(st.session_state.date_col)[price_col])
        
        with metric_col2:
            # Sentiment summary card
            sentiment_data = st.session_state.sentiment_results
            
            if safe_check_dataframe(sentiment_data):
                st.markdown("### Sentiment Overview")
                
                # Calculate sentiment metrics
                sentiment_counts = sentiment_data['sentiment'].value_counts()
                total = len(sentiment_data)
                
                # Display metrics
                metric_row2_1, metric_row2_2, metric_row2_3 = st.columns(3)
                
                with metric_row2_1:
                    pos_count = sentiment_counts.get('positive', 0)
                    pos_pct = (pos_count/total*100) if total > 0 else 0
                    st.metric("Positive", f"{pos_count} ({pos_pct:.1f}%)")
                
                with metric_row2_2:
                    neu_count = sentiment_counts.get('neutral', 0)
                    neu_pct = (neu_count/total*100) if total > 0 else 0
                    st.metric("Neutral", f"{neu_count} ({neu_pct:.1f}%)")
                
                with metric_row2_3:
                    neg_count = sentiment_counts.get('negative', 0)
                    neg_pct = (neg_count/total*100) if total > 0 else 0
                    st.metric("Negative", f"{neg_count} ({neg_pct:.1f}%)")
                
                # Sentiment chart
                fig = plot_sentiment_distribution(sentiment_data, MICROSOFT_COLORS)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="sent_dist_dash")
        
        # Combined sentiment and stock chart

        
        # Business segments breakdown
        st.markdown("### Business Segment Analysis")
        
        if 'business_line' in st.session_state.sentiment_results.columns:
            segment_fig = plot_sentiment_by_business_line(st.session_state.sentiment_results, MICROSOFT_COLORS)
            if segment_fig:
                st.plotly_chart(segment_fig, use_container_width=True, key="segment_chart_dash")
    else:
        st.info("No analysis results yet. Please run the analysis to see the dashboard.")

def render_stock_tab():
    """Render the stock analysis tab"""
    if safe_check_dataframe(st.session_state.get('stock_data')):
        stock_data = st.session_state.stock_data
        date_col = st.session_state.date_col
        price_col = st.session_state.price_col
        ohlc_cols = st.session_state.ohlc_cols
        
        st.header("Stock Price Analysis")
        
        # Get min and max dates for the range selector
        min_date = stock_data[date_col].min().date()
        max_date = stock_data[date_col].max().date()
        
        # Date range selection
        st.subheader("Chart Settings")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            default_start = max(min_date, max_date - timedelta(days=365))
            start_date = st.date_input(
                "Start Date (Stock Chart)",
                value=default_start,
                min_value=min_date,
                max_value=max_date,
                key="stock_start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date (Stock Chart)",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="stock_end_date"
            )
        
        # Chart type selection
        with col3:
            chart_type = st.selectbox(
                "Chart Type",
                ["Line", "Candlestick", "OHLC"],
                index=0
            )
        
        # Show moving averages and earnings dates options
        col4, col5 = st.columns(2)
        
        with col4:
            show_ma = st.checkbox("Show Moving Averages", value=True)
        
        with col5:
            show_earnings = st.checkbox("Show Earnings Dates", value=True)
        
        # Make sure end_date is not before start_date
        if start_date > end_date:
            st.error("End date must be after start date")
            end_date = start_date
        
        # Filter data based on selected date range
        mask = (stock_data[date_col].dt.date >= start_date) & (stock_data[date_col].dt.date <= end_date)
        filtered_df = stock_data[mask].copy()
        
        # Show how many data points are in the filtered range
        st.write(f"Showing {len(filtered_df)} data points from {start_date} to {end_date}")
        
        # Earnings date selector - Move this before creating the chart
        earnings_dates = [pd.to_datetime(date_str).date() for date_str in MSFT_EARNINGS_DATES]
        range_earnings_dates = [date for date in earnings_dates if start_date <= date <= end_date]
        
        # Initialize selected_earnings_date to None
        selected_earnings_date = None
        
        if range_earnings_dates:
            date_options = [date.strftime("%Y-%m-%d") for date in range_earnings_dates]
            
            selected_earnings = st.selectbox(
                "Select Earnings Date to Analyze", 
                date_options
            )
            
            # Convert the selected earnings date
            if selected_earnings:
                selected_earnings_date = datetime.strptime(selected_earnings, "%Y-%m-%d").date()
        
        # Create the main chart based on selected type
        fig = create_stock_chart(
            filtered_df, date_col, price_col, ohlc_cols, 
            chart_type, show_ma, show_earnings, 
            MSFT_EARNINGS_DATES, start_date, end_date,
            selected_earnings_date  # Pass the selected earnings date
        )
        
        # Display the main chart
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="stock_price_chart")
        
        # Earnings impact analysis section
        st.subheader("Earnings Call Impact Analysis")
        
        # Now use the already selected earnings date for the analysis
        if range_earnings_dates:
            # Analyze selected earnings date
            if selected_earnings_date:
                create_earnings_impact_analysis(
                    filtered_df, date_col, price_col, ohlc_cols, 
                    selected_earnings_date, MICROSOFT_COLORS, MSFT_EARNINGS_DATES
                )
        else:
            st.info("No earnings dates found in the selected date range")
        
        # Summary statistics
        st.markdown("---")
        st.subheader("Summary Statistics")
        
        # Layout statistics in columns
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        # Calculate statistics
        start_price = filtered_df[price_col].iloc[0] if not filtered_df.empty else 0
        end_price = filtered_df[price_col].iloc[-1] if not filtered_df.empty else 0
        percent_change = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
        min_price = filtered_df[price_col].min() if not filtered_df.empty else 0
        max_price = filtered_df[price_col].max() if not filtered_df.empty else 0
        
        # Display statistics with custom HTML metrics
        with stat_col1:
            st.markdown(f"<div style='font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>Start Price</div>"
                       f"<div style='font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;'>${start_price:.2f}</div>", 
                       unsafe_allow_html=True)
        
        # Use custom colored_metric for End Price
        with stat_col2:
            colored_metric(
                "End Price", 
                f"${end_price:.2f}", 
                percent_change
            )
        
        with stat_col3:
            st.markdown(f"<div style='font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>Min Price</div>"
                       f"<div style='font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;'>${min_price:.2f}</div>", 
                       unsafe_allow_html=True)
        
        with stat_col4:
            st.markdown(f"<div style='font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>Max Price</div>"
                       f"<div style='font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;'>${max_price:.2f}</div>", 
                       unsafe_allow_html=True)
        
        # Option to show the data table
        if st.checkbox("Show Data Table", value=False):
            st.dataframe(filtered_df)
    else:
        st.info("No stock data available. Please run the analysis first.")

def render_sentiment_tab():
    """Render the sentiment analysis tab"""
    if safe_check_dataframe(st.session_state.get('sentiment_results')):
        sentiment_data = st.session_state.sentiment_results
        
        st.header("Earnings Call Sentiment Analysis")
        
        # Overall sentiment section
        st.subheader("Overall Sentiment Distribution")
        
        # Display summary metrics
        sentiment_counts = sentiment_data['sentiment'].value_counts()
        total = len(sentiment_data)
        
        sent_col1, sent_col2, sent_col3 = st.columns(3)
        with sent_col1:
            pos_count = sentiment_counts.get('positive', 0)
            pos_pct = (pos_count/total*100) if total > 0 else 0
            st.metric("Positive", f"{pos_count} ({pos_pct:.1f}%)")
        
        with sent_col2:
            neu_count = sentiment_counts.get('neutral', 0)
            neu_pct = (neu_count/total*100) if total > 0 else 0
            st.metric("Neutral", f"{neu_count} ({neu_pct:.1f}%)")
        
        with sent_col3:
            neg_count = sentiment_counts.get('negative', 0)
            neg_pct = (neg_count/total*100) if total > 0 else 0
            st.metric("Negative", f"{neg_count} ({neg_pct:.1f}%)")
        
        # Create sentiment distribution chart
        fig = plot_sentiment_distribution(sentiment_data, MICROSOFT_COLORS)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="sent_dist_overall")
        
        # If we have a Q&A section, show that analysis separately

        
        # Business segments section
        st.markdown("---")
        st.subheader("Sentiment by Business Segment")
        
        # Create business segment chart
        segment_fig = plot_sentiment_by_business_line(sentiment_data, MICROSOFT_COLORS)
        if segment_fig:
            st.plotly_chart(segment_fig, use_container_width=True, key="segment_chart_sentiment")
        
        # Detailed sentiment analysis section
        st.markdown("---")
        st.subheader("Detailed Sentiment Analysis")
        
        # Show business line selector
        if 'business_line' in sentiment_data.columns:
            # Get business lines with counts for better display
            bl_counts = sentiment_data['business_line'].value_counts()
            business_lines = []
            
            for bl in sorted(sentiment_data['business_line'].unique()):
                count = bl_counts.get(bl, 0)
                business_lines.append(f"{bl} ({count})")
            
            selected_bl_with_count = st.selectbox(
                "Select Business Line",
                ["All Business Lines"] + business_lines
            )
            
            # Extract actual business line name without count
            if selected_bl_with_count == "All Business Lines":
                selected_bl = "All Business Lines"
            else:
                selected_bl = selected_bl_with_count.split(" (")[0]
            
            # Filter by selected business line
            if selected_bl == "All Business Lines":
                filtered_results = sentiment_data.copy()
            else:
                filtered_results = sentiment_data[
                    sentiment_data['business_line'] == selected_bl
                ].copy()
            
            # Enhanced search options
            search_col1, search_col2 = st.columns([3, 1])
            
            with search_col1:
                search_term = st.text_input("Search in statements:")
            
            with search_col2:
                sentiment_filter = st.selectbox(
                    "Filter by sentiment",
                    ["All", "Positive", "Neutral", "Negative"]
                )
            
            # Apply search filter if provided
            if search_term:
                display_results = filtered_results[
                    filtered_results['text'].str.contains(search_term, case=False, na=False)
                ].copy()
            else:
                display_results = filtered_results.copy()
            
            # Apply sentiment filter if selected
            if sentiment_filter != "All":
                sentiment_value = sentiment_filter.lower()
                display_results = display_results[display_results['sentiment'] == sentiment_value].copy()
            
            # Show filtered results count with better styling
            st.markdown(
                f"""
                <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 10px;">
                Showing <b>{len(display_results)}</b> of <b>{len(filtered_results)}</b> statements from 
                <b>{selected_bl}</b> with sentiment <b>{sentiment_filter}</b>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display sentiment summary for selection with icons
            sentiment_counts = display_results['sentiment'].value_counts()
            total = len(display_results)
            
            if total > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    pos_count = sentiment_counts.get('positive', 0)
                    pos_pct = (pos_count/total*100) if total > 0 else 0
                    st.metric("✅ Positive", f"{pos_count} ({pos_pct:.1f}%)")
                
                with col2:
                    neu_count = sentiment_counts.get('neutral', 0)
                    neu_pct = (neu_count/total*100) if total > 0 else 0
                    st.metric("⚖️ Neutral", f"{neu_count} ({neu_pct:.1f}%)")
                
                with col3:
                    neg_count = sentiment_counts.get('negative', 0)
                    neg_pct = (neg_count/total*100) if total > 0 else 0
                    st.metric("⚠️ Negative", f"{neg_count} ({neg_pct:.1f}%)")
            
            # Display statements with improved pagination
            st.markdown("#### Analyzed Statements")
            
            if display_results.empty:
                st.info("No statements match your search criteria")
            else:
                # Display statements with pagination
                page_size = 10
                total_pages = (len(display_results) + page_size - 1) // page_size
                
                if total_pages > 0:
                    # Create columns for pagination controls
                    page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
                    
                    with page_col2:
                        if total_pages > 1:
                            # normal slider when there is more than one page
                            page_num = st.slider(
                                "Page",
                                1,
                                total_pages,
                                1,
                                key="detail_page_slider"
                            )
                        else:
                            # only one page → skip the slider
                            page_num = 1
                            st.markdown("Page 1&nbsp;of&nbsp;1")  # small note so the layout doesn't shift
                    
                    # Calculate which sentences to show
                    start_idx = (page_num - 1) * page_size
                    end_idx = min(start_idx + page_size, len(display_results))
                    
                    # Show current page info
                    st.markdown(f"Showing statements {start_idx+1}-{end_idx} of {len(display_results)}")
                    
                    # Display statements with improved styling
                    for idx in range(start_idx, end_idx):
                        if idx < len(display_results):
                            row = display_results.iloc[idx]
                            
                            # Get appropriate sentiment color and icon
                            sentiment_color = {
                                'positive': MICROSOFT_COLORS['success'],
                                'neutral': MICROSOFT_COLORS['accent'],
                                'negative': MICROSOFT_COLORS['danger']
                            }.get(row['sentiment'], MICROSOFT_COLORS['accent'])
                            
                            sentiment_icon = {
                                'positive': "✅",
                                'neutral': "⚖️",
                                'negative': "⚠️"
                            }.get(row['sentiment'], "")
                            
                            # Format the statement with improved styling
                            st.markdown(
                                f"""
                                <div style='padding: 15px; margin: 10px 0; border-left: 4px solid {sentiment_color}; 
                                background-color: rgba({int(sentiment_color[1:3], 16)}, {int(sentiment_color[3:5], 16)}, {int(sentiment_color[5:7], 16)}, 0.05);
                                border-radius: 0 5px 5px 0;'>
                                <div style='display: flex; justify-content: space-between;'>
                                    <span style='font-weight: bold;'>{row['business_line'] if 'business_line' in row else ''}</span>
                                    <span style='color: {sentiment_color};'>{sentiment_icon} {row['sentiment'].upper() if 'sentiment' in row and row['sentiment'] else 'N/A'}</span>
                                </div>
                                <div style='margin-top: 8px;'>{row['text'] if 'text' in row else ''}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No statements match your search criteria")
        else:
            st.warning("No business line information available in the sentiment data")
    else:
        st.info("No sentiment data available. Please run the analysis first.")

def render_chatbot_tab():
    """Render the chatbot tab"""
    stock_data = st.session_state.get('stock_data')
    sentiment_data = st.session_state.get('sentiment_results')
    date_col = st.session_state.get('date_col')
    price_col = st.session_state.get('price_col')
    
    # Initialize the chat interface
    has_sentiment, has_stock = init_chat_interface(
        sentiment_data=sentiment_data,
        stock_data=stock_data,
        date_col=date_col,
        price_col=price_col
    )
    
    # Note: We don't handle the chat input here to avoid 
    # Streamlit container errors. The input is handled in the main function.

def render_try_yourself_tab():
    """Main tab rendering function"""
    # Initialize session state
    _init_state()
    
    # -----------------------------------------------------------------------
    # Header & Link Indicator
    # -----------------------------------------------------------------------
    st.markdown("## Microsoft Earnings Call & Stock Analysis")
    
    # Create tabs for the analysis UI
    dashboard_tab, stock_tab, sentiment_tab, chatbot_tab = st.tabs([
        "Dashboard", "Stock Analysis", "Sentiment Analysis", "Chatbot Assistant"
    ])
    
    # -----------------------------------------------------------------------
    # Configuration Section
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Configuration Section (Sidebar)
    # -----------------------------------------------------------------------
    st.sidebar.markdown("## Analysis Configuration")

    # Show data source information - but don't allow changing it
    st.sidebar.markdown(f"### Using Pre-categorized Data")
    st.sidebar.code(f"Data source: {SENTIMENT_DATA_PATH}", language=None)

    # In the sidebar, it's often better to stack controls vertically rather than using columns
    st.sidebar.text_input("Stock Ticker", value="MSFT", disabled=True)
    start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=90))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())

    # OpenAI API key input - use password field
    st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key", 
                help="Required for sentiment analysis. Your API key remains secured and is only used to process the sentences.")

    # Run analysis button
    if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
        if not st.session_state.openai_api_key:
            st.sidebar.error("Please enter your OpenAI API key")
        else:
            process_analysis_workflow(
                st.session_state.ticker_symbol,
                start_date,
                end_date,
                STOCK_DATA_PATH,
                SENTIMENT_DATA_PATH
            )
        
    # -----------------------------------------------------------------------
    # Tab Content
    # -----------------------------------------------------------------------
    with dashboard_tab:
        render_dashboard_tab()
    
    with stock_tab:
        render_stock_tab()
    
    with sentiment_tab:
        render_sentiment_tab()
    
    with chatbot_tab:
        render_chatbot_tab()
    
    # -----------------------------------------------------------------------
    # Chat Input (outside of tabs to avoid Streamlit container errors)
    # -----------------------------------------------------------------------
    # Placeholder for chat input
    chat_prompt = None
    
    # Display and handle chat outside of tab containers to avoid errors
    if safe_check_dataframe(st.session_state.get('sentiment_results')):
        chat_prompt = st.chat_input("Ask about the earnings call or stock performance")
    
    # Process chat input if provided
    if chat_prompt:
        process_chat_input(
            chat_prompt,
            st.session_state.sentiment_results,
            st.session_state.stock_data,
            st.session_state.date_col,
            st.session_state.price_col
        )

# If the file is run directly, set up a test environment
if __name__ == "__main__":
    st.set_page_config(
        page_title="Microsoft Earnings Call Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    # Call the main function
    render_try_yourself_tab()
