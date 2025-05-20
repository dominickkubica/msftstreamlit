# plotting.py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
from datetime import timedelta
import logging
import re

# Setup logger
logger = logging.getLogger("earnings_analyzer")

# Custom metric function with direct HTML color control
def colored_metric(label, value, delta, suffix="%", prefix="", vs_text=""):
    """Display a metric with colored delta values"""
    # Determine color and arrow based on delta value
    if delta < 0:
        color = "#D13438"  # Microsoft red
        arrow = "↓"
    else:
        color = "#107C10"  # Microsoft green
        arrow = "↑"
    
    # Format the delta with the colored arrow
    vs_part = f" {vs_text}" if vs_text else ""
    delta_html = f'<span style="color:{color}">{arrow} {abs(delta):.2f}{suffix}{vs_part}</span>'
    
    # Create HTML for the metric - ensure proper spacing
    html = f"""
    <div style="margin-bottom: 1rem;">
        <div style="font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);">{label}</div>
        <div style="font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;">{prefix}{value}</div>
        <div style="font-size: 0.9rem;">{delta_html}</div>
    </div>
    """
    
    # Display using markdown with HTML
    st.markdown(html, unsafe_allow_html=True)

# Format data text for consistent display
def format_stock_description(date, price, volume=None, high=None, low=None, opening=None):
    """Create consistently formatted stock descriptions with proper spacing"""
    if hasattr(date, "strftime"):
        date_str = date.strftime("%B %d, %Y")
    else:
        date_str = str(date)
    
    description = f"On {date_str}, the stock price was ${price:.2f}"
    
    # Add optional information with proper spacing
    if volume is not None:
        description += f", with a trading volume of {volume:,} shares"
    
    if high is not None and low is not None:
        description += f". The stock fluctuated between a low of ${low:.2f} and a high of ${high:.2f}"
    
    if opening is not None:
        description += f", opening at ${opening:.2f}"
    
    if not description.endswith("."):
        description += "."
        
    return description

# Create a properly formatted stock range description
def format_stock_range(start_date, start_price, end_date, end_price):
    """Create a properly formatted stock range description with consistent spacing"""
    start_date_str = start_date.strftime("%B %d, %Y") if hasattr(start_date, "strftime") else str(start_date)
    end_date_str = end_date.strftime("%B %d, %Y") if hasattr(end_date, "strftime") else str(end_date)
    
    description = f"From {start_date_str} to {end_date_str}, the stock price changed from ${start_price:.2f} to ${end_price:.2f}"
    
    percent_change = ((end_price - start_price) / start_price) * 100
    if percent_change > 0:
        description += f", an increase of {percent_change:.2f}%."
    elif percent_change < 0:
        description += f", a decrease of {abs(percent_change):.2f}%."
    else:
        description += ", with no change in price."
    
    return description

# Plot functions
def plot_sentiment_distribution(sentiment_data, microsoft_colors, title=None):
    """Create pie chart of sentiment distribution"""
    try:
        sentiment_counts = sentiment_data['sentiment'].value_counts()
        
        # Create pie chart
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': microsoft_colors['success'],
                'neutral': microsoft_colors['accent'],
                'negative': microsoft_colors['danger']
            },
            hole=0.4
        )
        
        # Update layout with proper title
        chart_title = title if title else "Sentiment Distribution"
        fig.update_layout(
            title=chart_title,
            legend=dict(orientation="h"),
            title_font=dict(size=20),
            template="plotly_white"
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating sentiment distribution chart: {e}")
        return None

def plot_sentiment_by_business_line(sentiment_data, microsoft_colors):
    """Create stacked bar chart of sentiment by business line"""
    try:
        # Calculate sentiment by business segment
        segment_sentiment = sentiment_data.groupby('business_line')['sentiment'].value_counts().unstack().fillna(0)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in segment_sentiment.columns:
                fig.add_trace(go.Bar(
                    x=segment_sentiment.index,
                    y=segment_sentiment[sentiment],
                    name=sentiment,
                    marker_color={
                        'positive': microsoft_colors['success'],
                        'neutral': microsoft_colors['accent'],
                        'negative': microsoft_colors['danger']
                    }[sentiment]
                ))
        
        # Update layout
        fig.update_layout(
            barmode='stack',
            legend=dict(orientation="h"),
            title="Sentiment by Business Segment",
            title_font=dict(size=20),
            template="plotly_white",
            xaxis_title="Business Segment",
            yaxis_title="Count",
            # Improve readability with rotated labels if many segments
            xaxis=dict(
                tickangle=-45 if len(segment_sentiment.index) > 4 else 0
            )
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating business segment chart: {e}")
        return None

def plot_sentiment_and_stock(sentiment_timeline, stock_data, date_col, price_col):
    """Create combined chart of sentiment and stock price"""
    if stock_data is None or sentiment_timeline.empty:
        return None
    
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add stock price line
        fig.add_trace(
            go.Scatter(
                x=stock_data[date_col], 
                y=stock_data[price_col],
                name='Stock Price',
                line=dict(color='#0078d4')
            ),
            secondary_y=True
        )
        
        # Add sentiment scores
        # Convert sentiment timeline date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(sentiment_timeline['date']):
            sentiment_timeline['date'] = pd.to_datetime(sentiment_timeline['date'])
        
        # Map categorical sentiment to numeric values for plotting
        sentiment_timeline['sentiment_value'] = sentiment_timeline['sentiment'].map({
            'positive': 1, 
            'neutral': 0, 
            'negative': -1
        })
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=sentiment_timeline['date'],
                y=sentiment_timeline['sentiment_value'],
                name='Sentiment',
                mode='lines+markers',
                text=sentiment_timeline['business_line'],
                marker=dict(
                    size=10,
                    color=[
                        {
                            'positive': '#107c10',  # green
                            'neutral': '#5c2d91',   # purple
                            'negative': '#d13438'   # red
                        }[s] for s in sentiment_timeline['sentiment']
                    ]
                ),
                line=dict(color='#107c10')
            ),
            secondary_y=False
        )
        
        # Update layout
        title = "Sentiment vs Stock Price"
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=500
        )
        
        # Set y-axes titles
        fig.update_yaxes(
            title_text="Sentiment Score (-1 to 1)",
            secondary_y=False,
            range=[-1.2, 1.2]
        )
        fig.update_yaxes(title_text="Stock Price ($)", secondary_y=True)
        
        return fig
    except Exception as e:
        logger.error(f"Error creating sentiment and stock chart: {e}")
        return None

def create_earnings_impact_analysis(stock_data, date_col, price_col, ohlc_cols, highlight_date, microsoft_colors, MSFT_EARNINGS_DATES):
    """Create detailed analysis of stock movement around earnings date"""
    from data_loader import find_closest_trading_day
    
    # Find closest trading day to the earnings date
    trading_day = find_closest_trading_day(stock_data, date_col, highlight_date)
    
    if not trading_day:
        return st.warning(f"No trading data found for {highlight_date}")
    
    # Get price on earnings day
    earnings_day_data = stock_data[stock_data[date_col].dt.date == trading_day]
    if earnings_day_data.empty:
        return st.warning(f"No stock data available for {trading_day}")
    
    earnings_price = earnings_day_data[price_col].iloc[0]
    
    # Create detailed earnings impact section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("### Stock Price Impact")
        
        # Define periods to analyze
        periods = [1, 3, 5, 10, 20, 60]
        period_data = []
        
        # Calculate pre-earnings price changes
        for days in periods:
            # Find date before earnings
            pre_date = trading_day - timedelta(days=days)
            pre_trading_day = find_closest_trading_day(stock_data, date_col, pre_date, 'before')
            
            if pre_trading_day:
                pre_data = stock_data[stock_data[date_col].dt.date == pre_trading_day]
                if not pre_data.empty:
                    pre_price = pre_data[price_col].iloc[0]
                    pre_change = ((earnings_price - pre_price) / pre_price) * 100
                    period_data.append({
                        "Period": f"{days}D Before",
                        "Price": pre_price,
                        "Change": pre_change,
                        "Direction": "pre"
                    })
        
        # Calculate post-earnings price changes
        for days in periods:
            # Find date after earnings
            post_date = trading_day + timedelta(days=days)
            post_trading_day = find_closest_trading_day(stock_data, date_col, post_date, 'after')
            
            if post_trading_day:
                post_data = stock_data[stock_data[date_col].dt.date == post_trading_day]
                if not post_data.empty:
                    post_price = post_data[price_col].iloc[0]
                    post_change = ((post_price - earnings_price) / earnings_price) * 100
                    period_data.append({
                        "Period": f"{days}D After",
                        "Price": post_price,
                        "Change": post_change,
                        "Direction": "post"
                    })
        
        # Create a DataFrame for better display
        period_df = pd.DataFrame(period_data)
        period_df['PeriodValue'] = period_df['Period'].str.extract(r'(\d+)').astype(int)
        
        # Group and format the data
        pre_df = period_df[period_df["Direction"] == "pre"].sort_values("PeriodValue")
        post_df = period_df[period_df["Direction"] == "post"].sort_values("PeriodValue")
        
        # Display the earnings day price using a properly formatted metric
        earnings_date_str = trading_day.strftime("%B %d, %Y")
        st.markdown(
            f"<div style='font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>Earnings Day Price ({earnings_date_str})</div>"
            f"<div style='font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;'>${earnings_price:.2f}</div>", 
            unsafe_allow_html=True
        )
        
        # Create a better visualization of the pre/post earnings changes
        st.write("#### Price Changes Around Earnings")
        
        # Create 2 columns for pre and post data
        pre_col, post_col = st.columns(2)
        
        # Use custom colored_metric for pre-earnings
        with pre_col:
            st.write("Pre-Earnings Performance:")
            for _, row in pre_df.iterrows():
                period = row["Period"]
                price = row["Price"]
                change = row["Change"]
                
                # Use our custom HTML-based colored metric
                colored_metric(
                    period, 
                    f"${price:.2f}", 
                    change
                )
        
        # Use custom colored_metric for post-earnings
        with post_col:
            st.write("Post-Earnings Performance:")
            for _, row in post_df.iterrows():
                period = row["Period"]
                price = row["Price"]
                change = row["Change"]
                
                # Use our custom HTML-based colored metric
                colored_metric(
                    period, 
                    f"${price:.2f}", 
                    change
                )
        
        # Create a zoomed-in chart showing the period around earnings
        st.write("#### Zoom View Around Earnings Date")
        
        # Define range for zoom chart: 30 days before to 30 days after
        zoom_start = trading_day - timedelta(days=30)
        zoom_end = trading_day + timedelta(days=30)
        
        # Find closest trading days to these dates
        zoom_start_day = find_closest_trading_day(stock_data, date_col, zoom_start, 'before')
        zoom_end_day = find_closest_trading_day(stock_data, date_col, zoom_end, 'after')
        
        if zoom_start_day and zoom_end_day:
            # Filter data for the zoom range
            zoom_mask = (stock_data[date_col].dt.date >= zoom_start_day) & (stock_data[date_col].dt.date <= zoom_end_day)
            zoom_df = stock_data[zoom_mask].copy()
            
            # Create zoom chart
            zoom_fig = go.Figure()
            
            # Add price line
            zoom_fig.add_trace(go.Scatter(
                x=zoom_df[date_col],
                y=zoom_df[price_col],
                mode='lines',
                name='MSFT',
                line=dict(color='#0078D4', width=2)
            ))
            
            # Add earnings day vertical line
            zoom_fig.add_vline(
                x=trading_day,
                line=dict(color='#FF0000', width=2)
            )
            
            # Add earnings day annotation
            zoom_fig.add_annotation(
                x=trading_day,
                y=zoom_df[price_col].max() * 1.02,
                text="Earnings Call",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#FF0000',
                font=dict(color='#FF0000', size=12),
                align="center"
            )
            
            # Update layout
            zoom_fig.update_layout(
                title=f"MSFT Stock Price Around {trading_day.strftime('%Y-%m-%d')} Earnings",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                template="plotly_white",
                hovermode="x unified"
            )
            
            # Display the zoom chart
            st.plotly_chart(zoom_fig, use_container_width=True, key="earnings_zoom_chart")
    
    with col2:
        st.write("### Key Metrics")
        
        # Opening Gap section (if available)
        if ohlc_cols and 'open' in ohlc_cols:
            earnings_open = earnings_day_data[ohlc_cols['open']].iloc[0]
            
            # Find previous trading day
            prev_date = trading_day - timedelta(days=1)
            prev_trading_day = find_closest_trading_day(stock_data, date_col, prev_date, 'before')
            
            if prev_trading_day:
                prev_data = stock_data[stock_data[date_col].dt.date == prev_trading_day]
                if not prev_data.empty:
                    prev_close = prev_data[price_col].iloc[0]
                    gap_pct = ((earnings_open - prev_close) / prev_close) * 100
                    
                    # Use custom colored_metric for Opening Gap
                    colored_metric(
                        "Opening Gap", 
                        f"${earnings_open:.2f}", 
                        gap_pct,
                        "% from prev close"
                    )
        
        # Historical Context section
        st.write("#### Historical Context")
        
        # MSFT earnings dates
        earnings_dates = [pd.to_datetime(date_str).date() for date_str in MSFT_EARNINGS_DATES]
        
        # Create comparison of earnings reactions
        earnings_reactions = []
        
        # Find post-earnings reaction for relevant earnings dates
        for e_date in earnings_dates:
            e_trading_day = find_closest_trading_day(stock_data, date_col, e_date)
            
            if e_trading_day:
                e_data = stock_data[stock_data[date_col].dt.date == e_trading_day]
                if not e_data.empty:
                    e_price = e_data[price_col].iloc[0]
                    
                    # Find price 3 days after earnings
                    post_date = e_trading_day + timedelta(days=3)
                    post_trading_day = find_closest_trading_day(stock_data, date_col, post_date, 'after')
                    
                    if post_trading_day:
                        post_data = stock_data[stock_data[date_col].dt.date == post_trading_day]
                        if not post_data.empty:
                            post_price = post_data[price_col].iloc[0]
                            pct_change = ((post_price - e_price) / e_price) * 100
                            
                            earnings_reactions.append({
                                "Date": e_trading_day,
                                "Change": pct_change,
                                "IsSelected": e_trading_day == trading_day
                            })
        
        if earnings_reactions:
            # Create a DataFrame for visualization
            reaction_df = pd.DataFrame(earnings_reactions)
            
            # Average reaction
            avg_reaction = reaction_df["Change"].mean()
            
            # Get current reaction
            current_reaction = 0
            if not reaction_df[reaction_df["IsSelected"]].empty:
                current_reaction = reaction_df[reaction_df["IsSelected"]]["Change"].iloc[0]
            
            # Use custom colored_metric for 3-Day Reaction
            colored_metric(
                "3-Day Reaction", 
                f"{current_reaction:.2f}%", 
                current_reaction - avg_reaction,
                "% vs avg"
            )
            
            # Create a comparison chart

def create_stock_chart(filtered_df, date_col, price_col, ohlc_cols, chart_type, show_ma, show_earnings, MSFT_EARNINGS_DATES, start_date, end_date, selected_earnings_date=None):
    """Create stock price chart based on selected parameters"""
    from data_loader import find_closest_trading_day
    
    if filtered_df.empty:
        return None
    
    # Create the main chart based on selected type
    if chart_type == "Line":
        # Simple line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df[date_col],
            y=filtered_df[price_col],
            mode='lines',
            name='MSFT',
            line=dict(color='#0078D4', width=2)  # Microsoft blue
        ))
        
        # Add moving averages if requested
        if show_ma and len(filtered_df) > 50:
            # 20-day moving average
            filtered_df['MA20'] = filtered_df[price_col].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=filtered_df[date_col],
                y=filtered_df['MA20'],
                mode='lines',
                name='20-Day MA',
                line=dict(color='#107C10', width=1.5, dash='dash')  # Microsoft green
            ))
            
            # 50-day moving average if enough data
            if len(filtered_df) > 100:
                filtered_df['MA50'] = filtered_df[price_col].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=filtered_df[date_col],
                    y=filtered_df['MA50'],
                    mode='lines',
                    name='50-Day MA',
                    line=dict(color='#5C2D91', width=1.5, dash='dash')  # Microsoft purple
                ))
        
    elif chart_type == "Candlestick":
        # Need OHLC data for candlestick
        if ohlc_cols and 'open' in ohlc_cols and 'high' in ohlc_cols and 'low' in ohlc_cols:
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=filtered_df[date_col],
                open=filtered_df[ohlc_cols['open']],
                high=filtered_df[ohlc_cols['high']],
                low=filtered_df[ohlc_cols['low']],
                close=filtered_df[price_col],
                name='MSFT',
                increasing=dict(line=dict(color='#107C10')),  # Microsoft green
                decreasing=dict(line=dict(color='#D13438'))   # Microsoft red
            )])
            
            # Add moving averages if requested
            if show_ma and len(filtered_df) > 50:
                # 20-day moving average
                filtered_df['MA20'] = filtered_df[price_col].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=filtered_df[date_col],
                    y=filtered_df['MA20'],
                    mode='lines',
                    name='20-Day MA',
                    line=dict(color='#FFFFFF', width=1.5, dash='dash')  # White for contrast
                ))
                
                # 50-day moving average if enough data
                if len(filtered_df) > 100:
                    filtered_df['MA50'] = filtered_df[price_col].rolling(window=50).mean()
                    fig.add_trace(go.Scatter(
                        x=filtered_df[date_col],
                        y=filtered_df['MA50'],
                        mode='lines',
                        name='50-Day MA',
                        line=dict(color='#FFB900', width=1.5, dash='dash')  # Microsoft yellow
                    ))
        else:
            st.warning("OHLC data not found, defaulting to line chart")
            fig = px.line(filtered_df, x=date_col, y=price_col, title=f"MSFT Stock Price ({start_date} to {end_date})")
    
    elif chart_type == "OHLC":
        # Similar to candlestick but different visual style
        if ohlc_cols and 'open' in ohlc_cols and 'high' in ohlc_cols and 'low' in ohlc_cols:
            # Create OHLC chart
            fig = go.Figure(data=[go.Ohlc(
                x=filtered_df[date_col],
                open=filtered_df[ohlc_cols['open']],
                high=filtered_df[ohlc_cols['high']],
                low=filtered_df[ohlc_cols['low']],
                close=filtered_df[price_col],
                name='MSFT',
                increasing=dict(line=dict(color='#107C10')),  # Microsoft green
                decreasing=dict(line=dict(color='#D13438'))   # Microsoft red
            )])
            
            # Add moving averages if requested
            if show_ma and len(filtered_df) > 50:
                # 20-day moving average
                filtered_df['MA20'] = filtered_df[price_col].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=filtered_df[date_col],
                    y=filtered_df['MA20'],
                    mode='lines',
                    name='20-Day MA',
                    line=dict(color='#FFFFFF', width=1.5, dash='dash')  # White for contrast
                ))
                
                # 50-day moving average if enough data
                if len(filtered_df) > 100:
                    filtered_df['MA50'] = filtered_df[price_col].rolling(window=50).mean()
                    fig.add_trace(go.Scatter(
                        x=filtered_df[date_col],
                        y=filtered_df['MA50'],
                        mode='lines',
                        name='50-Day MA',
                        line=dict(color='#FFB900', width=1.5, dash='dash')  # Microsoft yellow
                    ))
        else:
            st.warning("OHLC data not found, defaulting to line chart")
            fig = px.line(filtered_df, x=date_col, y=price_col, title=f"MSFT Stock Price ({start_date} to {end_date})")
    
    # Format date range for title with proper spacing
    start_date_str = start_date.strftime("%b %d, %Y") if hasattr(start_date, "strftime") else str(start_date)
    end_date_str = end_date.strftime("%b %d, %Y") if hasattr(end_date, "strftime") else str(end_date)
    
    # Update layout
    fig.update_layout(
        title=f"MSFT Stock Price ({start_date_str} to {end_date_str})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        template="plotly_white",
        xaxis_rangeslider_visible=False,  # Hide range slider for cleaner look
        hovermode="x unified"
    )
    
    # Add earnings date vertical lines if requested
    if show_earnings:
        # Convert hardcoded dates to datetime for comparison
        earnings_dates = [pd.to_datetime(date_str).date() for date_str in MSFT_EARNINGS_DATES]
        
        # Filter to only show earnings dates in the selected range
        range_earnings_dates = [date for date in earnings_dates if start_date <= date <= end_date]
        
        # Convert selected_earnings_date to datetime.date if it's not None
        if selected_earnings_date is not None and not isinstance(selected_earnings_date, type(start_date)):
            try:
                if isinstance(selected_earnings_date, str):
                    selected_earnings_date = pd.to_datetime(selected_earnings_date).date()
                elif hasattr(selected_earnings_date, 'date'):
                    selected_earnings_date = selected_earnings_date.date()
            except:
                selected_earnings_date = None
        
        if range_earnings_dates:
            # Find min and max prices for proper line positioning
            y_min = filtered_df[price_col].min()
            y_max = filtered_df[price_col].max()
            y_range = y_max - y_min
            
            for earnings_date in range_earnings_dates:
                # Find closest trading day to earnings date
                closest_date = find_closest_trading_day(filtered_df, date_col, earnings_date)
                
                if not closest_date:
                    continue
                
                # Format the date string properly
                date_str = closest_date.strftime("%b %d, %Y")
                
                # Determine if this is the selected earnings date
                is_selected = False
                if selected_earnings_date is not None:
                    # Check if this earnings date is the selected one (within 1 day tolerance)
                    is_selected = abs((closest_date - selected_earnings_date).days) <= 1
                
                # Choose color based on whether this is the selected earnings date
                line_color = "#FF0000" if is_selected else "#FFB900"  # Red if selected, yellow otherwise
                
                # Add vertical line for the earnings date
                fig.add_vline(
                    x=closest_date,
                    line=dict(
                        color=line_color,
                        width=1.5,
                        dash="dash"
                    )
                )
                
                # Add annotation for the earnings date
                fig.add_annotation(
                    x=closest_date,
                    y=y_max + (y_range * 0.02),  # Position above the highest point
                    text=f"Earnings: {date_str}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.5,
                    arrowcolor=line_color,
                    font=dict(color=line_color, size=10),
                    align="center"
                )
    
    return fig