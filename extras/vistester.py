import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np

# Set page config
st.set_page_config(page_title="MSFT Stock Chart", layout="wide")

# Custom metric function with direct HTML color control
def colored_metric(label, value, delta, suffix="%", prefix="", vs_text=""):
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
    
    # Create HTML for the metric
    html = f"""
    <div style="margin-bottom: 1rem;">
        <div style="font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);">{label}</div>
        <div style="font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;">{prefix}{value}</div>
        <div style="font-size: 0.9rem;">{delta_html}</div>
    </div>
    """
    
    # Display using markdown with HTML
    st.markdown(html, unsafe_allow_html=True)

# Main title
st.title("MSFT Stock Chart with Earnings Impact Analysis")
st.info("Select specific earnings dates to analyze their impact on stock price")

# Define file path
file_path = "C:\\Users\\kubic\\OneDrive\\Desktop\\MSFTSTREAMLIT\\HistoricalData_1747025804532.csv"

# Recent Microsoft earnings dates (updated with fiscal quarter dates)
# Format: 'YYYY-MM-DD' for approximately the last 3 years
MSFT_EARNINGS_DATES = [
    "2023-01-24", "2023-04-25", "2023-07-25", "2023-10-24",
    "2024-01-30", "2024-04-25", "2024-07-30", "2024-10-30",
    "2025-01-29", "2025-04-30"
]

# Function to load data
def load_stock_data(file_path):
    """Load stock data from CSV and perform minimal processing"""
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
        
    try:
        # Load raw data
        df = pd.read_csv(file_path)
        st.sidebar.success(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")
        
        # Display column names in expandable section
        with st.sidebar.expander("CSV Columns"):
            st.write(df.columns.tolist())
        
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

# Load data
data_result = load_stock_data(file_path)

if data_result:
    df, date_col, price_col, ohlc_cols = data_result
    
    # Get min and max dates for the range selector
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()
    
    # Sidebar - Main Controls
    st.sidebar.subheader("Chart Controls")
    
    # Date range selection in sidebar
    default_start = max(min_date, max_date - timedelta(days=365))  # Default to 1 year
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Make sure end_date is not before start_date
    if start_date > end_date:
        st.sidebar.error("End date must be after start date")
        end_date = start_date
    
    # Filter data based on selected date range
    mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
    filtered_df = df.loc[mask]
    
    # Show how many data points are in the filtered range
    st.sidebar.write(f"Showing {len(filtered_df)} data points from {start_date} to {end_date}")
    
    # Chart type selection
    chart_type = st.sidebar.selectbox(
        "Chart Type",
        ["Line", "Candlestick", "OHLC"],
        index=0
    )
    
    # Show moving averages option
    show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
    
    # Sidebar - Earnings Analysis Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Earnings Call Analysis")
    
    # Convert hardcoded dates to datetime for comparison
    earnings_dates = [datetime.strptime(date_str, "%Y-%m-%d").date() for date_str in MSFT_EARNINGS_DATES]
    
    # Filter to only show earnings dates in the selected range
    range_earnings_dates = [date for date in earnings_dates 
                          if start_date <= date <= end_date]
    
    # Format dates for display in selector
    if range_earnings_dates:
        date_options = [date.strftime("%Y-%m-%d") for date in range_earnings_dates]
        # Add a "Show All" option
        date_options = ["Show All Earnings"] + date_options
        
        # Create earnings date selector
        selected_earnings = st.sidebar.selectbox(
            "Select Earnings Date to Analyze", 
            date_options
        )
        
        # Determine which earnings date to highlight
        highlight_date = None
        if selected_earnings != "Show All Earnings":
            highlight_date = datetime.strptime(selected_earnings, "%Y-%m-%d").date()
    else:
        st.sidebar.warning("No earnings dates in selected range")
        selected_earnings = None
        highlight_date = None
    
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
        if 'open' in ohlc_cols and 'high' in ohlc_cols and 'low' in ohlc_cols:
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
        if 'open' in ohlc_cols and 'high' in ohlc_cols and 'low' in ohlc_cols:
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
    
    # Update layout
    fig.update_layout(
        title=f"MSFT Stock Price ({start_date} to {end_date})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        template="plotly_white",
        xaxis_rangeslider_visible=False,  # Hide range slider for cleaner look
        hovermode="x unified"
    )
    
    # Add earnings date vertical lines
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
            
            # Determine if this is the highlighted date
            is_highlighted = (highlight_date and earnings_date == highlight_date)
            
            # Set line properties based on whether this is the highlighted date
            line_width = 3 if is_highlighted else 1.5
            line_color = "#FF0000" if is_highlighted else "#FFB900"  # Red if highlighted, otherwise gold
            
            # Add vertical line for the earnings date
            fig.add_vline(
                x=closest_date,
                line=dict(
                    color=line_color,
                    width=line_width,
                    dash="solid" if is_highlighted else "dash"
                )
            )
            
            # Add annotation for the earnings date
            if is_highlighted or selected_earnings == "Show All Earnings":
                fig.add_annotation(
                    x=closest_date,
                    y=y_max + (y_range * 0.02),  # Position above the highest point
                    text=f"Earnings: {closest_date.strftime('%Y-%m-%d')}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=line_width,
                    arrowcolor=line_color,
                    font=dict(color=line_color, size=12 if is_highlighted else 10),
                    align="center"
                )
    
    # Display the main chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed earnings analysis if a specific date is selected
    if highlight_date:
        st.markdown("---")
        st.subheader(f"Earnings Call Analysis: {highlight_date.strftime('%B %d, %Y')}")
        
        # Find closest trading day to the earnings date
        trading_day = find_closest_trading_day(filtered_df, date_col, highlight_date)
        
        if trading_day:
            # Get price on earnings day
            earnings_day_data = filtered_df[filtered_df[date_col].dt.date == trading_day]
            if not earnings_day_data.empty:
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
                        pre_trading_day = find_closest_trading_day(filtered_df, date_col, pre_date, 'before')
                        
                        if pre_trading_day:
                            pre_data = filtered_df[filtered_df[date_col].dt.date == pre_trading_day]
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
                        post_trading_day = find_closest_trading_day(filtered_df, date_col, post_date, 'after')
                        
                        if post_trading_day:
                            post_data = filtered_df[filtered_df[date_col].dt.date == post_trading_day]
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
                    
                    # Group and format the data
                    pre_df = period_df[period_df["Direction"] == "pre"].sort_values("Period")
                    post_df = period_df[period_df["Direction"] == "post"].sort_values("Period")
                    
                    # Display the earnings day price using a simpler metric without delta
                    st.markdown(f"<div style='font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>Earnings Day Price</div>"
                               f"<div style='font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;'>${earnings_price:.2f}</div>", 
                               unsafe_allow_html=True)
                    
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
                    zoom_start_day = find_closest_trading_day(filtered_df, date_col, zoom_start, 'before')
                    zoom_end_day = find_closest_trading_day(filtered_df, date_col, zoom_end, 'after')
                    
                    if zoom_start_day and zoom_end_day:
                        # Filter data for the zoom range
                        zoom_mask = (filtered_df[date_col].dt.date >= zoom_start_day) & (filtered_df[date_col].dt.date <= zoom_end_day)
                        zoom_df = filtered_df.loc[zoom_mask]
                        
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
                        st.plotly_chart(zoom_fig, use_container_width=True)
                
                with col2:
                    st.write("### Key Metrics")
                    
                    # Opening Gap section (if available)
                    if 'open' in ohlc_cols:
                        earnings_open = earnings_day_data[ohlc_cols['open']].iloc[0]
                        
                        # Find previous trading day
                        prev_date = trading_day - timedelta(days=1)
                        prev_trading_day = find_closest_trading_day(filtered_df, date_col, prev_date, 'before')
                        
                        if prev_trading_day:
                            prev_data = filtered_df[filtered_df[date_col].dt.date == prev_trading_day]
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
                    
                    # Earnings Results section
                    st.write("#### Earnings Details")
                    
                    # Just a placeholder - in a real application you'd fetch this data
                    with st.expander("Earnings Results (Placeholder)"):
                        st.write("Note: This is placeholder data. In a real application, you would fetch actual earnings data.")
                        st.markdown("""
                        - **EPS**: $2.35 (Est: $2.27)
                        - **Revenue**: $56.2B (Est: $55.5B)
                        - **YoY Growth**: 15.3%
                        """)
                    
                    # Historical Context section
                    st.write("#### Historical Context")
                    
                    # Create comparison of earnings reactions
                    earnings_reactions = []
                    
                    # Find post-earnings reaction for all earnings dates
                    for e_date in range_earnings_dates:
                        e_trading_day = find_closest_trading_day(filtered_df, date_col, e_date)
                        
                        if e_trading_day:
                            e_data = filtered_df[filtered_df[date_col].dt.date == e_trading_day]
                            if not e_data.empty:
                                e_price = e_data[price_col].iloc[0]
                                
                                # Find price 3 days after earnings
                                post_date = e_trading_day + timedelta(days=3)
                                post_trading_day = find_closest_trading_day(filtered_df, date_col, post_date, 'after')
                                
                                if post_trading_day:
                                    post_data = filtered_df[filtered_df[date_col].dt.date == post_trading_day]
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
                        current_reaction = reaction_df[reaction_df["IsSelected"]]["Change"].iloc[0] if not reaction_df[reaction_df["IsSelected"]].empty else 0
                        
                        # Use custom colored_metric for 3-Day Reaction
                        colored_metric(
                            "3-Day Reaction", 
                            f"{current_reaction:.2f}%", 
                            current_reaction - avg_reaction,
                            "% vs avg"
                        )
                        
                        # Create a comparison chart
                        reaction_fig = go.Figure()
                        
                        # Add bars for all reactions
                        for _, row in reaction_df.iterrows():
                            date_str = row["Date"].strftime("%Y-%m-%d")
                            change = row["Change"]
                            is_selected = row["IsSelected"]
                            
                            # Color bars based on price movement (up is green, down is red)
                            bar_color = '#107C10' if change >= 0 else '#D13438'  # Green for price up, red for price down
                            
                            # Highlight selected date
                            if is_selected:
                                bar_color = '#FF0000'  # Highlight in red
                            
                            reaction_fig.add_trace(go.Bar(
                                x=[date_str],
                                y=[change],
                                name=date_str,
                                marker_color=bar_color,
                                width=0.7
                            ))
                        
                        # Add a line for the average
                        reaction_fig.add_shape(
                            type="line",
                            x0=-0.5,
                            y0=avg_reaction,
                            x1=len(reaction_df) - 0.5,
                            y1=avg_reaction,
                            line=dict(color="black", width=2, dash="dash")
                        )
                        
                        # Add annotation for the average
                        reaction_fig.add_annotation(
                            x=len(reaction_df) - 1,
                            y=avg_reaction,
                            text=f"Avg: {avg_reaction:.2f}%",
                            showarrow=False,
                            xanchor="right",
                            font=dict(color="black", size=10)
                        )
                        
                        # Update layout
                        reaction_fig.update_layout(
                            title="3-Day Reactions to Earnings",
                            xaxis_title="Earnings Date",
                            yaxis_title="% Change",
                            height=300,
                            template="plotly_white",
                            showlegend=False
                        )
                        
                        # Display the chart
                        st.plotly_chart(reaction_fig, use_container_width=True)
        else:
            st.warning(f"No trading data found for {highlight_date}")
    
    # Display summary statistics for the entire range
    st.markdown("---")
    st.subheader("Summary Statistics")
    
    # Layout statistics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate statistics
    start_price = filtered_df[price_col].iloc[0] if not filtered_df.empty else 0
    end_price = filtered_df[price_col].iloc[-1] if not filtered_df.empty else 0
    percent_change = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
    min_price = filtered_df[price_col].min() if not filtered_df.empty else 0
    max_price = filtered_df[price_col].max() if not filtered_df.empty else 0
    
    # Display statistics with custom HTML metrics
    with col1:
        st.markdown(f"<div style='font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>Start Price</div>"
                   f"<div style='font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;'>${start_price:.2f}</div>", 
                   unsafe_allow_html=True)
    
    # Use custom colored_metric for End Price
    with col2:
        colored_metric(
            "End Price", 
            f"${end_price:.2f}", 
            percent_change
        )
    
    with col3:
        st.markdown(f"<div style='font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>Min Price</div>"
                   f"<div style='font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;'>${min_price:.2f}</div>", 
                   unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"<div style='font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>Max Price</div>"
                   f"<div style='font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;'>${max_price:.2f}</div>", 
                   unsafe_allow_html=True)
    
    # Option to show the data table
    if st.checkbox("Show Data Table", value=False):
        st.dataframe(filtered_df)