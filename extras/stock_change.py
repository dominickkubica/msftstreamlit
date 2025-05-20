
#!/usr/bin/env python3
"""
stock_change.py

Usage:
    python stock_change.py TICKER DATE

Example:
    python stock_change.py AAPL 2025-05-05
"""

import argparse
import datetime
import sys

import yfinance as yf


def get_stock_change(ticker: str, date_str: str):
    # Parse the input date
    try:
        target = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        print(f"❌  Invalid date format: {date_str}. Use YYYY-MM-DD.")
        sys.exit(1)

    # We fetch a little extra history so we can find the prior trading day
    window_start = target - datetime.timedelta(days=7)
    window_end   = target + datetime.timedelta(days=1)

    # Download history
    hist = yf.download(
        tickers=ticker,
        start=window_start.strftime('%Y-%m-%d'),
        end=window_end.strftime('%Y-%m-%d'),
        progress=False,
        auto_adjust=False
    )

    if hist.empty:
        print(f"❌  No data found for {ticker} between "
              f"{window_start} and {window_end}.")
        sys.exit(1)

    # Ensure we have a row for the target date
    # yfinance index is Timestamp; match by date component
    day_rows = hist.loc[hist.index.date == target]
    if day_rows.empty:
        print(f"❌  {ticker} did not trade on {target} "
              "(weekend or holiday).")
        sys.exit(1)
    today = day_rows.iloc[0]

    # Find the last row *before* the target date
    prev = hist.loc[hist.index.date < target]
    if prev.empty:
        print(f"❌  No prior trading data available to compute change.")
        sys.exit(1)
    yesterday = prev.iloc[-1]

       # coerce to Python floats so formatting works
    open_price, high_price, low_price, close_price, prev_close = map(
        float,
        (
            today['Open'],
            today['High'],
            today['Low'],
            today['Close'],
            yesterday['Close'],
        )
    )

    pct_change = (close_price - prev_close) / prev_close * 100

    # Output
    print(f"\n📈  {ticker.upper()} on {target}")
    print(f"    Open:  ${open_price:,.2f}")
    print(f"    High:  ${high_price:,.2f}")
    print(f"    Low:   ${low_price:,.2f}")
    print(f"    Close: ${close_price:,.2f}")
    print(f"    Change vs prior close: {pct_change:+.2f}%\n")



def main():
    parser = argparse.ArgumentParser(
        description="Fetch a stock's percent move on a given date."
    )
    parser.add_argument(
        'ticker',
        help='Ticker symbol (e.g. AAPL, MSFT)'
    )
    parser.add_argument(
        'date',
        help='Date in YYYY-MM-DD format'
    )
    args = parser.parse_args()
    get_stock_change(args.ticker, args.date)


if __name__ == '__main__':
    main()
