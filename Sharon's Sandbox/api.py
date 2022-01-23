import numpy as np
import pandas as pd
# market data from Yahoo
# reference: https://algotrading101.com/learn/yfinance-guide/
import yfinance as yf
# Data viz
import plotly.graph_objs as go


def crypto_real_time_viz(tickers):
    # The full range of intervals available are:
    # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    data = yf.download(tickers, period='48h', interval='1m')

    # declare figure
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'], name='market data'))

    # Add titles
    fig.update_layout(
        title='live share price evolution',
        yaxis_title='Price (US Dollars)')

    # X-Axes
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=45, label="45m", step="minute", stepmode="backward"),
                dict(count=1, label="HTD", step="hour", stepmode="todate"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.show()
    return data  # return the dataframe
