# AUTHOR: ARMAAN KAPOOR
# Imc-Prosperity-2024 Visualizer
# Replace ```14: log_path``` with the path to a valid log file.

import re
from io import StringIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from plotly.subplots import make_subplots

log_path = "submissions/2024-04-15_14-35-58.log"


def read_file_sections(filepath):
    section_delimiters = ["Sandbox logs:", "Activities log:", "Trade History:"]
    current_section = None
    data = {key: [] for key in section_delimiters}

    with open(filepath, "r") as file:
        for line in file:
            line = line.strip()
            if line in section_delimiters:
                current_section = line
                continue
            if current_section:
                data[current_section].append(line)

    return data


def extract_trades(text):
    # Regex to find all blocks enclosed in curly braces
    trade_pattern = r"\{[^{}]*\}"
    # Regex to extract key-value pairs within curly braces
    kv_pattern = r"\"(\w+)\":\s*(\"[^\"]*\"|\d+)"

    trades = []
    matches = re.findall(trade_pattern, text)
    for match in matches:
        trade_data = {}
        for key, value in re.findall(kv_pattern, match):
            if value.startswith('"') and value.endswith('"'):
                value = value.strip('"')
            else:
                value = int(value)
            trade_data[key] = value
        trades.append(trade_data)
    return trades


sections = read_file_sections(log_path)
activities_data = "\n".join(sections["Activities log:"])
activities_df = pd.read_csv(StringIO(activities_data), delimiter=";")
trade_history_text = "\n".join(sections["Trade History:"])
trades = extract_trades(trade_history_text)
trade_history_df = pd.DataFrame(trades)


amethysts_df = activities_df[activities_df["product"] == "AMETHYSTS"]
starfruit_df = activities_df[activities_df["product"] == "STARFRUIT"]
orchids_df = activities_df[activities_df["product"] == "ORCHIDS"]


def plot_pnl(df, product_name, line_color):
    fig = px.line(
        df,
        x="timestamp",
        y="profit_and_loss",
        title=f"Profit and Loss Over Time for {product_name}",
        labels={"timestamp": "Timestamp", "profit_and_loss": "Profit and Loss"},
        template="plotly_white",
        color_discrete_sequence=[line_color],
    )
    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Profit and Loss",
        legend_title="Legend",
        yaxis=dict(tickformat=".2f"),
        xaxis=dict(tickangle=-45),
        hovermode="x",
    )
    fig.update_traces(mode="lines+markers", marker=dict(size=5, color=line_color))
    return fig


colors = {"AMETHYSTS": "blue", "STARFRUIT": "purple", "ORCHIDS": "green"}

fig_amethysts_pnl = plot_pnl(amethysts_df, "AMETHYSTS", colors["AMETHYSTS"])
fig_starfruit_pnl = plot_pnl(starfruit_df, "STARFRUIT", colors["STARFRUIT"])
fig_orchids_pnl = plot_pnl(orchids_df, "ORCHIDS", colors["ORCHIDS"])


amethysts_df = activities_df[activities_df["product"] == "AMETHYSTS"]
starfruit_df = activities_df[activities_df["product"] == "STARFRUIT"]
orchids_df = activities_df[activities_df["product"] == "ORCHIDS"]


def plot_product_prices(df, product_name, line_color):
    fig = go.Figure()

    # Add mid price line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["mid_price"],
            mode="lines",
            line=dict(color=line_color),
            name="Mid Price",
        )
    )

    # Add bid price markers
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["bid_price_1"],
            mode="markers",
            marker=dict(color="green", size=7),
            name="Bid Price",
        )
    )

    # Add ask price markers
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["ask_price_1"],
            mode="markers",
            marker=dict(color="red", size=7, symbol="x"),
            name="Ask Price",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Price Trends for {product_name}",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        legend_title="Price Type",
    )

    return fig


fig_price_trends_amethysts = plot_product_prices(amethysts_df, "AMETHYSTS", "blue")
fig_price_trends_starfruit = plot_product_prices(starfruit_df, "STARFRUIT", "purple")
fig_price_trends_orchids = plot_product_prices(orchids_df, "ORCHIDS", "green")


amethysts_df = activities_df[activities_df["product"] == "AMETHYSTS"]
starfruit_df = activities_df[activities_df["product"] == "STARFRUIT"]
orchids_df = activities_df[activities_df["product"] == "ORCHIDS"]


def plot_bid_prices_volumes_dual_axis(df, product_name):
    colors = [
        {"price": "darkblue", "volume": "lightblue"},
        {"price": "darkgreen", "volume": "lightgreen"},
        {"price": "darkred", "volume": "salmon"},
    ]

    fig = go.Figure()

    # Add line for each bid price and area for volumes
    for i in range(1, 4):  # Assuming there are up to 3 bid price levels
        bid_price_col = f"bid_price_{i}"
        bid_volume_col = f"bid_volume_{i}"

        if bid_price_col in df.columns and bid_volume_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[bid_price_col],
                    name=f"Bid Price {i} - {product_name}",
                    yaxis="y1",
                    line=dict(color=colors[i - 1]["price"], width=2),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[bid_volume_col],
                    name=f"Volume for Bid Price {i} - {product_name}",
                    yaxis="y2",
                    fill="tozeroy",
                    line=dict(color=colors[i - 1]["volume"], width=0.5),
                    opacity=0.5,
                )
            )

    fig.update_layout(
        title=f"Bid Prices and Volumes for {product_name}",
        xaxis_title="Timestamp",
        yaxis=dict(
            title="Bid Prices",
            side="left",
            showgrid=False,
        ),
        yaxis2=dict(
            title="Volumes",
            side="right",
            overlaying="y",
            showgrid=False,
        ),
        legend_title="Bid Levels and Volumes",
    )

    return fig


fig_bid_prices_volumes_amethysts = plot_bid_prices_volumes_dual_axis(
    amethysts_df, "AMETHYSTS"
)
fig_bid_prices_volumes_starfruit = plot_bid_prices_volumes_dual_axis(
    starfruit_df, "STARFRUIT"
)
fig_bid_prices_volumes_orchids = plot_bid_prices_volumes_dual_axis(
    orchids_df, "ORCHIDS"
)


def prepare_ticker_plots(trade_history_df, activities_df):
    unique_tickers = trade_history_df["symbol"].unique()
    figures = {}  # Dictionary to store figures

    for symbol in unique_tickers:
        # Filter data for the specific ticker
        ticker_trade_history_df = trade_history_df[trade_history_df["symbol"] == symbol]
        ticker_activities_df = activities_df[activities_df["product"] == symbol]

        # Merge the trade history with the activities data on timestamps
        merged_df = pd.merge(
            ticker_activities_df, ticker_trade_history_df, on="timestamp", how="outer"
        )

        # Create the plot
        fig = go.Figure()

        # Add the mid_price line
        fig.add_trace(
            go.Scatter(
                x=merged_df["timestamp"],
                y=merged_df["mid_price"],
                mode="lines",
                name="Mid Price",
                line=dict(color="blue"),
            )
        )

        # Prepare annotations for buys
        buy_annotations = [
            {
                "x": row["timestamp"],
                "y": row["mid_price"],
                "showarrow": True,
                "arrowhead": 1,
                "arrowsize": 2,
                "arrowwidth": 2,
                "arrowcolor": "green",
                "yshift": 10,
            }
            for index, row in merged_df[merged_df["buyer"] == "SUBMISSION"].iterrows()
        ]

        # Prepare annotations for sells
        sell_annotations = [
            {
                "x": row["timestamp"],
                "y": row["mid_price"],
                "showarrow": True,
                "arrowhead": 1,
                "arrowsize": 2,
                "arrowwidth": 2,
                "arrowcolor": "red",
                "yshift": -10,
            }
            for index, row in merged_df[merged_df["seller"] == "SUBMISSION"].iterrows()
        ]

        fig.update_layout(annotations=buy_annotations + sell_annotations)

        fig.update_layout(
            title=f"Trade Entries and Exits on Mid Price Timeseries for {symbol}",
            xaxis_title="Timestamp",
            yaxis_title="Price",
            legend_title="Legend",
        )

        # Store the figure in the dictionary with a key as the ticker name
        figures[f"fig_{symbol.lower()}_trades_made"] = fig

    return figures


# Prepare figures
trades_made_figures = prepare_ticker_plots(trade_history_df, activities_df)


def prepare_holdings_plots(trade_history_df, activities_df):
    unique_tickers = trade_history_df["symbol"].unique()
    figures = {}

    for symbol in unique_tickers:
        # Filter data for the specific ticker
        ticker_trade_history_df = trade_history_df[trade_history_df["symbol"] == symbol]
        ticker_activities_df = activities_df[activities_df["product"] == symbol]

        # Merge the trade history with the activities data on timestamps
        merged_df = pd.merge(
            ticker_activities_df, ticker_trade_history_df, on="timestamp", how="outer"
        )

        # Convert buy/sell actions into signed quantities
        merged_df["signed_quantity"] = merged_df.apply(
            lambda row: (
                row["quantity"]
                if row["buyer"] == "SUBMISSION"
                else -row["quantity"] if row["seller"] == "SUBMISSION" else 0
            ),
            axis=1,
        )

        # Calculate cumulative holdings over time
        merged_df["cumulative_holdings"] = merged_df["signed_quantity"].cumsum()

        # Create the plot
        fig = go.Figure()

        # Add a line for cumulative holdings
        fig.add_trace(
            go.Scatter(
                x=merged_df["timestamp"],
                y=merged_df["cumulative_holdings"],
                mode="lines",
                name="Cumulative Holdings",
                line=dict(color="green"),
            )
        )

        # Update layout
        fig.update_layout(
            title=f"Cumulative Holdings Over Time for {symbol}",
            xaxis_title="Timestamp",
            yaxis_title="Cumulative Holdings",
            legend_title="Legend",
        )

        # Store the figure in the dictionary with a key as the ticker name
        figures[f"fig_{symbol.lower()}_holdings"] = fig

    return figures


# Prepare figures
holdings_figures = prepare_holdings_plots(trade_history_df, activities_df)


app = Dash(__name__)

# App layout
app.layout = html.Div(
    children=[
        html.H1(children="Trading Analysis Dashboard", style={"textAlign": "center"}),
        html.Div(
            [
                html.H2("True Mid-Price and Bid-Ask Price/Volume Trends"),
                dcc.Graph(
                    id="price-trends-amethysts", figure=fig_price_trends_amethysts
                ),
                dcc.Graph(
                    id="price-trends-starfruit", figure=fig_price_trends_starfruit
                ),
                dcc.Graph(id="price-trends-orchids", figure=fig_price_trends_orchids),
                dcc.Graph(
                    id="bid-prices-volumes-amethysts",
                    figure=fig_bid_prices_volumes_amethysts,
                ),
                dcc.Graph(
                    id="bid-prices-volumes-starfruit",
                    figure=fig_bid_prices_volumes_starfruit,
                ),
                dcc.Graph(
                    id="bid-prices-volumes-orchids",
                    figure=fig_bid_prices_volumes_orchids,
                ),
            ],
            style={"padding": 10},
        ),
        html.Div(
            [
                html.H2("Strategy Analysis"),
                dcc.Graph(id="amethysts-pnl", figure=fig_amethysts_pnl),
                dcc.Graph(id="starfruit-pnl", figure=fig_starfruit_pnl),
                dcc.Graph(id="orchids-pnl", figure=fig_orchids_pnl),
                dcc.Graph(
                    id="amethysts-trades-made",
                    figure=trades_made_figures["fig_amethysts_trades_made"],
                ),
                dcc.Graph(
                    id="starfruit-trades-made",
                    figure=trades_made_figures["fig_starfruit_trades_made"],
                ),
                dcc.Graph(
                    id="orchids-trades-made",
                    figure=trades_made_figures["fig_orchids_trades_made"],
                ),
                dcc.Graph(
                    id="amethysts-holdings",
                    figure=holdings_figures["fig_amethysts_holdings"],
                ),
                dcc.Graph(
                    id="starfruit-holdings",
                    figure=holdings_figures["fig_starfruit_holdings"],
                ),
                dcc.Graph(
                    id="orchids-holdings",
                    figure=holdings_figures["fig_orchids_holdings"],
                ),
            ],
            style={"padding": 10},
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True, port="8081")
