import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from models.model import TD3
from models.portfolio_env import StockEnv
import glob
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Optimization - Inference", layout="wide")

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "NVDA",
    "ADBE",
    "CRM",
    "INTC",
    "AMD",
    "CSCO",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "BLK",
    "JNJ",
    "PFE",
    "UNH",
    "ABBV",
    "MRK",
    "AMZN",
    "WMT",
    "PG",
    "KO",
    "PEP",
    "XOM",
    "CVX",
    "COP",
    "CAT",
    "BA",
]


@st.cache_data
def load_model_list():
    """Get list of available models from models directory"""
    model_files = glob.glob("models/backup/*_actor")
    models = list(set([f.replace("_actor", "") for f in model_files]))
    return sorted(models)


@st.cache_data
def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data


def run_inference(model_path, env, initial_budget):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3(state_dim, action_dim, max_action, device=device)

    agent.load(model_path)

    state, _ = env.reset()
    done = False
    portfolio_values = []
    portfolio_weights = []

    while not done:
        action = agent.select_action(np.array(state))
        action = action / np.sum(action) if np.sum(action) > 0 else action
        portfolio_weights.append(action)

        state, reward, done, _, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])

    return portfolio_values, portfolio_weights


def main():
    st.title("Portfolio Optimization with TD3")
    st.write("Load a trained model and run inference on stock data")
    st.sidebar.header("Configuration")

    available_models = load_model_list()
    if not available_models:
        st.error("No trained models found in the 'models' directory!")
        return

    selected_model = st.sidebar.selectbox(
        "Select trained model", available_models, format_func=lambda x: os.path.basename(x)
    )

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    start_date = st.sidebar.date_input("Start date", start_date)
    end_date = st.sidebar.date_input("End date", end_date)

    if start_date >= end_date:
        st.error("Start date must be before end date!")
        return

    initial_budget = st.sidebar.number_input(
        "Initial Budget ($)", min_value=1000.0, max_value=1000000.0, value=10000.0, step=1000.0
    )

    st.sidebar.subheader("Selected Tickers")
    st.sidebar.write(", ".join(DEFAULT_TICKERS))

    if st.sidebar.button("Run Inference"):
        with st.spinner("Downloading stock data..."):
            data = download_data(DEFAULT_TICKERS, start_date, end_date)

        with st.spinner("Running inference..."):
            env = StockEnv(data, DEFAULT_TICKERS, initial_budget)
            portfolio_values, portfolio_weights = run_inference(selected_model, env, initial_budget)

        st.subheader("Portfolio Value Over Time")
        fig_portfolio_value = go.Figure(
            data=[
                go.Scatter(
                    x=list(range(len(portfolio_values))),
                    y=portfolio_values,
                )
            ]
        )
        fig_portfolio_value.update_layout(
            xaxis_title="Time Steps",
            yaxis_title="Portfolio Value ($)",
            height=600,
            width=400,
        )
        st.plotly_chart(fig_portfolio_value, use_container_width=True)

        total_return = (portfolio_values[-1] - initial_budget) / initial_budget * 100
        st.metric(
            "Total Return", f"{total_return:.2f}%", f"${portfolio_values[-1] - initial_budget:,.2f}"
        )

        st.subheader("Final Portfolio Allocation")
        final_weights = portfolio_weights[-1]

        threshold = 0.01
        significant_indices = final_weights > threshold
        significant_tickers = [
            DEFAULT_TICKERS[i] for i in range(len(DEFAULT_TICKERS)) if final_weights[i] > threshold
        ]
        significant_weights = final_weights[significant_indices]

        fig_portfolio_weights = go.Figure(
            data=[go.Pie(values=significant_weights, labels=significant_tickers)]
        )
        fig_portfolio_weights.update_layout(height=600, width=600)
        st.plotly_chart(fig_portfolio_weights, use_container_width=True)

        weight_data = {
            "Ticker": significant_tickers,
            "Weight": [f"{w*100:.2f}%" for w in significant_weights],
        }
        st.table(pd.DataFrame(weight_data))


if __name__ == "__main__":
    main()
