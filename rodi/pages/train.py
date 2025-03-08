import streamlit as st
import yfinance as yf
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from rodi.models import PROFILES, ModelProfile
from rodi.utils import DEFAULT_TICKERS
import pandas as pd
import os

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
st.set_page_config(page_title="Portfolio Optimization - Training", layout="wide")


@st.cache_data
def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data


def run_train(
    profile: ModelProfile,
    data: pd.DataFrame,
    initial_budget: int,
    max_timesteps: int,
    batch_size: int,
    exploration_noise: float,
    start_timesteps: int,
    eval_freq: int,
    save_model: int,
    progress_bar,
):
    env = profile.env(data, DEFAULT_TICKERS, initial_budget)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = profile.model(state_dim, action_dim, max_action, device=device)
    replay_buffer = profile.replay_buffer(state_dim, action_dim)

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    portfolio_values = []
    rewards = []

    state, _ = env.reset()

    for t in range(max_timesteps):
        episode_timesteps += 1

        progress_bar.progress((t + 1) / max_timesteps)
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                agent.select_action(np.array(state))
                + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            ).clip(0, 1)

            action = action / np.sum(action) if np.sum(action) > 0 else action

        next_state, reward, done, _, info = env.step(action)

        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state
        episode_reward += reward

        if "portfolio_value" in info:
            portfolio_values.append(info["portfolio_value"])

        if t >= start_timesteps:
            agent.train(replay_buffer, batch_size)

        if done:
            rewards.append(episode_reward)

            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % save_model == 0:
            prefix = profile.prefix_model_path
            model_path = f"rodi/models/backup/{prefix}_portfolio_{t+1}"
            agent.save(model_path)
            st.sidebar.success(f"Model saved: {model_path}")

    return portfolio_values, rewards


def main():
    st.title("Train Portfolio Optimization Model")
    st.write("Configure and train a new model for portfolio optimization")
    st.sidebar.header("Training Configuration")

    profile_name = st.sidebar.selectbox("Select model profile", list(PROFILES.keys()))
    max_timesteps = st.sidebar.number_input("Max Timesteps", 10, 1000000, 100000, 100)
    batch_size = st.sidebar.number_input("Batch Size", 16, 1024, 256, 16)
    exploration_noise = st.sidebar.slider("Exploration Noise", 0.0, 1.0, 0.1, 0.01)
    start_timesteps = st.sidebar.number_input("Start Timesteps", 10, 100000, 10000, 100)
    eval_freq = st.sidebar.number_input("Evaluation Frequency", 1, 50000, 5000, 100)
    save_model = st.sidebar.number_input("Save Model Frequency", 1, 50000, 20000, 100)

    initial_budget = st.sidebar.number_input(
        "Initial Budget ($)", min_value=1000, max_value=1000000, value=10000, step=1000
    )

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)

    start_date = st.sidebar.date_input("Start date", start_date)
    end_date = st.sidebar.date_input("End date", end_date)

    if start_date >= end_date:
        st.error("Start date must be before end date!")
        return

    Path("rodi/models/backup").mkdir(exist_ok=True)

    if st.sidebar.button("Start Training"):
        with st.spinner("Downloading stock data..."):
            data = download_data(DEFAULT_TICKERS, start_date, end_date)

        profile = PROFILES[profile_name]

        progress_bar = st.progress(0)
        st.write("Training progress:")

        portfolio_values, rewards = run_train(
            profile,
            data,
            initial_budget,
            max_timesteps,
            batch_size,
            exploration_noise,
            start_timesteps,
            eval_freq,
            save_model,
            progress_bar,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Portfolio Value Over Time")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(portfolio_values)
            ax1.set_xlabel("Time Steps")
            ax1.set_ylabel("Portfolio Value ($)")
            ax1.grid(True)
            st.pyplot(fig1)

        with col2:
            st.subheader("Episode Rewards")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(rewards)
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Total Reward")
            ax2.grid(True)
            st.pyplot(fig2)

        st.success("Training completed!")


if __name__ == "__main__":
    main()
