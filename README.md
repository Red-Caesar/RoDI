# Stake it 'til you make it

RoDI is a reinforcement learning approach to portfolio optimization.

## Inspiration

This project draws inspiration from the following research and implementations:

- [DDPG for stocks trading](https://arxiv.org/pdf/1811.07522): Practical Deep Reinforcement Learning Approach for Stock Trading
- [TD3 is better than DDPG](https://arxiv.org/pdf/1802.09477): Addressing Function Approximation Error in Actor-Critic Methods
- [Implementation of TD3](https://github.com/sfujim/TD3/tree/master).
- [DDPG + transformers](https://arxiv.org/pdf/2101.03138): Portfolio Optimization with 2D Relative-Attentional Gated Transformer

## Getting Started

### Local Deployment

We recommend using [uv](https://github.com/astral-sh/uv) for faster package installation:

```
uv pip install -e .
streamlit run rodi/app
```

### Docker Deployment

```
docker compose up --build
```

## Road Map
- [ ] Fix RL model and plots for training
- [ ] Add more models
