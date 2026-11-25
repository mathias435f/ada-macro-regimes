
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("../data")
FACT = ["size", "value", "quality", "momentum"]
SPLIT = "1999-12-31"

ACTIONS = {
    "poids_egaux": [0.25, 0.25, 0.25, 0.25], "momentum": [0, 0, 0, 1],
    "value+quality": [0, 0.5, 0.5, 0], "size+momentum": [0.5, 0, 0, 0.5],
    "quality+momentum": [0, 0, 0.5, 0.5], "value+momentum": [0, 0.5, 0, 0.5],
}
A = np.array(list(ACTIONS.values()))

def load_returns():
    factors = pd.read_csv(DATA / "factors_monthly.csv", index_col=0, parse_dates=True)
    regimes = pd.read_csv(DATA / "regimes_monthly.csv", index_col=0, parse_dates=True)["regime"]
    df = (factors[FACT] / 100).join(regimes.shift(1).rename("regime_lag")).dropna()
    df["regime_lag"] = df["regime_lag"].astype(int)
    return df[df.index <= SPLIT], df[df.index > SPLIT]

def perf(r):
    cum = (1 + r).cumprod()
    return pd.Series({
        "rendement_cumule": cum.iloc[-1] - 1,
        "sharpe": r.mean() / r.std() * np.sqrt(12),
        "max_drawdown": (cum / cum.cummax() - 1).min(),
    })

def train_q(R, S, n_states, n_epochs=200, alpha=0.1, gamma=0.9, eps=0.2, risk=10, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, len(A)))
    for ep in range(n_epochs):
        for t in range(len(R) - 1):
            s = S[t]
            a = rng.integers(len(A)) if rng.random() < eps else Q[s].argmax()
            r = R[t] @ A[a]
            reward = r - 0.5 * risk * r**2
            Q[s, a] += alpha * (reward + gamma * Q[S[t + 1]].max() - Q[s, a])
    return Q

def evaluer(Q, data, utiliser_regime=True):
    S = data["regime_lag"].values if utiliser_regime else np.zeros(len(data), dtype=int)
    W = A[Q.argmax(axis=1)[S]]
    return pd.Series((W * data[FACT].values).sum(axis=1), index=data.index)