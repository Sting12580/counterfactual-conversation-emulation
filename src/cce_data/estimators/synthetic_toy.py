"""
Synthetic toy bandit for Counterfactual Conversation Emulation (Phase 4).

Purpose: a controlled world with known ground truth V(pi_target), used as
a unit-test fixture for the DM / MIPS / OffCEM estimators in W6-W8.

Notation map to the paper plan:
    x          -- patient context (here: discrete index 0..K-1)
    a          -- clinician/agent action (here: discrete index 0..A-1)
    phi(a)     -- action embedding in R^d (stand-in for sentence-BERT / OpenAI)
    pi_b       -- behavior policy (clinician), known here (unknown in real)
    pi_target  -- target policy (agent), known here
    Y(x, a)    -- reward function (stand-in for LLM rubric judge)

Design choices that mirror the real problem:
  - Actions cluster in embedding space; most reward variation lives in clusters.
    -> MIPS's "No Direct Effect" assumption is APPROXIMATELY true.
  - A small residual reward depends on the raw action index, not the cluster.
    -> NDE is slightly violated -> MIPS will have small bias.
    -> OffCEM should beat MIPS by capturing the residual via DM.
"""
from __future__ import annotations

import numpy as np

# ----------------------------- World definition ------------------------------
RNG = np.random.default_rng(0)
N_CONTEXTS = 5
N_ACTIONS = 10
EMB_DIM = 4
N_CLUSTERS = 3

ACTION_EMB = RNG.normal(size=(N_ACTIONS, EMB_DIM))
CLUSTER_OF = np.array([
    int(np.argmax(np.abs(ACTION_EMB[a, :N_CLUSTERS]))) for a in range(N_ACTIONS)
])

CLUSTER_MEAN = np.array([0.3, 0.6, 0.8])
RESIDUAL = 0.05 * RNG.normal(size=(N_CONTEXTS, N_ACTIONS))


def expected_reward(x: int, a: int) -> float:
    """E[Y | x, a]. Mostly cluster-driven, plus a small residual."""
    return float(CLUSTER_MEAN[CLUSTER_OF[a]] + RESIDUAL[x, a])


def sample_reward(x: int, a: int) -> float:
    """Observed reward = E[Y|x,a] + Gaussian noise, clipped to [0,1]."""
    y = expected_reward(x, a) + 0.05 * RNG.normal()
    return float(np.clip(y, 0.0, 1.0))


# ----------------------------- Policies --------------------------------------
def softmax(z: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = z / temp
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


PI_B = np.zeros((N_CONTEXTS, N_ACTIONS))
for x in range(N_CONTEXTS):
    logits = np.array([CLUSTER_MEAN[CLUSTER_OF[a]] for a in range(N_ACTIONS)])
    logits[CLUSTER_OF == 2] -= 0.4
    PI_B[x] = softmax(logits, temp=0.3)

PI_TARGET = np.zeros((N_CONTEXTS, N_ACTIONS))
for x in range(N_CONTEXTS):
    logits = np.array([CLUSTER_MEAN[CLUSTER_OF[a]] for a in range(N_ACTIONS)])
    PI_TARGET[x] = softmax(logits, temp=0.15)

CONTEXT_PRIOR = np.ones(N_CONTEXTS) / N_CONTEXTS


# ----------------------------- Ground truth ----------------------------------
def true_value(pi: np.ndarray) -> float:
    """V(pi) = E_x [ sum_a pi(a|x) * E[Y|x,a] ], computed analytically."""
    v = 0.0
    for x in range(N_CONTEXTS):
        for a in range(N_ACTIONS):
            v += CONTEXT_PRIOR[x] * pi[x, a] * expected_reward(x, a)
    return v


V_TRUE_B = true_value(PI_B)
V_TRUE_TARGET = true_value(PI_TARGET)


# ----------------------------- Logged data -----------------------------------
def sample_logged_data(n: int) -> list[tuple[int, int, float]]:
    """Returns [(x_i, a_i, y_i)] sampled i.i.d. under pi_b."""
    data = []
    for _ in range(n):
        x = int(RNG.choice(N_CONTEXTS, p=CONTEXT_PRIOR))
        a = int(RNG.choice(N_ACTIONS, p=PI_B[x]))
        y = sample_reward(x, a)
        data.append((x, a, y))
    return data


# ----------------------------- Oracle IPS sanity check -----------------------
def oracle_ips(data, pi_target: np.ndarray, pi_b: np.ndarray) -> float:
    """V_hat_IPS = mean_i [ pi_target(a_i|x_i) / pi_b(a_i|x_i) * y_i ].

    Uses the TRUE pi_b. Validates the toy world: if even oracle IPS can't
    recover V_TRUE_TARGET, the data generator is broken.
    """
    weighted = [pi_target[x, a] / pi_b[x, a] * y for x, a, y in data]
    return float(np.mean(weighted))
