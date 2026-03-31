# -*- coding: utf-8 -*-

# =============================================================================
# EpiPivot: Learning to Control the Simplex Method under Epistemic Uncertainty
#
# This script implements the full EpiPivot framework, including:
#   - Size-invariant state feature extraction (25-dimensional)
#   - Trajectory-conditioned temporal self-attention encoder
#   - Epistemic Neural Network (ENN) with latent index variable
#   - Thompson sampling for uncertainty-aware pivot rule selection
#   - Supervised training via counterfactual rollout labels (Huber loss)
#
# Evaluation reports weighted pivot cost for EpiPivot vs. classical baselines
# (Largest Coefficient, Steepest Edge, Bland's Rule) on held-out LP instances.
#
# Output:
#   - Training progress printed to stdout
#   - Per-instance and summary results saved to CSV (epipivot_results.csv)
# =============================================================================

import copy
import time
import csv
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

# =============================================================================
# Hyperparameters
# =============================================================================

# Training problem size
M_TRAIN, N_TRAIN = 25, 50

# Number of LP instances: 100 training + 50 test (as described in Section 5.1)
NUM_TRAIN = 100
NUM_TEST  = 50

# Trajectory collection parameters
N_TRAJ_PER_PROBLEM = 25
MAX_DEPTH          = 70

# Rollout label parameters
ROLLOUT_T   = 100
HUBER_DELTA = 10.0

# Training parameters
EPOCHS     = 180
N_Z        = 8       # Number of ENN index samples per training step
BATCH_SIZE = 128
LR         = 3e-4

# Online evaluation
MAX_ONLINE_STEPS = 300

# History window length H (fixed to 4 based on ablation study, Section 5.3)
HISTORY_LEN = 4
STATE_DIM   = 25
RULE_DIM    = 3
TOKEN_DIM   = STATE_DIM + RULE_DIM

# Weighted per-step cost (Section 5.1):
# LC and Bland are assigned unit cost; Steepest Edge is assigned 1.15
# to reflect its additional per-iteration computational overhead.
RULE_NAMES     = {0: "LC", 1: "Steepest", 2: "Bland"}
RULE_STEP_COST = {0: 1.0, 1: 1.15, 2: 1.0}
FAIL_PENALTY   = float(2 * ROLLOUT_T * max(RULE_STEP_COST.values()))

GLOBAL_SEED = 101

# =============================================================================
# Utility Functions
# =============================================================================

def safe_quantile(x, q, default=0.0):
    """Compute quantile with a fallback default for empty arrays."""
    x = np.asarray(x)
    if x.size == 0:
        return float(default)
    return float(np.quantile(x, q))


def agg_stats(x):
    """Return summary statistics for a numeric array."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan,
                "p10": np.nan, "p90": np.nan}
    return {
        "mean":   float(np.mean(x)),
        "std":    float(np.std(x)),
        "median": float(np.median(x)),
        "p10":    float(np.quantile(x, 0.10)),
        "p90":    float(np.quantile(x, 0.90)),
    }


# =============================================================================
# Section 3.2: Size-Invariant State Representation (25-dimensional)
# =============================================================================

def extract_state_features(tab, steps):
    """
    Extract a fixed-dimensional, size-invariant feature vector from the
    current simplex tableau. The resulting 25-dimensional vector captures
    reduced cost statistics, right-hand-side properties, constraint matrix
    geometry, and joint rule-sensitive features. See Appendix A for full
    feature definitions.

    Parameters
    ----------
    tab   : np.ndarray, shape (m+1, n+m+1) — current simplex tableau
    steps : int — number of pivot steps performed so far

    Returns
    -------
    feats : np.ndarray, shape (25,)
    """
    rc   = tab[-1, :-1]
    rhs  = tab[:-1, -1]
    Abar = tab[:-1, :-1]

    improv_mask = (rc < -1e-8)
    cand        = np.where(improv_mask)[0]

    # --- Reduced cost features (7) ---
    num_improv  = float(np.sum(improv_mask))
    frac_improv = float(np.mean(improv_mask)) if rc.size > 0 else 0.0
    min_rc      = float(np.min(rc))           if rc.size > 0 else 0.0
    mean_rc     = float(np.mean(rc))          if rc.size > 0 else 0.0
    std_rc      = float(np.std(rc))           if rc.size > 0 else 0.0
    mean_abs_rc = float(np.mean(np.abs(rc)))  if rc.size > 0 else 0.0
    rc_gap      = float(np.sort(rc)[1] - np.sort(rc)[0]) if rc.size >= 2 else 0.0

    # --- Right-hand-side and degeneracy features (4) ---
    min_rhs   = float(np.min(rhs))              if rhs.size > 0 else 0.0
    mean_rhs  = float(np.mean(rhs))             if rhs.size > 0 else 0.0
    std_rhs   = float(np.std(rhs))              if rhs.size > 0 else 0.0
    frac_degen= float(np.mean(rhs <= 1e-6))     if rhs.size > 0 else 0.0

    # --- Geometric features of the constraint matrix (6) ---
    if Abar.size > 0:
        col_norms = np.linalg.norm(Abar, axis=0) + 1e-12
        row_norms = np.linalg.norm(Abar, axis=1) + 1e-12
    else:
        col_norms = np.array([1.0])
        row_norms = np.array([1.0])

    col_norm_mean = float(np.mean(col_norms))
    col_norm_std  = float(np.std(col_norms))
    col_norm_q90  = safe_quantile(col_norms, 0.9)
    row_norm_mean = float(np.mean(row_norms))
    row_norm_std  = float(np.std(row_norms))
    row_norm_q90  = safe_quantile(row_norms, 0.9)

    # --- Optimization context features (2) ---
    obj_const = float(tab[-1, -1])
    steps_f   = float(steps)

    # --- Joint rule-aware and feasibility features (6) ---
    if cand.size > 0 and Abar.size > 0:
        score      = rc[cand] / col_norms[cand]
        score_min  = float(np.min(score))
        score_mean = float(np.mean(score))
        score_gap  = float(np.sort(score)[1] - np.sort(score)[0]) if score.size >= 2 else 0.0

        pos_fracs = [float(np.mean(Abar[:, j] > 1e-12))
                     for j in cand[:min(10, cand.size)]]
        cand_pos_frac_mean = float(np.mean(pos_fracs))

        x = np.abs(rc).astype(float);  x = x - x.mean()
        y = col_norms.astype(float);   y = y - y.mean()
        denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12
        corr_absrc_norm = float(np.dot(x, y) / denom) if rc.size > 1 else 0.0
    else:
        score_min = score_mean = score_gap = cand_pos_frac_mean = corr_absrc_norm = 0.0

    feats = np.array([
        num_improv, frac_improv, min_rc, mean_rc, std_rc, mean_abs_rc, rc_gap,
        min_rhs, mean_rhs, std_rhs, frac_degen,
        col_norm_mean, col_norm_std, col_norm_q90,
        row_norm_mean, row_norm_std, row_norm_q90,
        obj_const, steps_f,
        score_min, score_mean, score_gap,
        cand_pos_frac_mean, corr_absrc_norm, float(cand.size),
    ], dtype=np.float32)

    # Guard against numerical instabilities in degenerate instances
    return np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


# =============================================================================
# Section 5.1: LP Instance Generators
# =============================================================================

def make_lp_standard(rng, m, n, rhs_range=(5, 20), coef_hi=4):
    """Generate a standard random LP instance."""
    A = rng.integers(0, coef_hi, size=(m, n)).astype(float)
    for j in range(n):
        if (A[:, j] == 0).all():
            A[rng.integers(0, m), j] = 1
    b = rng.integers(*rhs_range, size=m).astype(float)
    c = rng.integers(1, 6, size=n).astype(float)
    return A, b, c


def make_lp_hard(rng, m, n, rhs_range=(5, 20), coef_hi=4):
    """
    Generate a structurally challenging LP instance with high sparsity,
    strong degeneracy, and nearly dependent rows.
    """
    A = rng.integers(0, coef_hi, size=(m, n)).astype(float)
    for j in range(n):
        if (A[:, j] == 0).all():
            A[rng.integers(0, m), j] = 1.0

    # Apply sparsity
    mask = rng.random((m, n)) < 0.85
    A    = A * (~mask)
    for j in range(n):
        if (A[:, j] == 0).all():
            A[rng.integers(0, m), j] = 1.0

    # Introduce degeneracy
    b = rng.integers(rhs_range[0], rhs_range[1], size=m).astype(float)
    k = int(np.round(0.50 * m))
    if k > 0:
        idx     = rng.choice(m, size=k, replace=False)
        b[idx]  = 0.0

    # Introduce nearly dependent rows
    kdup = int(np.round(0.35 * m))
    if kdup > 0:
        src       = rng.choice(m, size=kdup, replace=True)
        dst       = rng.choice(m, size=kdup, replace=False)
        A[dst, :] = A[src, :] + rng.normal(0, 1e-3, size=(kdup, n))
        b[dst]    = b[src]

    # Normalize column norms
    col_norms     = np.linalg.norm(A, axis=0) + 1e-8
    scale         = np.clip(np.median(col_norms) / col_norms, 0.5, 2.0)
    A             = A * scale[None, :]

    c = rng.integers(1, 6, size=n).astype(float)
    return A, b, c


def make_lp_hard2(rng, m, n, rhs_range=(5, 20), coef_hi=4):
    """
    Generate a highly structured LP instance with block-correlated columns,
    log-normal column scaling, strong degeneracy, and near-dependent structure.
    """
    A = rng.integers(0, coef_hi, size=(m, n)).astype(float)
    for j in range(n):
        if (A[:, j] == 0).all():
            A[rng.integers(0, m), j] = 1.0

    # Sparsity
    mask = rng.random((m, n)) < 0.80
    A    = A * (~mask)
    for j in range(n):
        if (A[:, j] == 0).all():
            A[rng.integers(0, m), j] = 1.0

    # Degeneracy
    b = rng.integers(rhs_range[0], rhs_range[1], size=m).astype(float)
    k = int(np.round(0.55 * m))
    if k > 0:
        idx    = rng.choice(m, size=k, replace=False)
        b[idx] = 0.0

    # Near-dependent rows
    kdup = int(np.round(0.30 * m))
    if kdup > 0:
        src       = rng.choice(m, size=kdup, replace=True)
        dst       = rng.choice(m, size=kdup, replace=False)
        A[dst, :] = A[src, :] + rng.normal(0, 1e-4, size=(kdup, n))
        b[dst]    = b[src]

    # Block-correlated column structure
    nb = max(0, min(int(np.round(0.40 * n)), n))
    if nb >= 2:
        cols        = rng.choice(n, size=nb, replace=False)
        n_templates = max(1, nb // 5)
        templates   = []
        for _ in range(n_templates):
            t      = rng.integers(0, coef_hi, size=(m,)).astype(float)
            t_mask = rng.random((m,)) < 0.80
            t      = t * (~t_mask)
            if (t == 0).all():
                t[rng.integers(0, m)] = 1.0
            templates.append(t)
        templates = np.stack(templates, axis=1)
        assign    = rng.integers(0, n_templates, size=nb)
        for idx_c, j in enumerate(cols):
            A[:, j] = np.clip(
                templates[:, assign[idx_c]] + rng.normal(0, 0.10, size=(m,)),
                0.0, None
            )
            if (A[:, j] == 0).all():
                A[rng.integers(0, m), j] = 1.0

    # Near-dependent columns
    for _ in range(4):
        j1 = int(rng.integers(0, n))
        j2 = (j1 + 1) % n
        A[:, j2] = np.clip(A[:, j1] + rng.normal(0, 1e-4, size=(m,)), 0.0, None)
        if (A[:, j2] == 0).all():
            A[rng.integers(0, m), j2] = 1.0

    # Log-normal column scaling
    scales = np.clip(np.exp(rng.normal(0.0, 1.0, size=(n,))), 0.05, 20.0)
    A      = A * scales[None, :]

    c = rng.integers(1, 6, size=n).astype(float)
    return A, b, c


def sample_lp_mixed(rng, m, n):
    """
    Sample an LP instance from the mixed training distribution (Section 5.1):
    50% standard, 25% hard, 25% hard2.
    """
    u = float(rng.random())
    if u < 0.50:
        return make_lp_standard(rng, m, n)
    elif u < 0.75:
        return make_lp_hard(rng, m, n)
    else:
        return make_lp_hard2(rng, m, n)


# =============================================================================
# Simplex Environment
# =============================================================================

class SimplexEnv:
    """
    Primal simplex method environment for standard-form LP:
        max c^T x  s.t. Ax <= b, x >= 0.

    Slack variables are added automatically to form the initial basis.
    Supports three pivot rules: LC (0), Steepest Edge (1), Bland (2).
    """

    def __init__(self, A, b, c):
        self.A, self.b, self.c = map(np.asarray, (A, b, c))
        self.m, self.n = self.A.shape
        self.reset()

    def reset(self):
        A_slack      = np.hstack([self.A, np.eye(self.m)])
        self.tab     = np.zeros((self.m + 1, self.n + self.m + 1))
        self.tab[:-1, :-1] = A_slack
        self.tab[:-1, -1]  = self.b
        self.tab[-1, :self.n] = -self.c
        self.basis   = list(range(self.n, self.n + self.m))
        self.steps   = 0

    def get_state(self):
        return extract_state_features(self.tab, self.steps)

    def is_optimal(self):
        return (self.tab[-1, :-1] >= -1e-8).all()

    def step(self, rule, eps=1e-8):
        """
        Perform one simplex pivot using the specified rule.
        Returns True if the pivot was successful, False otherwise.
        """
        rc   = self.tab[-1, :-1]
        cand = np.where(rc < -1e-8)[0]
        if cand.size == 0:
            return False

        # Select entering variable
        if rule == 0:    # Largest Coefficient (LC)
            enter = int(cand[np.argmin(rc[cand])])
        elif rule == 1:  # Steepest Edge
            norms = np.linalg.norm(self.tab[:-1, :-1], axis=0) + 1e-8
            enter = int(cand[np.argmin(rc[cand] / norms[cand])])
        elif rule == 2:  # Bland's rule
            enter = int(cand[0])
        else:
            raise ValueError(f"Unknown pivot rule: {rule}")

        # Ratio test to select leaving variable
        col = self.tab[:-1, enter]
        pos = col > eps
        if not np.any(pos):
            return False

        ratio        = np.full(self.m, np.inf)
        ratio[pos]   = self.tab[:-1, -1][pos] / col[pos]
        leave        = int(np.argmin(ratio))
        if not np.isfinite(ratio[leave]):
            return False

        piv = self.tab[leave, enter]
        if (not np.isfinite(piv)) or abs(piv) <= 1e-9:
            return False

        # Pivot operation
        self.steps     += 1
        self.tab[leave] /= piv
        for i in range(self.m + 1):
            if i != leave:
                self.tab[i] -= self.tab[i, enter] * self.tab[leave]

        if not np.isfinite(self.tab).all():
            return False

        self.tab       = np.clip(self.tab, -1e12, 1e12)
        self.basis[leave] = enter
        return True


# =============================================================================
# Section 3.4: Rollout Labels via Counterfactual Simulation
# =============================================================================

def compute_rollout_label(env, rule, rollout_t, fail_penalty):
    """
    Estimate the finite-horizon weighted pivot cost C^tau(s_t, a) by
    simulating rule `rule` forward for up to `rollout_t` steps from a
    deep copy of `env`. Returns `fail_penalty` if the pivot fails.
    """
    env_copy = copy.deepcopy(env)
    cost     = 0.0
    for _ in range(rollout_t):
        if env_copy.is_optimal():
            return float(cost)
        ok = env_copy.step(rule)
        if not ok:
            return float(fail_penalty)
        cost += float(RULE_STEP_COST[int(rule)])
    return float(cost)


# =============================================================================
# Section 3.3.1: Trajectory Token Construction
# =============================================================================

def build_history_tokens(history_states, rule, history_len=HISTORY_LEN):
    """
    Construct a right-aligned token sequence of length `history_len` from
    the recent state history. The candidate pivot rule is injected as a
    one-hot vector at the last (current) token position.

    Returns
    -------
    tokens : np.ndarray, shape (history_len, TOKEN_DIM)
    mask   : np.ndarray, shape (history_len,) — 1 for real tokens, 0 for padding
    """
    tail   = history_states[-history_len:]
    T_real = len(tail)

    hist  = [tail[-1]] * history_len
    mask  = np.zeros(history_len, dtype=np.float32)
    start = history_len - T_real
    hist[start:] = tail
    mask[start:] = 1.0

    tokens = []
    for t in range(history_len):
        s        = np.asarray(hist[t], dtype=np.float32)
        rule_vec = np.eye(3, dtype=np.float32)[rule] if t == history_len - 1 \
                   else np.zeros(3, dtype=np.float32)
        tokens.append(np.concatenate([s, rule_vec]))

    return np.stack(tokens).astype(np.float32), mask


# =============================================================================
# Training Dataset Construction
# =============================================================================

def build_trajectory_dataset(bank, n_traj_per_problem, history_len, max_depth, seed):
    """
    Collect training groups by performing random-action rollouts on each LP
    instance. At each visited state, a counterfactual snapshot is stored for
    all three pivot rules. Labels are computed separately via precomputation.
    """
    rng    = np.random.default_rng(seed)
    groups = []

    for A, b, c in bank:
        for _ in range(n_traj_per_problem):
            env            = SimplexEnv(A, b, c)
            history_states = [env.get_state()]

            for _ in range(max_depth):
                env_snapshot = copy.deepcopy(env)
                group        = []
                for r in range(3):
                    tokens, mask = build_history_tokens(history_states, r, history_len)
                    group.append((tokens, mask, env_snapshot))
                groups.append(group)

                if env.is_optimal():
                    break

                ok = env.step(int(rng.integers(0, 3)))
                if not ok:
                    break

                history_states.append(env.get_state())
                if len(history_states) > history_len:
                    history_states = history_states[-history_len:]

    return groups


def precompute_labels(groups, rollout_t, fail_penalty):
    """
    Precompute rollout labels y = [C^tau_LC, C^tau_Steepest, C^tau_Bland]
    for all training groups. Labels are computed once before training to
    avoid redundant simulation during the training loop.
    """
    all_y = []
    for g in groups:
        env  = copy.deepcopy(g[0][2])
        y    = [compute_rollout_label(env, r, rollout_t, fail_penalty) for r in range(3)]
        all_y.append(y)
    return np.asarray(all_y, dtype=np.float32)


# =============================================================================
# Section 3.3: Temporal Attention ENN
# =============================================================================

class TemporalAttnENN:
    """
    Trajectory-conditioned cost predictor combining:
      - Positional embedding + masked multi-head self-attention
      - Masked mean pooling over the attended sequence
      - ENN head: concatenation with latent index z, followed by MLP

    Input
    -----
    x    : (B, T, TOKEN_DIM) — token sequence
    mask : (B, T)            — binary padding mask (1 = real, 0 = padded)
    z    : (index_dim,)      — ENN latent index variable

    Output
    ------
    Scalar cost prediction per input, shape (B,)
    """

    def __init__(self, index_dim=8, d_model=64, num_heads=4, key_size=16):
        self.index_dim = index_dim
        self.d_model   = d_model
        self.num_heads = num_heads
        self.key_size  = key_size

        def net(x, mask, z):
            B, T, _ = x.shape

            # Token projection and positional embedding
            h = hk.Linear(self.d_model, name="tok_proj")(x)
            pos_emb = hk.get_parameter(
                "pos_emb",
                shape=(HISTORY_LEN, self.d_model),
                init=hk.initializers.TruncatedNormal(stddev=0.02),
            )
            h = h + pos_emb[None, :T, :]

            # Zero out padded tokens before attention
            h = h * mask[..., None]

            # Attention mask: block attention to and from padded positions
            attn_mask = (mask[:, :, None] * mask[:, None, :] > 0.5)[:, None, :, :]

            mha = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=self.d_model,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                name="self_attn",
            )
            h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                             name="ln")(h + mha(h, h, h, mask=attn_mask))

            # Zero out padded positions after attention to prevent leakage
            h = h * mask[..., None]

            # Masked mean pooling
            m      = mask.astype(jnp.float32)[..., None]
            h_pool = jnp.sum(h * m, axis=1) / (jnp.sum(m, axis=1) + 1e-8)

            # ENN head: concatenate trajectory embedding with latent index
            h_cat = jnp.concatenate(
                [h_pool, jnp.broadcast_to(z, (B, self.index_dim))], axis=-1
            )
            return hk.nets.MLP([64, 64, 1], name="head_mlp")(h_cat).squeeze(-1)

        self._net = hk.transform_with_state(net)

    def indexer(self, rng):
        """Sample a latent index z ~ N(0, I)."""
        return jax.random.normal(rng, shape=(self.index_dim,))

    def init(self, rng, x, mask, z):
        return self._net.init(rng, x, mask, z)

    def apply(self, params, state, x, mask, z):
        return self._net.apply(params, state, None, x, mask, z)


# =============================================================================
# Section 3.4: Training
# =============================================================================

def train(enn_net, groups, group_labels, epochs, batch_size, lr, n_z, seed):
    """
    Train the ENN using Huber loss averaged over multiple index samples.
    Labels are precomputed rollout costs; no simulation occurs during training.
    """
    np_rng = np.random.default_rng(seed)
    rng    = jax.random.PRNGKey(seed)

    # Initialize parameters
    dummy_x = jnp.array(groups[0][0][0])[None, :, :]
    dummy_m = jnp.array(groups[0][0][1])[None, :]
    rng, k1, k2 = jax.random.split(rng, 3)
    params, state = enn_net.init(k1, dummy_x, dummy_m, enn_net.indexer(k2))

    opt       = optax.adam(lr)
    opt_state = opt.init(params)

    def huber(x, delta):
        absx = jnp.abs(x)
        quad = jnp.minimum(absx, delta)
        return 0.5 * quad ** 2 + delta * (absx - quad)

    @jax.jit
    def train_step(params, state, opt_state, xb, mb, yb, rng):
        rng, z_key = jax.random.split(rng)
        z_batch    = jax.vmap(enn_net.indexer)(jax.random.split(z_key, n_z))

        def loss_fn(p):
            def f_one_z(z):
                preds, _ = enn_net.apply(p, state, xb, mb, z)
                return preds.reshape(-1, 3)
            preds_z = jax.vmap(f_one_z)(z_batch)       # (n_z, B, 3)
            return jnp.mean(huber(preds_z - yb[None], HUBER_DELTA))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        _, new_state = enn_net.apply(params, state, xb, mb, z_batch[0])
        return params, new_state, opt_state, rng, loss

    num_groups = len(groups)
    print(f"[Train] Training on {num_groups} decision points for {epochs} epochs.")

    for ep in range(epochs):
        idx = np_rng.permutation(num_groups)
        last_loss = None

        for i in range(0, num_groups, batch_size):
            b_idx = idx[i: i + batch_size]
            if len(b_idx) == 0:
                continue

            x_batch, m_batch, y_batch = [], [], []
            for j in b_idx:
                for r in range(3):
                    x_batch.append(groups[j][r][0])
                    m_batch.append(groups[j][r][1])
                y_batch.append(group_labels[j])

            xb = jnp.array(x_batch)
            mb = jnp.array(m_batch)
            yb = jnp.array(y_batch)

            params, state, opt_state, rng, loss = train_step(
                params, state, opt_state, xb, mb, yb, rng
            )
            last_loss = float(loss)

        if ep % 40 == 0:
            print(f"  epoch {ep:03d}  loss = {last_loss:.4f}")

    return params, state


# =============================================================================
# Section 3.3.4: Online Pivot-Rule Selection (Thompson Sampling)
# =============================================================================

def run_epipivot(env, enn_net, params, state, rng, max_steps=MAX_ONLINE_STEPS):
    """
    Run EpiPivot on a given LP instance using Thompson sampling.
    At each iteration, one latent index z is sampled and the pivot rule
    minimizing the predicted cost under z is selected (Algorithm 2).
    """
    history_states = []
    total_cost     = 0.0
    pivot_time     = 0.0
    t0_total       = time.time()

    while not env.is_optimal() and env.steps < max_steps:
        history_states.append(env.get_state())
        if len(history_states) > HISTORY_LEN:
            history_states = history_states[-HISTORY_LEN:]

        # Sample one latent index z (Thompson sampling)
        rng, z_key = jax.random.split(rng)
        z = enn_net.indexer(z_key)

        # Build token sequences for all three candidate rules
        x_list = []
        m_list = []
        for r in range(3):
            tokens, mask = build_history_tokens(history_states, r)
            x_list.append(tokens)
            m_list.append(mask)

        x3 = jnp.array(x_list)
        m3 = jnp.array(m_list)

        preds, _ = enn_net.apply(params, state, x3, m3, z)
        costs    = jnp.clip(preds.reshape(-1), 0.0, FAIL_PENALTY)
        rule     = int(jnp.argmin(costs))

        total_cost += float(RULE_STEP_COST[rule])
        t0          = time.time()
        ok          = env.step(rule)
        pivot_time += time.time() - t0

        if not ok:
            total_cost = float(FAIL_PENALTY)
            break

    return {
        "steps":      int(env.steps),
        "cost":       float(total_cost),
        "success":    bool(env.is_optimal()),
        "total_time": float(time.time() - t0_total),
        "pivot_time": float(pivot_time),
    }


def run_fixed_rule(env, rule, max_steps=MAX_ONLINE_STEPS):
    """Run a fixed classical pivot rule as a baseline."""
    total_cost = 0.0
    pivot_time = 0.0
    t0_total   = time.time()

    while not env.is_optimal() and env.steps < max_steps:
        total_cost += float(RULE_STEP_COST[int(rule)])
        t0          = time.time()
        ok          = env.step(int(rule))
        pivot_time += time.time() - t0
        if not ok:
            total_cost = float(FAIL_PENALTY)
            break

    return {
        "steps":      int(env.steps),
        "cost":       float(total_cost),
        "success":    bool(env.is_optimal()),
        "total_time": float(time.time() - t0_total),
        "pivot_time": float(pivot_time),
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================

def main():
    rng = np.random.default_rng(GLOBAL_SEED)

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    print("[Data] Generating LP instances...")
    train_bank = [sample_lp_mixed(rng, M_TRAIN, N_TRAIN) for _ in range(NUM_TRAIN)]
    test_bank  = [sample_lp_mixed(rng, M_TRAIN, N_TRAIN) for _ in range(NUM_TEST)]
    print(f"[Data] {NUM_TRAIN} training instances, {NUM_TEST} test instances generated.")

    print("[Data] Building trajectory dataset...")
    groups = build_trajectory_dataset(
        train_bank,
        n_traj_per_problem=N_TRAJ_PER_PROBLEM,
        history_len=HISTORY_LEN,
        max_depth=MAX_DEPTH,
        seed=GLOBAL_SEED,
    )

    print("[Data] Precomputing rollout labels...")
    group_labels = precompute_labels(groups, ROLLOUT_T, FAIL_PENALTY)
    print(f"[Data] Dataset ready: {len(groups)} decision points.\n")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    enn_net      = TemporalAttnENN(index_dim=N_Z)
    params, state = train(
        enn_net, groups, group_labels,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, n_z=N_Z, seed=GLOBAL_SEED,
    )
    print("\n[Train] Training complete.\n")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    test_modes = ["base"]
    results    = []

    for mode in test_modes:
        # Generate test instances for this distribution
        if mode == "base":
            instances = [make_lp_standard(rng, M_TRAIN, N_TRAIN) for _ in range(NUM_TEST)]
        elif mode == "hard2":
            instances = [make_lp_hard2(rng, M_TRAIN, N_TRAIN)    for _ in range(NUM_TEST)]
        else:
            instances = [make_lp_hard(rng, M_TRAIN, N_TRAIN)     for _ in range(NUM_TEST)]

        epi_cost, epi_steps, epi_succ = [], [], []
        base_cost = {r: [] for r in range(3)}
        base_succ = {r: [] for r in range(3)}
        base_steps= {r: [] for r in range(3)}

        rng_key = jax.random.PRNGKey(GLOBAL_SEED + 1)

        for lp_id, (A, b, c) in enumerate(instances, 1):
            rng_key, subkey = jax.random.split(rng_key)

            # EpiPivot (H = 4, Thompson sampling)
            res = run_epipivot(SimplexEnv(A, b, c), enn_net, params, state, subkey)
            epi_cost.append(res["cost"])
            epi_steps.append(res["steps"])
            epi_succ.append(int(res["success"]))

            results.append(dict(
                dist=mode, lp_id=lp_id, method="EpiPivot",
                cost=res["cost"], steps=res["steps"], success=int(res["success"]),
                pivot_time=res["pivot_time"],
            ))

            # Classical baselines
            for r in range(3):
                resb = run_fixed_rule(SimplexEnv(A, b, c), r)
                base_cost[r].append(resb["cost"])
                base_steps[r].append(resb["steps"])
                base_succ[r].append(int(resb["success"]))
                results.append(dict(
                    dist=mode, lp_id=lp_id, method=RULE_NAMES[r],
                    cost=resb["cost"], steps=resb["steps"], success=int(resb["success"]),
                    pivot_time=resb["pivot_time"],
                ))

        # Print summary for this distribution
        print(f"=== Results: (m,n)=({M_TRAIN},{N_TRAIN}) | dist={mode} ===")
        print(f"  {'Method':<12}  {'Cost (mean)':>12}  {'Cost (std)':>10}  "
              f"{'Steps':>7}  {'Success':>8}")
        print(f"  {'-'*56}")

        s = agg_stats(epi_cost)
        print(f"  {'EpiPivot':<12}  {s['mean']:>12.2f}  {s['std']:>10.2f}  "
              f"{np.mean(epi_steps):>7.1f}  {100*np.mean(epi_succ):>7.1f}%")

        for r in range(3):
            s = agg_stats(base_cost[r])
            print(f"  {RULE_NAMES[r]:<12}  {s['mean']:>12.2f}  {s['std']:>10.2f}  "
                  f"{np.mean(base_steps[r]):>7.1f}  {100*np.mean(base_succ[r]):>7.1f}%")
        print()


if __name__ == "__main__":
    main()
