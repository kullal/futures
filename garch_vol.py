"""GARCH(1,1) Volatility Filter for BTC 4H bars (Ciaian et al. 2018).

Fits GARCH(1,1) on 4H log returns to forecast 1-step conditional volatility.
Used to filter out entries during extreme volatility spikes (e.g. >95th percentile).
"""
import numpy as np
import pandas as pd

try:
    from arch import arch_model
    _HAS_ARCH = True
except ImportError:
    _HAS_ARCH = False


def _manual_garch_vol(returns: np.ndarray) -> np.ndarray:
    """Fallback GARCH(1,1) conditional volatility via manual MLE / standard params."""
    from scipy.optimize import minimize

    # Scale returns to percent
    r = np.asarray(returns)
    var_0 = np.var(r) if np.var(r) > 0 else 1.0

    def nll(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e10
        h = var_0
        ll = 0.0
        for ret in r:
            ll += -0.5 * (np.log(2 * np.pi) + np.log(h) + (ret ** 2) / h)
            h = omega + alpha * (ret ** 2) + beta * h
        return -ll

    try:
        opt = minimize(
            nll, [0.05 * var_0, 0.05, 0.90],
            bounds=[(1e-6, None), (1e-6, 0.5), (0.5, 0.999)],
            method="L-BFGS-B",
        )
        omega, alpha, beta = opt.x
    except Exception:
        omega, alpha, beta = 0.05 * var_0, 0.07, 0.91

    # Generate 1-step forecast series
    h = var_0
    forecasts = np.empty_like(r)
    for i, ret in enumerate(r):
        h = omega + alpha * (ret ** 2) + beta * h
        forecasts[i] = np.sqrt(h)
    return forecasts


def compute_garch_vol(df: pd.DataFrame, p: int = 1, q: int = 1) -> pd.Series:
    """Compute 1-step ahead GARCH(p,q) conditional volatility forecast on 4H log returns.

    Returns volatility in percent per 4H bar, aligned with df index.
    """
    close = df["close"]
    log_ret = (np.log(close / close.shift(1)).fillna(0.0) * 100).to_numpy()

    if _HAS_ARCH:
        try:
            am = arch_model(log_ret, vol="Garch", p=p, q=q, dist="normal", rescale=False)
            res = am.fit(disp="off")
            omega = float(res.params["omega"])
            alpha = float(res.params["alpha[1]"])
            beta = float(res.params["beta[1]"])
            resid = np.asarray(res.resid)
            cond_vol = np.asarray(res.conditional_volatility)
            # 1-step ahead forecast: sigma_{t+1}^2 = omega + alpha * e_t^2 + beta * sigma_t^2
            f_var = omega + alpha * (resid ** 2) + beta * (cond_vol ** 2)
            vol_arr = np.sqrt(f_var)
            return pd.Series(vol_arr, index=df.index, name="garch_vol")
        except Exception:
            pass

    vol_arr = _manual_garch_vol(log_ret)
    return pd.Series(vol_arr, index=df.index, name="garch_vol")


def attach_garch_vol(df: pd.DataFrame, p: int = 1, q: int = 1) -> pd.DataFrame:
    """Attach 'garch_vol' column to df if not already present."""
    df_out = df.copy()
    if "garch_vol" not in df_out.columns:
        df_out["garch_vol"] = compute_garch_vol(df_out, p=p, q=q)
    return df_out


if __name__ == "__main__":
    from data import load_klines
    from strategy import to_df

    df_k = to_df(load_klines())
    v = compute_garch_vol(df_k)
    p95 = np.percentile(v.dropna(), 95)
    print(f"GARCH(1,1) on {len(v)} bars: mean={v.mean():.3f}%, median={v.median():.3f}%, 95th pct={p95:.3f}%")
