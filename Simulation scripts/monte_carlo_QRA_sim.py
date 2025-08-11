"""
FAST Chama Monte-Carlo Driver
=============================
Key changes (⚡ = performance win)

1. ⚡ Move *all* BLAS-thread env-vars **before** NumPy/SciPy import.
2. ⚡ Use `threadpoolctl` inside each worker to enforce 1 math-thread.
3. ⚡ Schedule **one worker-process per block of hours** (≈ len(hours)/n_jobs),
   so each process re-uses its imported modules and RAM.
4. ⚡ Inside a worker: generate leak rates once per hour, loop samples *serially*
   (no extra spawn), then **write that hour’s result straight to CSV** named
   `<YYYYMMDD_HHMM>.csv` → minimal IPC / memory.  The parent prints the file
   list; concatenate later if you wish.
"""

# ----------  1. LIMIT INNER THREADING (must be before NumPy import) ----------
import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

# ----------  2. Standard imports ---------------------------------------------
import math
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from typing import List, Optional, Sequence, Tuple

from simulation_modified import Grid, Source, GaussianPuff   # heavy code
from scipy.stats import norm

def generate_mc_leak_rates(
    tau=1081.1,
    t0=210,
    duration=3600,
    dt=10,
    n_samples=1000,
    *,
    A0_hat=0.001347103726986978,  # user estimate for triangular mode
    renormalize=False,
    eps=1e-12,
):
    """
    Monte Carlo leak-rate profiles: plateau A0 then exponential decay.

    A0 ~ Triangular(low=A0_hat/2, mode=A0_hat, high=1/t0).
    (The high endpoint is slightly shrunk by `eps` to avoid roundoff
     that could make A1 negative.)

    Parameters
    ----------
    tau : float
        Decay time constant (s).
    t0 : float
        Plateau duration (s).
    duration : int
        Total profile duration (s).
    dt : int
        Time step (s).
    n_samples : int
        Number of MC samples.
    A0_hat : float
        Mode of triangular distribution.
    random_seed : int or None
        RNG seed.
    renormalize : bool
        If True, scale each discrete profile so ∑ rate*dt = 1.
    eps : float
        Fractional shrink for the high endpoint to keep A1 ≥ 0 numerically.

    Returns
    -------
    rates : ndarray, shape (n_samples, n_steps)
        Leak‑rate profiles (fraction per second).
        (Callers expecting only `rates` continue to work.)
    """

    # time grid (inclusive of duration)
    t = np.arange(0, duration + dt, dt)
    n_steps = t.size

    # triangular params
    low  = 0.5 * A0_hat
    mode = A0_hat
    high = (1.0 / t0) * (1.0 - eps)  # slight shrink for numeric safety

    # ensure ordering (in case user passes unusual A0_hat)
    if not (low < mode < high):
        raise ValueError(f"Triangular params invalid: low={low}, mode={mode}, high={high}.")

    # sample plateau rates
    A0_samples = np.random.default_rng().triangular(low, mode, high, size=n_samples)

    # decay amplitude (continuous mass balance assumption)
    A1_samples = (1.0 - A0_samples * t0) / tau
    A1_samples = np.clip(A1_samples, 0.0, None)  # numeric guard

    # build profiles
    t_mat  = t[None, :]              # (1, n_steps)
    A0_mat = A0_samples[:, None]     # (n_samples, 1)
    A1_mat = A1_samples[:, None]     # (n_samples, 1)

    plateau_mask = (t_mat <= t0)     # (1, n_steps)
    rates = np.empty((n_samples, n_steps), dtype=float)

    # plateau region
    rates[:, plateau_mask[0]] = A0_mat

    # decay region
    decay_times = t_mat - t0         # ≤0 before t0, >0 after
    post_mask = ~plateau_mask[0]
    rates[:, post_mask] = A1_mat * np.exp(-decay_times[:, post_mask] / tau)

    # discrete renormalization (accounts for truncation at `duration`)
    if renormalize:
        mass = rates.sum(axis=1) * dt
        mass = np.where(mass > 0, mass, 1.0)
        rates /= mass[:, None]

    return rates


def dose_approximation(conc_mod, n_pr=2):#, beta_pr=0.75, alpha_pr=-9.4):
    """
    conc_mod: DataFrame with columns ['x','y',...,'T','C'] at least.
    T is sorted ascendingly for each spatial group.
    """
    # Compute ∫ C(t)^n dt per (x, y, ...) group using trapezoidal rule
    trap_area = (
        conc_mod
        .groupby(['X', 'Y'])[['S', 'T']]  # explicitly specify
        .apply(lambda group: np.trapz((group['S'] * 10**6) ** n_pr, x=group['T']))
        .reset_index(name='integral')
    )
    result = trap_area.reset_index()
    return result


def dose_to_probit(integral_df,beta_pr=0.75, alpha_pr=-9.4):
    """
    integral_df: DataFrame with columns ['X', 'Y', 'integral'].
    Returns a DataFrame with probit values.
    """
    integral_df['probit'] = (
        beta_pr * np.log(integral_df['integral']) + alpha_pr
    )
    return integral_df

def probit_to_probability(integral_df):
    """
    integral_df: DataFrame with columns ['X', 'Y', 'integral', 'probit'].
    Returns a DataFrame with probability values.
    """
    integral_df['probability'] = (
        norm.cdf(integral_df['probit'])
    )
    return integral_df

def simulation_using_different_inputs(wind_direction, wind_speed, stability_class,rate_vector):
    # grid -----------------------------------------------------------------
    x_grid = np.linspace(-1000, 1000, 200)
    y_grid = np.linspace(-1000, 1000, 200)
    z_grid = np.array([1.75])
    grid_mod = Grid(x_grid, y_grid, z_grid)

    source_mod  = Source(0, 0, 1, rate_vector)

    # met data -------------------------------------------------------------
    atm = pd.DataFrame({'Wind Direction': [wind_direction],
                        'Wind Speed':     [wind_speed],
                        'Stability Class':[stability_class]},
                    index=[0])

    # run patched Gaussian-puff -------------------------------------------
    gpuff_mod = GaussianPuff(grid_mod, source_mod, atm, tpuff=10, tend=60*60)
    gpuff_mod.run(grid_mod, 10)            # 10-s reporting step
    result = gpuff_mod.conc
    result = dose_approximation(result)
    result = dose_to_probit(result)
    result = probit_to_probability(result)
    return result



# ----------  3. YOUR EXISTING HELPER FNS (unchanged) -------------------------
# ... (generate_mc_leak_rates, dose_approximation, etc.) ...

# keep generate_mc_leak_rates definition from your script
# keep dose_approximation, dose_to_probit, probit_to_probability
# keep simulation_using_different_inputs (unchanged)

# ----------  4. Worker: run a chunk of hours & RETURN ONE DATAFRAME ----------
from typing import Optional  # ensure imported near top

def _run_hours_chunk(
    weather_df: pd.DataFrame,
    hours_subset: Sequence[pd.Timestamp],
    *,
    n_samples: int,
    out_dir: Optional[Path] = None,   # kept for API compat; unused
) -> pd.DataFrame:


    """Process a subset of hours in this worker; return a single concatenated DataFrame."""
    chunk_outs = []

    # enforce single math thread inside this process
    with threadpool_limits(limits=1):
        for t in hours_subset:
            row = weather_df.loc[weather_df["time"] == t].iloc[0]
            wd, ws, sc = row["wind_direction"], row["wind_speed"], row["stability_class"]

            rates_matrix = generate_mc_leak_rates(n_samples=n_samples)

            # loop MC samples serially (memory‑cheaper than spawning more jobs)
            sample_outs = []
            for s_idx, rv in enumerate(rates_matrix):
                result = simulation_using_different_inputs(
                    wd, ws, sc, pd.Series(rv)
                ).assign(weather_time=t, sample=s_idx)
                sample_outs.append(result)

            hour_df = pd.concat(sample_outs, ignore_index=True)
            chunk_outs.append(hour_df)

    # one DataFrame back to parent
    return pd.concat(chunk_outs, ignore_index=True)
# ---------------------------------------------------------------------------


# ----------  5. Top-level scheduler ------------------------------------------
# ----------  5. Top-level scheduler ------------------------------------------
def run_mc_dispersion(
    csv_file: str,
    hours: List[str],
    *,
    timestamp_col="time",
    wind_dir_col="wind_direction",
    wind_speed_col="wind_speed",
    stability_col="stability_class",
    n_samples: int = 100,
    n_jobs: int   = 8,
    out_file: str = "mc_results.parquet",   # NEW: single combined output file
    out_dir: str  = None,                   # deprecated; ignored unless you need it
    parquet_engine: str = "pyarrow",        # override if needed
    parquet_compression: str = "zstd",      # "snappy" if zstd not available
) -> pd.DataFrame:
    """
    Parallel Monte Carlo dispersion over requested hours.
    Returns *and* writes a single combined DataFrame.

    Saving to Parquet is recommended (fast load, small disk footprint).
    """
    # load weather once (shared read-only copy)
    df_all = pd.read_csv(csv_file, parse_dates=[timestamp_col])
    df_all = df_all.rename(
        columns={
            timestamp_col: "time",
            wind_dir_col: "wind_direction",
            wind_speed_col: "wind_speed",
            stability_col: "stability_class",
        }
    )
    df_all["time"] = pd.to_datetime(df_all["time"])

    # parse requested hours
    hours_ts = pd.to_datetime(hours)
    missing  = hours_ts[~hours_ts.isin(df_all["time"])]
    if len(missing):
        raise ValueError(f"Missing hours in weather file: {list(missing)}")

    # split hours list into n_jobs roughly equal chunks
    chunk_size = math.ceil(len(hours_ts) / n_jobs)
    hour_chunks = [
        hours_ts[i : i + chunk_size] for i in range(0, len(hours_ts), chunk_size)
    ]

    print(f"Scheduling {len(hours_ts)} hours → {len(hour_chunks)} worker processes")

    # parallel execution: each worker returns a DataFrame
    chunk_dfs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_hours_chunk)(
            df_all, chunk, n_samples=n_samples, out_dir=None
        )
        for chunk in hour_chunks
    )

    # combine all worker results
    df_combined = pd.concat(chunk_dfs, ignore_index=True)

    # write single Parquet file
    df_combined.to_parquet(
        out_file,
        engine=parquet_engine,
        compression=parquet_compression,
        index=False,
    )
    print(f"✓ Wrote combined results: {out_file}")

    return df_combined
# ---------------------------------------------------------------------------
# ----------  6. EXAMPLE CALL --------------------------------------------------
if __name__ == "__main__":
    # Full June 2024 hourly interval
    START = "2024-09-01 00:00"
    END   = "2024-12-31 23:00"
    HOURS = pd.date_range(START, END, freq="H").strftime("%Y-%m-%d %H:%M").tolist()

    df_all = run_mc_dispersion(
        csv_file="ERA5_malmo/era5_malmo_2024_combined.csv",
        hours=HOURS,
        n_samples=1,     # MC samples per hour
        n_jobs=14,       # you can tune; start w/ physical cores if memory tight
        out_file="mc_malmo_jan_dec_2024.parquet",
    )
    # quick peek
    print(df_all.head())
