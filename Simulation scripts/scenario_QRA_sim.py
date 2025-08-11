import pandas as pd
import numpy as np
from datetime import time as dtime

# ------------ 0. Load data -------------
df = pd.read_csv("ERA5_malmo/era5_malmo_2024_combined.csv", parse_dates=["time"])

# ------------ 1. 12 wind-direction sectors -------------
sector_centers = np.arange(0, 360, 30)
ranges = []
for c in sector_centers:
    start = (c - 14) % 360
    end   = (c + 15) % 360
    ranges.append((c, start, end))

def direction_to_sector(d):
    # Treat calm (dir==0 & speed==0) separately if you wish. Here we still bin it.
    for center, start, end in ranges:
        if start <= end:
            if start <= d <= end:
                return center
        else:  # wrap (346–360, 0–15)
            if d >= start or d <= end:
                return center
    return np.nan

df["sector"] = df["wind_direction"].apply(direction_to_sector)

# proportions of each sector
dir_prop = df["sector"].value_counts(normalize=True).sort_index()
dir_prop = dir_prop.dropna()

# ------------ 2. Speed classes + weather type codes -------------
bins   = [-np.inf, 3, 7, np.inf]
labels = ["≤3 m/s (låg)", "3–7 m/s (medel)", ">7 m/s (hög)"]
df["speed_class"] = pd.cut(df["wind_speed"], bins=bins, labels=labels)

DAY_W  = 0.44
NIGHT_W= 0.56

# speed-class distribution from historical data (whole year)
class_prop_tot = df["speed_class"].value_counts(normalize=True).reindex(labels).fillna(0)

# map to Purple Book weather types
weather_codes_day   = {"≤3 m/s (låg)": "2B", "3–7 m/s (medel)": "5C", ">7 m/s (hög)": "8D"}
weather_codes_night = {"≤3 m/s (låg)": "2F", "3–7 m/s (medel)": "5D", ">7 m/s (hög)": "8D"}

weather_rows = []
for cls in labels:
    weather_rows.append({"speed_class": cls, "day_part": "Dag",  "code": weather_codes_day[cls],
                         "stability_letter": weather_codes_day[cls][-1],  # B/C/D
                         "prop": class_prop_tot[cls] * DAY_W})
    weather_rows.append({"speed_class": cls, "day_part": "Natt", "code": weather_codes_night[cls],
                         "stability_letter": weather_codes_night[cls][-1],  # F/D/D
                         "prop": class_prop_tot[cls] * NIGHT_W})

weather_df = pd.DataFrame(weather_rows)
weather_df["prop"] /= weather_df["prop"].sum()   # ensure Σ=1

# ------------ 3. Final matrix (optional to inspect) -------------
mat = pd.DataFrame({
    (r["speed_class"], r["day_part"], r["code"]): dir_prop.values * r["prop"]
    for _, r in weather_df.iterrows()
}, index=dir_prop.index)

mat_pct = (mat * 100)
mat_pct[("Totalt","","")] = mat_pct.sum(axis=1)
mat_pct.loc["Total",:]    = mat_pct.sum(axis=0)
# display(mat_pct)  # uncomment in notebook

# --- Build SCENARIOS only (no simulations yet) ---

import pandas as pd
import numpy as np
from itertools import product

# Assumes you already have:
# dir_prop  -> Series indexed by sector (0,30,...,330) with proportions (sum=1)
# weather_df -> DataFrame with columns: ['speed_class','day_part','code','prop'] (sum(prop)=1)

# 1) Map code -> v_sim and stability letter (take only the LETTER for stability)
code_map = {
    "2B": {"v_sim": 2.0, "stab": "B"},
    "2F": {"v_sim": 2.0, "stab": "F"},
    "5C": {"v_sim": 5.0, "stab": "C"},
    "5D": {"v_sim": 5.0, "stab": "D"},
    "8D": {"v_sim": 8.0, "stab": "D"},
    "8N": {"v_sim": 8.0, "stab": "D"},  # if you ever use 8N, keep D or change to wanted letter
}

# 2) Cartesian product: each sector × each weather type row
sectors = dir_prop.index.to_list()
weather_rows = weather_df.to_dict("records")

rows = []
for sec, w in product(sectors, weather_rows):
    code = w["code"]
    info = code_map[code]
    rows.append({
        "sector": sec,
        "code": code,
        "speed_class": w["speed_class"],
        "day_part": w["day_part"],
        "v_sim": info["v_sim"],
        "stability_letter": info["stab"],
        "dir_prop": dir_prop.loc[sec],
        "weather_prop": w["prop"],
    })

scenarios = pd.DataFrame(rows)

# 3) Cell weight (will sum to 1.0)
scenarios["cell_weight"] = scenarios["dir_prop"] * scenarios["weather_prop"]

# Optional sanity checks
assert np.isclose(scenarios["cell_weight"].sum(), 1.0), "Weights do not sum to 1!"
# display(scenarios.head())

# --- A0 generator: returns exactly 3 profiles (min/mode/high) ---
def generate_mc_leak_rates_3_type(
    tau=1081.1,
    t0=210,
    duration=3600,
    dt=10,
    A0_hat=0.001347103726986978,
    renormalize=False,
    eps=1e-12,
):
    t = np.arange(0, duration + dt, dt)
    low  = 0.5 * A0_hat
    mode = A0_hat
    high = (1.0 / t0) * (1.0 - eps)

    A0_samples = np.array([low, mode, high])
    A1_samples = (1.0 - A0_samples * t0) / tau
    A1_samples = np.clip(A1_samples, 0.0, None)

    plateau_mask = t <= t0
    rates = np.empty((3, t.size))
    rates[:, plateau_mask]  = A0_samples[:, None]
    rates[:, ~plateau_mask] = (A1_samples[:, None] *
                               np.exp(-(t[~plateau_mask] - t0) / tau))

    if renormalize:
        mass = rates.sum(axis=1) * dt
        rates = rates / mass[:, None]

    return rates, t

from simulation_modified import Grid, Source, GaussianPuff   # heavy code
from scipy.stats import norm

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

# Replace your old helper with this:
def to_rate_series(rate_vec, t_index):
    """Make a pandas Series the model can .loc on."""
    return pd.Series(rate_vec, index=t_index)
# Get the three A0 profiles
rates_matrix, t_grid = generate_mc_leak_rates_3_type()
rate_tags = ["A0_min", "A0_mode", "A0_high"]

# --- run ONLY the first two scenarios and SAVE THE FULL GRID for each run ---
results_grid = []   # list of DataFrames (one per scenario × A0_case)
print("simulations starts")
# for _, row in scenarios.iloc[:1].iterrows():
for _, row in scenarios.iterrows():
    for i, tag in enumerate(rate_tags):
        # leak-rate vector as Series (index = t_grid)
        rate_series = to_rate_series(rates_matrix[i], t_grid)

        # run simulation
        sim_out = simulation_using_different_inputs(
            wind_direction = row["sector"],
            wind_speed     = row["v_sim"],
            stability_class= row["stability_letter"],   # ONLY letter
            rate_vector    = rate_series
        )

        # add scenario metadata to every grid row
        sim_out = sim_out.assign(
            sector        = row["sector"],
            code          = row["code"],
            speed_class   = row["speed_class"],
            day_part      = row["day_part"],
            A0_case       = tag,
            stability     = row["stability_letter"],
            v_sim         = row["v_sim"],
            cell_weight   = row["cell_weight"]
        )

        results_grid.append(sim_out)

# combine to one big DataFrame
sim_results_grid = pd.concat(results_grid, ignore_index=True)
print(sim_results_grid.head())
# Parquet (smaller & faster to load)
sim_results_grid.to_parquet("sim_results_grid_classic_QRA.parquet", index=False)