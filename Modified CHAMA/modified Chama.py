"""
The simulation module contains methods to run Gaussian air dispersion models. 
Chama can also integrate simulations from third party software for additional
sensor placement applications.

.. rubric:: Contents

.. autosummary::

    Grid
    Source
    GaussianPlume
    GaussianPuff
"""

from __future__ import print_function, division
import numpy as np
import pandas as pd
# from scipy import integrate


def _calculate_sigma(x, stability_class):
    """
    Calculates sigmay and sigmaz as a function of grid points in the 
    direction of travel (x) for stability class A through F.

    Parameters
    ---------------
    x: numpy array
        Grid points in the direction of travel (m)
    stability_class : string
        Stability class, A through F
        
    Returns
    ---------
    sigmay: numpy array
        Standard deviation of the Gaussian distribution in the horizontal
        (crosswind) direction (m)
    sigmaz: numpy array
        Standard deviation of the Gaussian distribution in the vertical
        direction (m)
    """
    if stability_class == 'A':
        k = [0.250, 927, 0.189, 0.1020, -1.918]
    elif stability_class == 'B':
        k = [0.202, 370, 0.162, 0.0962, -0.101]
    elif stability_class == 'C':
        k = [0.134, 283, 0.134, 0.0722, 0.102]
    elif stability_class == 'D':
        k = [0.0787, 707, 0.135, 0.0475, 0.465]
    elif stability_class == 'E':
        k = [0.0566, 1070, 0.137, 0.0335, 0.624]
    elif stability_class == 'F':
        k = [0.0370, 1170, 0.134, 0.0220, 0.700]
    else:
        return

    sigmay = k[0] * x / (1 + x / k[1]) ** k[2]
    sigmaz = k[3] * x / (1 + x / k[1]) ** k[4]

    return sigmay, sigmaz


def _modify_grid(model, wind_direction, wind_speed):
    """
    Rotates grid to account for wind direction.
    Translates grid to account for source location.
    
    Parameters
    ---------------
    model: chama GaussianPlume
        GaussianPlume object
    wind_direction: float
        Wind direction (degrees)
    wind_speed: float
        Wind speed (m/s)

    Returns
    ---------
    gridx: numpy array
        x values in the grid (m)
    gridy: numpy array
        y values in the grid (m)
    gridz: numpy array
        z values in the grid (m)
    """

    angle_rad = wind_direction / 180.0 * np.pi
    gridx = (model.grid.x - model.source.x) * np.cos(angle_rad) \
            + (model.grid.y - model.source.y) * np.sin(angle_rad)
    gridy = - (model.grid.x - model.source.x) * np.sin(angle_rad) \
            + (model.grid.y - model.source.y) * np.cos(angle_rad)

    gridx[gridx < 0] = 0

    gridz = _calculate_z_with_buoyancy(model, gridx, wind_speed)
        
    return gridx, gridy, gridz


def _calculate_z_with_buoyancy(model, x, wind_speed, emission_rate):
    """
    Buoyancy‐adjusted plume-centre height z(x).

    Parameters
    ----------
    model : GaussianPlume | GaussianPuff
        Only used for constants (g, densities, source.z).
    x : ndarray | float
        Downwind distance (m).
    wind_speed : float
        Wind speed at release height (m s-1).
    emission_rate : float
        Leak rate for *this puff* (kg s-1).  **Must be scalar.**

    Returns
    -------
    z : ndarray
        Centre-line height (m).  Shape follows `x`.
    """
    buoyancy_parameter = (model.gravity * emission_rate / np.pi) * (
        1.0 / model.density_eff - 1.0 / model.density_air
    )

    z = model.source.z + (
        1.6 * buoyancy_parameter ** (1.0 / 3.0) * x ** (2.0 / 3.0)
    ) / wind_speed

    return z



class Grid(object):

    def __init__(self, x, y, z):
        """
        Defines the receptor grid.
        
        Parameters
        --------------
        x: numpy array
            x values in the grid (m)
        y: numpy array
            y values in the grid (m)
        z: numpy array
            z values in the grid (m)
        """
        self.x, self.y, self.z = np.meshgrid(x, y, z)


class Source(object):

    def __init__(self, x, y, z, rate):
        """
        Defines the source location and leak rate.
        
        Parameters
        -------------
        x: float
            x location of the source (m)
        y: float
            y location of the source (m)
        z: float
            z location of the source (m)
        rate: float
            source leak rate (kg/s)
        """
        self.x = x
        self.y = y
        self.z = z
        self.rate = rate


class GaussianPlume:
    
    def __init__(self, grid, source, atm,
                 gravity=9.81, density_eff=0.769, density_air=1.225):
        """
        Defines the Gaussian plume model.
        
        Parameters
        ---------------
        grid: chama Grid
            Grid points at which concentrations should be calculated
        source: chama Source
            Source location and leak rate
        atm: pandas DataFrame 
            Atmospheric conditions for the simulation. Columns include 
            'Wind Direction', 'Wind Speed', and 'Stability Class' indexed by 
            the time that changes occur.
        gravity: float
            Gravity (m2/s), default = 9.81 m2/s
        density_eff: float
            Effective density of the leaked species (kg/m3),
            default = 0.769 kg/m3
        density_eff: float
            Effective density of air (kg/m3), default = 1.225 kg/m3
        """
        self.grid = grid
        self.source = source
        self.atm = atm
        self.gravity = gravity
        self.density_eff = density_eff
        self.density_air = density_air
        self.conc = pd.DataFrame()
        self.run()

    def run(self):
        """
        Computes the concentrations of a Gaussian plume.
        """

        conc = pd.DataFrame()
        for t in self.atm.index:
            
            wind_direction = self.atm.at[t, 'Wind Direction']
            wind_speed = self.atm.at[t, 'Wind Speed']
            stability_class = self.atm.at[t, 'Stability Class']

            X2, Y2, h = _modify_grid(self, wind_direction, wind_speed)
            sigmay, sigmaz = _calculate_sigma(X2, stability_class)
            
            a = np.zeros(X2.shape)
            b = np.zeros(X2.shape)
            c = np.zeros(X2.shape)

            a[X2 > 0] = self.source.rate / \
                    (2 * np.pi * wind_speed * sigmay[X2 > 0] * sigmaz[X2 > 0])
            b[X2 > 0] = np.exp(-Y2[X2 > 0] ** 2 / (2 * sigmay[X2 > 0] ** 2))
            c[X2 > 0] = np.exp(-(self.grid.z[X2 > 0] - h[X2 > 0]) ** 2 /
                               (2 * sigmaz[X2 > 0] ** 2)) \
                      + np.exp(-(self.grid.z[X2 > 0] + h[X2 > 0]) ** 2 /
                               (2 * sigmaz[X2 > 0] ** 2))
            
            conc_at_t = a * b * c
            conc_at_t[np.isnan(conc_at_t)] = 0
            conc_at_t = pd.DataFrame(data=np. transpose([self.grid.x.ravel(),
                self.grid.y.ravel(), self.grid.z.ravel(), conc_at_t.ravel()]), 
                columns=['X', 'Y', 'Z', 'S'])
            conc_at_t['T'] = t
            
            conc = pd.concat([conc, conc_at_t], ignore_index=True)

        self.conc = conc
        self.conc = self.conc[['X', 'Y', 'Z', 'T', 'S']]

class GaussianPuff:
    """
    Re-implemented so that each puff is
    • created at its release time t₀ using the source *rate at t₀*  
    • propagated through every later reporting step  
    • added to the running concentration field before the next puff is made
    """

    def __init__(self, grid=None, source=None, atm=None, tpuff=1, tend=None,
                 tstep=10, gravity=9.81, density_eff=0.769, density_air=1.225):

        self.grid = grid
        self.source = source
        self.atm = atm.sort_index()                        # must include Wind Dir/Speed & Stability
        self.tpuff = tpuff
        self.tend = max(atm.index) if tend is None else tend
        self.tstep = tstep
        self.gravity = gravity
        self.density_eff = density_eff
        self.density_air = density_air

        # ➊ build a light-weight “puff table” (one row = one release event)
        self._make_puff_table()

        # ➋ run the dispersion if a grid is present
        self.conc = pd.DataFrame()
        if self.grid is not None:
            self.run()


    # ------------------------------------------------------------------
    # helper that always gets “the last known” met row at or before t
    # ------------------------------------------------------------------
    def _atm_row(self, t):
        """
        Return the atmospheric row valid for integer-second *t*
        (forward-fills if t is between two rows).
        """
        if t < self.atm.index[0]:
            raise KeyError(f"No met data before t={t}s")
        return self.atm.loc[:t].iloc[-1]     # last row whose index ≤ t

    def _atm_values(self, t):
        row = self._atm_row(t)
        return row['Wind Direction'], row['Wind Speed'], row['Stability Class']

    # ------------------------------------------------------------------
    # leak-rate helper  (scalar, time-aware)
    # ------------------------------------------------------------------
    def _rate_at(self, t):
        """
        Return the leak rate (kg s-1) valid *at integer-second t* as a float.

        Priority:
        1.  'Rate' column in the atm table (already scalar)
        2.  self.source.rate[t]             if self.source.rate is a Series
        3.  float(self.source.rate)         if it's a scalar
        """
        # 1 ── time-varying rates stored in atm
        if 'Rate' in self.atm.columns:
            return float(self._atm_row(t)['Rate'])

        # 2 ── time-varying rates supplied via Source(rate=<Series>)
        rate_obj = self.source.rate
        if isinstance(rate_obj, (pd.Series, pd.DataFrame, np.ndarray)):
            try:
                return float(rate_obj.loc[t])
            except Exception:
                # fallback: last known value before t
                return float(rate_obj.loc[:t].iloc[-1])

        # 3 ── plain scalar
        return float(rate_obj)



    def _make_puff_table(self):
        """One record per puff: release time, meteo, emission mass Q."""
        releases = np.arange(0, self.tend + self.tpuff, self.tpuff)
        rows = []
        for t0 in releases:
            wd, ws, stab = self._atm_values(int(t0))
            rows.append({
                'T_release': t0,
                'wd': wd,
                'ws': ws,
                'angle': np.deg2rad(wd),
                'stab': stab,
                'Q': self._rate_at(int(t0)) * self.tpuff
            })
        self.puffs = pd.DataFrame(rows)

        # ------------------------------------------------------------------
    # main solver -- “create-then-propagate” puff logic
    # ------------------------------------------------------------------
    def run(self, grid=None, tstep=None):
        """
        Populate self.conc.

        Parameters
        ----------
        grid : Grid, optional
            If supplied, overrides the grid stored in self.grid.
        tstep : int | float, optional
            Reporting-step in seconds.  Overrides self.tstep if given.
        """
        # allow caller overrides -------------------------------------------------
        if grid  is not None:
            self.grid = grid
        if tstep is not None:
            self.tstep = tstep

        # ─────────────────────────────────────────────────────────────────────────
        #  From this point on, **ONLY NumPy ndarrays** are carried forward.
        # ─────────────────────────────────────────────────────────────────────────
        gx = np.asarray(self.grid.x)      # shape (nx, ny)  or (nx,)  depending on Grid impl.
        gy = np.asarray(self.grid.y)
        gz = np.asarray(self.grid.z)

        times  = np.arange(0, self.tend + self.tstep, self.tstep, dtype=float)
        accum  = [np.zeros(gx.shape, dtype=float) for _ in times]

        # outer-loop: puff  |  inner-loop: every reporting time ≥ t₀ --------------
        for _, p in self.puffs.iterrows():
            t0, ws, ang, stab, Q = p[['T_release', 'ws', 'angle', 'stab', 'Q']]

            for idx, t in enumerate(times):
                if t < t0:                      # puff not born yet
                    continue

                dt  = t - t0
                D   = ws * dt                   # travel distance
                xk  = self.source.x + D*np.cos(ang)
                yk  = self.source.y + D*np.sin(ang)
                rate_scalar = Q / self.tpuff

                zk = _calculate_z_with_buoyancy(self, D, ws, rate_scalar)


                sigy, sigz = _calculate_sigma(D, stab)
                if sigy == 0.0 or sigz == 0.0:
                    continue                    # avoids divide-by-zero

                x_part = np.exp(-((xk - gx)**2) / (2.0 * sigy**2))
                y_part = np.exp(-((yk - gy)**2) / (2.0 * sigy**2))
                z_part = np.exp(-((zk - gz)**2) / (2.0 * sigz**2))
                z_refl = np.exp(-((zk + gz)**2) / (2.0 * sigz**2))

                kern   = (1.0 / (sigy**2 * sigz)) * x_part * y_part * (z_part + z_refl)
                accum[idx] += kern * Q / ((2.0*np.pi)**1.5)

        # flatten into long-form DataFrame identical to original API -------------
        records = []
        for t, arr in zip(times, accum):
            records.append(pd.DataFrame({
                'X': gx.ravel(),
                'Y': gy.ravel(),
                'Z': gz.ravel(),
                'T': t,
                'S': arr.ravel()
            }))
        self.conc = pd.concat(records, ignore_index=True)[['X', 'Y', 'Z', 'T', 'S']]
