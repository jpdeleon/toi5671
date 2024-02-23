#!/usr/bin/env python
import os
import json
from multiprocessing import Pool
from urllib.request import urlopen
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from matplotlib.ticker import AutoMinorLocator
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import minimize
import pandas as pd
import emcee
import corner
import seaborn as sb
import astropy.units as u
from pytransit import QuadraticModel
from ldtk import LDPSetCreator, BoxcarFilter
import seaborn as sb
from numba import njit

sb.set(
    context="paper",
    style="ticks",
    palette="deep",
    font="sans-serif",
    font_scale=1.5,
    color_codes=True,
)
sb.set_style({"xtick.direction": "in", "ytick.direction": "in"})
sb.set_context(rc={"lines.markeredgewidth": 1})
pl.rcParams["font.size"] = 26

os.environ["OMP_NUM_THREADS"] = "1"
os.nice(19)

PRIOR_K_MIN = 0.01
PRIOR_K_MAX = 1.0
LOG_TWO_PI = np.log(2*np.pi)

# bandpasses in increasing wavelength
"""
V: 484.0,606.4
gp: 391.4,556.3
rp: 546.0,703.4
ip: 682.9,860.0
zs: 811.0,927.5
Iz: 716.3397,1034.5211
TESS: 585.0,1056.0
gp: 401.5,552.5
rp: 562.5,697.5
ip: 705.0,817.5
zs: 825.0,922.5
"""
filter_widths = {
    "gp": (400, 550),
    "V": (480, 600),
    "rp": (560, 700),
    "ip": (700, 820),
    "zs": (825, 920),
    "I+z": (720, 1030),
    "TESS": (585, 1050),
}
all_bands = list(filter_widths.keys())
if True:
    #manual assignment
    colors = {
        'gp': 'purple',
        'V': 'blue',
        'rp': 'green',
        'ip': 'yellow',
        'zs': 'darkorange',
        'Iz': 'red',
        'TESS': 'black'
    }
else:
    cmap = cm.get_cmap("RdBu_r")
    color = cmap(np.linspace(0.1, 1.0, len(filter_widths)))
    colors = {k: c for k,c in zip(filter_widths,color)}
        
    
lco_phot_mapping = {
    'BJD_TDB': 'BJD_TDB',
    'rel_flux_T1': 'Flux',
    'rel_flux_err_T1': 'Err',
    'AIRMASS': 'Airmass',
    #'Peak_T1', 
    # 'FWHM_T1', 
    # 'X(IJ)_T1', 
    # 'Y(IJ)_T1', 
    # 'Source-Sky_T1'
}

@dataclass
class TransitFit:
    name: str
    data: Dict = field(repr=False)
    star_params: Dict[str, Tuple[float, float]] = field(repr=False)
    planet_params: Dict[str, Tuple[float, float]] = field(repr=False)
    alias: str = ".01"
    # bands: List[str] = None
    model: str = "achromatic"
    covariate: str = "Airmass"
    lm_order: int = 1
    mask_start: float = None
    mask_end: float = None
    outdir: str = "results"
    use_r1r2: bool=False
    DEBUG: bool = field(repr=False, default=False)
    
    def __post_init__(self):
        # super().__post_init__()
        self._validate_inputs()
        self._init_data()
        self._init_params()
        self._init_ldc()
        self._mcmc_samples = None

    def _validate_inputs(self):
        """Make sure inputs are correct"""
        assert isinstance(self.data, dict), "data must be a dict"

        # sort data in griz order
        self.data = dict(sorted(self.data.items(), key=lambda x: all_bands.index(x[0])))
        self.bands = list(self.data.keys()) #if self.bands is None else self.bands
        errmsg = f"`bands` can only use the given keys in `data`: {self.data.keys()}"
        assert len(self.bands) <= len(self.data.keys()), errmsg
        for band in self.bands:
            errmsg = f"{band} not in `data` keys: {self.data.keys()}"
            assert band in self.data.keys(), errmsg
        self.nband = len(self.bands)

        assert isinstance(self.star_params, dict)
        assert isinstance(self.planet_params, dict)
        assert (
            np.array(self.planet_params["tdur"]) < 1
        ).all(), "`tdur` must be in days"
        assert (np.array(self.planet_params["rprs"]) < 1).all(), "Check `rprs`"
        assert self.planet_params["a_Rs"][0] > 1, "`a/Rs` is <1?"
        # errmsg = f"{self.covariate} not in data[band].columns, errmsg
        
        self.teff = self.star_params["teff"]
        self.logg = self.star_params["logg"]
        self.feh = self.star_params["feh"]

        models = ["achromatic", "chromatic"]
        errmsg = f"model is either {' or '.join(models)}"
        assert self.model in models, errmsg

        if self.DEBUG:
            print("==========")
            print("DEBUG MODE")
            print("==========")
            print("bands", self.bands)
            print("nband", self.nband)
        
    def _init_data(self):
        """initialize user-given photometry data"""
        self.times = {}
        self.fluxes = {}
        self.flux_errs = {}
        self.covariates = {}
        self.transit_models = {}
        self.epoch_tcs = {}
        for band,dfs in self.data.items():
            self.transit_models[band] = {}
            self.times[band] = {}
            self.fluxes[band] = {}
            self.flux_errs[band] = {}
            self.covariates[band] = {}
            self.epoch_tcs[band] = {}
            for inst,df in dfs.items():
                t = df["BJD_TDB"].values 
                f = df["Flux"].values
                e = df["Err"].values
                z = df[self.covariate].values
                self.times[band][inst] = t 
                self.fluxes[band][inst] = f/np.median(f)
                self.flux_errs[band][inst] = e
                self.covariates[band][inst] = z
                self.transit_models[band][inst] = QuadraticModel()
                self.transit_models[band][inst].set_data(t) 
                self.epoch_tcs[band][inst] = get_epoch_tc(t, 
                                                       self.planet_params["period"][0], 
                                                       self.planet_params["t0"][0])
        
    def _init_params(self, return_dict=False):
        """initialize parameters of the model"""
        
        if self.use_r1r2:
            imp = self.planet_params.get("imp", (0, 0.1))
            k = self.planet_params["rprs"]
            r1, r2 = imp_k_to_r1r2(imp[0], k[0], k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX)
            r1_err, r2_err = imp_k_to_r1r2(
                imp[1], k[1], k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX
            )
            params = {
                "tc": self.planet_params["t0"],
                "per": self.planet_params["period"],
                "a_Rs": self.planet_params["a_Rs"],
                "r1": (r1, r1_err),  # optional
            }
            self.k_idx = len(params)
            if self.model == "chromatic":
                params.update({"r2_" + b: (r2, r2_err) for b in self.bands})
            else:
                params.update({"r2": (r2, r2_err)})
        else:
            params = {
                "t0": self.planet_params["t0"],
                "period": self.planet_params["period"],
                "a_Rs": self.planet_params["a_Rs"],
                "imp": self.planet_params.get("imp", (0, 0.1)),  # optional
            }
            self.k_idx = len(params)
            if self.model == "chromatic":
                params.update(
                    {"k_" + b: self.planet_params["rprs"] for b in self.bands}
                )
            elif self.model == "achromatic":
                params.update({"k": self.planet_params["rprs"]})
        self.d_idx = len(params)
        self.lin_model_offsets = list(map(tuple,np.zeros(shape=(self.ndatasets,2))))
        i=0
        for band,dfs in self.data.items():
            for inst,df in dfs.items():
                params.update({f"d_{band}_{inst}": self.lin_model_offsets[i]})
                i+=1
        self.model_param_names = list(params.keys())
        self.ndim = len(params)
        self.model_params = params
        # print(params.items())
        self.pv_updated = [v[0] for k, v in params.items()]
        self.pv_init = [v[0] for k, v in params.items()]
        return params if return_dict else [v[0] for k, v in params.items()]
         
    def _init_ldc(self):
        """initialize quadratic limb-darkening coefficients"""
        self.ldc = {}
        self.ldtk_filters = [BoxcarFilter(b, *filter_widths[b]) for b in self.bands]
        sc = LDPSetCreator(
            teff=self.teff,
            logg=self.logg,
            z=self.feh,
            filters=self.ldtk_filters,
        )
        # Create the limb darkening profiles
        self.ldtk_profiles = sc.create_profiles()
        # Estimate quadratic law coefficients
        #cq, eq = ps.coeffs_qd(do_mc=True)
        qc, qe = self.ldtk_profiles.coeffs_qd()
        for i, band in enumerate(self.ldtk_profiles._filters):
            q1, q2 = qc[i][0], qc[i][1]
            if self.DEBUG:
                print(f"{ps._filters[i]}: q1,q2=({q1:.2f}, {q2:.2f})")
            self.ldc[band] = (q1,q2)

    @property
    def ndatasets(self):
        return sum([len(v) for k,v in self.data.items()])
        
    @property
    def ndata(self):
        return sum([len(v) for b,d in self.data.items() for i, v in d.items()])

    @property
    def instruments(self):
        return list(set([i for b, d in self.data.items() for i, v in d.items()]))
        
    @property
    def ninsts(self):
        return len(self.instruments)

    @property
    def mcmc_samples(self):
        return self._mcmc_samples
        
    def get_chi2_linear_baseline(self, p0):
        """
        p0 : list
            parameter vector

        chi2 of linear baseline model
        """
        t0 = self.planet_params["t0"][0]
        period = self.planet_params["period"][0]
        
        chi2 = 0.0
        i = 0
        for band,lcs in self.data.items():
            for inst,lc in lcs.items():
                t = self.times[band][inst]
                tc = self.epoch_tcs[band][inst]
                f = self.fluxes[band][inst]
                e = self.flux_errs[band][inst]
                z = self.covariates[band][inst]

                idx = (f>0.95) & (f<1.1)
                flux_time = p0[i] * t[idx]
                c = np.polyfit(z[idx], f[idx] - flux_time, self.lm_order)
                linear_model = np.polyval(c, z)[idx]
                chi2 += np.sum((f[idx] - linear_model + flux_time) ** 2 / e[idx] ** 2)
                i+=1
        return chi2

    def optimize_chi2_linear_baseline(self, p0=None, repeat=1):
        """
        p0 : list
            parameter vector
        """
        p0 = np.zeros(self.ndatasets) if p0 is None else p0
        assert len(p0) == self.ndatasets
        for i in range(repeat):
            p = p0 if i == 0 else res_lin.x
            res_lin = minimize(self.get_chi2_linear_baseline, p, method="Nelder-Mead")
            print(res_lin.fun, res_lin.success, res_lin.x)

        npar_lin = len(res_lin.x)
        # print('npar(linear) = ', npar_lin)
        self.bic_lin = res_lin.fun + npar_lin * np.log(self.ndata)
        # print('BIC(linear) = ', self.bic_lin)
        self.lin_model_offsets = res_lin.x
        self.pv_updated[self.d_idx:] = res_lin.x
        if self.DEBUG:
            print(f"updated pv: {self.pv_updated}")            

    def plot_linear_baseline(self):
        i = 0
        for band,dfs in self.data.items():
            for inst,df in dfs.items():
                fig, ax = pl.subplots()
                t = tf.times[band][inst]
                tc = self.epoch_tcs[band][inst]
                f = tf.fluxes[band][inst]
                e = tf.flux_errs[band][inst]
                z = tf.covariates[band][inst]
                d = tf.lin_model_offsets[i][0]
                idx = (f>0.95) & (f<1.1)
                flux_time = d * t[idx]
                c = np.polyfit(z[idx], f[idx] - flux_time, 1)
                linear_model = np.polyval(c, z)[idx]
                ax.plot(t[idx], linear_model)
                ax.plot(t, f)
                ax.set_title(f"{inst}/{band}")
                i+=1
        
    def unpack_parameters(self, pv):
        """
        pv : list
            parameter vector from MCMC or optimization

        Unpack commonly used parameters for transit and systematics models
        """
        assert len(pv) == self.ndim
        t0, period, a_Rs, imp = pv[: self.k_idx]
        if self.model == "chromatic":
            k = np.array([pv[self.k_idx + i] for i in range(self.nband)])
        elif self.model == "achromatic":
            k = np.zeros(self.nband) + pv[self.k_idx]
        d = np.array(pv[self.d_idx : self.d_idx + self.ndatasets])
        return t0, period, a_Rs, imp, k, d

    # def pack_parameters(self, pv):
    #     d = {}
    #     i = 0
    #     for band in self.data:
    #         d[band] = {}
    #         for inst in self.data[band]:
    #             d[band][inst] = pv[i]
    #             i+=1
    #     return d

    def plot_raw_data(self, marker='-', binsize=None, figsize=(10, 10), ylims=(0.9, 1.05)):
        fig = pl.figure(figsize=figsize)
        ncol = 2 if self.ndatasets > 1 else 1
        nrow = int(np.ceil(self.ndatasets/2)) if self.ndatasets > 2 else 1
        i = 0
        for band,dfs in self.data.items():
            for inst,df in dfs.items():
                ax = fig.add_subplot(nrow, ncol, 1+i)
                i+=1
                t = self.times[band][inst]
                f = self.fluxes[band][inst]
                e = self.flux_errs[band][inst]
                tc = self.epoch_tcs[band][inst]
                ax.plot(t-tc, f, marker, color=colors[band])
                if binsize is not None:
                    tbin, ybin, yebin = binning_equal_interval(t-tc, f, e, binsize, tc)
                    ax.errorbar(tbin, ybin, yerr=yebin, marker="o", c=colors[band], ls="")
                ax.set_title(f'{inst}/{band}')
            if ylims:
                ax.set_ylim(*ylims)
        fig.tight_layout()
        return fig

    def plot_lightcurves(self, pv, binsize=600 / 86400, ylims=None, figsize=(8, 5)):
        """
        pv : list
            parameter vector from MCMC or optimization

        Raw and detrended lightcurves using `pv`

        See also `plot_detrended_data_and_transit()`
        """
        assert len(pv) == self.ndim
        # unpack free parameters
        if self.use_r1r2:
            t0, period, a_Rs, r1, r2, d = self.unpack_parameters(pv)
            imp, k = r1r2_to_imp_k(r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX)
        else:
            t0, period, a_Rs, imp, k, d = self.unpack_parameters(pv)
        # derived
        inc = np.arccos(imp / a_Rs)
        depth = self.planet_params["rprs"][0] ** 2

        trends = self.get_trend_models(pv)
        transits = self.get_upsampled_transit_models(pv)
        i = 0
        for band,dfs in self.data.items():
            for inst,df in dfs.items():
                fig, ax = pl.subplots(1, 2, sharey=True, figsize=figsize)
                t = self.times[band][inst]
                f = self.fluxes[band][inst]
                e = self.flux_errs[band][inst]
                # z = self.covariates[band][inst]

                tc = self.epoch_tcs[band][inst]
                # transit
                flux_tr = self.transit_models[band][inst].evaluate_ps(
                    k[i], self.ldc[band], tc, period, a_Rs, inc, e=0, w=0
                )
                # raw data and binned
                tbin, ybin, _ = binning_equal_interval(t-tc, f, e, binsize, int(tc))
    
                ax[0].plot(t-tc, f, ".k", alpha=0.1)
                ax[0].plot(tbin, ybin, "o", color=colors[band], alpha=0.5)
                # flux with trend
                ax[0].plot(t-tc, flux_tr * trends[band][inst], lw=3, c=colors[band])
                # ax[0].set_xlabel(f"BJD-{int(tc)}")
                ax[0].set_xlabel("Time from mid-transit (d)")
                ax[0].set_ylabel("Normalized Flux")
                if ylims:
                    ax[0].set_ylim(*ylims)
    
                # detrended and binned
                tbin, ybin, _ = binning_equal_interval(t-tc, f / trends[band][inst], e, binsize, t0)
                ax[1].plot(t-tc, f / trends[band][inst], ".k", alpha=0.1)
                ax[1].plot(tbin, ybin, "o", color=colors[band], alpha=0.5)
                # upsampled transit
                xmodel, ymodel = transits[band][inst]
                ax[1].plot(xmodel-tc, ymodel, lw=3, c=colors[band])
                _ = self.plot_ing_egr(ax=ax[1], ymin=0.9, ymax=1.0, color="C0")
                ax[1].axhline(
                    1 - depth,
                    color="blue",
                    linestyle="dashed",
                    # label="TESS",
                    alpha=0.5,
                )
                if ylims:
                    ax[1].set_ylim(*ylims)
                # ax[1].set_xlabel(f"BJD-{int(tc)}")
                ax[1].set_xlabel("Time from mid-transit (d)")
                # ax[1].legend(loc="best")
                fig.suptitle(f"{inst}/{band}")
            i+=1
        # fig.tight_layout()
        return fig

    def get_upsampled_transit_models(self, pv, npoints=200):
        """
        pv : list
            parameter vector from MCMC or optimization

        returns a dict = {band: (time,flux_transit)}
        """
        assert len(pv) == self.ndim
        # unpack free parameters
        if self.use_r1r2:
            t0, period, a_Rs, r1, r2, d = self.unpack_parameters(pv)
            imp, k = r1r2_to_imp_k(r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX)
        else:
            t0, period, a_Rs, imp, k, d = self.unpack_parameters(pv)
        # derived
        inc = np.arccos(imp / a_Rs)

        models = {}
        i = 0
        for band,dfs in self.data.items():
            models[band] = {}
            for inst,df in dfs.items():
                t = self.times[band][inst]
                tc = self.epoch_tcs[band][inst]
                
                xmodel = np.linspace(np.min(t), np.max(t), npoints)
                tmodel = QuadraticModel()
                tmodel.set_data(xmodel)
                ymodel = tmodel.evaluate_ps(
                    k[i], self.ldc[band], tc, period, a_Rs, inc, e=0, w=0
                )
                models[band][inst] = (xmodel, ymodel)
            i+=1
        return models

    def get_trend_models(self, pv):
        """
        pv : list
            parameter vector from MCMC or optimization

        returns a dict {band: (time,flux_transit)}
        """
        assert len(pv) == self.ndim
        # unpack free parameters
        if self.use_r1r2:
            t0, period, a_Rs, r1, r2, d = self.unpack_parameters(pv)
            imp, k = r1r2_to_imp_k(r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX)
        else:
            t0, period, a_Rs, imp, k, d = self.unpack_parameters(pv)
        # derived
        inc = np.arccos(imp / a_Rs)
        
        models = {}
        i = 0
        for band,dfs in self.data.items():
            models[band] = {}
            for inst,df in dfs.items():
                t = self.times[band][inst]
                f = self.fluxes[band][inst]
                z = self.covariates[band][inst]
                tc = self.epoch_tcs[band][inst]
                
                flux_tr = self.transit_models[band][inst].evaluate_ps(
                    k[i], self.ldc[band], tc, period, a_Rs, inc, e=0, w=0
                )
                flux_tr_time = d[i] * (t - tc) * flux_tr
                c = np.polyfit(z, (f - flux_tr_time) / flux_tr, self.lm_order)
                trend = np.polyval(c, z) + d[i] * (t - tc)
                models[band][inst] = trend
            i+=1
        return models        
        
    def get_chi2_transit(self, pv):
        """
        fixed parameters
        ----------------
        period : fixed

        free parameters
        ---------------
        t0 : mid-transit
        imp : impact parameter
        a_Rs : scaled semi-major axis
        k : radius ratio = Rp/Rs
        d : linear model coefficients

        TODO: add argument to set (normal) priors on T0, Tdur, and a_Rs
        """
        # unpack free parameters
        if self.use_r1r2:
            t0, period, a_Rs, r1, r2, d = self.unpack_parameters(pv)
            imp, k = r1r2_to_imp_k(r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX)

            # uniform priors
            if (r1 < 0.0) or (r1 > 1.0):
                if self.DEBUG:
                    print(f"Error (r1): 0<{r1:.2f}<1")
                return np.inf
            if np.any(r2 < 0.0) or np.any(r2 > 1.0):
                if self.DEBUG:
                    print(f"Error (r2): 0<{r2}<1")
                return np.inf
        else:
            # use imp and k
            t0, period, a_Rs, imp, k, d = self.unpack_parameters(pv)

            # uniform priors
            if np.any(k < PRIOR_K_MIN):
                if self.DEBUG:
                    print(f"Error (k_min): {k:.2f}<{PRIOR_K_MIN}")
                    print(f"You may need to decrease PRIOR_K_MIN={PRIOR_K_MIN}")
                return np.inf
            if np.any(k > PRIOR_K_MAX):
                if self.DEBUG:
                    print(f"Error (k_max): {k:.2f}>{PRIOR_K_MAX}")
                    print(f"You may need to increase PRIOR_K_MAX={PRIOR_K_MAX}")
                return np.inf
            if (imp < 0.0) or (imp > 1.0):
                if self.DEBUG:
                    print(f"Error (imp): 0<{imp:.2f}<1")
                return np.inf

        # derived
        inc = np.arccos(imp / a_Rs)
        tdur = tdur_from_per_imp_aRs_k(period, imp, a_Rs, np.mean(k))
        if a_Rs <= 0.0:
            if self.DEBUG:
                print(f"Error (a_Rs): {a_Rs:.2f}<0")
            return np.inf
        if imp / a_Rs >= 1.0:
            if self.DEBUG:
                print(f"Error (imp/a_Rs): {imp:.2f}/{a_Rs:.2f}>1")
            return np.inf
        # if np.any(d < -0.1) or np.any(d > 0.1):
        #     return np.inf
            
        chi2 = 0.0
        i = 0
        for band,dfs in self.data.items():
            for inst,df in dfs.items():
                t = self.times[band][inst]
                f = self.fluxes[band][inst]
                e = self.flux_errs[band][inst]
                z = self.covariates[band][inst]
                tc = self.epoch_tcs[band][inst]
                
                flux_tr = self.transit_models[band][inst].evaluate_ps(
                    k[i], self.ldc[band], tc, period, a_Rs, inc, e=0, w=0
                )
                flux_tr_time = d[i] * (t - tc) * flux_tr
                c = np.polyfit(z, (f - flux_tr_time) / flux_tr, self.lm_order)
                trend = np.polyval(c, z) + d[i] * (t - tc)
                model = trend * flux_tr
                chi2 = chi2 + np.sum((f - model) ** 2 / e**2)
            i+=1
        # add normal priors
        epoch0 = self.planet_params['t0']
        period0 = self.planet_params['period']
        tdur0 = self.planet_params['tdur']
        a_Rs0 = self.planet_params['a_Rs']
        # tc shouldn't be more or less than half a period
        if abs(t0-epoch0[0]) > period0[0]/2:
            if self.DEBUG:
                    errmsg = f"Error (tc more or less than half of period): "
                    errmsg += f"abs({t0:.4f}-{epoch0[0]:.4f}) > {period0[0]/2:.4f}"
                    print(errmsg)
            return np.inf
        else:
            chi2 += ((tc - epoch0[0])/epoch0[1])**2
        # if abs(period-period0[0]) > 1e-1:
        if period<0:
            if self.DEBUG:
                print("Error (period<0)")
            return np.inf
        else:
            chi2 += ((period - period0[0])/period0[1])**2
        if a_Rs > 0.:
            chi2 += ((a_Rs - a_Rs0[0]) / a_Rs0[1]) ** 2
        if tdur > 0.:
            chi2 += ((tdur - tdur0[0]) / tdur0[1]) ** 2
        return chi2

    def plot_ing_egr(self, ax, ymin=0.9, ymax=1.0, color="C0"):
        """
        plot ingress and egress timings over detrended light curve plot
        """
        tdur, tdure = self.planet_params["tdur"]
        ing = - tdur / 2
        egr = tdur / 2
        
        ax.axvspan(
            ing - tdure / 2,
            ing + tdure / 2,
            alpha=1,
            ymin=ymin,
            ymax=ymax,
            color=color,
        )
        ax.axvspan(
            ing - 3 * tdure / 2,
            ing + 3 * tdure / 2,
            alpha=0.5,
            ymin=ymin,
            ymax=ymax,
            color=color,
        )
        ax.axvspan(
            egr - tdure / 2,
            egr + tdure / 2,
            alpha=1,
            ymin=ymin,
            ymax=ymax,
            color=color,
        )
        ax.axvspan(
            egr - 3 * tdure / 2,
            egr + 3 * tdure / 2,
            alpha=0.5,
            ymin=ymin,
            ymax=ymax,
            color=color,
        )
        return ax

    def neg_loglikelihood(self, pv):
        raise NotImplementedError("unstable")
        return -np.log(self.get_chi2_transit(pv))

    def neg_likelihood(self, pv):
        return -self.get_chi2_transit(pv)

    def optimize_chi2_transit(self, p0, method="Nelder-Mead"):
        """
        Optimize parameters using `scipy.minimize`
        Uses previous optimized parameters if run again
        """
        if hasattr(self, "opt_result"):
            pv = self.opt_result.x
        elif p0 is not None:
            assert len(p0) == self.ndim
            pv = p0
        else:
            pv = self.pv_updated

        self.opt_result = minimize(
            self.get_chi2_transit, pv, method=method
        )
        if self.opt_result.success:
            for a,(n,i,j) in enumerate(zip(self.model_param_names, 
                                           self.pv_init, 
                                           self.opt_result.x)):
                if a==0:
                    tc = int(i)
                    n=f"t0-{tc}"
                    i=i-tc
                    j=j-tc
                    print("Parameter        Init.     Opt.  %Diff")
                print(f"{n:12}: {i:8.4f}{j:8.4f}{((i-j)/j)*100:8.1f}")
            self.optimum_params = self.opt_result.x
        else:
            print("Caution: Optimization **NOT** successful!")

    def sample_mcmc(self, pv=None, nsteps=1_000, nwalkers=None):
        """
        pv : list
            parameter vector (uses optimized values if None)
        """
        if self.DEBUG:
            print("Setting DEBUG=False.")
            self.DEBUG = False
        self.nwalkers = 10 * self.ndim if nwalkers is None else nwalkers
        # if hasattr(self, 'sampler'):
        #     params = self.sampler
        self.nsteps = nsteps
        if pv is not None:
            params = pv
        elif hasattr(self, 'optimum_params'):
            params = self.optimum_params if pv is None else pv
        else:
            raise ValueError('Run `optimize_chi2_transit()` first.')
        assert len(params) == self.ndim
        pos = [
            params + 1e-5 * np.random.randn(self.ndim) for i in range(self.nwalkers)
        ]
        with Pool(self.ndim) as pool:
            self.sampler = emcee.EnsembleSampler(
                self.nwalkers, self.ndim, self.neg_likelihood, pool=pool
            )
            state = self.sampler.run_mcmc(pos, self.nsteps // 2, progress=True)
            # if reset:
            self.sampler.reset()
            self.sampler.run_mcmc(state, self.nsteps, progress=True)

        # Extract and analyze the results
        self.analyze_mcmc_results()

    def analyze_mcmc_results(self):
        log_prob = self.sampler.get_log_prob()
        argmax = np.argmax(log_prob)
        self.best_fit_params = self.sampler.flatchain[argmax]
        # compute bic
        j = int(argmax / self.nwalkers)
        i = argmax - self.nwalkers * j
        self.chi2_best = -log_prob[j, i]
        # print(chi2_best)
        npar_tr = len(self.best_fit_params)  # +4
        # print('ndata = ', self.ndata)
        # print('npar(transit+linear) = ', npar_tr)
        self.bic = self.chi2_best + npar_tr * np.log(self.ndata)
        # print('BIC(transit+linear) = ', bic_tr)
        if not hasattr(self, "bic_lin"):
            self.optimize_chi2_linear_baseline()
        self.bic_delta = self.bic_lin - self.bic
        # print('delta_BIC = ', delta_bic)

    def get_mcmc_samples(self, discard=1, thin=1):
        """
        samples are converted from r1,r2 to imp and k
        """
        # FIXME: using get_chain() overwrites the chain somehow!
        # fc = self.sampler.get_chain(flat=True, discard=discard, thin=thin).copy()
        if hasattr(self, "mcmc_samples"):
            fc = self.sampler.flatchain.copy()
            param_names = self.model_param_names.copy()
            fc = fc.reshape(self.nsteps, self.nwalkers, -1)
            fc = fc[discard::thin].reshape(-1, self.ndim)
            df = pd.DataFrame(fc, columns=self.model_param_names)
            if self.use_r1r2:
                print("Converting r1,r2 --> imp,k")
                r1s = df["r1"].values
                if self.model == "chromatic":
                    for band in self.bands:
                        col = f"r2_{band}"
                        r2s = df[col].values
                        imps = np.zeros_like(r1s)
                        ks = np.zeros_like(r1s)
                        for i, (r1, r2) in enumerate(zip(r1s, r2s)):
                            imps[i], ks[i] = r1r2_to_imp_k(
                                r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX
                            )
                        df[f"k_{band}"] = ks
                        df = df.drop(labels=col, axis=1)
                else:
                    col = "r2"
                    r2s = df[col].values
                    imps = np.zeros_like(r1s)
                    ks = np.zeros_like(r1s)
                    for i, (r1, r2) in enumerate(zip(r1s, r2s)):
                        imps[i], ks[i] = r1r2_to_imp_k(
                            r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX
                        )
                    df["k"] = ks
                    df = df.drop(labels=col, axis=1)
                df["imp"] = imps
                param_names = [s.replace("r1", "imp") for s in param_names]
                param_names = [s.replace("r2", "k") for s in param_names]
            self._mcmc_samples = df[param_names]
            return df[param_names]
        else:
            return self.mcmc_samples[discard::thin]
    
    def plot_chain(self, start=0, end=None, figsize=(10, 10)):
        """
        visualize MCMC walkers

        start : int
            parameter id (0 means first)
        end : int
            parameter id
        """
        end = self.ndim if end is None else end
        fig, axes = pl.subplots(end - start, figsize=figsize, sharex=True)
        samples = self.sampler.get_chain()
        for i in np.arange(start, end):
            ax = axes[i]
            ax.plot(samples[:, :, start + i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.model_param_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        return fig

    def plot_corner(self, transform=True, discard=1, thin=1, start=0, end=None):
        """
        corner plot of MCMC chain

        start : int
            parameter id (0 means first)
        end : int
            parameter id

        TODO show priors in `truths` argument of corner
        """
        end = self.ndim if end is None else end
        if self.use_r1r2 and transform:
            df = self.get_mcmc_samples(discard=discard, thin=thin)
            labels = df.columns.values.copy()
            fig = corner.corner(
                df.iloc[:, start:end], labels=labels[start:end], show_titles=True
            )
        else:
            labels = self.model_param_names.copy()
            fc = self.sampler.flatchain.copy()
            if discard > 1:
                fc = fc.reshape(self.nsteps, self.nwalkers, -1)
                fc = fc[discard::thin].reshape(-1, self.ndim)
            fig = corner.corner(
                fc[:, start:end], labels=labels[start:end], show_titles=True
            )
        return fig

    def plot_detrended_data_and_transit(
        self,
        pv: list,
        title: str = None,
        xlims: tuple = None,
        ylims: tuple = None,
        binsize: float = 600 / 86400,
        msize: int = 5,
        font_size: int = 20,
        title_height: float = 0.95,
        figsize: tuple = (10, 10),
    ):
        """
        pv : list
            parameter vector (uses optimized values if None)

        - 2x2 plot of detrended data with transit model
        - time is in units of hours

        See also `plot_detrended_data_and_transit()`
        """
        title = (
            f"{self.name}{self.alias}"
            if title is None
            else title
        )
        ncols = 2 if self.ndatasets > 1 else 1
        nrows = int(np.ceil(self.ndatasets/2)) if self.ndatasets > 2 else 1
        fig, axs = pl.subplots(
            ncols,
            nrows,
            figsize=figsize,
            sharey="row",
            sharex="col",
            tight_layout=True,
        )
                
        ax = axs.flatten()
        depth = self.planet_params["rprs"][0] ** 2
        t0, period, _, _, _, _ = self.unpack_parameters(pv)

        i = 0
        for band,dfs in self.data.items():
            for inst,df in dfs.items():
                t = self.times[band][inst]
                f = self.fluxes[band][inst]
                e = self.flux_errs[band][inst]
                tc = self.epoch_tcs[band][inst]
                detrended_flux = f / self.get_trend_models(pv)[band][inst]
                ax[i].plot((t - tc) * 24, detrended_flux, "k.", alpha=0.2)
                # raw data and binned
                tbin, ybin, yebin = binning_equal_interval(
                    t, detrended_flux, e, binsize, tc
                )
                ax[i].errorbar(
                    (tbin - tc) * 24, ybin, yerr=yebin, fmt="ok", markersize=msize
                )
                xmodel, ymodel = self.get_upsampled_transit_models(pv, npoints=500)[band][inst]
                ax[i].plot((xmodel - tc) * 24, ymodel, "-", lw=3, color=colors[band])
                ax[i].axhline(
                    1 - depth,
                    color="blue",
                    linestyle="dashed",
                    # label="TESS",
                    alpha=0.5,
                )
                if self.nband % 2 == 1:
                    ax[i].set_ylabel("Relative Flux", fontsize=font_size * 0.8)
                ax[i].set_xlabel(
                    "Time from transit center (hours)", fontsize=font_size * 0.8
                )
                if (self.nband > 2) and (i < 2):
                    ax[i].set_xlabel("")
    
                if xlims is None:
                    xmin, xmax = ax[i].get_xlim()
                else:
                    ax[i].set_xlim(*xlims)
                    xmin, xmax = xlims
    
                if ylims is None:
                    ymin, ymax = ax[i].get_ylim()
                else:
                    ax[i].set_ylim(*ylims)
                    ymin, ymax = ylims
                ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
                ax[i].yaxis.set_minor_locator(AutoMinorLocator(5))
                ax[i].tick_params(labelsize=font_size * 0.8)
                tx = xmin + (xmax - xmin) * 0.75
                ty = ymin + (ymax - ymin) * 0.9
                ax[i].text(
                    tx, ty, f"{inst}/{band}", color=colors[band], fontsize=font_size * 0.8
                )
        # ax[1].set_title(f"", loc="right", fontsize=font_size)
        # ax[i].legend(loc="best")
        fig.suptitle(title, fontsize=font_size, y=title_height)
        return fig

    def plot_posteriors(
            self,
            title: str = None,
            figsize: tuple = (12, 5),
            font_size: float = 12,
            nsigma: float = 3
        ):
            """
            plot Rp/Rs, Tc and impact parameter posteriors
            """
            errmsg = "Valid only for chromatic model"
            if not self.model == "chromatic":
                raise ValueError(errmsg)
            title = (
                f"{self.name}{self.alias}"
                if title is None
                else title
            )
    
            df = self.get_mcmc_samples()
    
            fig, axs = pl.subplots(1, 3, figsize=figsize)
            ax = axs.flatten()
    
            ############# Rp/Rs
            for i, band in enumerate(self.bands):
                k = df["k_" + b].values
                k_med = df["k_" + b].median()
                k_percs = percentile(k)
                # med, low1, hig1, low2, hig2, low3, hig3
                k_err1 = k_percs[0] - k_percs[1], k_percs[2] - k_percs[0]
                k_err2 = k_percs[0] - k_percs[3], k_percs[4] - k_percs[0]
                k_err3 = k_percs[0] - k_percs[5], k_percs[6] - k_percs[0]
                ax[0].errorbar(
                    i,
                    k_med,
                    yerr=np.c_[k_err1].T,
                    elinewidth=40,
                    fmt="none",
                    alpha=0.5,
                    zorder=1,
                    color=colors[band],
                )
                ax[0].errorbar(
                    i,
                    k_med,
                    yerr=np.c_[k_err2].T,
                    elinewidth=40,
                    fmt="none",
                    alpha=0.3,
                    zorder=2,
                    color=colors[band],
                )
                ax[0].errorbar(
                    i,
                    k_med,
                    yerr=np.c_[k_err3].T,
                    elinewidth=40,
                    fmt="none",
                    alpha=0.1,
                    zorder=3,
                    color=colors[band],
                )
                print(f"Rp/Rs({band})^2 = {1e3*k_med**2:.2f} ppt")
    
            k0 = self.planet_params["rprs"][0]
            ax[0].axhline(k0, linestyle="dashed", color="black", label="TESS")
            ax[0].legend(loc="best")
            ax[0].set_xlim(-0.5, 3.5)
            ax[0].set_xticks(range(self.nband))
            ax[0].set_xticklabels(self.bands)
            ax[0].set_xlabel("Band", fontsize=font_size * 1.5)
            ax[0].set_ylabel("Radius ratio", fontsize=font_size * 1.5)
    
            ax[0].text(
                0.0,
                1.12,
                title,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax[0].transAxes,
                fontsize=font_size * 1.5,
            )
            text = f"{self.model.title()} transit fit, "
            text += f"$\Delta$BIC (non-transit - transit) = {self.bic_delta:.1f}"
            ax[0].text(
                0.0,
                1.05,
                text,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax[0].transAxes,
                fontsize=font_size,
            )
            ############# Mid-transit
            # posterior
            t0 = df["t0"].values
            t0 = tc - int(tc)
            n = self.plot_kde(t0, ax=ax[1], label="Posterior")
            # prediction
            t00, t0_err0 = np.array(self.planet_params['t0'])-int(tc)
            xmodel = np.linspace(t00 - nsigma * t0_err0, t0 + nsigma * t0_err0, 200)
            ymodel = n * np.exp(-((xmodel - t00) ** 2) / t0_err0**2)
            ax[1].plot(xmodel, ymodel, label="Prediction", color="C1", lw=3, zorder=0)
            ax[1].set_xlabel(
                f"T0 (BJD-{int(tc)})",
                fontsize=font_size * 1.5,
            )
            ax[1].legend(loc="best")
    
            imp = df["imp"].values
            _ = self.plot_kde(imp, ax=ax[2], color="C0")
            ax[2].set_xlabel("Impact parameter", fontsize=font_size * 1.5)
            # ax[2].set_title(f"", loc="right", fontsize=font_size * 1.5)
            return fig
    
    def plot_final_fit(
        self,
        discard: int = 1,
        thin: int = 1,
        nsamples: int = 100,
        ylims_top: tuple = (0.9, 1.02),
        ylims_bottom: tuple = (0.9, 1.02),
        msize: int = 5,
        font_size: int = 25,
        title: str = None,
        figsize: tuple = (16, 12),
        binsize: float = 600 / 86400
    ):
        ymin1, ymax1 = ylims_top
        ymin2, ymax2 = ylims_bottom
    
        fig, ax = pl.subplots(
            2, self.ndatasets, figsize=figsize, sharey="row", sharex="col"
        )
        pl.subplots_adjust(hspace=0.1, wspace=0)
        
        # unpack free parameters
        if not hasattr(self, "best_fit_params"):
            raise ValueError("Run `sample_mcmc()` first.")
    
        pv = self.best_fit_params
        if self.use_r1r2:
            t0_best, per_best, a_Rs_best, r1, r2, d_best = self.unpack_parameters(pv)
            imp_best, k_best = r1r2_to_imp_k(
                r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX
            )
        else:
            t0_best, per_best, a_Rs_best, imp_best, k_best, d_best = self.unpack_parameters(pv)
    
        # derived
        inc_best = np.arccos(imp_best / a_Rs_best)
    
        trends_best = self.get_trend_models(self.best_fit_params)
        transits_best = self.get_upsampled_transit_models(self.best_fit_params)
    
        # fc = self.sampler.get_chain(flat=True, discard=discard, thin=thin)
        fc = self.sampler.flatchain.copy()
        if discard > 1:
            fc = fc.reshape(self.nsteps, self.nwalkers, -1)
            fc = fc[discard::thin].reshape(-1, self.ndim)
        i = 0
        for n,(band,dfs) in enumerate(self.data.items()):
            for inst, df in dfs.items():
                t = self.times[band][inst]
                f = self.fluxes[band][inst]
                e = self.flux_errs[band][inst]
                z = self.covariates[band][inst]
                tc = self.epoch_tcs[band][inst]
                dt = t-int(tc)
                # raw and binned data
                tbin, ybin, yebin = binning_equal_interval(dt, f, e, binsize, tc)
                ax[0, i].plot(dt, f, ".k", alpha=0.1)
                ax[0, i].errorbar(tbin, ybin, yerr=yebin, fmt="ok", markersize=msize)
    
                # plot each random mcmc samples
                rand = np.random.randint(len(fc), size=nsamples)
                for j in range(len(rand)):
                    idx = rand[j]
                    # unpack free parameters
                    if self.use_r1r2:
                        tc, per, a_Rs, r1 = fc[idx, : self.k_idx]
                        if self.model == "chromatic":
                            r2 = fc[idx, self.k_idx : self.d_idx]
                        elif self.model == "achromatic":
                            r2 = np.zeros(self.nband) + fc[idx, self.k_idx]
                        imp, k = r1r2_to_imp_k(
                            r1, r2, k_lo=PRIOR_K_MIN, k_up=PRIOR_K_MAX
                        )
                    else:
                        t0, per, a_Rs, imp = fc[idx, : self.k_idx]
                        if self.model == "chromatic":
                            k = fc[idx, self.k_idx : self.d_idx]
                        elif self.model == "achromatic":
                            k = np.zeros(self.nband) + fc[idx, self.k_idx]
                    d = fc[idx, self.d_idx : self.d_idx + self.nband]
                    # derived parameters
                    inc = np.arccos(imp / a_Rs)
                    # transit
                    flux_tr = self.transit_models[band][inst].evaluate_ps(
                        k[n], self.ldc[band], tc, per, a_Rs, inc, e=0, w=0
                    )
                    flux_tr_time = d[i] * (t - tc) * flux_tr
                    c = np.polyfit(z, (f - flux_tr_time) / flux_tr, self.lm_order)
                    # transit with trend
                    ax[0, i].plot(
                        dt,
                        flux_tr * (np.polyval(c, z) + d[i] * (t - tc)),
                        alpha=0.05,
                        color=colors[band],
                    )
    
                # best-fit transit model
                flux_tr = self.transit_models[band][inst].evaluate_ps(
                    k_best[i],
                    self.ldc[band],
                    t0_best,
                    per_best,
                    a_Rs_best,
                    inc_best,
                    e=0,
                    w=0,
                )
    
                tbin, ybin, yebin = binning_equal_interval(
                    dt, f / trends_best[band][inst], e, binsize, t0
                )
                # detrended flux
                ax[1, i].plot(dt, f / trends_best[band][inst], ".k", alpha=0.1)
                ax[1, i].errorbar(tbin, ybin, yerr=yebin, fmt="ok", markersize=msize)
                # super sampled best-fit transit model
                xmodel, ymodel = transits_best[band][inst]
                ax[1, i].plot(xmodel-int(tc), ymodel, color=colors[band], linewidth=3)
                ax[0, i].yaxis.set_minor_locator(AutoMinorLocator(5))
                ax[1, i].yaxis.set_minor_locator(AutoMinorLocator(5))
                ax[0, i].xaxis.set_minor_locator(AutoMinorLocator(5))
                ax[1, i].xaxis.set_minor_locator(AutoMinorLocator(5))
                ax[0, i].set_ylim(ymin1, ymax1)
                ax[1, i].set_ylim(ymin2, ymax2)
    
                tx = np.min(dt) + (np.max(dt) - np.min(dt)) * 0.75
                ty = ymin1 + (ymax1 - ymin1) * 0.9
                ax[0, i].text(
                    tx, ty, f"{inst}/{band}", color=colors[band], fontsize=font_size * 0.6
                )
                tx = np.min(dt) + (np.max(dt) - np.min(dt)) * 0.02
                ty = ymin2 + (ymax2 - ymin2) * 0.8
                ax[1, i].text(tx, ty, "Detrended", fontsize=font_size * 0.6)
    
                rms = np.std(
                    f - flux_tr * (np.polyval(c, z) + d_best[i] * (t - tc))
                )
                rms_text = f"rms = {rms:.4f}"
                ty = ymin2 + (ymax2 - ymin2) * 0.1
                ax[1, i].text(tx, ty, rms_text, fontsize=font_size * 0.6)
                depth = self.planet_params["rprs"][0] ** 2
                ax[1, i].axhline(
                    1 - depth,
                    color="blue",
                    linestyle="dashed",
                    label="TESS",
                    alpha=0.5,
                )
                _ = self.plot_ing_egr(ax=ax[1, i], ymin=0.9, ymax=1.0, color="C0")
                if i == 0:
                    ax[0, i].set_ylabel("Flux ratio", fontsize=font_size)
                    ax[1, i].set_ylabel("Flux ratio", fontsize=font_size)
                    ax[0, i].tick_params(labelsize=16)
                    ax[1, i].tick_params(labelsize=16)
                    target_name = (
                        f"{self.name}{self.alias}"
                        if title is None
                        else title
                    )
                    ax[0, i].text(
                        0.0,
                        1.14,
                        target_name,
                        horizontalalignment="left",
                        verticalalignment="center",
                        transform=ax[0, i].transAxes,
                        fontsize=font_size,
                    )
                    text = f"{self.model.title()} transit fit,  "
                    text += f"$\Delta$BIC (non-transit - transit) = {self.bic_delta:.1f}"
                    ax[0, i].text(
                        0.0,
                        1.05,
                        text,
                        horizontalalignment="left",
                        verticalalignment="center",
                        transform=ax[0, i].transAxes,
                        fontsize=font_size * 0.6,
                    )
                # if (i > 0) and (i == self.nband - 1):
                #     ax[0, i].set_title("", loc="right", fontsize=font_size * 0.8)
                if i > 0:
                    ax[0, i].tick_params(
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        labelsize=16,
                    )
                    ax[1, i].tick_params(
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        labelsize=16,
                    )
                ax[1, i].set_xlabel(
                    f"BJD - {int(tc):.0f}",
                    labelpad=30,
                    fontsize=font_size,
                )
            i+=1
        ax[1, i].legend(loc="lower right", fontsize=12)
        return fig

    def plot_radii(
        self, rstar, rstar_err, fill=False, unit=u.Rjup, figsize=(5, 5), alpha=0.5
    ):
        fig, ax = plt.subplots(figsize=figsize)
        df = self.get_mcmc_samples()
        rstar_samples = np.random.normal(rstar, rstar_err, size=len(df))
        if self.model == "chromatic":
            cols = [f"k_{band}" for band in self.bands]
            d = df[cols].apply(lambda x: x * rstar_samples * u.Rsun.to(unit))
            d.columns = self.bands
            for column, color in colors.items():
                _ = self.plot_kde(
                    d[column].values,
                    ax=ax,
                    color=color,
                    label=column,
                    alpha=alpha,
                    fill=fill,
                )
        else:
            d = df["k"] * rstar_samples * u.Rsun.to(unit)
            d.name = "achromatic Rp"
            _ = self.plot_kde(
                d.values, ax=ax, color=color, label=column, alpha=alpha, fill=fill
            )

        ax.set_xlabel(f"Companion radius ({unit._format['latex']})")
        ax.set_title(
            f"(assuming Rs={rstar:.2f}+/-{rstar_err:.2f}" + r" $R_{\odot}$)"
        )
        Rp_tfop = self.planet_params["rprs"][0] * rstar * u.Rsun.to(unit)
        ax.axvline(
            Rp_tfop, 0, 1, c="k", ls="--", lw=2, label=f"Rp={Rp_tfop:.1f}\n(TFOP)"
        )
        ax.set_ylabel("Density")
        ax.legend()
        return fig
        
@dataclass
class Star:
    name: str
    star_params: Dict[str, Tuple[float, float]] = None
    source: str = "tic"

    def __post_init__(self):
        if self.star_params is None:
            self.get_star_params()

        sources = set(
            [
                p.get("prov")
                for i, p in enumerate(self.data_json["stellar_parameters"])
            ]
        )
        errmsg = f"{self.source} must be in {sources}"
        assert self.source in sources, errmsg

    def get_tfop_data(self):
        base_url = "https://exofop.ipac.caltech.edu/tess"
        self.exofop_url = (
            f"{base_url}/target.php?id={self.name.replace(' ','')}&json"
        )
        response = urlopen(self.exofop_url)
        assert response.code == 200, "Failed to get data from ExoFOP-TESS"
        try:
            data_json = json.loads(response.read())
            return data_json
        except Exception:
            raise ValueError(f"No TIC data found for {self.name}")

    def get_star_params(self):
        if not hasattr(self, "data_json"):
            self.data_json = self.get_tfop_data()

        self.ra = float(self.data_json["coordinates"].get("ra"))
        self.dec = float(self.data_json["coordinates"].get("dec"))
        self.ticid = self.data_json["basic_info"].get("tic_id")

        idx = 1
        for i, p in enumerate(self.data_json["stellar_parameters"]):
            if p.get("prov") == self.source:
                idx = i + 1
                break
        star_params = self.data_json["stellar_parameters"][idx]

        try:
            self.rstar = tuple(
                map(
                    float,
                    (
                        star_params.get("srad", np.nan),
                        star_params.get("srad_e", np.nan),
                    ),
                )
            )
            self.mstar = tuple(
                map(
                    float,
                    (
                        star_params.get("mass", np.nan),
                        star_params.get("mass_e", np.nan),
                    ),
                )
            )
            # stellar density in rho_sun
            self.rhostar = (
                self.mstar[0] / self.rstar[0] ** 3,
                np.sqrt(
                    (1 / self.rstar[0] ** 3) ** 2 * self.mstar[1] ** 2
                    + (3 * self.mstar[0] / self.rstar[0] ** 4) ** 2
                    * self.rstar[1] ** 2
                ),
            )
            print(f"Mstar=({self.mstar[0]:.2f},{self.mstar[1]:.2f}) Msun")
            print(f"Rstar=({self.rstar[0]:.2f},{self.rstar[1]:.2f}) Rsun")
            print(f"Rhostar=({self.rhostar[0]:.2f},{self.rhostar[1]:.2f}) rhosun")
            self.teff = tuple(
                map(
                    float,
                    (
                        star_params.get("teff", np.nan),
                        star_params.get("teff_e", 500),
                    ),
                )
            )
            self.logg = tuple(
                map(
                    float,
                    (
                        star_params.get("logg", np.nan),
                        star_params.get("logg_e", 0.1),
                    ),
                )
            )
            val = star_params.get("feh", 0)
            val = 0 if (val is None) or (val == "") else val
            val_err = star_params.get("feh_e", 0.1)
            val_err = 0.1 if (val is None) or (val_err == "") else val_err
            self.feh = tuple(map(float, (val, val_err)))
            print(f"teff=({self.teff[0]:.0f},{self.teff[1]:.0f}) K")
            print(f"logg=({self.logg[0]:.2f},{self.logg[1]:.2f}) cgs")
            print(f"feh=({self.feh[0]:.2f},{self.feh[1]:.2f}) dex")
        except Exception as e:
            print(e)
            raise ValueError(f"Check exofop: {self.exofop_url}")

    def get_gaia_sources(self, rad_arcsec=30):
        target_coord = SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg)
        if (not hasattr(self, "gaia_sources")) or (rad_arcsec > 30):
            msg = f'Querying Gaia sources {rad_arcsec}" around {self.name}: '
            msg += f"({self.ra:.4f}, {self.dec:.4f}) deg."
            print(msg)
            self.gaia_sources = Catalogs.query_region(
                target_coord,
                radius=rad_arcsec * u.arcsec,
                catalog="Gaia",
                version=3,
            ).to_pandas()
            self.gaia_sources["distance"] = self.gaia_sources[
                "distance"
            ] * u.arcmin.to(u.arcsec)
        self.gaia_sources = self.gaia_sources[
            self.gaia_sources["distance"] <= rad_arcsec
        ]
        assert len(self.gaia_sources) > 1, "gaia_sources contains single entry"
        return self.gaia_sources

    def params_to_dict(self):
        return {
            "rstar": self.rstar,
            "mstar": self.rstar,
            "rhostar": self.rhostar,
            "teff": self.teff,
            "logg": self.logg,
            "feh": self.feh,
        }


@dataclass
class Planet(Star):
    name: str
    star_params: Dict[str, Tuple[float, float]]
    alias: str = ".01"
    planet_params: Dict[str, Tuple[float, float]] = None
    source: str = "toi"

    def __post_init__(self):
        self.get_planet_params()

    def get_planet_params(self):
        if not hasattr(self, "data_json"):
            data_json = self.get_tfop_data()

        sources = set(
            [p.get("prov") for i, p in enumerate(data_json["planet_parameters"])]
        )
        errmsg = f"{self.source} must be in {sources}"
        assert self.source in sources, errmsg

        # try:
        #     idx = int(self.alias.replace('.', ''))
        # except:
        #     idx = 1
        idx = 1
        for i, p in enumerate(data_json["planet_parameters"]):
            if p.get("prov") == self.source:
                idx = i + 1
                break
        planet_params = data_json["planet_parameters"][idx]

        try:
            self.t0 = tuple(
                map(
                    float,
                    (
                        planet_params.get("epoch", np.nan),
                        planet_params.get("epoch_e", 0.1),
                    ),
                )
            )
            self.period = tuple(
                map(
                    float,
                    (
                        planet_params.get("per", np.nan),
                        planet_params.get("per_e", 0.1),
                    ),
                )
            )
            self.tdur = (
                np.array(
                    tuple(
                        map(
                            float,
                            (
                                planet_params.get("tdur", 0),
                                planet_params.get("dur_e", 0),
                            ),
                        )
                    )
                )
                / 24
            )
            self.rprs = np.sqrt(
                np.array(
                    tuple(
                        map(
                            float,
                            (
                                planet_params.get("dep_p", 0),
                                planet_params.get("dep_p_e", 0),
                            ),
                        )
                    )
                )
                / 1e6
            )
            self.imp = tuple(
                map(
                    float,
                    (
                        (
                            0
                            if planet_params.get("imp", 0) == ""
                            else planet_params.get("imp", 0)
                        ),
                        (
                            0.1
                            if planet_params.get("imp_e", 0.1) == ""
                            else planet_params.get("imp_e", 0.1)
                        ),
                    ),
                )
            )
            print(f"t0={self.t0} BJD\nP={self.period} d\nRp/Rs={self.rprs}")
            rhostar = self.star_params["rhostar"]
            self.a_Rs = (
                (rhostar[0] / 0.01342 * self.period[0] ** 2) ** (1 / 3),
                1
                / 3
                * (1 / 0.01342 * self.period[0] ** 2) ** (1 / 3)
                * rhostar[0] ** (-2 / 3)
                * rhostar[1],
            )
        except Exception as e:
            print(e)
            raise ValueError(f"Check exofop: {self.exofop_url}")

    def params_to_dict(self):
        return {
            "t0": self.t0,
            "period": self.period,
            "tdur": self.tdur,
            "imp": self.imp,
            "rprs": self.rprs,
            "a_Rs": self.a_Rs,
        }

def read_tess(phot_dir):
    fp = f'{phot_dir}/tess_cpm_s16s23s50.csv'
    df = pd.read_csv(fp)
    df = df.rename({
        'time': 'BJD_TDB',
        'flux': 'Flux', 
        'err': 'Err',
    }, axis=1)
    df['Airmass'] = np.ones(len(df))
    return df


def read_lco1_mcd_g(phot_dir):
    fp = f'{phot_dir}/TIC23863105-01_20230428_LCO-MCD-1m0_gp_measurements.tbl'
    df = pd.read_csv(fp, delim_whitespace=True)
    df = df.rename(lco_phot_mapping, axis=1)
    cols = lco_phot_mapping.values()
    return df[cols]
    
def read_lco0_tei_z(phot_dir):
    fp = f'{phot_dir}/TIC119585136-01_20220618_LCO-TEID-0m4_zs_measurements.tbl'
    df = pd.read_csv(fp, delim_whitespace=True)
    df = df.rename(lco_phot_mapping, axis=1)
    cols = lco_phot_mapping.values()
    return df[cols]

def read_muscat3(phot_dir):
    dfs = {}
    cols = ['BJD_TDB', 'Flux', 'Err', 'Airmass']
    for band in ['g','r','i','z']:
        fp = f'{phot_dir}/TOI5671.01L-jd20220630_MuSCAT3_{band}_measurements.csv'
        dfs[band] = pd.read_csv(fp)[cols]    
    return dfs

def read_muscat2(phot_dir, date):
    """"""
    data = {}
    bands = ['r','i','zs'] if date==230510 else ['g','r','i','zs']
    for band in bands:
        fp = f'{phot_dir}/muscat2/phot/TOI5634-01_20{date}_{band}_TCS_MuSCAT2_Raw.dat'
        band = 'z' if band=='zs' else band
        data[band] = pd.read_csv(fp, delim_whitespace=True, 
                                  names=['BJD_TDB','Flux', 'Err'])
    return data

def read_keplercam_i(phot_dir):
    fp = f'{phot_dir}/TIC23863106-01_20220608_KeplerCam_ip.dat'
    df = pd.read_csv(fp, delim_whitespace=True)
    df = df.rename({'rel_flux_T2': 'Flux',
                    'rel_flux_err_T2': 'Err',
                    'AIRMASS': 'Airmass'
                   }, axis=1)
    return df[['BJD_TDB','Flux','Err','Airmass']]
    
def read_lco0_hal_z(phot_dir):
    fp = f'{phot_dir}/TIC23863105-11_20220621_LCO-HAL-0m4_zs_measurements.tbl'
    df = pd.read_csv(fp, delim_whitespace=True)
    df = df.rename(lco_phot_mapping, axis=1)
    cols = lco_phot_mapping.values()
    return df[cols]

def read_lco1_mcd_V(phot_dir):
    fp = f'{phot_dir}/TIC23863105-01_20230326_LCO-McD-1m0_V_measurements.tbl'
    df = pd.read_csv(fp, delim_whitespace=True)
    df = df.rename(lco_phot_mapping, axis=1)
    cols = lco_phot_mapping.values()
    return df[cols]


def read_trappist_I_z1(phot_dir):
    fp = f'{phot_dir}/TIC_119585136-01_20230113_TRAPPIST-North-0.6m_I+z.txt'
    df = pd.read_csv(fp, delim_whitespace=True)
    mapping = {'#BJD-TDB': 'BJD_TDB', 
                'DIFF_FLUX': 'Flux',
                'ERROR': 'Err',
                'AIRMASS': 'Airmass'                    
               }
    df = df.rename(mapping, axis=1)
    cols = mapping.values()
    return df[cols]

def read_trappist_I_z2(phot_dir):
    fp = f'{phot_dir}/TIC_119585136-01_20230124_TRAPPIST-North-0.6m_I+z_measurments.txt'
    df = pd.read_csv(fp, delim_whitespace=True)
    mapping = {'#BJD-TDB': 'BJD_TDB', 
                'DIFF_FLUX': 'Flux',
                'ERROR': 'Err',
                'AIRMASS': 'Airmass'                    
               }
    df = df.rename(mapping, axis=1)
    cols = mapping.values()
    return df[cols]

def read_spec_g(phot_dir):
    fp = f'{phot_dir}/TIC_119585136-01_20230124_SPECULOOS-North-1.0m_gp_Artemis_measurments.txt'
    df = pd.read_csv(fp, delim_whitespace=True)
    mapping = {'#BJD-TDB': 'BJD_TDB', 
                'DIFF_FLUX': 'Flux',
                'ERROR': 'Err',
                'AIRMASS': 'Airmass'                    
               }
    df = df.rename(mapping, axis=1)
    cols = mapping.values()
    return df[cols]

def read_spec_z(phot_dir):
    fp = f'{phot_dir}/TIC_119585136-01_20230327_SPECULOOS-North-1m0_z_measurments.txt'
    df = pd.read_csv(fp, delim_whitespace=True)
    mapping = {'#BJD-TDB': 'BJD_TDB', 
                'DIFF_FLUX': 'Flux',
                'ERROR': 'Err',
                'AIRMASS': 'Airmass'                    
               }
    df = df.rename(mapping, axis=1)
    cols = mapping.values()
    return df[cols]
    
def read_spec_I_z(phot_dir):
    fp = f'{phot_dir}/TIC_23863105-01_20230414_SPECULOOS-North-1m0_I+z_measurments.txt'
    df = pd.read_csv(fp, delim_whitespace=True)
    mapping = {'#BJD-TDB': 'BJD_TDB', 
                'DIFF_FLUX': 'Flux',
                'ERROR': 'Err',
                'AIRMASS': 'Airmass'                    
               }
    df = df.rename(mapping, axis=1)
    cols = mapping.values()
    return df[cols]

def read_all_phot(phot_dir, sort_by='band'):
    """"""
    lc_lco1_mcd_V = read_lco1_mcd_V(phot_dir)
    lc_lco1_mcd_g = read_lco1_mcd_g(phot_dir)
    lc_spec_g = read_spec_g(phot_dir)    
    lcs_muscat3 = read_muscat3(phot_dir)
    lcs_muscat2_230510 = read_muscat2(phot_dir, date=230510)
    lcs_muscat2_240217 = read_muscat2(phot_dir, date=240217)
    lc_keplercam_i = read_keplercam_i(phot_dir)
    lc_spec_z = read_spec_z(phot_dir)
    lc_lco0_tei_z = read_lco0_tei_z(phot_dir)
    lc_lco0_hal_z = read_lco0_hal_z(phot_dir)
    lc_trappist_I_z1 = read_trappist_I_z1(phot_dir)
    lc_trappist_I_z2 = read_trappist_I_z2(phot_dir)
    lc_spec_I_z = read_spec_I_z(phot_dir)
    lc_tess = read_tess(phot_dir)

    lcs = {}
    if sort_by=='band':
        lcs['gp'] = {'muscat3': lcs_muscat3['g'],
                     # 'muscat2': lcs_muscat2_240217['g'],
                     'lco1m': lc_lco1_mcd_g, 
                     #'speculoos': lc_spec_g
                    }
        lcs['V']  = {'lco1m': lc_lco1_mcd_V,}
        lcs['rp'] = {'muscat3': lcs_muscat3['r'],
                     # 'muscat2_a': lcs_muscat2_230510['r'],
                     # 'muscat2_b': lcs_muscat2_240217['r'],
                    }
        lcs['ip'] = {'muscat3': lcs_muscat3['i'], 
                     # 'muscat2_a': lcs_muscat2_230510['i'],
                     # 'muscat2_b': lcs_muscat2_240217['i'],
                     'keplercam': lc_keplercam_i,
                    }
        #lcs['I+z'] = {'speculoos': lc_spec_I_z,
        #             'trappist_a': lc_trappist_I_z1,
        #             'trappist_b': lc_trappist_I_z2,
        #            }
        lcs['zs'] = {'muscat3': lcs_muscat3['z'], 
                     # 'muscat2_a': lcs_muscat2_230510['z'],
                     # 'muscat2_b': lcs_muscat2_240217['z'],
                     #'speculoos': lc_spec_z,
                     #'lco0.4m_a': lc_lco0_tei_z, 
                     'lco0.4m_b': lc_lco0_hal_z
                    }
        lcs['TESS'] = {'TESS': lc_tess}
    elif sort_by=='inst':
        lcs['muscat3'] = {'gp': [lcs_muscat3['g']], 
                          'rp': [lcs_muscat3['r']], 
                          'ip': [lcs_muscat3['i']], 
                          'zs': [lcs_muscat3['z']]}
        # lcs['muscat2_a'] = {#'gp': [lcs_muscat2_230510['g']], 
        #                   'rp': [lcs_muscat2_230510['r']], 
        #                   'ip': [lcs_muscat2_230510['i']], 
        #                   'zs': [lcs_muscat2_230510['z']]}
        # lcs['muscat2_b'] = {'gp': [lcs_muscat2_240217['g']], 
        #                   'rp': [lcs_muscat2_240217['r']], 
        #                   'ip': [lcs_muscat2_240217['i']], 
        #                   'zs': [lcs_muscat2_240217['z']]}
        lcs['lco0.4m'] = {'zs': [lc_lco0_tei_z, lc_lco0_hal_z]}
        lcs['lco1m'] = {'gp': [lc_lco1_mcd_g],
                        'V': [lc_lco1_mcd_V], 
                        'zs': [lc_lco0_tei_z, lc_lco0_hal_z]}
        lcs['speculoos'] = {'gp': [lc_spec_g],
                            'Iz': [lc_spec_I_z], 
                            'zs': [lc_spec_z]}
        lcs['trappist'] = {'Iz': [lc_trappist_I_z1, lc_trappist_I_z2]}
        lcs['keplercam'] = {'ip': [lc_keplercam_i]}
        lcs['tess'] = {'tess': [lc_tess]}
    elif sort_by=='date':
        raise ValueError('sort_by `band` or `inst` for now.')
    else:
        raise ValueError('sort_by `band` or `inst` for now.')
    return lcs

def binning_equal_interval(t, y, ye, binsize, t0):
    intt = np.floor((t - t0) / binsize)
    intt_unique = np.unique(intt)
    n_unique = len(intt_unique)
    tbin = np.zeros(n_unique)
    ybin = np.zeros(n_unique)
    yebin = np.zeros(n_unique)

    for i in range(n_unique):
        index = np.where(intt == intt_unique[i])
        tbin[i] = t0 + float(intt_unique[i]) * binsize + 0.5 * binsize
        w = 1 / ye[index] / ye[index]
        ybin[i] = np.sum(y[index] * w) / np.sum(w)
        yebin[i] = np.sqrt(1 / np.sum(w))

    return tbin, ybin, yebin

def r1r2_to_imp_k(r1, r2, k_lo=0.01, k_up=0.5):
    """
    Efficient Joint Sampling of Impact Parameters and
    Transit Depths in Transiting Exoplanet Light Curves
    Espinosa+2018: RNAAS, 2 209
    https://iopscience.iop.org/article/10.3847/2515-5172/aaef38
    """
    Ar = (k_up - k_lo) / (2.0 + k_lo + k_up)
    if r1 > Ar:
        imp = (1.0 + k_lo) * (1.0 + ((r1 - 1.0) / (1.0 - Ar)))
        k = (1.0 - r2) * k_lo + r2 * k_up
    else:
        q1 = r1 / Ar
        imp = (1.0 + k_lo) + np.sqrt(q1) * r2 * (k_up - k_lo)
        k = k_up + (k_lo - k_up) * np.sqrt(q1) * (1.0 - r2)
    return imp, k


def imp_k_to_r1r2(imp, k, k_lo=0.01, k_up=0.5):
    """
    Inverse function for r1r2_to_imp_k function.
    """
    Ar = (k_up - k_lo) / (2.0 + k_lo + k_up)
    discriminant = 1.0 + 4.0 * (1.0 - Ar) * (imp - (1.0 + k_lo))

    if discriminant >= 0:
        # Case r1 > Ar
        r1 = (1.0 + k_lo + np.sqrt(discriminant)) / (2.0 * (1.0 - Ar))
        r2 = (k - (1.0 - r1) * k_lo) / (r1 * k_up)
    else:
        # Case r1 <= Ar
        q1 = ((1.0 + k_lo) - imp) / (k_up - k_lo)
        q2 = (k_up - k) / (k_up - k_lo * np.sqrt(q1))
        r1 = Ar * q1
        r2 = q2

    return r1, r2

def tdur_from_per_imp_aRs_k(per, imp, a_Rs, k):
    # inc = np.arccos(imp / a_Rs)
    cosi = imp / a_Rs
    sini = np.sqrt(1.0 - cosi**2)
    return (
        per
        / np.pi
        * np.arcsin(1.0 / a_Rs * np.sqrt((1.0 + k) ** 2 - imp**2) / sini)
    )

def get_epoch_tc(times, period, t0):
    n = int(np.floor((max(times)-t0)/period))
    return t0+n*period

@njit(cache=False)
def lnlike_normal(o, m, e):
    return -np.sum(np.log(e)) -0.5*o.size*LOG_TWO_PI - 0.5*np.sum((o-m)**2/e**2)


@njit(cache=False)
def lnlike_normal_s(o, m, e):
    return -o.size*np.log(e) -0.5*o.size*LOG_TWO_PI - 0.5*np.sum((o-m)**2)/e**2