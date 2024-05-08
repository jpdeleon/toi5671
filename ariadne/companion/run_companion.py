"""
Run allesfitter with and without Teff/logg/met priors

See docs: https://github.com/jvines/astroARIADNE#fitter-setup
"""
from astroARIADNE.star import Star
from astroARIADNE.fitter import Fitter
from astroARIADNE.plotter import *

from urllib.request import urlopen
import numpy as np
import sys
import json

target_name = 'TOI 5648'
gaia_id = 1485436090253565056
ra, dec = 211.009596, 38.620245

plot_only = True
rerun_fit = False
use_priors = False
if use_priors:
    out_folder = 'with_prior'
else:
    out_folder = 'without_prior'
    
# Setting up plotter, which is independent to the main fitting routine
if plot_only:
    artist = SEDPlotter(out_folder+'/BMA.pkl', out_folder, pdf=True)
    artist.plot_SED_no_model()  # Plots the stellar SED without the model
    artist.plot_SED()  # Plots stellar SED with model included
    artist.plot_bma_hist()  # Plots bayesian model averaging histograms
    artist.plot_bma_HR(10)  # Plots HR diagram with 10 samples from posterior
    artist.plot_corner()  # Corner plot of the posterior parameters
    sys.exit()

# For stars with Teff > 4000 K BT-Settl, BT-NextGen and BT-Cond are identical and thus only BT-Settl is used
# Kurucz and Castelli & Kurucz are known to work poorly on stars with Teff < 4000 K
models = [
	'phoenix',
	'btsettl',
	'btnextgen',
	'btcond',
	'kurucz',
	'ck04'
]

# remove photometry where the target is not resolved (s<7")
# https://github.com/jvines/astroARIADNE/blob/master/filters.md
mag_dict = {
	'GaiaDR2v2_G': (15.0191,0.000352),
	'GaiaDR2v2_BP': (16.2031,0.0026),
	'GaiaDR2v2_RP': (13.9376,0.0013),
	'2MASS_J': (12.504,0.018),
	'2MASS_H': (11.823,0.024),
	'2MASS_K': (11.633,0.023),
	'WISE_RSR_W1': (11.535,0.026),
	'WISE_RSR_W2': (11.484,0.026),
	}

engine = 'dynesty'
nlive = 500
dlogz = 0.1
bound = 'multi'
sample = 'rwalk'
threads = 30
dynamic = False
dustmap = 'SFD' #'Bayestar'

setup = [engine, nlive, dlogz, bound, sample, threads, dynamic]

#url = f"https://exofop.ipac.caltech.edu/tess/target.php?id={target_name.replace(' ','-')}&json"
#response = urlopen(url)
#data_json = json.loads(response.read())

#ra = float(data_json['coordinates']['ra'])
#dec = float(data_json['coordinates']['dec'])

#dist = float(data_json['stellar_parameters'][1]['dist'])
#dist_err = float(data_json['stellar_parameters'][1]['dist_e'])

s = Star(target_name, ra, dec, g_id=gaia_id, dustmap=dustmap, mag_dict=mag_dict)

f = Fitter()
f.star = s
f.setup = setup
f.av_law = 'fitzpatrick'
f.bma = True #PHOENIXv2 evidence is ~1 to try False?
f.models = models
f.n_samples = 10_000


f.prior_setup = {
	# 'dist': ('normal', dist, dist_err), 
        'dist': ('default'), #from Gaia DR2
	'rad': ('default'), #U[0.5,20] Rsun
	'Av': ('default') #[0, map_max], where map_max is max of line-of-sight as per SFD map
}

if use_priors:
    f.prior_setup['teff'] = ('normal', 3571, 157) #Gaia
    f.prior_setup['logg'] = ('normal', 4.717, 0.01) #TIC
    #f.prior_setup['z'] = ('normal', 0.31, 0.33) #
else:
    # default is RAVE
    f.prior_setup['teff'] = ('default')
    f.prior_setup['logg'] = ('default')
    f.prior_setup['z'] = ('default')

f.out_folder = out_folder
if rerun_fit:
    f.initialize()
    f.fit_bma()
