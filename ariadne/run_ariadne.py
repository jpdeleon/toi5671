from astroARIADNE.star import Star
from astroARIADNE.fitter import Fitter
from astroARIADNE.plotter import *

from urllib.request import urlopen
import json

target_name = 'TOI 5671'
gaia_id = 1485436090253564928

rerun_fit = False
use_TIC_priors = False
use_IRD_priors = False

engine = 'dynesty'
nlive = 500
dlogz = 0.1
bound = 'multi'
sample = 'rwalk'
threads = 6
dynamic = False
dustmap = 'SFD' #'Bayestar'

# remove photometry where the target is not resolved (s<7")
# https://github.com/jvines/astroARIADNE/blob/master/filters.md
mag_dict = {
    #'GALEX_NUV': (22.5280, 0.4720),
    #'SDSS_u': (20.5810, 0.0490),
    #'SDSS_g': (18.0420, 0.0060),
    'GaiaDR2v2_BP': (17.4410, 0.0058),
    #'SDSS_r': (16.5480, 0.0050),
    'GaiaDR2v2_G': (16.0232, 0.0028),
    #'SDSS_i': (15.3390, 0.0050),
    'GaiaDR2v2_RP': (14.8500, 0.0040),
    #'TESS': (14.8084, 0.0075),
    #'SDSS_z': (14.6240, 0.0050),
    '2MASS_J': (13.2470, 0.0280),
    '2MASS_H': (12.6590, 0.0230),
    '2MASS_Ks': (12.3620, 0.0230),
    'WISE_RSR_W1': (12.2270, 0.0340),
    'WISE_RSR_W2': (12.1120, 0.0340),
    }
mags_to_remove = ['GALEX_NUV',
                  'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z',
                  'TESS',
                  #'2MASS_J','2MASS_H','2MASS_Ks',
                  #'WISE_RSR_W1',
                  #'WISE_RSR_W2',
                  ]
setup = [engine, nlive, dlogz, bound, sample, threads, dynamic]

url = f"https://exofop.ipac.caltech.edu/tess/target.php?id={target_name.replace(' ','-')}&json"
response = urlopen(url)
data_json = json.loads(response.read())

ra = float(data_json['coordinates']['ra'])
dec = float(data_json['coordinates']['dec'])

dist = float(data_json['stellar_parameters'][1]['dist'])
dist_err = float(data_json['stellar_parameters'][1]['dist_e'])

s = Star(target_name, ra, dec, g_id=gaia_id, mag_dict=mag_dict, dustmap=dustmap)
#[s.remove_mag(m) for m in mags_to_remove]

# For stars with Teff > 4000 K BT-Settl, BT-NextGen and BT-Cond are identical and thus only BT-Settl is used
# Kurucz and Castelli & Kurucz are known to work poorly on stars with Teff < 4000 K
models = [
	'phoenix',
	'btsettl',
	'btnextgen',
	'btcond',
	#'kurucz',
	#'ck04'
]

f = Fitter()
f.star = s
f.setup = setup
f.av_law = 'fitzpatrick'
f.bma = True #PHOENIXv2 evidence is ~1 to try False?
f.models = models
f.n_samples = 10_000

f.prior_setup = {
	'dist': ('normal', dist, 3*dist_err), #from Gaia DR2
	#'rad': ('default'), #U[0.5,20] Rsun
    'rad': ('uniform', 0.3, 10),
	'Av': ('default') #[0, map_max], where map_max is max of line-of-sight as per SFD map
}

if use_TIC_priors:
    out_folder = 'TIC_priors'
    f.prior_setup['teff'] = ('normal', 3380, 157)
    f.prior_setup['logg'] = ('normal', 4.8, 0.1)
    f.prior_setup['z'] = ('default') 
elif use_IRD_priors:
    out_folder = 'IRD_priors'
    raise NotImplementedError()
else:
    # default is RAVE
    out_folder = 'wo_spec_priors'
    f.prior_setup['teff'] = ('default')
    f.prior_setup['logg'] = ('default')
    f.prior_setup['z'] = ('default')

f.out_folder = out_folder
if rerun_fit:
    f.initialize()
    f.fit_bma()



# Setting up plotter, which is independent to the main fitting routine
if True:
    artist = SEDPlotter(out_folder+'/BMA.pkl', out_folder, pdf=True)
    artist.plot_SED_no_model()  # Plots the stellar SED without the model
    #artist.plot_SED()  # Plots stellar SED with model included
    artist.plot_bma_hist()  # Plots bayesian model averaging histograms
    artist.plot_bma_HR(10)  # Plots HR diagram with 10 samples from posterior
    artist.plot_corner()  # Corner plot of the posterior parameters
