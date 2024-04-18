import sys
sys.path.insert(0, '/home/jp/github/research/project/chronos/chronos/')
import numpy as np
from gls import Gls
import pandas as pd

period= (3.307821,2.8e-06)
epoch = (2459401.27988518, 0.000372)
duration = (1.8008/24, 0.0097/24)

def get_transit_mask(time, period, t0, dur):
    """
    lc : lk.LightCurve
        lightcurve that contains time and flux properties
    """
    if dur >= 1:
        raise ValueError("dur should be in days")

    mask = []
    t0 += np.ceil((time[0] - dur - t0) / period) * period
    for t in np.arange(t0, time[-1] + dur, period):
        mask.extend(np.where(np.abs(time - t) < dur / 2.0)[0])

    return np.array(mask)

df1 = pd.read_csv('./tess_s50.csv', names=['time','flux','err'])
df1 = df1[(df1.flux>0.9) & (df1.flux<1.05)]
time, flux, err = df1.time.values, df1.flux.values, df1.err.values
tmask = get_transit_mask(time, period[0], epoch[0], duration[0])
data = data = (time[~tmask], flux[~tmask], err[~tmask])
gls = Gls(data, Pbeg=0.1)
gls.plot(block=True)
