#!/usr/bin/env python
"""
from A. Fukui

/ut3/afukui/analysis/SED_fit_v2/sed_fitting.py
"""
import numpy as np
from scipy import interpolate

path = '/ut3/afukui/analysis/SED_fit_v2/'

def load_bt_settl(t,g,m,a):

    if m==0:
        m_str='-0.0'
    elif m>0:
        m_str='+{:.1f}'.format(m)
    else:
        m_str='{:.1f}'.format(m)

    if t < 2600:
        filename = path + 'BT-Settl/bt-settl-agss/lte{0:03d}-{1:.1f}{2}.BT-Settl.7.dat.txt'\
                    .format(int(t/100),g,m_str)
    elif t >= 2600 and g >= 5.0 and m == -0.5:
        filename = path + 'BT-Settl/bt-settl-agss/lte{0:03d}-{1:.1f}{2}a+0.2.BT-Settl.7.dat.txt'\
                    .format(int(t/100),g,m_str)
    else:
        filename = path + 'BT-Settl/bt-settl-agss/lte{0:03d}-{1:.1f}{2}a+0.0.BT-Settl.7.dat.txt'\
                    .format(int(t/100),g,m_str)
    wl, spec = np.loadtxt(filename, unpack=True, dtype=float)
    return wl, spec


def interp_spectra_1D(spec1,spec2,frac):
    return spec1 + (spec2-spec1)*frac


def interp_spectra_3D(t,g,m, spec):
    ## for Teff >= 2600 K
    t_brckt=np.zeros(2)
    g_brckt=np.zeros(2)
    m_brckt=np.zeros(2)
    if t>7000:
        t_brckt[0] = np.floor(t/200.)*200
        t_brckt[1] = t_brckt[0] + 200
        dt = 200
    else:
        t_brckt[0] = int(t/100.)*100
        t_brckt[1] = t_brckt[0] + 100
        dt = 100
    g_brckt[0] = np.floor(g/0.5)*0.5
    g_brckt[1] = g_brckt[0] + 0.5
    dg = 0.5
    if ((m>0.0) & (m<0.3)):
        m_brckt[0] = 0.0
        m_brckt[1] = 0.3
        dm = 0.3
    elif(m==0.0):
        m_brckt[0] = 0.0
        m_brckt[1] = 0.5
        dm = 0.5
    elif(m>=0.3):
        m_brckt[0] = 0.3
        m_brckt[1] = 0.5
        dm = 0.2
    else:
        m_brckt[0] = -0.5
        m_brckt[1] = 0.0
        dm = 0.5
    spec_gm=[]
    for i in range(2):
        spec_gm.append([])
        for j in range(2):
            label0 = '{0:04d}-{1:.1f}-{2:.1f}'.format(int(t_brckt[0]), g_brckt[i], m_brckt[j])
            label1 = '{0:04d}-{1:.1f}-{2:.1f}'.format(int(t_brckt[1]), g_brckt[i], m_brckt[j])

            spec_t0 = spec[label0]
            spec_t1 = spec[label1]
            frac = (t-t_brckt[0])/dt
            spec_gm[i].append(interp_spectra_1D(spec_t0, spec_t1, frac))

    spec_m=[]
    for i in range(2):
        frac = (g-g_brckt[0])/dg
        spec_m.append(interp_spectra_1D(spec_gm[0][i], spec_gm[1][i], frac))
    frac = (m-m_brckt[0])/dm
    spec_intp = interp_spectra_1D(spec_m[0], spec_m[1], frac)
    return spec_intp


def load_filter_response(mission, band, wl_resample):
    response_file = path + 'Filter_trans/{0}_trans_{1}.dat'.format(mission, band)
    wl, trans = np.loadtxt(response_file, usecols=(0,1), unpack=True)
    if mission == '2MASS':
        wl *= 1e4 # Angstrom
    elif mission == 'Gaia':
        wl *= 1e4 # Angstrom
    elif mission == 'WISE':
        wl *= 1e4 # Angstrom
    intp_func = interpolate.interp1d(wl, trans, kind='cubic', bounds_error=False, fill_value=0)
    trans_resample = intp_func(wl_resample)
    trans_resample[trans_resample < 0] = 0
    return trans_resample

def load_filter_response_M2(wl_resample):
    response_file = path + 'Filter_trans/TP_MuSCAT2.csv'
    wl, tp_g, tp_r, tp_i, tp_z\
        = np.genfromtxt(response_file, usecols=(0,5,6,7,8), skip_header=1,\
                        delimiter=',', filling_values=0.0, unpack=True, dtype=float)
    wl *= 10 # Angstrom

    tp = [tp_g, tp_r, tp_i, tp_z]
    intp_func=[]
    tp_resample=[]
    for i in range(4):
        intp_func.append(interpolate.interp1d(wl, tp[i], kind='cubic', bounds_error=False, fill_value=0))
        tp_resample.append(intp_func[i](wl_resample))
        tp_resample[i][tp_resample[i] < 0] = 0
    return tp_resample

def zero_mag_flux(mission, band):
    zero_mag_flux = 0
    if mission == '2MASS' and band == 'J':
        zero_mag_flux = 3.129e-13 # W cm^-2 um^-1
        zero_mag_flux *= 1e7 / 1e4 # erg s^-1 cm^-2 A^-1
    elif mission == '2MASS' and band == 'H':
        zero_mag_flux = 1.133e-13 # W cm^-2 um^-1
        zero_mag_flux *= 1e7 / 1e4 # erg s^-1 cm^-2 A^-1
    elif mission == '2MASS' and band == 'K':
        zero_mag_flux = 4.283e-14 # W cm^-2 um^-1
        zero_mag_flux *= 1e7 / 1e4 # erg s^-1 cm^-2 A^-1

    ## for Vega mag, taken from https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photProc/cu5pho_ssec_photCal.html
    elif mission == 'Gaia' and band == 'BP':
        zero_mag_flux = 4.10957e-11 # W m^-2 nm^-1
        zero_mag_flux *= 1e7 / 1e5 # erg s^-1 cm^-2 A^-1

    elif mission == 'Gaia' and band == 'G':
        zero_mag_flux = 2.53545e-11 # W m^-2 nm^-1
        zero_mag_flux *= 1e7 / 1e5 # erg s^-1 cm^-2 A^-1

    elif mission == 'Gaia' and band == 'RP':
        zero_mag_flux = 1.29851e-11 # W m^-2 nm^-1
        zero_mag_flux *= 1e7 / 1e5 # erg s^-1 cm^-2 A^-1

   ## taken from https://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec4_3g.html#WISEZMA
    elif mission == 'WISE' and band == 'W1':
        zero_mag_flux = 8.1787e-15 # W cm^-2 um^-1
        zero_mag_flux *= 1e7 / 1e4 # erg s^-1 cm^-2 A^-1

    elif mission == 'WISE' and band == 'W2':
        zero_mag_flux = 2.4150e-15 # W cm^-2 um^-1
        zero_mag_flux *= 1e7 / 1e4 # erg s^-1 cm^-2 A^-1

    elif mission == 'WISE' and band == 'W3':
        zero_mag_flux = 6.5151e-17 # W cm^-2 um^-1
        zero_mag_flux *= 1e7 / 1e4 # erg s^-1 cm^-2 A^-1

    elif mission == 'WISE' and band == 'W4':
        zero_mag_flux = 5.0901e-18 # W cm^-2 um^-1
        zero_mag_flux *= 1e7 / 1e4 # erg s^-1 cm^-2 A^-1

    return zero_mag_flux


def calc_extinction(wave_eff, Av):
    Rv = 3.1
    x = 1./float(wave_eff)
    a = 0.
    b = 0.

    if x >= 0.3 and x <= 1.1:
        a = 0.574 * x**1.61
        b = -0.527 * x**1.61
    elif x > 1.1 and x <= 3.3:
        y = x - 1.82
        a = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4\
            + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
        b = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4\
            - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    elif x < 0.3:
        a = 0.
        b = 0.

    return (a + b/Rv) * Av
