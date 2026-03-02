"""

FUNCTIONS TO CALCULATE BIAS AND CORRELATION FUNCTION 
(2-HALO TERM)

"""

#from __future__ import print_function
#from __future__ import division

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splrep,splev
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline as _spline

from colossus.cosmology import cosmology

from .constants import *



def wf(y):
    """
    Tophat window function
    """
    w = 3.0*(np.sin(y) - y*np.cos(y))/y**3.0
    if (y>100.0):
        w = 0.0
    return w


def siny_ov_y(y):
    s = np.sin(y)/y
    if (y>100):
        s = 0.0
    return s

def rhoc_of_z(param):
    """
    Redshift dependence of critical density
    (in comoving units where rho_b=const; same as in AHF)
    """
    Om = param.cosmo.Om
    z  = param.cosmo.z
    
    return RHOC*(Om*(1.0+z)**3.0 + (1.0-Om))/(1.0+z)**3.0


def hubble(a,param):
    """
    Hubble parameter
    """
    Om = param.cosmo.Om
    Ol = 1.0-Om
    H0 = 100.0*param.cosmo.h0
    H  = H0 * (Om/(a**3.0) + (1.0 - Om - Ol)/(a**2.0) + Ol)**0.5
    return H


def growth_factor(a, param):
    """
    Growth factor from Longair textbook (Eq. 11.56)
    """
    Om = param.cosmo.Om
    itd = lambda aa: 1.0/(aa*hubble(aa,param))**3.0
    itl = quad(itd, 0.0, a, epsrel=5e-3, limit=100)
    return hubble(a,param)*(5.0*Om/2.0)*itl[0]


def bias(var,dcz):
    """
    bias function from Cooray&Sheth Eq.68
    """
    q  = 0.707
    p  = 0.3
    nu = dcz**2.0/var
    e1 = (q*nu - 1.0)/dcz
    E1 = 2.0*p/dcz/(1.0 + (q*nu)**p)
    b1 = 1.0 + e1 + E1
    return b1


def variance(r,TF_tck,Anorm,param):
    """
    variance of density perturbations at z=0
    """
    ns = param.cosmo.ns
    kmin = param.code.kmin
    kmax = param.code.kmax
    itd = lambda logk: np.exp((3.0+ns)*logk) * splev(np.exp(logk),TF_tck)**2.0 * wf(np.exp(logk)*r)**2.0
    itl = quad(itd, np.log(kmin), np.log(kmax), epsrel=5e-3, limit=100)
    var = Anorm*itl[0]/(2.0*np.pi**2.0)
    return var


def correlation(r,TF_tck,Anorm,param):
    """
    Correlation function at z=0
    """
    ns = param.cosmo.ns
    kmin = param.code.kmin
    kmax = param.code.kmax
    itd = lambda logk: np.exp((3.0+ns)*logk) * splev(np.exp(logk),TF_tck)**2.0 * siny_ov_y(np.exp(logk)*r)
    itl = quad(itd, np.log(kmin), np.log(kmax), epsrel=5e-3, limit=100)
    corr = Anorm*itl[0]/(2.0*np.pi**2.0)
    return corr

def ks_func(k, z, cosmo):
    return cosmo.sigma(1.0 / k, z, filt = 'gaussian') - 1.0

def ks_calc(z, cosmo):
    z = scipy.optimize.root_scalar(ks_func, args = (z, cosmo), method = 'brentq', bracket = (1.0e-2, 1.0e5))

    return z.root
    
#
# revised halofit (Takahashi+2012)
# delta2k_halofit = (k^3 / 2pi^2) * pk_halofit, k in units of h/Mpc
# 
def delta2k_halofit(k, z, cosmo):
    ks = ks_calc(z, cosmo)
    neff = (-2.0) * cosmo.sigma(1.0 / ks, z, filt = 'gaussian', derivative = True) - 3.0
    #
    # use spline
    #
    #lnr = np.linspace(np.log(0.3 / ks), np.log(3.0 / ks), 100)
    #lns = cosmo.sigma(np.exp(lnr), z, filt = 'gaussian', derivative = True)
    #sig_spline = _spline(lnr, lns, k = 5)
    #dsdr2 = sig_spline.derivatives(np.log(1.0 / ks))[1]
    #
    # get derivative from central difference
    #
    h = 1.0e-4
    sp = cosmo.sigma((1.0 / ks) * (1.0 + h), z, filt = 'gaussian', derivative = True)
    sm = cosmo.sigma((1.0 / ks) * (1.0 - h), z, filt = 'gaussian', derivative = True)
    dsdr2 = (sp - sm) / (np.log((1.0 / ks) * (1.0 + h)) - np.log((1.0 / ks) * (1.0 - h)))
    #
    C = (-2.0) * dsdr2 
    
    om = cosmo.Om(z)
    ow = cosmo.Ode(z)
    w  = cosmo.wz(z)
         
    an = 10.0 ** (1.5222 + 2.8553 * neff + 2.3706 * neff * neff + 0.9903 * neff * neff * neff + 0.2250  * neff * neff * neff * neff - 0.6038 * C + 0.1749 * ow * (1.0 + w))
    bn = 10.0 ** (-0.5642 + 0.5864 * neff + 0.5716 * neff * neff - 1.5474 * C + 0.2279 * ow * (1.0 + w))
    cn = 10.0 ** (0.3698 + 2.0404 * neff + 0.8161 * neff * neff + 0.5869 * C)
    gamman = 0.1971 - 0.0843 * neff + 0.8460 * C
    alphan = np.abs(6.0835 + 1.3373 * neff - 0.1959 * neff * neff - 5.5274 * C)
    betan  = 2.0379 - 0.7354 * neff + 0.3157 * neff * neff + 1.2490 * neff * neff * neff + 0.3980 * neff * neff * neff * neff - 0.1682 * C
    mun    = 0.
    nun    = 10.0 ** (5.2105 + 3.6902 * neff)
    f1 = om ** (-0.0307)
    f2 = om ** (-0.0585)
    f3 = om ** 0.0743

    y  = k / ks
    fy = (y / 4.0) + (y * y / 8.0)

    # colossus use the Eisenstein & Hu transfer function by default
    pk_lin = cosmo.matterPowerSpectrum(k, z)
    deltal = k * k * k * pk_lin / (2.0 * np.pi * np.pi)
    
    deltaq = deltal * (((1.0 + deltal) ** betan) / (1.0 + alphan * deltal)) * np.exp((-1.0) * fy)
    deltad = (an * y ** (3.* f1)) / (1.0 + bn * (y ** f2) + (cn * f3 * y) ** (3.0 - gamman)) 
    deltah = deltad / (1.0 + (mun / y) + (nun / (y * y)))
    
    return deltaq + deltah

def pk_halofit(k, z, cosmo):
    return 2.0 * np.pi * np.pi * delta2k_halofit(k, z, cosmo) / (k * k * k)



def correlation_halofit(r, z, colossus_cosmo, param):
    """
    Matter correlation function xi_mm(r,z) from the non-linear HALOFIT P_mm.
    """
    kmin = param.code.kmin
    kmax = param.code.kmax
    itd = lambda logk: np.exp(3.0 * logk) * pk_halofit(np.exp(logk), z, colossus_cosmo) * siny_ov_y(np.exp(logk) * r)
    itl = quad(itd, np.log(kmin), np.log(kmax), epsrel=5e-3, limit=100)
    return itl[0] / (2.0 * np.pi**2.0)


def _set_colossus_cosmology(param):
    """
    Configure a Colossus cosmology from param.cosmo.
    """
    Om = param.cosmo.Om
    Ob = param.cosmo.Ob
    h0 = param.cosmo.h0
    s8 = param.cosmo.s8
    ns = param.cosmo.ns

    cosmo_name = getattr(param.cosmo, 'colossus_name', 'bar19_custom')
    flat = getattr(param.cosmo, 'flat', True)
    cosmo_par = {
        'flat': flat,
        'H0': 100.0 * h0,
        'Om0': Om,
        'Ob0': Ob,
        'sigma8': s8,
        'ns': ns,
    }
    if (not flat) and ('Ode0' not in cosmo_par):
        cosmo_par['Ode0'] = 1.0 - Om

    if hasattr(param.cosmo, 'colossus_params'):
        extra = param.cosmo.colossus_params
        if isinstance(extra, dict):
            cosmo_par.update(extra)

    try:
        return cosmology.setCosmology(cosmo_name, cosmo_par)
    except Exception:
        # Fallback to deterministic naming if the requested name already
        # exists in memory with different parameters.
        hash_input = tuple(sorted((key, repr(value)) for key, value in cosmo_par.items()))
        hash_key = abs(hash(hash_input)) % 10**8
        fallback_name = '{0}_{1}'.format(cosmo_name, hash_key)
        return cosmology.setCosmology(fallback_name, cosmo_par)


def cosmo(param):
    """
    Calculate halo bias and matter correlation function with Colossus
    for an arbitrary cosmology in param.cosmo.
    Write results to temporary file.
    """
    try:
        from colossus.lss import bias as colossus_bias
    except ImportError:
        print('IOERROR: colossus is required to calculate cosmological functions.')
        print('Install with: pip install colossus')
        exit()

    Om = param.cosmo.Om
    z = param.cosmo.z
    rmin = param.code.rmin
    rmax = param.code.rmax

    if (rmin <= 0.0) or (rmax <= rmin):
        print('ERROR: invalid radial range; require 0 < rmin < rmax.')
        exit()

    bin_N = 100
    bin_r = np.logspace(np.log(rmin), np.log(rmax), bin_N, base=np.e)
    bin_m = 4.0*np.pi*Om*rhoc_of_z(param)*bin_r**3.0/3.0

    try:
        colossus_cosmo = _set_colossus_cosmology(param)
    except Exception:
        print('ERROR: unable to configure Colossus cosmology from param.cosmo.')
        print('Check Om, Ob, h0, ns, s8 and optional colossus_params.')
        exit()

    bias_model = getattr(param.code, 'bias_model', 'tinker10')
    bias_mdef = f'{param.code.deltavir:.0f}c'
    try:
        bin_bias = colossus_bias.haloBias(bin_m, z=z, model=bias_model, mdef=bias_mdef)
    except Exception:
        print('ERROR: unable to evaluate halo bias with Colossus.')
        print('Check param.cosmo.bias_model and param.cosmo.bias_mdef.')
        exit()

    try:
        bin_corr = colossus_cosmo.correlationFunction(bin_r, z=z)
    except Exception:
        print('ERROR: unable to evaluate linear matter correlation with Colossus.')
        exit()

    bin_bias = np.asarray(bin_bias)
    bin_corr = np.asarray(bin_corr)
    bin_corr *= np.sqrt((1 + 1.17*bin_corr)**1.49/(1 + 0.69*bin_corr)**2.09) # correct for scale-dependent bias (Tinker+2010, Eq. B7)
    # COSMOfile = param.files.cosmofct
    # try:
    #     np.savetxt(COSMOfile,np.transpose([bin_r,bin_m,bin_bias,bin_corr]))
    # except IOError:
    #     print('IOERROR: cannot write Cosmofct file in a non-existing directory!')
    #     exit()
    return bin_r, bin_m, bin_bias, bin_corr
