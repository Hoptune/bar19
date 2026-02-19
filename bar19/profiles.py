"""

PROFILES AND FRACTIONS FOR BARIONIC CORRECTIONS

"""
#from __future__ import print_function
#from __future__ import division

import numpy as np
from scipy.special import erf
from scipy.integrate import simps, cumtrapz
from scipy.optimize import fsolve
from scipy.interpolate import splrep,splev
from .constants import *
from .cosmo import *



"""
GENERAL FUNCTIONS REALTED TO THE NFW PROFILE
"""

def r500_fct(rvir,c,deltavir):
    """
    From r_delta to r500 assuming a NFW profile
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 500./deltavir*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return y0*rvir


def rvir_fct(rvir,c,deltavir):
    """
    From r_delta to rvir (delta=96) assuming a NFW profile
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 96.0/deltavir*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return y0*rvir


def M500_fct(Mvir,c,deltavir):
    """
    From M200 to M500 assuming a NFW profiles
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 500./deltavir*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return 500./deltavir*Mvir*y0**3.0


#def cvir_fct(mvir,param):
#    """
#    Concentrations form Dutton+Maccio (2014)
#    c200 (200 times RHOC)
#    Assumes PLANCK cosmology
#    """
#    z = param.cosmo.z
#    
#    A = 0.520 + (0.905-0.520)*np.exp(-0.617*z**1.21)
#    B = -0.101 + 0.026*z
#    return 10.0**A*(mvir/1.0e12)**(B)


"""
STELLAR FRACTIONS
"""

def fSTAR_fct(Mvir,param,eta=0.3):
    NN = param.baryon.Nstar
    M1 = param.baryon.Mstar
    return NN/(Mvir/M1)**(eta)


#def fSTAR_fct(Mvir,eta):
#    """
#    Total stellar fraction (central and satellite galaxies).
#    Free model parameter eta.
#    (Function inspired by Moster+2013, Eq.2)
#    """
#    NN = 0.0351
#    M1 = 10.0**11.4351/0.704
#    zeta = 1.376
#    return 2*NN*((Mvir/M1)**(-zeta)+(Mvir/M1)**(eta))**(-1.0)



"""
Generalised (truncated) NFW profiles
"""

def uNFWtr_fct(rbin,cvir,t,Mvir,param):
    """
    Truncated NFW density profile. Normalised.
    """

    rvir = (3.0*Mvir/(4.0*np.pi*param.sim.deltavir*rhoc_of_z(param)))**(1.0/3.0)
    x = cvir*rbin/rvir
    return 1.0/(x * (1.0+x)**2.0 * (1.0+x**2.0/t**2.0)**2.0)


def rhoNFW_fct(rbin,cvir,Mvir,param):
    """
    NFW density profile.
    """
    rvir = (3.0*Mvir/(4.0*np.pi*param.sim.deltavir*rhoc_of_z(param)))**(1.0/3.0)
    rho0 = param.sim.deltavir*rhoc_of_z(param)*cvir**3.0/(3.0*np.log(1.0+cvir)-3.0*cvir/(1.0+cvir))
    x = cvir*rbin/rvir
    return rho0/(x * (1.0+x)**2.0)


def mNFWtr_fct(x,t):
    """
    Truncated NFW mass profile. Normalised.
    """
    pref   = t**2.0/(1.0+t**2.0)**3.0/2.0
    first  = x/((1.0+x)*(t**2.0+x**2.0))*(x-2.0*t**6.0+t**4.0*x*(1.0-3.0*x)+x**2.0+2.0*t**2.0*(1.0+x-x**2.0))
    second = t*((6.0*t**2.0-2.0)*np.arctan(x/t)+t*(t**2.0-3.0)*np.log(t**2.0*(1.0+x)**2.0/(t**2.0+x**2.0)))
    return pref*(first+second)


def mNFW_fct(x):
    """
    NFW mass profile. Normalised.
    """
    return (np.log(1.0+x)-x/(1.0+x))


def mTOTtr_fct(t):
    """
    Normalised total mass (from truncated NFW)
    """
    pref   = t**2.0/(1.0+t**2.0)**3.0/2.0
    first  = (3.0*t**2.0-1.0)*(np.pi*t-t**2.0-1.0)
    second = 2.0*t**2.0*(t**2.0-3.0)*np.log(t)
    return pref*(first+second)


def MNFWtr_fct(rbin,cvir,t,Mvir,param):
    """
    Truncateed NFW mass profile.
    """
    
    rvir = (3.0*Mvir/(4.0*np.pi*param.sim.deltavir*rhoc_of_z(param)))**(1.0/3.0)
    return Mvir*mNFWtr_fct(cvir*rbin/rvir,t)/mNFWtr_fct(cvir,t)


def MNFW_fct(rbin,cvir,Mvir,param):
    """
    NFW mass profile
    """
    rvir = (3.0*Mvir/(4.0*np.pi*param.sim.deltavir*rhoc_of_z(param)))**(1.0/3.0)
    x = cvir*rbin/rvir
    return (np.log(1.0+x) - x/(1.0+x))/(np.log(1.0+cvir)-cvir/(1.0+cvir))*Mvir



"""
GAS PROFILE
"""

def beta_fct(Mvir,param):
    """
    Parametrises slope of gas profile
    Two models (0), (1)
    """
    z  = param.cosmo.z
    Mc = param.baryon.Mc
    mu = param.baryon.mu
    nu = param.baryon.nu
    Mc_of_z = Mc*(1+z)**nu
    
    if (param.code.beta_model==0):

        dslope = 3.0
        beta = dslope - (Mc_of_z/Mvir)**mu
        if (beta<-10.0):
            beta = -10.0

    elif (param.code.beta_model==1):

        dslope = 3.0
        beta = dslope*(Mvir/Mc)**mu/(1+(Mvir/Mc)**mu)

    else:
        print('ERROR: beta model not defined!')
        exit()
        
    return beta


def uHGA_fct(rbin,Mvir,param):
    """
    Normalised gas density profile
    """
    thej = param.baryon.thej
    thco = param.baryon.thco
    al   = param.baryon.alpha
    be   = beta_fct(Mvir,param)
    ga   = param.baryon.gamma
    de   = param.baryon.delta 
    
    rvir = (3.0*Mvir/(4.0*np.pi*param.sim.deltavir*rhoc_of_z(param)))**(1.0/3.0)
    rco  = thco*rvir
    rej  = thej*rvir
    
    x = rbin/rco
    y = rbin/rej
    return 1.0/(1.0+x**al)**(be)/(1.0+y**ga)**((de-al*be)/ga)


"""
STELLAR PROFILE
"""

def uCGA_fct(rbin,Mvir,param):
    """
    Normalised density profile of central galaxy
    """
    rvir = (3.0*Mvir/(4.0*np.pi*param.sim.deltavir*rhoc_of_z(param)))**(1.0/3.0)
    R12 = param.baryon.rcga*rvir
    return np.exp(-(rbin/R12/2.0)**2.0)/rbin**2.0


def MCGA_fct(rbin,Mvir,param):
    """
    Normalised mass profile of central galaxy
    (needs to be multiplied with Mtot)
    """
    rvir = (3.0*Mvir/(4.0*np.pi*param.sim.deltavir*rhoc_of_z(param)))**(1.0/3.0)
    R12 = param.baryon.rcga*rvir
    return erf(rbin/R12/2.0)


"""
TOTAL PROFILE
"""

def profiles(rbin,Mvir,cvir,cosmo_corr,cosmo_bias,param):

    """
    Calculates fractions, density and mass profiles as a function of radius
    Returns a dictionary
    """
    #parameters
    Om      = param.cosmo.Om
    Ob      = param.cosmo.Ob
    eps     = param.code.eps
    eta     = param.baryon.eta
    deta    = param.baryon.deta
    
    #radii
    tau  = eps*cvir
    rvir = (3.0*Mvir/(4.0*np.pi*param.sim.deltavir*rhoc_of_z(param)))**(1.0/3.0)
    r500 = r500_fct(rvir,cvir)
    M500 = MNFW_fct(r500,cvir,Mvir,param)
    #M500 = MNFWtr_fct(r500,cvir,tau,Mvir,param)

    #total fractions
    fbar  = Ob/Om
    fcdm  = (Om-Ob)/Om
    fstar = fSTAR_fct(Mvir,param,eta)
    fcga  = fSTAR_fct(Mvir,param,eta+deta) #Moster13
    fsga  = fstar-fcga #satellites and intracluster light
    if(fsga<0):
        print('ERROR: negative fraction of satellite galaxies')
        exit()
    fhga  = fbar-fcga-fsga

    #total dark-matter-only mass
    Mtot = Mvir*mTOTtr_fct(tau)/mNFWtr_fct(cvir,tau)

    #Initial density and mass profiles
    rho0NFWtr = param.sim.deltavir*rhoc_of_z(param)*cvir**3.0/(3.0*mNFWtr_fct(cvir,tau))
    rhoNFW = rho0NFWtr*uNFWtr_fct(rbin,cvir,tau,Mvir,param)
    rho2h = (cosmo_bias*cosmo_corr + 1.0)*Om*RHOC #rho_b=const in comoving coord.
    rhoDMO = rhoNFW + rho2h
    MNFW   = MNFWtr_fct(rbin,cvir,tau,Mvir,param)
    M2h    = cumtrapz(4.0*np.pi*rbin**2.0*rho2h,rbin,initial=rbin[0])
    MDMO   = MNFW + M2h

    #Final density and mass profiles
    uHGA =  uHGA_fct(rbin,Mvir,param)
    rho0HGA = Mtot/(4.0*np.pi*simps(rbin**2.0*uHGA,rbin))
    rhoHGA   = rho0HGA*uHGA
    R12      = param.baryon.rcga*rvir
    rho0CGA  = Mtot/(4.0*np.pi**(3.0/2.0)*R12)
    rhoCGA   = rho0CGA*uCGA_fct(rbin,Mvir,param)
    MHGA     = cumtrapz(4.0*np.pi*rbin**2.0*rhoHGA,rbin,initial=rbin[0]) + M2h
    MCGA     = Mtot*MCGA_fct(rbin,Mvir,param) + M2h
    MHGA_tck = splrep(rbin, MHGA, s=0, k=3)
    MCGA_tck = splrep(rbin, MCGA, s=0, k=3)

    #Adiabatic contraction/expansion (Gnedin 2004, see also Teyssier et al 2011)
    #MNFWri = MDMO
    #aa = 0.68
    #func = lambda x: (x-1.0) - aa*(MNFWri/((fcdm+fsga)*MNFWri + fcga*splev(x*rbin,MCGA_tck,der=0) + fhga*splev(x*rbin,MHGA_tck,der=0)) - 1.0)
    ##Adiabatic contraction/expansion (after Abadi et al 2010)
    MNFWri = MDMO
    aa = 0.3
    nn = 2.0
    func = lambda x: (x-1.0) - aa*((MNFWri/((fcdm+fsga)*MNFWri + fcga*splev(x*rbin,MCGA_tck,der=0) + fhga*splev(x*rbin,MHGA_tck,der=0)))**nn - 1.0)
    if (isinstance(rbin, float)):
        xi = 1.0
    else:
        xi = np.empty(len(rbin)); xi.fill(1.0)
    xx = fsolve(func, xi, fprime=None)
    MACM     = MNFWtr_fct(rbin/xx,cvir,tau,Mvir,param)
    MACM_tck = splrep(rbin, MACM, s=0, k=3)
    rhoACM   = splev(rbin,MACM_tck,der=1)/(4.0*np.pi*rbin**2.0)
    MACM     = MACM + M2h

    #total profile
    rhoBAR   = (fcdm+fsga)*rhoACM + fhga*rhoHGA + fcga*rhoCGA
    rhoDMB   = rhoBAR + rho2h
    MDMB     = (fcdm+fsga)*MACM + fhga*MHGA + fcga*MCGA
    MDMB_tck = splrep(rbin, MDMB, s=0, k=3)
    MDMBinv_tck = splrep(MDMB, rbin, s=0, k=3)

    #define dictionaries
    frac = { 'CDM':fcdm, 'CGA':fcga, 'SGA':fsga, 'HGA':fhga }
    dens = { 'NFW':rhoNFW, 'BG':rho2h, 'DMO':rhoDMO, 'ACM':rhoACM, 'CGA':rhoCGA, 'HGA':rhoHGA, 'DMB':rhoDMB }
    mass = { 'NFW':MNFW, 'BG':M2h, 'DMO':MDMO, 'ACM':(fcdm+fsga)*MACM, 'CGA':fcga*MCGA, 'HGA':fhga*MHGA, 'DMB':MDMB }
    return frac, dens, mass

