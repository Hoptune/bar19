"""
External Parameters
"""

class Bunch(object):
    """
    translates dic['name'] into dic.name 
    """

    def __init__(self, data):
        self.__dict__.update(data)


def cosmo_par():
    par = {
        "z": 0.0,
        "Om": 0.315,
        "Ob": 0.049,
        "s8": 0.83,
        "h0": 0.673,
        "ns": 0.963,
        "dc": 1.675,
        }
    return Bunch(par)

def baryon_par():
    par = {
        "Mc": 3.0e13,     # beta(M,z): critical mass scale
        "mu": 0.3,        # beta(M,z): critical mass scale
        "nu": 0.0,        # beta(M,c): redshift dependence
        "thej": 4.0,      # ejection factor thej=rej/rvir
        "thco": 0.1,      # core factor thco=rco/rvir
        "alpha": 1.0,     # index in gas profile [default: 1.0]
        "gamma": 2.0,     # index in gas profile [default: 2.0]
        "delta": 7.0,     # index in gas profile [default: 7.0 -> same asympt. behav. than NFWtrunc profile]  
        "rcga": 0.015,    # half-light radius of central galaxy (ratio to rvir)
        "Nstar": 0.04,    # Stellar normalisation param [fstar = Nstar*(Mstar/Mvir)**eta]
        "Mstar": 2.5e11,  # Stellar critical mass [fstar = Nstar*(Mstar/Mvir)**eta]
        "eta": 0.32,      # exponent of total stellar fraction [fstar = Nstar*(Mstar/Mvir)**eta]
        "deta": 0.28,     # exponent of central stellar fraction [fstar = Nstar*(Mstar/Mvir)**(eta+deta)]
        }
    return Bunch(par)

def io_files():
    par = {
        "transfct": 'CDM_PLANCK_tk.dat',
        "cosmofct": 'cosmofct.dat',
        "displfct": 'displfct.dat',
        "partfile_in": 'partfile_in.std',
        "partfile_out": 'partfile_out.std',
        "partfile_format": 'tipsy',
        "halofile_in": 'file_halo',
        "halofile_format": 'AHF-ASCII',
    }
    return Bunch(par)

def code_par():
    par = {
        "kmin": 0.01,
        "kmax": 100.0,
        "rmin": 0.005,
        "rmax": 50.0,
        "rbuffer": 10.0, # buffer size to take care of boundary conditions
        "eps": 4.0,      # truncation factor: eps=rtr/rvir 
        "beta_model": 0, # 0: old model from Schneider+18 1: new model
        }
    return Bunch(par)

def sim_par():
    par = {
        "Lbox": 128.0,   #box size of partfile_in
        "rbuffer": 10.0, #buffer size to take care of boundary conditions
        "Mhalo_min": 2.5e11, # minimum halo mass
        "N_chunk": 1,     #number of chunks (for multiprocesser: n_core = N_chunk^3)
        "deltavir": 200.0, # overdensity relative to critical density
        }
    return Bunch(par)

def par():
    par = Bunch({"cosmo": cosmo_par(),
        "baryon": baryon_par(),
        "files": io_files(),
        "code": code_par(),
        "sim": sim_par(),
        })
    return par
