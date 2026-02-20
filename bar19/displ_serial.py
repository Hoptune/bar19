"""
CALCULATE DISPLACEMENT FUNCTION FOR A GRID OF M AND C
PRINT INFORMATION INTO TEMPORARY FILE

"""

from __future__ import print_function
from __future__ import division

import numpy as np
import os
import multiprocessing
from scipy import spatial
from scipy.interpolate import splrep,splev

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


from .params import par
from .constants import *
from .profiles import *

"""
READING/WRITING FILES
"""

def _is_master_process():
    """
    Only render progress bars from the main process and MPI rank 0.
    """
    mpi_rank_vars = (
        'OMPI_COMM_WORLD_RANK',
        'PMI_RANK',
        'PMIX_RANK',
        'MV2_COMM_WORLD_RANK',
        'SLURM_PROCID',
    )
    for var in mpi_rank_vars:
        rank = os.environ.get(var)
        if (rank is not None) and (rank != '') and (rank != '0'):
            return False
    try:
        return multiprocessing.current_process().name == 'MainProcess'
    except Exception:
        return True


def _progress(iterable, total=None, desc=None, disable=False):
    """
    Wrap iterables with a progress bar when tqdm is available.
    """
    if disable or (tqdm is None) or (not _is_master_process()):
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)

def _write_par_to_hdf5(root, param):
    """
    Write parameter sections (except files) into an HDF5 file/group.
    """
    if param is None:
        return
    par_group = root.create_group('par')
    for section_name in ('cosmo', 'baryon', 'code', 'sim'):
        section = getattr(param, section_name, None)
        if section is None:
            continue
        section_group = par_group.create_group(section_name)
        for key, value in getattr(section, '__dict__', {}).items():
            if key.startswith('_'):
                continue
            try:
                section_group.create_dataset(key, data=np.asarray(value))
            except Exception:
                section_group.attrs[key] = str(value)

def read_nbody_file(param):
    """
    Read in N-body output and adopt units
    Only supports tispy file format for the moment
    """
    nbody_file_in = param.files.partfile_in
    nbody_file_format = param.files.partfile_format
    Lbox = param.sim.Lbox
    if (nbody_file_format=='hdf5' or nbody_file_format=='catalog-hdf5'):
        try:
            import h5py
        except ImportError:
            print('IOERROR: h5py is required for partfile_format=hdf5.')
            print('Install with: pip install h5py')
            exit()
        try:
            with h5py.File(nbody_file_in, 'r') as f:
                g = f['dm'] if 'dm' in f else f
                x = np.asarray(g['x'], dtype=np.float32)
                y = np.asarray(g['y'], dtype=np.float32)
                z = np.asarray(g['z'], dtype=np.float32)
        except (OSError, KeyError):
            print('IOERROR: Cannot read HDF5 catalog with required x/y/z datasets.')
            print('Define par.files.partfile_in = "/path/to/catalog.hdf5"')
            exit()
        if (len(x) != len(y)) or (len(x) != len(z)):
            print('IOERROR: x, y, z must have identical lengths.')
            exit()
        p_header = {'a': 1.0, 'npart': len(x), 'ndim': 3, 'ng': 0, 'nd': len(x), 'ns': 0, 'buffer': 0}
        p_dt = np.dtype([('x','>f'),('y','>f'),('z','>f')])
        p = np.zeros(len(x), dtype=p_dt)
        p['x'] = np.mod(x, Lbox)
        p['y'] = np.mod(y, Lbox)
        p['z'] = np.mod(z, Lbox)
        print('Reading user-supplied HDF5 catalog done!')
        return p, p_header

    try:
        f = open(nbody_file_in, 'r')
    except IOError:
        print('IOERROR: N-body particle file does not exist!')
        print('Define par.files.partfile_in = "/path/to/file"')
        exit()
    if (nbody_file_format=='tipsy'):
        #header
        h_dt = np.dtype([('a','>d'),('npart','>i'),('ndim','>i'),('ng','>i'),('nd','>i'),('ns','>i'),('buffer','>i')])
        p_header = np.fromfile(f, dtype=h_dt, count=1, sep='')
        #particles
        p_dt = np.dtype([('mass','>f'),("x",'>f'),("y",'>f'),("z",'>f'),("vx",'>f'),("vy",'>f'),("vz",'>f'),("eps",'>f'),("phi",'>f')])
        p = np.fromfile(f, dtype=p_dt, count=p_header['npart'], sep='')
        #from tipsy units to [0,Lbox] in units of Lbox
        p['x']=Lbox*(p['x']+0.5)
        p['y']=Lbox*(p['y']+0.5)
        p['z']=Lbox*(p['z']+0.5)
        print('Reading tipsy-file done!')
    elif (nbody_file_format=='gadget'):
        print('Reading gadget files not implemented. Exit!')
        exit()
    else:
        print('Unknown file format. Exit!')
        exit()
    return p, p_header


def write_nbody_file(p,p_header,param):
    """
    Write N-body outputs with displaced particles
    Adopts units. Only tipsy file format for the moment.
    """
    nbody_file_out = param.files.partfile_out
    nbody_file_format = param.files.partfile_format
    Lbox = param.sim.Lbox
    if (nbody_file_format=='hdf5' or nbody_file_format=='catalog-hdf5'):
        try:
            import h5py
        except ImportError:
            print('IOERROR: h5py is required for partfile_format=hdf5.')
            print('Install with: pip install h5py')
            exit()
        try:
            with h5py.File(nbody_file_out, 'w') as f:
                g_dm = f.create_group('dm')
                g_dm.create_dataset('x', data=p['x'].astype(np.float32))
                g_dm.create_dataset('y', data=p['y'].astype(np.float32))
                g_dm.create_dataset('z', data=p['z'].astype(np.float32))
                _write_par_to_hdf5(f, param)
        except OSError:
            print('IOERROR: Cannot write HDF5 catalog output file!')
            print('Define par.files.partfile_out = "/path/to/output.hdf5"')
            exit()
        print('Writing HDF5 catalog output done!')
        return

    try:
        f = open(nbody_file_out, 'wb')
    except IOError:
        print('IOERROR: Path to output file does not exist!')
        print('Define par.files.partfile_out = "/path/to/file"')
        exit()
    if (nbody_file_format=='tipsy'):
        #back to tipsy units
        p['x']=(p['x']/Lbox-0.5).astype(np.float32)
        p['y']=(p['y']/Lbox-0.5).astype(np.float32)
        p['z']=(p['z']/Lbox-0.5).astype(np.float32)
        p_header.tofile(f,sep='')
        p.tofile(f,sep='')
        #f.write(p_header)
        #f.write(p)
    elif (nbody_file_format=='gadget'):
        print('Writing gadget files not implemented. Exit!')
        exit()
    else:
        print('Unknown file format. Exit!')
        exit()



def read_halo_file(param):
    """
    Read in halo file, adopt units.
    Select for hosts above a minimum halo mass.
    Rstricted to AHF for the moment.
    """
    halo_file_in = param.files.halofile_in
    halo_file_format = param.files.halofile_format
    Mhalo_min = param.sim.Mhalo_min
    if (halo_file_format=='AHF-ASCII'):
        try:
            names = "ID,IDhost,Mvir,Nvir,x,y,z,rvir,cvir"
            h = np.genfromtxt(halo_file_in,usecols=(0,1,3,4,5,6,7,11,42),comments='#',dtype=None,names=names)
        except IOError:
            print('IOERROR: Halo file does not exist!')
            print('Define par.files.halofile_in = "/path/to/file"')
            exit()
        #adopt units
        h['x']    = h['x']/1000.0
        h['y']    = h['y']/1000.0
        h['z']    = h['z']/1000.0
        h['rvir'] = h['rvir']/1000.0
        print('Nhalo = ',len(h['Mvir']))
        #select haloes above minimum mass
        gID  = np.where(h['Mvir'] >= Mhalo_min)
        h = h[gID]
        #select haloes with reasonable concentration
        gID  = np.where(h['cvir'] > 0)
        h = h[gID]
        #select main haloes (only if ahf calculates host) 
        gID  = np.where(h['IDhost'] < 0.0)
        h = h[gID] 
        gID  = np.where(h['Mvir'] >= Mhalo_min)
        h = h[gID]
        gID  = np.where(h['cvir'] > 0)
        h = h[gID]
        print('Nhalo = ',len(h['Mvir']))
        return h

    elif (halo_file_format=='hdf5' or halo_file_format=='halo-hdf5' or halo_file_format=='catalog-hdf5'):
        try:
            import h5py
        except ImportError:
            print('IOERROR: h5py is required for halofile_format=hdf5.')
            print('Install with: pip install h5py')
            exit()

        try:
            with h5py.File(halo_file_in, 'r') as f:
                g = f['halos'] if 'halos' in f else f
                x = np.asarray(g['x'])
                y = np.asarray(g['y'])
                z = np.asarray(g['z'])
                Mvir = np.asarray(g['Mvir'])
                rvir = np.asarray(g['rvir'])
                cvir = np.asarray(g['cvir'])
                if 'IDhost' in g:
                    host_id = np.asarray(g['IDhost'])
                elif 'upid' in g:
                    host_id = np.asarray(g['upid'])
                else:
                    host_id = None
        except (OSError, KeyError):
            print('IOERROR: Cannot read HDF5 halo catalog with required datasets.')
            print('Define par.files.halofile_in = "/path/to/halos.hdf5"')
            exit()

        if any(len(arr) != len(x) for arr in (y, z, Mvir, rvir, cvir)):
            print('IOERROR: HDF5 halo datasets must have identical lengths.')
            exit()

        h_dt = np.dtype([('Mvir', '<f8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
                         ('rvir', '<f8'), ('cvir', '<f8')])
        h = np.zeros(len(x), dtype=h_dt)
        h['x'] = x/1000.0
        h['y'] = y/1000.0
        h['z'] = z/1000.0
        h['Mvir'] = Mvir
        h['rvir'] = rvir/1000.0
        h['cvir'] = cvir

        print("Using param.sim.Mhalo_min to filter haloes.")

        #select haloes above minimum mass
        gID  = np.where(h['Mvir'] >= Mhalo_min)
        h = h[gID]
        #select haloes with reasonable concentration
        gID  = np.where(h['cvir'] > 0)
        h = h[gID]
        #select main haloes when a host id is provided
        if host_id is not None:
            host_id = host_id[gID]
            gID  = np.where(host_id < 0.0)
            h = h[gID]
        else:
            print('WARNING: HDF5 halo catalog has no IDhost/upid. Keeping all haloes.')

        print('Nhalo = ',len(h['Mvir']))
        return h

    else:
        print('Unknown halo file format. Exit!')
        exit()



"""
BUILD BUFFER FOR N-BODY FILE
"""

def build_buffer(h,param):
    """
    Takes care of boundary problems by Building buffer around simulation box.
    Only haloes are dublicated.
    Size of buffer > max radius of displacement.
    """
    Lbox = param.sim.Lbox
    rbuffer = param.sim.rbuffer
    ID = np.where((h['x']>(Lbox-rbuffer)) & (h['x']<=Lbox))
    h  = np.append(h,h[ID])
    if (len(ID[0])>0):
        h['x'][-len(ID[0]):] = h['x'][-len(ID[0]):]-Lbox
    ID = np.where((h['x']>0) & (h['x']<rbuffer))
    h  = np.append(h,h[ID])
    if (len(ID[0])>0):
        h['x'][-len(ID[0]):] = h['x'][-len(ID[0]):]+Lbox

    ID = np.where((h['y']>(Lbox-rbuffer)) & (h['y']<=Lbox))
    h  = np.append(h,h[ID])
    if (len(ID[0])>0):
        h['y'][-len(ID[0]):] = h['y'][-len(ID[0]):]-Lbox
    ID = np.where((h['y']>0) & (h['y']<rbuffer))
    h  = np.append(h,h[ID])
    if (len(ID[0])>0):
        h['y'][-len(ID[0]):] = h['y'][-len(ID[0]):]+Lbox

    ID = np.where((h['z']>(Lbox-rbuffer)) & (h['z']<=Lbox))
    h  = np.append(h,h[ID])
    if (len(ID[0])>0):
        h['z'][-len(ID[0]):] = h['z'][-len(ID[0]):]-Lbox
    ID = np.where((h['z']>0) & (h['z']<rbuffer))
    h  = np.append(h,h[ID])
    if (len(ID[0])>0):
        h['z'][-len(ID[0]):] = h['z'][-len(ID[0]):]+Lbox
    print('Nhalo (incl buffer) = ', len(h['Mvir']))
    return h




"""
DISPLACEMENT FUNCTION
"""

def displ(rbin,MDMO,MDMB):
    """
    Calculates the displacement of all particles as a function
    of the radial distance from the halo centre
    """
    MDMB_tck = splrep(rbin, MDMB, s=0, k=3)
    MDMBinv_tck=splrep(MDMB, rbin, s=0, k=3)
    rDMB = splev(MDMO,MDMBinv_tck,der=0)
    DDMB = rDMB - rbin
    return DDMB




"""
WRITE FILE WITH DISPLACMENTS FOR A GRID OF M AND C 
"""

def displ_file(param):
    """
    Calculates displacement for a gid of different halo masses and concentrations
    Writes outcome to file
    """

    #relevant parameters
    Mc   = param.baryon.Mc
    mu   = param.baryon.mu
    nu   = param.baryon.nu
    thej = param.baryon.thej
    red  = param.cosmo.z

    #Read cosmic variance/nu/correlation and interpolate
    cosmofile = param.files.cosmofct
    try:
        vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(cosmofile, usecols=(0,1,2,3), unpack=True)
        bias_tck = splrep(vc_m, vc_bias, s=0)
        corr_tck = splrep(vc_r, vc_corr, s=0)
    except IOError:
        print('IOERROR: Cosmofct file does not exist!')
        print('Define par.files.cosmofct = "/path/to/file"')
        print('Run: cosmo(params) to create file')
        exit()

    #mass bins
    N_Mvir = 50
    Mvir = np.logspace(12,15.5,N_Mvir,base=10)

    #radius bins (the same for all haloes)
    N_rbin = 100
    rmin = param.code.rmin
    rmax = param.code.rmax
    rbin = np.logspace(np.log10(rmin),np.log10(rmax),N_rbin,base=10)

    #concentration bins
    N_cvir = 20

    #loop over grid (Mvir, cvir) and write to file
    displfile = param.files.displfct
    try:
        displfct_file = open(displfile, 'w')
    except IOError:
        print('IOERROR: cannot write displfct file in a non-existing directory!')
        exit()
    rbin_print = rbin[np.where(rbin > 0.05)]
    rbin_print = rbin_print[np.where(rbin_print < 60.0)]
    print(N_Mvir, N_cvir,file=displfct_file)
    print(' '.join(map(str, rbin_print)),file=displfct_file)
    #loop over halo mass
    for i in range(len(Mvir)):
        dc = 10**0.4  #Dutton2014 (Fig15)
        cvir = np.linspace(cvir_fct(Mvir[i],red)/dc,cvir_fct(Mvir[i],red)*dc,N_cvir)
        cosmo_bias = splev(Mvir[i],bias_tck)
        cosmo_corr = splev(rbin,corr_tck)
        #loop over concentrations
        for k in range(len(cvir)):
            #print>>displfct_file, Mvir[i], cvir[k]
            print(Mvir[i], cvir[k],file=displfct_file)
            frac, dens, mass = profiles(rbin,Mvir[i],cvir[k],cosmo_corr,cosmo_bias,param)
            DDMB = displ(rbin,mass['DMO'],mass['DMB'])
            #check consistency of fhga
            #r500 = r500_fct(rvir[i],cvir[k])
            MHGA_tck = splrep(rbin, mass['HGA'], s=0, k=3)
            MDMB_tck = splrep(rbin, mass['DMB'], s=0, k=3)
            #print to file (reduced resolution)
            rbin_print = rbin[np.where(rbin > 0.05)]
            DDMB_print = DDMB[np.where(rbin > 0.05)]
            rbin_print = rbin_print[np.where(rbin_print < 60.0)]
            DDMB_print = DDMB_print[np.where(rbin_print < 60.0)]
            #print>>displfct_file, ' '.join(map(str, DDMB_print))
            print(' '.join(map(str, DDMB_print)),file=displfct_file)

    displfct_file.close()
    print('Writing to displacemnent file done!')



"""
DISPLACE PARTICLES IN N_BODY SIM USING DISPLACEMENT FUNCTION
"""

def displace_from_displ_file(param):
    """
    Read in file generated with displ_file() and displace 
    particles in N-body output according to this file
    """
    
    #relavant parameters
    Lbox = param.sim.Lbox
    
    #Read in displacement file
    displfile = param.files.displfct
    try:
        displ_file  = open(displfile, 'r')
    except IOError:
        print('IOERROR: displfct file does not exist!')
        print('Define par.files.displfct = "/path/to/file"')
        print('Run: displ_file(params) to create file')
        exit()
 
    #read displacement fct for a grid of Mvir and cvir
    N_Mvir, N_cvir  = map(int,displ_file.readline().split())
    rbin = np.array(map(float,displ_file.readline().split()))
    D_array = np.zeros((len(rbin),N_Mvir,N_cvir))
    Mvir = []
    cvir = []
    for i in range(N_Mvir):
        Mvir += [0.0]
        cvir_vec = []
        for k in range(N_cvir):
            Mv, cv = map(float,displ_file.readline().split())
            if (Mv != Mvir[i]):
                Mvir[i] = Mv
            cvir_vec += [cv]
            #fill in displ in D_array
            D_array[:,i,k] = np.array(map(float,displ_file.readline().split()))
        #fill in matrix with vector cvir(Mvir)
        cvir += [cvir_vec]
    displ_file.close()
    Mvir = np.array(Mvir)
    cvir = np.array(cvir)

    #Read in N-body particle file
    p, p_header = read_nbody_file(param)

    #Copy into p_temp
    Dp_dt = np.dtype([("x",'>f'),("y",'>f'),("z",'>f')])
    Dp    = np.zeros(len(p),dtype=Dp_dt)
    
    #Read in halo file
    h = read_halo_file(param)

    #Create buffer to account for box boundaries
    h  = build_buffer(h,param)

    #Build tree
    print('building tree..')
    p_tree = spatial.cKDTree(zip(p['x'],p['y'],p['z']), leafsize=100)
    print('...done!')

    #Loop over haloes and displace
    n_halo = len(h['Mvir'])
    for k in _progress(range(n_halo), total=n_halo, desc='Displacing halos',
                       disable=(n_halo <= 1)):
        #find displacement function from D_array (no interpolation for the moment)
        idx_Mvir = abs(Mvir-h['Mvir'][k]).argmin()
        idx_cvir = abs(cvir[idx_Mvir]-h['cvir'][k]).argmin()
        DDMB     = D_array[:,idx_Mvir,idx_cvir]
        DDMB_tck = splrep(rbin, DDMB,s=0,k=3)
        #define maximum displacement
        smallestD = 0.01 #Mpc/h
        #array of idx with D>Dsmallest
        idx = np.where(abs(DDMB) > smallestD)
        idx = idx[:][0]
        if (len(idx)>1):
            idx_largest = idx[-1]
            rball = rbin[idx_largest]
        else:
            rball = 0.0
        #consistency check:
        if (rball>Lbox/2.0):
            print('rball = ', rball)
            print('ERROR: REDUCE RBALL!')
            exit()
        #particle ids within rball
        ipbool = p_tree.query_ball_point((h['x'][i],h['y'][i],h['z'][i]),rball)
        #update displacement
        rpDMB  = ((p['x'][ipbool]-h['x'][i])**2.0 + (p['y'][ipbool]-h['y'][i])**2.0 + (p['z'][ipbool]-h['z'][i])**2.0)**0.5
        if (rball>0.0 and len(rpDMB)):
            DrpDMB = splev(rpDMB,DDMB_tck,der=0,ext=1)
            Dp['x'][ipbool] += (p['x'][ipbool]-h['x'][i])*DrpDMB/rpDMB
            Dp['y'][ipbool] += (p['y'][ipbool]-h['y'][i])*DrpDMB/rpDMB
            Dp['z'][ipbool] += (p['z'][ipbool]-h['z'][i])*DrpDMB/rpDMB
    #displace particles
    p['x'] += Dp['x']
    p['y'] += Dp['y']
    p['z'] += Dp['z']
    #periodic bounsdaries
    p['x'][p['x']>Lbox] -= Lbox
    p['x'][p['x']<0.0]  += Lbox
    p['y'][p['y']>Lbox] -= Lbox
    p['y'][p['y']<0.0]  += Lbox
    p['z'][p['z']>Lbox] -= Lbox
    p['z'][p['z']<0.0]  += Lbox
    print('Calculating Displacement fields done!')
    #write N-body file with displacements
    write_nbody_file(p,p_header,param)





"""
CALCULATE DISPLACEMENT AND DISPLACE DIRECTLY. PARTICLES ARE DISPLACED
MULTIPLE TIMES.
"""

def displace(param):
    """
    Reading in N-body and halo files, looping over haloes, calculateing
    displacements, and dispalcing particles.
    Combines functions displ_file() and displace_from_displ_file()
    """

    #relevant parameters
    Mc   = param.baryon.Mc
    mu   = param.baryon.mu
    nu   = param.baryon.nu
    thej = param.baryon.thej
    Lbox = param.sim.Lbox

    #Read cosmic variance/nu/correlation and interpolate
    cosmofile = param.files.cosmofct
    try:
        vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(cosmofile, usecols=(0,1,2,3), unpack=True)
        bias_tck = splrep(vc_m, vc_bias, s=0)
        corr_tck = splrep(vc_r, vc_corr, s=0)
    except IOError:
        print('IOERROR: Cosmofct file does not exist!')
        print('Define par.files.cosmofct = "/path/to/file"')
        print('Run: cosmo(params) to create file')
        exit()

    #Read in N-body particle file
    p, p_header = read_nbody_file(param)

    #Read in halo file
    h = read_halo_file(param)

    #Create buffer to account for box boundaries
    h  = build_buffer(h,param)

    #Build tree
    print('building tree..')
    p_tree = spatial.cKDTree(zip(p['x'],p['y'],p['z']), leafsize=100)
    print('...done!')

    #Loop over haloes, calculate displacement, and displace partricles
    n_halo = len(h['Mvir'])
    for i in _progress(range(n_halo), total=n_halo, desc='Displacing halos',
                       disable=(n_halo <= 1)):
        #range where we consider displacement
        rmax = param.code.rmax
        rmin = (0.001*h['rvir'][i] if 0.001*h['rvir'][i]>param.code.rmin else param.code.rmin)
        rmax = (20.0*h['rvir'][i] if 20.0*h['rvir'][i]<param.code.rmax else param.code.rmax)
        rbin = np.logspace(np.log10(rmin),np.log10(rmax),100,base=10)
        #calculate displacement
        cosmo_bias = splev(h['Mvir'][i],bias_tck)
        cosmo_corr = splev(rbin,corr_tck)
        frac, dens, mass = profiles(rbin,h['Mvir'][i],h['cvir'][i],Mc,mu,nu,thej,cosmo_corr,cosmo_bias,param)
        DDMB = displ(rbin,mass['DMO'],mass['DMB'])
        DDMB_tck = splrep(rbin, DDMB,s=0,k=3)
        #define minimum displacement
        smallestD = 0.01 #Mpc/h
        #array of idx with D>Dsmallest
        idx = np.where(abs(DDMB) > smallestD)
        idx = idx[:][0]
        if (len(idx)>1):
            idx_largest = idx[-1]
            rball = rbin[idx_largest]
        else:
            rball = 0.0
        #consistency check:
        if (rball>Lbox/2.0):
            print('rball = ', rball)
            print('ERROR: REDUCE RBALL!')
            exit()
        #particles within rball
        ipbool = p_tree.query_ball_point((h['x'][i],h['y'][i],h['z'][i]),rball)
        phx = p['x'][ipbool]
        phy = p['y'][ipbool]
        phz = p['z'][ipbool]
        #haloes within rball
        hdis = ((h['x']-h['x'][i])**2.0 + (h['y']-h['y'][i])**2.0 + (h['z']-h['z'][i])**2.0)**0.5
        ihbool = np.all([(hdis<rball),(hdis>0)],axis=0)
        hhx = h['x'][ihbool]
        hhy = h['y'][ihbool]
        hhz = h['z'][ihbool]
        #displace particles
        rpDMB  = ((phx-h['x'][i])**2.0 + (phy-h['y'][i])**2.0 + (phz-h['z'][i])**2.0)**0.5
        if (rball>0.0 and len(rpDMB)):
            DrpDMB = splev(rpDMB,DDMB_tck,der=0,ext=1)
            DxpDMB = (phx-h['x'][i])*DrpDMB/rpDMB
            DypDMB = (phy-h['y'][i])*DrpDMB/rpDMB
            DzpDMB = (phz-h['z'][i])*DrpDMB/rpDMB
            phx    = (phx + DxpDMB)
            phy    = (phy + DypDMB)
            phz    = (phz + DzpDMB)
        #periodic bounsdaries
        phx[phx>Lbox] -= Lbox
        phx[phx<0.0]  += Lbox
        phy[phy>Lbox] -= Lbox
        phy[phy<0.0]  += Lbox
        phz[phz>Lbox] -= Lbox
        phz[phz<0.0]  += Lbox
        #displace halo positions
        if (len(hhx)>0):
            rhDMB  = ((hhx-h['x'][i])**2.0 + (hhy-h['y'][i])**2.0 + (hhz-h['z'][i])**2.0)**0.5
            DrhDMB = splev(rhDMB,DDMB_tck,der=0)
            DxhDMB = (hhx-h['x'][i])*DrhDMB/rhDMB
            DyhDMB = (hhy-h['y'][i])*DrhDMB/rhDMB
            DzhDMB = (hhz-h['z'][i])*DrhDMB/rhDMB
            hhx    = (hhx + DxhDMB)
            hhy    = (hhy + DyhDMB)
            hhz    = (hhz + DzhDMB)
        #putting particles back into main array
        p['x'][ipbool]=phx
        p['y'][ipbool]=phy
        p['z'][ipbool]=phz
        #putting haloes back into main array
        h['x'][ihbool]=hhx
        h['y'][ihbool]=hhy
        h['z'][ihbool]=hhz
    print('Calculating Displacement fields done!')
    #write N-body file with displacements
    write_nbody_file(p,p_header,param)




"""
CALCULATE MULTIPLE DISPLACEMENTS AND AND DISPLACE ONCE AT THE END.
MORE MEMORY INTENSIVE.
"""

def displace_allinone(param):
    """
    Reading in N-body and halo files, looping over haloes, calculating
    displacements. Displace once at the end.
    """

    #relevant parameters
    Mc   = param.baryon.Mc
    mu   = param.baryon.mu
    nu   = param.baryon.nu
    thej = param.baryon.thej
    Lbox = param.sim.Lbox

    #Read cosmic variance/nu/correlation and interpolate
    cosmofile = param.files.cosmofct
    try:
        vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(cosmofile, usecols=(0,1,2,3), unpack=True)
        bias_tck = splrep(vc_m, vc_bias, s=0)
        corr_tck = splrep(vc_r, vc_corr, s=0)
    except IOError:
        print('IOERROR: Cosmofct file does not exist!')
        print('Define par.files.cosmofct = "/path/to/file"')
        print('Run: cosmo(params) to create file')
        exit()

    #Read in N-body particle file
    p, p_header = read_nbody_file(param)

    #Copy into p_temp
    Dp_dt = np.dtype([("x",'>f'),("y",'>f'),("z",'>f')])
    Dp    = np.zeros(len(p),dtype=Dp_dt)

    #Read in halo file
    h = read_halo_file(param)

    #Create buffer to account for box boundaries
    h  = build_buffer(h,param)

    #Build tree
    print('building tree..')
    p_tree = spatial.cKDTree(zip(p['x'],p['y'],p['z']), leafsize=100)
    print('...done!')

    #Loop over haloes, calculate displacement, and displace partricles
    n_halo = len(h['Mvir'])
    for i in _progress(range(n_halo), total=n_halo, desc='Displacing halos',
                       disable=(n_halo <= 1)):
        #range where we consider displacement
        rmax = param.code.rmax
        rmin = (0.001*h['rvir'][i] if 0.001*h['rvir'][i]>param.code.rmin else param.code.rmin)
        rmax = (20.0*h['rvir'][i] if 20.0*h['rvir'][i]<param.code.rmax else param.code.rmax)
        rbin = np.logspace(np.log10(rmin),np.log10(rmax),100,base=10)
        #calculate displacement
        cosmo_bias = splev(h['Mvir'][i],bias_tck)
        cosmo_corr = splev(rbin,corr_tck)
        frac, dens, mass = profiles(rbin,h['Mvir'][i],h['cvir'][i],Mc,mu,nu,thej,cosmo_corr,cosmo_bias,param)
        DDMB = displ(rbin,mass['DMO'],mass['DMB'])
        DDMB_tck = splrep(rbin, DDMB,s=0,k=3)
        #define minimum displacement
        smallestD = 0.01 #Mpc/h
        #array of idx with D>Dsmallest
        idx = np.where(abs(DDMB) > smallestD)
        idx = idx[:][0]
        if (len(idx)>1):
            idx_largest = idx[-1]
            rball = rbin[idx_largest]
        else:
            rball = 0.0
        #consistency check:
        if (rball>Lbox/2.0):
            print('rball = ', rball)
            print('ERROR: REDUCE RBALL!')
            exit()
        #particle ids within rball
        ipbool = p_tree.query_ball_point((h['x'][i],h['y'][i],h['z'][i]),rball)
        #update displacement
        rpDMB  = ((p['x'][ipbool]-h['x'][i])**2.0 + (p['y'][ipbool]-h['y'][i])**2.0 + (p['z'][ipbool]-h['z'][i])**2.0)**0.5
        if (rball>0.0 and len(rpDMB)):
            DrpDMB = splev(rpDMB,DDMB_tck,der=0,ext=1)
            Dp['x'][ipbool] += (p['x'][ipbool]-h['x'][i])*DrpDMB/rpDMB
            Dp['y'][ipbool] += (p['y'][ipbool]-h['y'][i])*DrpDMB/rpDMB
            Dp['z'][ipbool] += (p['z'][ipbool]-h['z'][i])*DrpDMB/rpDMB
    #displace particles
    p['x'] += Dp['x']
    p['y'] += Dp['y']
    p['z'] += Dp['z']
    #periodic bounsdaries
    p['x'][p['x']>Lbox] -= Lbox
    p['x'][p['x']<0.0]  += Lbox
    p['y'][p['y']>Lbox] -= Lbox
    p['y'][p['y']<0.0]  += Lbox
    p['z'][p['z']>Lbox] -= Lbox
    p['z'][p['z']<0.0]  += Lbox
    #write N-body file with displacements
    write_nbody_file(p,p_header,param)




"""
CALCULATE DISPLACEMENTS FOR DIFFERENT COMPONENTS INDIVIDUALLY. 
DISPLACE ONCE AT THE END.
"""

def displace_components(param):
    """
    Reading in N-body and halo files, looping over haloes, calculating
    displacements, and dispalcing particles according to components.
    """

    #relevant parameters
    Mc   = param.baryon.Mc
    mu   = param.baryon.mu
    nu   = param.baryon.nu
    thej = param.baryon.thej
    Lbox = param.sim.Lbox

    #Read cosmic variance/nu/correlation and interpolate
    cosmofile = param.files.cosmofct
    try:
        vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(cosmofile, usecols=(0,1,2,3), unpack=True)
        bias_tck = splrep(vc_m, vc_bias, s=0)
        corr_tck = splrep(vc_r, vc_corr, s=0)
    except IOError:
        print('IOERROR: Cosmofct file does not exist!')
        print('Define par.files.cosmofct = "/path/to/file"')
        print('Run: cosmo(params) to create file')
        exit()

    #Read in N-body particle file
    p, p_header = read_nbody_file(param)

    #Copy into p_temp
    Dp_dt = np.dtype([("x",'>f'),("y",'>f'),("z",'>f')])
    Dp    = np.zeros(len(p),dtype=Dp_dt)

    #Read in halo file
    h = read_halo_file(param)

    #Create buffer to account for box boundaries
    h  = build_buffer(h,param)

    #Define gas particles (random shuffle and then first nHGA particles)
    np.take(p,np.random.rand(p.shape[0]).argsort(),axis=0,out=p)
    fhga = param.cosmo.Ob/param.cosmo.Om # - SUBTACT STARS
    nHGA = int(round(fhga*p.shape[0],0))

    p_header['ng'] = np.uint64(nHGA)
    #print type(p_header), type(p_header['ng'][0])
    p_header['nd'] = np.uint64(p_header['npart'] - np.uint64(nHGA))
    #print p_header
    #exit()

    #fhga  = param.cosmo.Ob/param.cosmo.Om # - MORE STIUFF STARS
    #nHGA  = int(round(fhga*p.shape[0],0))
    #irand = np.random.choice(p.shape[0],nHGA)
    #np.put(p,irand,p[irand])

    #Build tree
    print('building tree..')
    p_tree = spatial.cKDTree(zip(p['x'],p['y'],p['z']), leafsize=100)
    print('...done!')

    #Loop over haloes, calculate displacement, and displace partricles
    n_halo = len(h['Mvir'])
    for i in _progress(range(n_halo), total=n_halo, desc='Displacing halos',
                       disable=(n_halo <= 1)):
        #range where we consider displacement
        rmax = param.code.rmax
        rmin = (0.001*h['rvir'][i] if 0.001*h['rvir'][i]>param.code.rmin else param.code.rmin)
        rmax = (20.0*h['rvir'][i] if 20.0*h['rvir'][i]<param.code.rmax else param.code.rmax)
        rbin = np.logspace(np.log10(rmin),np.log10(rmax),100,base=10)
        #calculate displacement
        cosmo_bias = splev(h['Mvir'][i],bias_tck)
        cosmo_corr = splev(rbin,corr_tck)
        frac, dens, mass = profiles(rbin,h['Mvir'][i],h['cvir'][i],Mc,mu,nu,thej,cosmo_corr,cosmo_bias,param)
        #gas displacement
        DHGA = displ(rbin,frac['HGA']*mass['DMO'],mass['HGA'])
        DHGA_tck = splrep(rbin, DHGA,s=0,k=3)
        #nongas displacement
        DNGA = displ(rbin,(frac['CDM']+frac['CGA']+frac['SGA'])*mass['DMO'],mass['ACM']+mass['CGA'])
        DNGA_tck = splrep(rbin, DNGA,s=0,k=3)
        #define minimum displacement
        smallestD = 0.01 #Mpc/h
        #array of idx with D>Dsmallest
        idxHGA = np.where(abs(DHGA) > smallestD)
        idxHGA = idxHGA[:][0]
        if (len(idxHGA)>1):
            idxHGA_largest = idxHGA[-1]
            rballHGA = rbin[idxHGA_largest]
        else:
            rballHGA = 0.0
        idxNGA = np.where(abs(DNGA) > smallestD)
        idxNGA = idxNGA[:][0]
        if (len(idxNGA)>1):
            idxNGA_largest = idxNGA[-1]
            rballNGA = rbin[idxNGA_largest]
        else:
            rballNGA = 0.0
        rball = max(rballHGA,rballNGA)
        #consistency check:
        if (rball>Lbox/2.0):
            print('rball = ', rball)
            print('ERROR: REDUCE RBALL!')
            exit()
        #particle ids within rball
        ipbool = np.array(p_tree.query_ball_point((h['x'][i],h['y'][i],h['z'][i]),rball))
        ipHGA  = ipbool[np.where(ipbool < nHGA)]
        ipNGA  = ipbool[np.where(ipbool >= nHGA)]
        #update displacement
        if (rball>0.0 and len(ipHGA)>0 and len(ipNGA)>0):
            rpHGA  = ((p['x'][ipHGA]-h['x'][i])**2.0 + (p['y'][ipHGA]-h['y'][i])**2.0 + (p['z'][ipHGA]-h['z'][i])**2.0)**0.5
            rpNGA  = ((p['x'][ipNGA]-h['x'][i])**2.0 + (p['y'][ipNGA]-h['y'][i])**2.0 + (p['z'][ipNGA]-h['z'][i])**2.0)**0.5
            DrpHGA = splev(rpHGA,DHGA_tck,der=0,ext=1)
            DrpNGA = splev(rpNGA,DNGA_tck,der=0,ext=1)
            Dp['x'][ipHGA] += (p['x'][ipHGA]-h['x'][i])*DrpHGA/rpHGA
            Dp['y'][ipHGA] += (p['y'][ipHGA]-h['y'][i])*DrpHGA/rpHGA
            Dp['z'][ipHGA] += (p['z'][ipHGA]-h['z'][i])*DrpHGA/rpHGA
            Dp['x'][ipNGA] += (p['x'][ipNGA]-h['x'][i])*DrpNGA/rpNGA
            Dp['y'][ipNGA] += (p['y'][ipNGA]-h['y'][i])*DrpNGA/rpNGA
            Dp['z'][ipNGA] += (p['z'][ipNGA]-h['z'][i])*DrpNGA/rpNGA
    #displace particles
    p['x'] += Dp['x']
    p['y'] += Dp['y']
    p['z'] += Dp['z']
    #p['x'] = (p['x'] + Dp['x']).astype(np.float32)
    #p['y'] = (p['y'] + Dp['y']).astype(np.float32)
    #p['z'] = (p['z'] + Dp['z']).astype(np.float32)
    #periodic bounsdaries
    p['x'][p['x']>Lbox] -= Lbox
    p['x'][p['x']<0.0]  += Lbox
    p['y'][p['y']>Lbox] -= Lbox
    p['y'][p['y']<0.0]  += Lbox
    p['z'][p['z']>Lbox] -= Lbox
    p['z'][p['z']<0.0]  += Lbox
    print(p['mass'])
    #write N-body file with displacements
    p_header['ng'] = np.uint64(nHGA)
    p_header['nd'] = np.uint64(p_header['npart']) - np.uint64(nHGA)
    print(p_header)
    write_nbody_file(p,p_header,param)
