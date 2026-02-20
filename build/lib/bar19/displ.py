"""
CALCULATE DISPLACEMENT FUNCTION FOR A GRID OF M AND C
PRINT INFORMATION INTO TEMPORARY FILE

"""

#from __future__ import print_function
#from __future__ import division

import numpy as np
import os
import multiprocessing
from scipy import spatial
from scipy.interpolate import splrep,splev
from numpy.lib.recfunctions import append_fields

import schwimmbad
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

#from .params import par
from .constants import *
from .profiles import *

"""
READING/WRITING FILES
"""

def _progress(iterable, total=None, desc=None, disable=False):

    """
    Wrap iterables with a progress bar when tqdm is available.
    """

    if disable or (tqdm is None) or (not _is_master_process()):
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


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


def read_nbody_file(param):

    """
    Read in N-body output, adopt units, and build chunks 
    (for multi-processor mode)
    Only supports tispy file format for the moment
    """

    nbody_file_in = param.files.partfile_in
    nbody_file_format = param.files.partfile_format
    Lbox   = param.sim.Lbox
    N_chunk = param.sim.N_chunk
    L_chunk = Lbox/N_chunk

    #read in file
    if (nbody_file_format=='tipsy'):

        try:
            f = open(nbody_file_in, 'r')
        except IOError:
            print('IOERROR: N-body tipsy file does not exist!')
            print('Define par.files.partfile_in = "/path/to/file"')
            exit()

        #header
        p_header_dt = np.dtype([('a','>d'),('npart','>i'),('ndim','>i'),('ng','>i'),('nd','>i'),('ns','>i'),('buffer','>i')])
        p_header = np.fromfile(f, dtype=p_header_dt, count=1, sep='')

        #particles
        p_dt = np.dtype([('mass','>f'),("x",'>f'),("y",'>f'),("z",'>f'),("vx",'>f'),("vy",'>f'),("vz",'>f'),("eps",'>f'),("phi",'>f')])
        p = np.fromfile(f, dtype=p_dt, count=int(p_header['npart']), sep='')

        #from tipsy units to [0,Lbox] in units of Lbox
        p['x']=Lbox*(p['x']+0.5)
        p['y']=Lbox*(p['y']+0.5)
        p['z']=Lbox*(p['z']+0.5)

        print('Reading tipsy-file done!')

    elif (nbody_file_format=='npy'):

        try:
            p = np.load(nbody_file_in)
        except IOError:
            print('IOERROR: N-body npy file does not exist!')
            print('Define par.files.partfile_in = "/path/to/file"')
            exit()

        #header (placeholder)
        p_header = {'a': 1.0, 'npart': 1, 'ndim': 3, 'ng': 0, 'nd': 1, 'ns': 0,'buffer': 0}

        #require structured arrays containing x/y/z fields
        if (p.dtype.names is None) or any(name not in p.dtype.names for name in ("x", "y", "z")):
            print('IOERROR: npy particle catalog must be a structured array with x/y/z fields.')
            exit()

        print('Reading npy-file done!')

    elif (nbody_file_format=='hdf5' or nbody_file_format=='catalog-hdf5'):

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

    elif (nbody_file_format=='gadget'):

        print('Reading gadget files not implemented. Exit!')
        exit()

    else:
        print('Unknown file format. Exit!')
        exit()

    # Fast chunking in O(N) assignment + grouping by chunk ID.
    n_total_chunks = int(N_chunk**3)
    x = np.asarray(p['x'])
    y = np.asarray(p['y'])
    z = np.asarray(p['z'])

    # Match legacy behavior: keep only coordinates in [0, 1.00001*Lbox),
    # where x==Lbox/y==Lbox/z==Lbox are assigned to the last chunk.
    upper = 1.00001 * Lbox
    valid = (x >= 0.0) & (x < upper) & (y >= 0.0) & (y < upper) & (z >= 0.0) & (z < upper)

    p_list = [None] * n_total_chunks
    for i in range(n_total_chunks):
        p_list[i] = p[:0]

    particles_assigned = int(np.sum(valid))
    if particles_assigned > 0:
        xv = x[valid]
        yv = y[valid]
        zv = z[valid]

        ix = np.floor(xv / L_chunk).astype(np.int64)
        iy = np.floor(yv / L_chunk).astype(np.int64)
        iz = np.floor(zv / L_chunk).astype(np.int64)
        ix = np.clip(ix, 0, N_chunk - 1)
        iy = np.clip(iy, 0, N_chunk - 1)
        iz = np.clip(iz, 0, N_chunk - 1)

        flat_chunk_id = ((ix * N_chunk) + iy) * N_chunk + iz
        valid_idx = np.nonzero(valid)[0]
        order = np.argsort(flat_chunk_id, kind='stable')
        sorted_particle_idx = valid_idx[order]

        counts = np.bincount(flat_chunk_id, minlength=n_total_chunks)
        starts = np.empty(n_total_chunks + 1, dtype=np.int64)
        starts[0] = 0
        starts[1:] = np.cumsum(counts, dtype=np.int64)

        for chunk_id in _progress(range(n_total_chunks), total=n_total_chunks,
                                  desc='Loading chunked particles',
                                  disable=(n_total_chunks <= 1)):
            i0 = int(starts[chunk_id])
            i1 = int(starts[chunk_id + 1])
            p_list[chunk_id] = p[sorted_particle_idx[i0:i1]]

    if (particles_assigned != len(p)):
        print('Chunking: particle number not conserved! Exit.')
        exit()
    return p_list, p_header



def write_nbody_file(p_list,p_header,param):

    """
    Combine chunks and write N-body outputs with displaced 
    particles. Only tipsy file format for the moment.
    p_list = list of p with legth = Nchunk 
    """

    nbody_file_out = param.files.partfile_out
    nbody_file_format = param.files.partfile_format
    Lbox = param.sim.Lbox
    N_chunk = param.sim.N_chunk

    #combine chunks
    #if (N_chunk > 1):
    #    p = np.concatenate(p_list)
    #else:
    #    p = p_list

    p = np.concatenate(p_list)
    #print(type(p))
    #print(p['x'][0])
    #correct for periodic boundaries
    p['x'][p['x']>=Lbox] -= Lbox
    p['x'][p['x']<0.0]  += Lbox
    p['y'][p['y']>=Lbox] -= Lbox
    p['y'][p['y']<0.0]  += Lbox
    p['z'][p['z']>=Lbox] -= Lbox
    p['z'][p['z']<0.0]  += Lbox

    #write output
    if (nbody_file_format=='tipsy'):

        try:
            f = open(nbody_file_out, 'wb')
        except IOError:
            print('IOERROR: Path to output file does not exist!')
            print('Define par.files.partfile_out = "/path/to/file"')
            exit()

        #back to tipsy units
        p['x']=(p['x']/Lbox-0.5).astype(np.float32)
        p['y']=(p['y']/Lbox-0.5).astype(np.float32)
        p['z']=(p['z']/Lbox-0.5).astype(np.float32)
        p_header.tofile(f,sep='')
        p.tofile(f,sep='')

    elif (nbody_file_format=='npy'):
        try:
            np.save(nbody_file_out,p)
        except IOError:
            print('IOERROR: Path to output file does not exist!')
            print('Define par.files.partfile_out = "/path/to/file"')
            exit()

    elif (nbody_file_format=='hdf5' or nbody_file_format=='catalog-hdf5'):
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
        except OSError:
            print('IOERROR: Cannot write HDF5 catalog output file!')
            print('Define par.files.partfile_out = "/path/to/output.hdf5"')
            exit()

        print('Writing HDF5 catalog output done!')

    elif (nbody_file_format=='gadget'):
        print('Writing gadget files not implemented. Exit!')
        exit()

    else:
        print('Unknown file format. Exit!')
        exit()


def read_halo_file(param):

    """
    Only supports AHF-file format for the moment
    Read in halo file, adopt units, build buffer around 
    chunks (for multi-processor mode)
    Select for hosts above a minimum halo mass.
    Restricted to AHF for the moment.
    """

    #read files 
    halo_file_in = param.files.halofile_in
    halo_file_format = param.files.halofile_format
    Mhalo_min = param.sim.Mhalo_min
    Lbox = param.sim.Lbox

    if (halo_file_format=='AHF-ASCII'):
        try:
            names = "ID,IDhost,Mvir,Nvir,x,y,z,rvir,cvir"
            h = np.genfromtxt(halo_file_in,usecols=(0,1,3,4,5,6,7,11,42),comments='#',dtype=None,names=names)
        except IOError:
            print('IOERROR: AHF-ASCII file does not exist!')
            print('Define par.files.halofile_in = "/path/to/file"')
            exit()

        #adopt units
        h['x']    = h['x']/1000.0
        h['y']    = h['y']/1000.0
        h['z']    = h['z']/1000.0
        h['rvir'] = h['rvir']/1000.0

        #select haloes above minimum mass
        gID  = np.where(h['Mvir'] >= Mhalo_min)
        h = h[gID]
        #select haloes with reasonable concentration
        gID  = np.where(h['cvir'] > 0)
        h = h[gID]
        #select main haloes (only if ahf calculates host) 
        gID  = np.where(h['IDhost'] < 0.0)
        h = h[gID] 
        print('Nhalo = ',len(h['Mvir']))

    elif (halo_file_format=='ROCKSTAR-NPY'):
        try:
            h = np.load(halo_file_in)
        except IOError:
            print('IOERROR: ROCKSTAR-NPY file does not exist!')
            print('Define par.files.halofile_in = "/path/to/file"')
            exit()

        h_dt = np.dtype([('halo_id', '<i8'), ('upid', '<i8'), ('x', '<f8'), ('y', '<f8'), 
                         ('z', '<f8'), ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'), 
                         ('Mv', '<f8'), ('mpeak', '<f8'), ('vmp', '<f8'), ('r', '<f8'), 
                         ('sm', '<f8'), ('icl', '<f8'), ('sfr', '<f8'), ('ssfr', '<f8'), 
                         ('pid', '<f8'), ('Mvir', '<f8'), ('rvir', '<f8'), 
                         ('rs_hlist', '<f8'), ('scale_half_mass', '<f8'), ('scale_last_mm', '<f8'), 
                         ('m200b_hlist', '<f8'), ('m200c_hlist', '<f8'), ('gamma_inst', '<f8'), 
                         ('gamma_100myr', '<f8'), ('gamma_1tdyn', '<f8'), ('gamma_2tdyn', '<f8'), 
                         ('gamma_mpeak', '<f8'), ('vmax_mpeak', '<f8'), ('halo_hostid', '<i8'), 
                         ('mhalo_host', '<f8'), ('mask_central', '?'), ('mtot_galaxy', '<f8'), 
                         ('mstar_mhalo', '<f8'), ('logms_gal', '<f8'), ('logms_icl', '<f8'), 
                         ('logms_tot', '<f8'), ('logms_halo', '<f8'), ('logmh_vir', '<f8'), 
                         ('logmh_peak', '<f8'), ('logmh_host', '<f8')])
        h = np.array(h,dtype=h_dt)
        h = append_fields(h, 'cvir', h['rvir']/h['rs_hlist'])

        print("Using param.sim.Mhalo_min to filter haloes.")

        #select haloes with reasonable concentration
        gID  = np.where(np.isfinite(h['cvir']))
        h = h[gID]
        #select main haloes (only if ahf calculates host)
        gID  = np.where(h['upid'] < 0.0)
        h = h[gID]

        #select haloes above minimum mass
        gID  = np.where(h['Mvir'] >= Mhalo_min)
        h = h[gID]

        print('Nhalo = ',len(h['Mvir']))

        #adopt units
        h['rvir'] = h['rvir']/1000
        h['rs_hlist'] = h['rs_hlist']/1000
        
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
        h['x'] = x
        h['y'] = y
        h['z'] = z
        h['Mvir'] = Mvir
        h['rvir'] = rvir
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

    else:
        print('Unknown halo file format. Exit!')
        exit()
    if param.code.return_bcmmass:
        print('WARNING: param.code.return_bcmmass is True. Returning halo masses in h["Mvir_bcm"].')
        h = append_fields(h, 'Mvir_bcm', np.zeros(len(h['Mvir'])))
    #build buffer
    rbuffer = param.code.rbuffer
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

    #separate into chunks
    N_chunk = param.sim.N_chunk
    L_chunk = Lbox/N_chunk
    chunk_bounds = []
    for x_min in np.linspace(0,Lbox-L_chunk,N_chunk):
        x_max = x_min + L_chunk
        for y_min in np.linspace(0,Lbox-L_chunk,N_chunk):
            y_max = y_min + L_chunk
            for z_min in np.linspace(0,Lbox-L_chunk,N_chunk):
                z_max = z_min + L_chunk
                chunk_bounds.append((x_min, x_max, y_min, y_max, z_min, z_max))

    h_list = []
    for x_min, x_max, y_min, y_max, z_min, z_max in _progress(
            chunk_bounds,
            total=len(chunk_bounds),
            desc='Loading chunked halos',
            disable=(len(chunk_bounds) <= 1)):
        ID = np.where((h['x']>=(x_min-rbuffer)) & (h['x']<(x_max+rbuffer)) & \
                      (h['y']>=(y_min-rbuffer)) & (h['y']<(y_max+rbuffer)) & \
                      (h['z']>=(z_min-rbuffer)) & (h['z']<(z_max+rbuffer)))
        h_list += [h[ID]]
    return h_list

def write_halo_file(h_list, param):

    """
    Combine displaced halo chunks and write a halo catalog in HDF5 format.
    """

    try:
        import h5py
    except ImportError:
        print('IOERROR: h5py is required to write halo output in HDF5 format.')
        print('Install with: pip install h5py')
        exit()

    # Accept either a list of chunks or a single structured array.
    if isinstance(h_list, (list, tuple)):
        if len(h_list) == 0:
            print('WARNING: Empty halo list. Writing empty halo catalog.')
            h = np.zeros(
                0,
                dtype=[('Mvir', '<f8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
                       ('rvir', '<f8'), ('cvir', '<f8'), ('Mvir_bcm', '<f8')],
            )
        else:
            h = np.concatenate(h_list)
    else:
        h = h_list

    if (h.dtype.names is None):
        print('IOERROR: Halo output must be a structured array.')
        exit()

    # Chunks overlap by a buffer; remove duplicate rows and keep first-seen order.
    if len(h) > 0:
        _, unique_idx = np.unique(h, return_index=True)
        h = h[np.sort(unique_idx)]

    halo_file_out = getattr(param.files, 'halofile_out', None)
    if halo_file_out is None:
        part_out = getattr(param.files, 'partfile_out', 'partfile_out.hdf5')
        base, _ = os.path.splitext(part_out)
        halo_file_out = base + '_halos.hdf5'
        print('WARNING: param.files.halofile_out not set; using', halo_file_out)

    if not (halo_file_out.endswith('.hdf5') or halo_file_out.endswith('.h5')):
        base, _ = os.path.splitext(halo_file_out)
        halo_file_out = base + '.hdf5'

    try:
        with h5py.File(halo_file_out, 'w') as f:
            g_halo = f.create_group('halos')
            for field in h.dtype.names:
                g_halo.create_dataset(field, data=np.asarray(h[field]))
    except OSError:
        print('IOERROR: Cannot write HDF5 halo output file!')
        print('Define par.files.halofile_out = "/path/to/output_halos.hdf5"')
        exit()

    print('Writing HDF5 halo output done! Nhalo =', len(h))




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



def displace(param):

    """
    Reading in N-body and halo files, defining chunks, 
    looping over haloes in single or multi-processor mode,
    dispalcing particles, combining chunks, 
    writing N-body file
    """

    #Read in N-body particle file and build chunks
    p_list, p_header = read_nbody_file(param)

    #Read in halo file, build chunks and buffer
    h_list = read_halo_file(param)

    #split work on cpus and perform displacement
    N_chunk = param.sim.N_chunk
    N_cpu   = int(N_chunk**3)
    print('N_cpu = ',N_cpu)

    if (N_cpu == 1):
        ph_displ = [displace_chunk(p_list[0],h_list[0],p_header,param)]

    elif (N_cpu > 1):
        pool = schwimmbad.choose_pool(mpi=False, processes=N_cpu)
        tasks = [(i, p_list[i], h_list[i], p_header, param) for i in range(N_cpu)]
        ph_displ = [None] * N_cpu

        if hasattr(pool, 'imap_unordered'):
            task_results = pool.imap_unordered(worker, tasks)
            for chunk_id, p_chunk_displ, h_chunk_displ in _progress(
                    task_results,
                    total=N_cpu,
                    desc='Displacing chunks',
                    disable=(N_cpu <= 1)):
                ph_displ[chunk_id] = (p_chunk_displ, h_chunk_displ)
        else:
            task_results = pool.map(worker, tasks)
            for chunk_id, p_chunk_displ, h_chunk_displ in _progress(
                    task_results,
                    total=N_cpu,
                    desc='Displacing chunks',
                    disable=(N_cpu <= 1)):
                ph_displ[chunk_id] = (p_chunk_displ, h_chunk_displ)

        pool.close()

    #combine chunks and write output
    p_displ = [p for p,h in ph_displ]
    h_displ = [h for p,h in ph_displ]
    write_nbody_file(p_displ,p_header,param)
    write_halo_file(h_displ,param)


def worker(task):

    """
    Worker for multi-processing
    """

    if len(task) == 5:
        chunk_id, p_chunk, h_chunk, p_header, param = task
        p_displ, h_displ = displace_chunk(p_chunk,h_chunk,p_header,param)
        return chunk_id, p_displ, h_displ

    p_chunk, h_chunk, p_header, param = task
    return displace_chunk(p_chunk,h_chunk,p_header,param)



def displace_chunk(p_chunk,h_chunk,p_header,param):

    """
    Reading in N-body and halo files, looping over haloes, calculating
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

    #Copy into p_temp
    Dp_dt = np.dtype([("x",'>f'),("y",'>f'),("z",'>f')])
    Dp    = np.zeros(len(p_chunk),dtype=Dp_dt)

    #Build tree
    print('building tree..')
    p_tree = spatial.cKDTree(list(zip(p_chunk['x'],p_chunk['y'],p_chunk['z'])), leafsize=100)
    print('...done!')

    #Loop over haloes, calculate displacement, and displace partricles
    n_halo = len(h_chunk['Mvir'])
    for i in _progress(range(n_halo), total=n_halo, desc='Displacing halos',
                       disable=(n_halo <= 1)):

        #range where we consider displacement
        rmax = param.code.rmax
        rmin = (0.001*h_chunk['rvir'][i] if 0.001*h_chunk['rvir'][i]>param.code.rmin else param.code.rmin)
        rmax = (20.0*h_chunk['rvir'][i] if 20.0*h_chunk['rvir'][i]<param.code.rmax else param.code.rmax)
        rbin = np.logspace(np.log10(rmin),np.log10(rmax),100,base=10)

        #calculate displacement
        cosmo_bias = splev(h_chunk['Mvir'][i],bias_tck)
        cosmo_corr = splev(rbin,corr_tck)
        frac, dens, mass = profiles(rbin,h_chunk['Mvir'][i],h_chunk['cvir'][i],cosmo_corr,cosmo_bias,param)
        DDMB = displ(rbin,mass['DMO'],mass['DMB'])
        DDMB_tck = splrep(rbin, DDMB,s=0,k=3)
        if (param.code.return_bcmmass):
            enc_dens_ov_rhocrit = mass['DMB'] / rhoc_of_z(param) / (4*np.pi/3.0*rbin**3)
            M_deltaout = 10**np.interp(param.sim.deltavir, enc_dens_ov_rhocrit[::-1], np.log10(mass['DMB'])[::-1], left=np.inf, right=-np.inf)
            h_chunk['Mvir_bcm'][i] = M_deltaout

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
        ipbool = p_tree.query_ball_point((h_chunk['x'][i],h_chunk['y'][i],h_chunk['z'][i]),rball)

        #update displacement
        rpDMB  = ((p_chunk['x'][ipbool]-h_chunk['x'][i])**2.0 + 
                  (p_chunk['y'][ipbool]-h_chunk['y'][i])**2.0 + 
                  (p_chunk['z'][ipbool]-h_chunk['z'][i])**2.0)**0.5

        if (rball>0.0 and len(rpDMB)):
            DrpDMB = splev(rpDMB,DDMB_tck,der=0,ext=1)
            Dp['x'][ipbool] += (p_chunk['x'][ipbool]-h_chunk['x'][i])*DrpDMB/rpDMB
            Dp['y'][ipbool] += (p_chunk['y'][ipbool]-h_chunk['y'][i])*DrpDMB/rpDMB
            Dp['z'][ipbool] += (p_chunk['z'][ipbool]-h_chunk['z'][i])*DrpDMB/rpDMB

    #displace particles
    p_chunk['x'] += Dp['x']
    p_chunk['y'] += Dp['y']
    p_chunk['z'] += Dp['z']

    return p_chunk, h_chunk
