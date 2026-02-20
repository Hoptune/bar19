Baryonification code forked from https://bitbucket.org/aurelschneider/baryonification/src/master/, based on https://arxiv.org/abs/1810.08629 and https://arxiv.org/abs/1510.06034.

Modifications to the original codes include:

## Read and write hdf5 halo and particle catalogs
e.g., minimal HDF5 particle catalogs:

Particle I/O now supports:

- `par.files.partfile_format = "hdf5"` (or `"catalog-hdf5"`)
- input and output files with datasets `dm/x`, `dm/y`, `dm/z` (or `x`, `y`, `z` at root)

Only positions are read/written for HDF5 catalogs.

## Customization of the mass definition from params, and support exporting baryonified halo mass.
