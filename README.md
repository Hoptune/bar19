Baryonification code forked from https://bitbucket.org/aurelschneider/baryonification/src/master/, based on https://arxiv.org/abs/1810.08629 and https://arxiv.org/abs/1510.06034.

## Minimal HDF5 particle catalogs

Particle I/O now supports:

- `par.files.partfile_format = "hdf5"` (or `"catalog-hdf5"`)
- input and output files with datasets `dm/x`, `dm/y`, `dm/z` (or `x`, `y`, `z` at root)

Only positions are read/written for HDF5 catalogs.
