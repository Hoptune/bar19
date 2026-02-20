import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


def _load_displ_module():
    repo_root = Path(__file__).resolve().parents[1]
    displ_path = repo_root / "bar19" / "displ.py"

    pkg = types.ModuleType("bar19")
    pkg.__path__ = [str(repo_root / "bar19")]
    sys.modules.setdefault("bar19", pkg)

    # Stub dependencies not needed for halo output tests.
    sys.modules.setdefault("bar19.profiles", types.ModuleType("bar19.profiles"))
    sys.modules.setdefault("schwimmbad", types.ModuleType("schwimmbad"))

    spec = importlib.util.spec_from_file_location("bar19.displ", displ_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bar19.displ"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _make_param(halofile_out, lbox):
    return types.SimpleNamespace(
        files=types.SimpleNamespace(
            halofile_out=str(halofile_out),
            partfile_out=str(halofile_out).replace("_halos", "_parts"),
        ),
        sim=types.SimpleNamespace(
            Lbox=float(lbox),
        ),
        code=types.SimpleNamespace(
            return_bcmmass=False,
        ),
        cosmo=types.SimpleNamespace(),
        baryon=types.SimpleNamespace(),
    )


class HaloOutputCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.displ = _load_displ_module()
        self.halo_dtype = np.dtype(
            [
                ("ID", np.int64),
                ("hostID", np.int64),
                ("Mvir", np.float64),
                ("x", np.float64),
                ("y", np.float64),
                ("z", np.float64),
                ("rvir", np.float64),
                ("cvir", np.float64),
            ]
        )

    @unittest.skipIf(h5py is None, "h5py is required for HDF5 halo output test")
    def test_write_halo_file_writes_coordinates_within_box(self):
        lbox = 128.0
        h = np.zeros(4, dtype=self.halo_dtype)
        h["ID"] = [1, 2, 2, 3]
        h["hostID"] = [-1, -1, -1, -1]
        h["Mvir"] = [1.0e12, 2.0e12, 2.2e12, 3.0e12]
        h["x"] = [-0.1, 128.0, 999.0, 130.5]
        h["y"] = [129.5, -2.0, 999.0, 127.9]
        h["z"] = [256.0, 127.0, 999.0, -1.0]
        h["rvir"] = [0.2, 0.3, 0.3, 0.4]
        h["cvir"] = [5.0, 6.0, 6.1, 7.0]

        with tempfile.TemporaryDirectory() as tmpdir:
            halo_out = Path(tmpdir) / "output_halos.hdf5"
            param = _make_param(halo_out, lbox=lbox)
            self.displ.write_halo_file([h], param)

            with h5py.File(halo_out, "r") as f:
                g = f["halos"]
                self.assertEqual(len(np.asarray(g["ID"])), 3)
                for axis in ("x", "y", "z"):
                    arr = np.asarray(g[axis])
                    self.assertTrue(np.all(arr >= 0.0))
                    self.assertTrue(np.all(arr < lbox))


if __name__ == "__main__":
    unittest.main()
