import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


def _legacy_chunk_particles(p, lbox, n_chunk):
    l_chunk = lbox / n_chunk
    chunks = []
    for x_min in np.linspace(0, lbox - l_chunk, n_chunk):
        x_max = x_min + l_chunk
        if x_max == lbox:
            x_max = 1.00001 * x_max
        for y_min in np.linspace(0, lbox - l_chunk, n_chunk):
            y_max = y_min + l_chunk
            if y_max == lbox:
                y_max = 1.00001 * y_max
            for z_min in np.linspace(0, lbox - l_chunk, n_chunk):
                z_max = z_min + l_chunk
                if z_max == lbox:
                    z_max = 1.00001 * z_max
                idx = np.where(
                    (p["x"] >= x_min)
                    & (p["x"] < x_max)
                    & (p["y"] >= y_min)
                    & (p["y"] < y_max)
                    & (p["z"] >= z_min)
                    & (p["z"] < z_max)
                )
                chunks.append(p[idx])
    return chunks


def _load_displ_module():
    repo_root = Path(__file__).resolve().parents[1]
    displ_path = repo_root / "baryonification" / "displ.py"

    pkg = types.ModuleType("baryonification")
    pkg.__path__ = [str(repo_root / "baryonification")]
    sys.modules.setdefault("baryonification", pkg)

    # Stub dependencies not needed by read_nbody_file in these tests.
    sys.modules.setdefault("baryonification.profiles", types.ModuleType("baryonification.profiles"))
    sys.modules.setdefault("schwimmbad", types.ModuleType("schwimmbad"))

    spec = importlib.util.spec_from_file_location("baryonification.displ", displ_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["baryonification.displ"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _make_param(partfile_in, lbox, n_chunk):
    return types.SimpleNamespace(
        files=types.SimpleNamespace(
            partfile_in=str(partfile_in),
            partfile_format="npy",
        ),
        sim=types.SimpleNamespace(
            Lbox=float(lbox),
            N_chunk=int(n_chunk),
        ),
        cosmo=types.SimpleNamespace(z=0.0),
    )


def _assert_chunks_identical(testcase, actual, expected):
    testcase.assertEqual(len(actual), len(expected))
    for i, (a, e) in enumerate(zip(actual, expected)):
        testcase.assertEqual(a.dtype, e.dtype, "dtype mismatch in chunk {}".format(i))
        testcase.assertTrue(np.array_equal(a, e), "content mismatch in chunk {}".format(i))


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.displ = _load_displ_module()

    def _read_chunks_from_npy(self, p, lbox, n_chunk, filename):
        with tempfile.TemporaryDirectory() as tmpdir:
            partfile = Path(tmpdir) / filename
            np.save(partfile, p)
            param = _make_param(partfile, lbox=lbox, n_chunk=n_chunk)
            p_list, p_header = self.displ.read_nbody_file(param)
        return p_list, p_header

    def test_chunking_matches_legacy_random_points(self):
        lbox = 128.0
        for n_chunk in (1, 2, 3):
            with self.subTest(n_chunk=n_chunk):
                rng = np.random.default_rng(42 + n_chunk)
                n_part = 500
                p_dt = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
                p = np.zeros(n_part, dtype=p_dt)
                p["x"] = rng.uniform(0.0, lbox, size=n_part).astype(np.float32)
                p["y"] = rng.uniform(0.0, lbox, size=n_part).astype(np.float32)
                p["z"] = rng.uniform(0.0, lbox, size=n_part).astype(np.float32)

                p_list, _ = self._read_chunks_from_npy(
                    p,
                    lbox=lbox,
                    n_chunk=n_chunk,
                    filename="particles.npy",
                )
                expected = _legacy_chunk_particles(p, lbox=lbox, n_chunk=n_chunk)
                _assert_chunks_identical(self, p_list, expected)

    def test_chunking_matches_legacy_many_random_configs(self):
        lbox = 256.0
        for n_chunk in (1, 2, 3, 4):
            for seed in (3, 11, 29, 47):
                with self.subTest(n_chunk=n_chunk, seed=seed):
                    rng = np.random.default_rng(seed)
                    n_part = 800
                    p = np.zeros(
                        n_part,
                        dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
                    )
                    p["x"] = rng.uniform(0.0, lbox, size=n_part).astype(np.float32)
                    p["y"] = rng.uniform(0.0, lbox, size=n_part).astype(np.float32)
                    p["z"] = rng.uniform(0.0, lbox, size=n_part).astype(np.float32)

                    p_list, _ = self._read_chunks_from_npy(
                        p,
                        lbox=lbox,
                        n_chunk=n_chunk,
                        filename="particles_many.npy",
                    )
                    expected = _legacy_chunk_particles(p, lbox=lbox, n_chunk=n_chunk)
                    _assert_chunks_identical(self, p_list, expected)

    def test_chunking_matches_legacy_boundary_points(self):
        lbox = 128.0
        n_chunk = 2
        p = np.array(
            [
                (0.0, 0.0, 0.0),
                (lbox, 0.0, 0.0),
                (0.0, lbox, 0.0),
                (0.0, 0.0, lbox),
                (lbox, lbox, lbox),
                (lbox - 1e-6, lbox - 1e-6, lbox - 1e-6),
            ],
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
        )

        p_list, _ = self._read_chunks_from_npy(
            p,
            lbox=lbox,
            n_chunk=n_chunk,
            filename="particles_boundary.npy",
        )
        expected = _legacy_chunk_particles(p, lbox=lbox, n_chunk=n_chunk)
        _assert_chunks_identical(self, p_list, expected)

    def test_chunking_with_extra_fields_matches_legacy(self):
        lbox = 100.0
        n_chunk = 3
        rng = np.random.default_rng(99)
        p = np.zeros(
            400,
            dtype=[
                ("id", np.int64),
                ("mass", np.float32),
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
            ],
        )
        p["id"] = np.arange(len(p))
        p["mass"] = rng.uniform(1.0, 2.0, size=len(p)).astype(np.float32)
        p["x"] = rng.uniform(0.0, lbox, size=len(p)).astype(np.float32)
        p["y"] = rng.uniform(0.0, lbox, size=len(p)).astype(np.float32)
        p["z"] = rng.uniform(0.0, lbox, size=len(p)).astype(np.float32)

        p_list, _ = self._read_chunks_from_npy(
            p,
            lbox=lbox,
            n_chunk=n_chunk,
            filename="particles_extra_fields.npy",
        )
        expected = _legacy_chunk_particles(p, lbox=lbox, n_chunk=n_chunk)
        _assert_chunks_identical(self, p_list, expected)

    def test_chunking_preserves_row_order_within_each_chunk(self):
        lbox = 64.0
        n_chunk = 2
        p = np.array(
            [
                (10, 1.0, 1.0, 1.0),
                (11, 2.0, 2.0, 2.0),
                (12, 3.0, 3.0, 3.0),
                (13, 40.0, 40.0, 40.0),
                (14, 41.0, 41.0, 41.0),
                (15, 4.0, 4.0, 4.0),
            ],
            dtype=[("id", np.int64), ("x", np.float32), ("y", np.float32), ("z", np.float32)],
        )
        p_list, _ = self._read_chunks_from_npy(
            p,
            lbox=lbox,
            n_chunk=n_chunk,
            filename="particles_order.npy",
        )
        expected = _legacy_chunk_particles(p, lbox=lbox, n_chunk=n_chunk)
        _assert_chunks_identical(self, p_list, expected)

    def test_chunking_preserves_particle_count_and_chunk_count(self):
        lbox = 64.0
        n_chunk = 4
        rng = np.random.default_rng(7)
        p = np.zeros(1000, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)])
        p["x"] = rng.uniform(0.0, lbox, size=len(p))
        p["y"] = rng.uniform(0.0, lbox, size=len(p))
        p["z"] = rng.uniform(0.0, lbox, size=len(p))

        p_list, _ = self._read_chunks_from_npy(
            p,
            lbox=lbox,
            n_chunk=n_chunk,
            filename="particles_count.npy",
        )
        self.assertEqual(len(p_list), n_chunk**3)
        self.assertEqual(sum(len(chunk) for chunk in p_list), len(p))

    def test_chunking_handles_empty_input(self):
        lbox = 128.0
        n_chunk = 3
        p = np.zeros(0, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)])

        p_list, p_header = self._read_chunks_from_npy(
            p,
            lbox=lbox,
            n_chunk=n_chunk,
            filename="particles_empty.npy",
        )
        self.assertEqual(len(p_list), n_chunk**3)
        self.assertTrue(all(len(chunk) == 0 for chunk in p_list))
        self.assertEqual(sum(len(chunk) for chunk in p_list), 0)
        self.assertEqual(p_header["ndim"], 3)

    def test_chunking_exits_for_out_of_bounds_particles(self):
        lbox = 128.0
        n_chunk = 2
        upper = 1.00001 * lbox
        p = np.array(
            [
                (0.1, 0.1, 0.1),
                (-1e-3, 0.2, 0.2),
                (upper, 0.3, 0.3),
            ],
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            partfile = Path(tmpdir) / "particles_bad.npy"
            np.save(partfile, p)
            param = _make_param(partfile, lbox=lbox, n_chunk=n_chunk)
            with self.assertRaises(SystemExit):
                self.displ.read_nbody_file(param)


if __name__ == "__main__":
    unittest.main()
