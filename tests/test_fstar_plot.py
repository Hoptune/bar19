import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _ensure_scipy_integrate_compat():
    import scipy.integrate as integrate

    if not hasattr(integrate, "simps") and hasattr(integrate, "simpson"):
        integrate.simps = integrate.simpson
    if not hasattr(integrate, "cumtrapz") and hasattr(integrate, "cumulative_trapezoid"):
        integrate.cumtrapz = integrate.cumulative_trapezoid


def _load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_params_and_profiles():
    repo_root = Path(__file__).resolve().parents[1]
    pkg_path = repo_root / "bar19"

    pkg = sys.modules.get("bar19")
    if pkg is None:
        pkg = types.ModuleType("bar19")
        sys.modules["bar19"] = pkg
    pkg.__path__ = [str(pkg_path)]

    _ensure_scipy_integrate_compat()
    _load_module("bar19.constants", pkg_path / "constants.py")
    _load_module("bar19.cosmo", pkg_path / "cosmo.py")
    params_mod = _load_module("bar19.params", pkg_path / "params.py")
    profiles_mod = _load_module("bar19.profiles", pkg_path / "profiles.py")
    return params_mod, profiles_mod


@unittest.skipIf(plt is None, "matplotlib is required for plotting tests")
class FstarPlotTests(unittest.TestCase):
    def setUp(self):
        self.params_mod, self.profiles_mod = _load_params_and_profiles()

    def test_plot_fstar_central_vs_halo_mass_with_default_parameters(self):
        param = self.params_mod.par()
        mhalo = np.logspace(11.0, 15.0, 128)
        fstar_central = self.profiles_mod.fSTAR_fct(mhalo, param, param.baryon.eta_high_cen)
        fstar_total = self.profiles_mod.fSTAR_fct(mhalo, param, param.baryon.eta_high_tot)

        self.assertTrue(np.all(np.isfinite(fstar_central)))
        self.assertTrue(np.all(fstar_central > 0.0))
        self.assertTrue(np.all(fstar_central < 1.0))
        self.assertTrue(np.all(np.isfinite(fstar_total)))
        self.assertTrue(np.all(fstar_total > 0.0))
        self.assertTrue(np.all(fstar_total < 1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = Path(tmpdir) / "fstar_central_default.png"

            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.plot(mhalo, fstar_total, linewidth=2.0, label=r"$f_{\rm STAR}^{\rm tot}$")
            ax.plot(mhalo, fstar_central, linewidth=2.0, label=r"$f_{\rm STAR}^{\rm cen}$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1.0e11, 1.0e15)
            ax.set_xlabel(r"$M_{\rm halo}\,[{\rm M}_\odot/h]$")
            ax.set_ylabel(r"$f_{\rm STAR}^{\rm cen}$")
            ax.legend(fontsize=11)
            fig.tight_layout()
            fig.savefig(outfile, dpi=120)

            self.assertEqual(len(ax.lines), 2)
            np.testing.assert_allclose(ax.lines[0].get_xdata(), mhalo)
            np.testing.assert_allclose(ax.lines[0].get_ydata(), fstar_total)
            np.testing.assert_allclose(ax.lines[1].get_xdata(), mhalo)
            np.testing.assert_allclose(ax.lines[1].get_ydata(), fstar_central)
            self.assertTrue(outfile.exists())
            self.assertGreater(outfile.stat().st_size, 0)

            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
