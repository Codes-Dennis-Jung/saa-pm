"""Library behaviour: no global side effects, in-memory inputs, optional extras."""

import dataclasses
import logging
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import saa_pm
from saa_pm import MandateData, load_mandate
from saa_pm.sample_data import write_sample_data


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sample")
    write_sample_data(str(d), seed=42)
    return str(d)


def test_import_does_not_touch_matplotlib():
    """Importing saa_pm must not load matplotlib or switch the user's backend."""
    code = "import sys, saa_pm\nprint('matplotlib' in sys.modules)\n"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_charts_leave_global_state_alone(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    backend, font = matplotlib.get_backend(), dict(matplotlib.rcParams)
    from saa_pm.reporting import charts

    w = pd.DataFrame({"a": [0.6, 0.4], "b": [0.5, 0.5]}, index=["x", "y"])
    path = charts.allocation_by_class(w, {"x": "Class 1", "y": "Class 2"}, str(tmp_path / "c.png"))
    assert (tmp_path / "c.png").exists() and path.endswith("c.png")
    assert matplotlib.get_backend() == backend
    assert dict(matplotlib.rcParams) == font


def test_library_is_silent_by_default(data_dir, capsys):
    logging.getLogger("saa_pm").handlers = [
        h for h in logging.getLogger("saa_pm").handlers if isinstance(h, logging.NullHandler)
    ]
    saa_pm.SAAStudy(data_dir, None, verbose=False).run(steps=("allocation",), write=False)
    assert capsys.readouterr().out == ""


def test_from_frames_matches_folder(data_dir):
    def rd(name, **kw):
        return pd.read_csv(f"{data_dir}/{name}.csv", **kw)

    cfg = rd("config").set_index("key")["value"].to_dict()
    mem = MandateData.from_frames(
        assets=rd("assets"),
        prices=rd("prices", index_col=0, parse_dates=True),
        holdings=rd("holdings"),
        config=cfg,
        groups=rd("groups"),
        pacing=rd("pacing"),
        commitment_future=rd("commitment_future"),
    )
    disk = load_mandate(data_dir)
    assert mem.folder == "<in-memory>"
    assert mem.universe.names == disk.universe.names
    assert np.allclose(mem.returns.values, disk.returns.values, equal_nan=True)
    assert mem.holdings.total == disk.holdings.total
    # field by field, NaN == NaN: since Python 3.13 dataclass __eq__ compares fields
    # with ==, so unset (NaN) goal fields would make identical settings unequal
    for f in dataclasses.fields(mem.settings.goals):
        a, b = getattr(mem.settings.goals, f.name), getattr(disk.settings.goals, f.name)
        assert a == b or (np.isnan(a) and np.isnan(b)), f.name
    assert len(mem.schedule) == len(disk.schedule)


def test_version_single_source():
    from importlib.metadata import PackageNotFoundError, version

    try:
        assert version("saa-pm") == saa_pm.__version__
    except PackageNotFoundError:  # running from a source checkout
        assert saa_pm.__version__.count(".") == 2


def test_runs_without_report_extras(data_dir, tmp_path, monkeypatch):
    """Without matplotlib / openpyxl the study still writes CSVs and the summary."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, *args, **kwargs):
        importer = (globals or {}).get("__name__", "")
        if importer.startswith("saa_pm") and name.startswith(("matplotlib", "openpyxl")):
            raise ImportError(name)  # simulate the extras not being installed
        return real_import(name, globals, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import saa_pm.reporting

    monkeypatch.delitem(sys.modules, "saa_pm.reporting.charts", raising=False)
    monkeypatch.delattr(saa_pm.reporting, "charts", raising=False)  # set by an earlier import
    out = tmp_path / "res"
    res = saa_pm.SAAStudy(data_dir, str(out), verbose=False).run(steps=("allocation",))
    assert (out / "SUMMARY.md").exists()
    assert not (out / "saa_results.xlsx").exists()
    assert res.charts == []
