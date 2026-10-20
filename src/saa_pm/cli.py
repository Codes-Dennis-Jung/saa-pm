"""Command line: ``saa-pm run``, ``saa-pm check``, ``saa-pm sample-data``.

saa-pm check  examples/data                 validate an input folder
saa-pm run    examples/data -o results      full study
saa-pm run    my_data --steps allocation projection
saa-pm sample-data my_data                  write a synthetic example folder
"""

from __future__ import annotations

import argparse
import sys

from .study import ALL_STEPS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="saa-pm", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="validate an input folder")
    c.add_argument("data")

    r = sub.add_parser("run", help="run the study and write results")
    r.add_argument("data")
    r.add_argument("-o", "--out", default="results")
    r.add_argument("--steps", nargs="+", default=list(ALL_STEPS), choices=list(ALL_STEPS))
    r.add_argument("--paths", type=int, default=10_000, help="Monte Carlo paths")
    r.add_argument("-q", "--quiet", action="store_true")

    s = sub.add_parser("sample-data", help="write a synthetic example input folder")
    s.add_argument("out")
    s.add_argument("--seed", type=int, default=42)

    a = ap.parse_args(argv)
    if a.cmd == "check":
        from .io.loaders import load_mandate
        from .io.validation import validate

        rep = validate(load_mandate(a.data))
        print(rep.coverage.to_string(float_format=lambda x: f"{x:.4f}"))
        print(rep.text())
        return 0 if rep.ok else 1
    if a.cmd == "run":
        from .study import SAAStudy

        SAAStudy(a.data, a.out, verbose=not a.quiet).run(a.steps, mc_paths=a.paths)
        return 0
    if a.cmd == "sample-data":
        from .sample_data import write_sample_data

        write_sample_data(a.out, seed=a.seed)
        print(f"sample input folder written to {a.out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
