#!/usr/bin/env python3
"""Generate pressure/temperature EOS configurations for PWidom."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPOSITIONS = {
    "ice": "h2o",
    "h225": "h2.25o",
    "H": "h",
}

MIN_POINTS = 5


def load_inventory(filename):
    df = pd.read_csv(filename)
    required = {"good", "Path", "T_bin", "True_P_GPa", "V_per_atom",
                "comp_snap", "N_H", "N_O", "Other"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Binned inventory missing composition/provenance fields: {sorted(missing)}")
    # Never trust a snapped label alone in mixed-material inventories.
    nh = pd.to_numeric(df["N_H"], errors="coerce")
    no = pd.to_numeric(df["N_O"], errors="coerce")
    mask = (df["Other"].fillna("").astype(str).str.strip().eq("")
            & (nh > 0) & (no >= 0))
    if "N_ions" in df:
        mask &= np.isclose(nh + no, pd.to_numeric(df["N_ions"], errors="coerce"))
    ratio = nh / no.replace(0, np.nan)
    valid = ((df["comp_snap"].eq("h") & no.eq(0))
             | (df["comp_snap"].eq("h2o") & np.isclose(ratio, 2, rtol=.04))
             | (df["comp_snap"].eq("h2.25o") & np.isclose(ratio, 2.25, rtol=.04)))
    df = df[mask & valid].copy()

    df = df[
        df["good"].astype(str).str.lower().isin(["true", "1"])
    ].copy()

    df = df[
        ~df["Path"].astype(str).str.contains(
            "lambda_", case=False, na=False
        )
    ].copy()

    for col in ("T_bin", "True_P_GPa", "V_per_atom"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def select(df, comp, temperature):
    d = df[
        (df["comp_snap"] == comp)
        & np.isclose(df["T_bin"], temperature)
    ].copy()

    if comp != "h":
        paths = d["Path"].astype(str)
        mask = (
            paths.str.contains("fcc", case=False, na=False)
            & ~paths.str.contains(
                "liquid|solid|melt", case=False, na=False
            )
        )
        d = d[mask]

    d = d[
        np.isfinite(d["True_P_GPa"])
        & np.isfinite(d["V_per_atom"])
        & (d["True_P_GPa"] > 0)
        & (d["V_per_atom"] > 0)
    ]

    return d


def fit_eos(d, comp, temperature):
    if len(d) < MIN_POINTS:
        raise ValueError(
            f"{comp} at {temperature:g} K: "
            f"only {len(d)} usable DFT points"
        )

    v = d["V_per_atom"].to_numpy(dtype=float)
    p = d["True_P_GPa"].to_numpy(dtype=float)

    if len(np.unique(v)) < 3:
        raise ValueError("EOS needs at least three distinct volumes")
    x = np.log(v)
    y = np.log(p)

    coeff = np.polyfit(x, y, 2)
    degree = 2

    # Require pressure to decrease with increasing volume.
    xx = np.linspace(x.min(), x.max(), 200)

    if np.any(np.polyval(np.polyder(coeff), xx) >= 0):
        coeff = np.polyfit(x, y, 1)
        degree = 1

    if np.any(np.polyval(np.polyder(coeff), xx) >= 0):
        raise ValueError(
            f"{comp} {temperature:g} K: nonphysical EOS slope"
        )

    rmse = float(
        np.sqrt(np.mean((np.polyval(coeff, x) - y) ** 2))
    )

    return {
        "coeff": coeff,
        "deriv": np.polyder(coeff),
        "xmin": float(x.min()),
        "xmax": float(x.max()),
        "pmin": float(p.min()),
        "pmax": float(p.max()),
        "n": int(len(d)),
        "T": float(temperature),
        "degree": degree,
        "rmse": rmse,
    }


def volume_at_pressure(fit, pressure):
    pmin = fit["pmin"]
    pmax = fit["pmax"]

    if not np.isfinite(pressure) or pressure <= 0:
        raise ValueError("Pressure must be finite and positive")

    extrapolated = not (pmin <= pressure <= pmax)

    if extrapolated:
        print(
            f"    WARNING: PRESSURE EXTRAPOLATION "
            f"{pmin:.1f}-{pmax:.1f} -> {pressure:g} GPa"
        )

    # Solve the fitted polynomial directly: no arbitrary log-volume search cap.
    coeff = np.array(fit["coeff"], copy=True)
    coeff[-1] -= np.log(pressure)
    roots = [float(z.real) for z in np.roots(coeff)
             if abs(z.imag) < 1e-10
             and np.polyval(fit["deriv"], z.real) < 0]
    xmin, xmax = fit["xmin"], fit["xmax"]

    if not roots:
        raise ValueError(
            f"No stable EOS root at {pressure:g} GPa, "
            f"T={fit['T']:g} K"
        )

    centre = (xmin + xmax) / 2
    root = min(roots, key=lambda z: abs(z - centre))

    slope = float(np.polyval(fit["deriv"], root))
    volume = float(np.exp(root))
    bulk = float(-pressure * slope)

    if not np.isfinite(volume) or not np.isfinite(bulk):
        raise ValueError("Nonfinite EOS result")

    if volume <= 0 or bulk <= 0:
        raise ValueError("Nonphysical EOS result")

    return volume, bulk, extrapolated


def available_temperatures(df, comp):
    temps = sorted(
        float(t) for t in df["T_bin"].dropna().unique()
    )

    return [
        t for t in temps
        if len(select(df, comp, t)) >= MIN_POINTS
    ]


def state_at(df, comp, pressure, temperature):
    if not np.isfinite([pressure, temperature]).all() or min(pressure, temperature) <= 0:
        raise ValueError("P and T must be finite and positive")
    temps = available_temperatures(df, comp)

    if not temps:
        raise ValueError(f"No usable DFT data for {comp}")

    exact = [
        t for t in temps if np.isclose(t, temperature)
    ]

    if exact:
        selected = [exact[0]]
    else:
        below = [t for t in temps if t < temperature]
        above = [t for t in temps if t > temperature]

        if below and above:
            selected = [max(below), min(above)]
        elif len(temps) >= 2:
            selected = (
                temps[:2] if temperature < temps[0]
                else temps[-2:]
            )

        else:
            raise ValueError(
                f"{comp}: cannot estimate {temperature:g} K "
                f"from available temperatures {temps}"
            )

    fitted = []

    for t in selected:
        fit = fit_eos(select(df, comp, t), comp, t)
        volume, bulk, extrap = volume_at_pressure(
            fit, pressure
        )
        fitted.append((t, volume, bulk, extrap, fit))

    if len(fitted) == 1:
        t, volume, bulk, extrap, fit = fitted[0]
        source = f"DFT EOS {comp} {t:g} K"
        method = "direct"
    else:
        t0, v0, b0, e0, f0 = fitted[0]
        t1, v1, b1, e1, f1 = fitted[1]

        fraction = (temperature - t0) / (t1 - t0)

        volume = v0 + fraction * (v1 - v0)

        # Interpolate dV/dP, then calculate B = -V/(dV/dP).
        dvdp = (
            (1 - fraction) * (-v0 / b0)
            + fraction * (-v1 / b1)
        )

        bulk = -volume / dvdp

        method = (
            "temperature interpolation"
            if 0 <= fraction <= 1
            else "temperature extrapolation"
        )

        source = (
            f"DFT EOS {comp} {method} "
            f"from {t0:g}/{t1:g} K"
        )

    if not np.isfinite([volume, bulk]).all():
        raise ValueError("Nonfinite interpolated EOS")

    if volume <= 0 or bulk <= 0:
        raise ValueError("Nonphysical interpolated EOS")

    print(
        f"  {comp:7s}: V/atom={volume:.7f}, "
        f"B={bulk:.2f} GPa"
    )

    for t, v, b, extrap, fit in fitted:
        print(
            f"    T={t:g} K, n={fit['n']}, "
            f"P range={fit['pmin']:.1f}-"
            f"{fit['pmax']:.1f} GPa, "
            f"log-RMSE={fit['rmse']:.4f}, "
            f"degree={fit['degree']}"
            + (" [PRESSURE EXTRAPOLATED]" if extrap else "")
        )

    if method != "direct":
        print(f"    WARNING: {method}")

    return {
        "vpa": float(volume),
        "B_GPa": float(bulk),
        "source": source,
        "pressure_extrapolated": bool(
            any(row[3] for row in fitted)
        ),
        "temperature_method": method,
        "temperature_extrapolated": method == "temperature extrapolation",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument(
        "--P", nargs="+", type=float, default=[400, 500]
    )
    ap.add_argument(
        "--T", nargs="+", type=float, default=[2000]
    )
    ap.add_argument(
        "--inventory",
        default="runs/inventory_binned.csv",
    )
    ap.add_argument("--outdir", default="envs")
    ap.add_argument("--force", action="store_true")

    args = ap.parse_args()

    df = load_inventory(args.inventory)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for pressure in args.P:
        for temperature in args.T:
            if not np.isfinite([pressure, temperature]).all() or pressure <= 0 or temperature <= 0:
                raise ValueError("P and T must be positive")

            filename = (
                f"pwidom_{pressure:g}GPa_"
                f"{temperature:g}K.json"
            )
            output = outdir / filename

            if output.exists() and not args.force:
                print(f"SKIP existing {output}")
                continue

            print(
                f"\n=== {pressure:g} GPa / "
                f"{temperature:g} K ===",
                flush=True,
            )

            state = {
                "P": pressure,
                "T": temperature,
                "inventory": str(Path(args.inventory).resolve()),
            }

            for key, comp in COMPOSITIONS.items():
                state[key] = state_at(
                    df, comp, pressure, temperature
                )

            state["ice"]["nrep"] = 3
            state["h225"]["nrep"] = 3
            state["H"]["natoms"] = 250

            temporary = output.with_suffix(".json.tmp")

            with temporary.open("w") as handle:
                json.dump(state, handle, indent=2)
                handle.write("\n")

            temporary.replace(output)
            print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
