"""80/20 split and force/stress-weighted fine-tuning launcher (MACE)."""
import random
import shlex

from ase.io import read, write


def split(extxyz, train_out, valid_out, frac=0.8, seed=0):
    frames = read(extxyz, index=":")
    random.Random(seed).shuffle(frames)
    k = int(len(frames) * frac)
    write(train_out, frames[:k], format="extxyz")
    write(valid_out, frames[k:], format="extxyz")
    return k, len(frames) - k


def mace_command(train, valid, name="h3o_ft", foundation="medium",
                 energy_weight=1.0, forces_weight=100.0, stress_weight=10.0):
    """Return a `mace_run_train` command line (run on a Perlmutter A100 node)."""
    args = [
        "mace_run_train", f"--name={name}", f"--foundation_model={foundation}",
        f"--train_file={train}", f"--valid_file={valid}",
        "--energy_key=energy", "--forces_key=forces", "--stress_key=stress",
        "--loss=universal", f"--energy_weight={energy_weight}",
        f"--forces_weight={forces_weight}", f"--stress_weight={stress_weight}",
        "--lr=0.0001", "--max_num_epochs=100", "--device=cuda",
        "--default_dtype=float64", "--ema", "--amsgrad",
    ]
    return " ".join(shlex.quote(a) for a in args)
