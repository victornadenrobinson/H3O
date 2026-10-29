"""List the foundation models each installed package can load (run with that env's python; no GPU needed).

usage:  ~/work/envs/mace-ng/bin/python mlp_pipeline/list_models.py mace
        ~/work/envs/mattersim-env/bin/python mlp_pipeline/list_models.py mattersim
        ~/grace-venv/bin/python mlp_pipeline/list_models.py grace
"""
import glob, importlib, os, re, shutil, subprocess, sys


def mace():
    fm = importlib.import_module("mace.calculators.foundations_models")
    print("mace file:", fm.__file__)
    for k, v in vars(fm).items():
        if isinstance(v, dict) and v and all(isinstance(x, str) for x in v.values()) and any("http" in x for x in v.values()):
            print(f"\n{k} ({len(v)} entries):")
            for n, u in v.items():
                print(f"  {n:34s} {u.rsplit('/', 2)[-2]}/{u.rsplit('/', 1)[-1]}")
    from mace.calculators import mace_mp
    doc = (mace_mp.__doc__ or "")
    m = re.search(r"model.*?:(.*?)(?:\n\s*\n|\Z)", doc, re.S)
    print("\nmace_mp docstring, model argument:", (m.group(1)[:900] if m else doc[:900]))
    print("\nTo see the heads of a multihead model, load it and print calc.models[0].heads; select with name@head (e.g. mace:mace-mh-1@<head>).")
    print("cached model files:", [os.path.basename(p) for p in glob.glob(os.path.expanduser("~/.cache/mace/*"))])


def mattersim():
    import mattersim
    root = os.path.dirname(mattersim.__file__)
    names = set()
    for p in glob.glob(root + "/**/*.py", recursive=True):
        try:
            names |= set(re.findall(r"[Mm]atter[Ss]im-v[\w.\-]+\.pth", open(p, errors="ignore").read()))
        except OSError:
            pass
    print("checkpoint names referenced in the package:", sorted(names))
    print("downloaded:", sorted(os.path.basename(p) for p in glob.glob(os.path.expanduser("~/.local/mattersim/pretrained_models/*"))))
    print("package version:", getattr(mattersim, "__version__", "?"))


def grace():
    exe = shutil.which("grace_models") or os.path.join(os.path.dirname(sys.executable), "grace_models")
    r = subprocess.run([exe, "list"], capture_output=True, text=True)
    print((r.stdout or r.stderr)[-6000:])


if __name__ == "__main__":
    {"mace": mace, "mattersim": mattersim, "grace": grace}[sys.argv[1]]()
