# Environments (one per MLP; they conflict, so keep them separate)

Versions below are starting points from memory, not verified: check each package's current install docs.

```bash
# MACE  (zero-shot + fine-tune)
python -m venv ~/venvs/mace && source ~/venvs/mace/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121        # match the cluster's CUDA
pip install mace-torch ase pandas matplotlib tqdm

# GRACE  (tensorpotential; TensorFlow-based, own env)
python -m venv ~/venvs/grace && source ~/venvs/grace/bin/activate
pip install tensorpotential ase pandas tqdm                                   # see github.com/ICAMS/grace-tensorpotential

# MatterSim
python -m venv ~/venvs/mattersim && source ~/venvs/mattersim/bin/activate
pip install mattersim ase pandas tqdm

# CPU-only analysis (scripts 00-03, 06-11)
pip install numpy pandas ase matplotlib tqdm
```
Scripts 04 (benchmark) and 05 (fine-tune) are the only ones needing a model env; 04's GRACE/MatterSim calls are unverified.
Foundation weights are downloaded on first use, which needs internet on the node: pre-download on a login node if compute nodes are offline.
