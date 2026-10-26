"""G_fcc - G_bcc from two tdi_*.json files:  python mlp_pipeline/tdi/compare.py runs/tdi/tdi_fcc_*.json runs/tdi/tdi_bcc_*.json"""
import json, sys
f, b = (json.load(open(x)) for x in sys.argv[1:3])
print(f"fcc {f['tag']}: G {f['G_meV_atom']:.2f} +- {f['G_err_meV_atom']:.2f} meV/atom  (MLP P {f['P_mlp']:.1f} GPa, O {f['O_struct']})")
print(f"bcc {b['tag']}: G {b['G_meV_atom']:.2f} +- {b['G_err_meV_atom']:.2f} meV/atom  (MLP P {b['P_mlp']:.1f} GPa, O {b['O_struct']})")
for hf, hb in zip(f["hist"], b["hist"]):
    print(f"  t {hf['t_ps']:5.2f} ps   G_fcc - G_bcc = {hf['G_meV_atom'] - hb['G_meV_atom']:+8.2f} meV/atom")
d = f["G_meV_atom"] - b["G_meV_atom"]; e = (f["G_err_meV_atom"] ** 2 + b["G_err_meV_atom"] ** 2) ** .5
print(f"G_fcc - G_bcc = {d:+.2f} +- {e:.2f} meV/atom   (literature, 3000 K: -22.1 at 400 GPa, -26.5 at 500 GPa; Cheng 2021 / Wilson 2013)")
