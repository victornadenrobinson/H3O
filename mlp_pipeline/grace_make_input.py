"""Fill GRACE's packaged input_template.yaml for a frozen-weights fine-tune, bypassing the interactive wizard.
Settings follow the gracemaker fine-tuning docs (finetune_foundation_model, reduce_elements, shift: auto, Adam lr 1e-3 + EMA + cosine, 2L frozen set).
GUESSES (check against `gracemaker -cm` and the first log): cutoff, loss weights/types, how placeholders expand.
usage: ~/grace-venv/bin/python mlp_pipeline/grace_make_input.py runs/grace/input.yaml --train train_s.pkl.gz --test valid_s.pkl.gz
"""
import argparse, glob, os, re, site, sys

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--model", default="GRACE-2L-SMAX-OMAT-medium")
ap.add_argument("--cutoff", default="7.5"); ap.add_argument("--updates", type=int, default=400); ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--wE", default="1"); ap.add_argument("--wF", default="5"); ap.add_argument("--wS", default="10")
ap.add_argument("--lr", default="0.001"); ap.add_argument("--naive", action="store_true", help="train all weights instead of the frozen set")
ap.add_argument("--template", default=None)
a = ap.parse_args()
tpl = a.template or next(iter(glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/tensorpotential/resources/input_template.yaml"))), None)
if not tpl:
    sys.exit("input_template.yaml not found: pass --template")
t = open(tpl).read()
frozen = "" if a.naive else 'trainable_variable_names: ["I2/reducing_", "rho/reducing_", "I1/reducing_"]'
lr = float(a.lr)
sub = {
    "CUTOFF": a.cutoff, "TRAIN_FILENAME": a.train, "TEST_DATA": f"test_filename: {a.test}", "SAVE_DATASET": "False",
    "PARAM_DTYPE": "float32",
    "POTENTIAL_SETTINGS": f"finetune_foundation_model: {a.model}\n  reduce_elements: True\n  shift: auto",
    "ENERGY_LOSS_WEIGHT": a.wE, "FORCE_LOSS_WEIGHT": a.wF, "LOSS_TYPE": "huber", "EXTRA_E_ARGS": ", delta: 0.01",
    "STRESS_LOSS": f"stress: {{ weight: {a.wS}, type: huber, delta: 0.01 }},", "SWITCH_LOSS": "", "WEIGHTING_SCHEME": "",
    "TARGET_TOTAL_UPDATES": str(a.updates),
    "OPTIMIZER_BLOCK": (f"  optimizer: Adam\n  opt_params: {{learning_rate: {lr}, use_ema: True, ema_momentum: 0.99, weight_decay: null, clipnorm: 1.0}}\n"
                        f"  scheduler: cosine_decay\n  scheduler_params: {{minimum_learning_rate: {lr / 12:.3g}}}"),
    "COMPUTE_CONVEX_HULL": "False", "BATCH_SIZE": str(a.batch), "TEST_BATCH_SIZE": str(a.batch), "eval_init_stats": "True",
    "RESET_EPOCH": "False", "TRAINABLE_VARIABLE_NAMES": frozen,
}
miss = set(re.findall(r"\{\{(\w+)\}\}", t)) - set(sub)
if miss:
    print("WARNING: placeholders not filled:", sorted(miss))
for k, v in sub.items():
    t = t.replace("{{" + k + "}}", v)
open(a.out, "w").write(t)
print(f"wrote {a.out} from {tpl}")
