# Sourced by the Slurm array scripts (young_tdi_*.sbatch) right after the cd. sbatch exports the submitting shell's environment to every array task, so a run knob left
# over from an interactive run (K=..., NREP=2, HPER=2.25, INIT=..., VPA=...) would silently change the state point or the tag of EVERY task in the table.
# Knobs that define the system or its files are dropped here (the tables set what they need inline); deliberate overrides that the scripts document (DT where a table reads ${DT:-...}, PS, EQUIL, HOT,
# NODES/RULE/NGAUSS/ONLYNODES for a Kronrod extension, MODEL, PLIST/TLIST) pass through and are echoed so the log shows them. FRESH is refused by young_tdi_par.sh in arrays.
for _v in TAG K VPA BULK NREP NATOMS HPER NGHOST INIT PREPHOLD PREPFREE ONLY0 MINRUNS POWER CHUNK PS1; do
  case " $ARRAY_KEEP " in *" $_v "*) [ -n "${!_v}" ] && echo "NOTE: $_v=${!_v} taken from the submitting environment"; continue;; esac   # ARRAY_KEEP: knobs this script documents as overrides
  if [ -n "${!_v}" ]; then echo "NOTE: $_v=${!_v} from the submitting environment is ignored by this array script (set it in its table)"; unset $_v; fi
done
for _v in MODEL SEED DT PS EQUIL HOT HOTT NODES NPAR RULE NGAUSS ONLYNODES PLIST TLIST; do
  [ -n "${!_v}" ] && echo "NOTE: $_v=${!_v} taken from the submitting environment"
done
unset _v
