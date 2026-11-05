#!/bin/bash
#SBATCH -A m3349_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH --gpus 1
#SBATCH -J lagtest
#SBATCH -o lagtest_%j.out
# Perlmutter GPU-node health test: shell startup, filesystems, small-file speed, GPU, TensorFlow import. Everything goes to the output file, nothing long is typed or pasted.
#   sbatch mlp_pipeline/perlmutter_lagtest.sh        then:  head -60 lagtest_<jobid>.out
echo "== $(hostname) $(date)"; uptime; echo "-- top CPU"; ps -eo pcpu,comm --sort=-pcpu | head -5
echo "-- shell startup"; ( time bash --noprofile --norc -c exit ) 2>&1 | grep real; ( time bash -lic exit ) 2>&1 | grep real
echo "-- ls home / scratch"; ( time ls ~ >/dev/null ) 2>&1 | grep real; ( time ls $SCRATCH >/dev/null ) 2>&1 | grep real
echo "-- write 200 MB to scratch and /tmp"
( time dd if=/dev/zero of=$SCRATCH/lag_dd bs=1M count=200 conv=fsync ) 2>&1 | grep -E "copied|real"; rm -f $SCRATCH/lag_dd
( time dd if=/dev/zero of=/tmp/lag_dd bs=1M count=200 conv=fsync ) 2>&1 | grep -E "copied|real"; rm -f /tmp/lag_dd
echo "-- many small files (create/read 2000)"; d=$SCRATCH/lag_small; mkdir -p $d
( time (for i in $(seq 2000); do echo x > $d/$i; done; cat $d/* >/dev/null) ) 2>&1 | grep real; rm -rf $d
echo "-- GPU"; nvidia-smi -L; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
V=${VENV:-$SCRATCH/h3o/grace-venv}
if [ -f $V/bin/activate ]; then source $V/bin/activate; which python
  ( time python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" ) 2>&1 | grep -E "GPU|real"
else echo "no venv at $V (set VENV=)"; fi
echo "== done $(date)"
