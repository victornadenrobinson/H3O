#!/bin/bash
# Same as smoke_cpu.sh but on the GPU of the current node (run inside a GPU srun/sbatch session).
DEVICE=cuda exec bash "$(dirname "$0")/smoke_cpu.sh" "$@"
