#!/bin/bash
# What has been submitted, what is running, what finished: run on the Young LOGIN node from the repo root and paste the output to the chat.   bash mlp_pipeline/jobs_status.sh [days, default 3]
days=${1:-3}
echo "== last submissions (runs/JOBS_LOG.tsv: date, job id, script, sbatch options, run knobs, git, note)"; [ -f runs/JOBS_LOG.tsv ] && tail -n 25 runs/JOBS_LOG.tsv || echo "(no runs/JOBS_LOG.tsv yet: jobs submitted without submit.sh are not recorded there)"
echo; echo "== queue by job name and state (array tasks counted)"; squeue -u $USER -r -h -o "%j %T" | sort | uniq -c
echo; echo "== running now (job id, name, elapsed, node, time limit)"; squeue -u $USER -t R -h -o "%i %j %M %N %l" | sort -k2
echo; echo "== finished in the last $days days by job name and state; GPU-hours = elapsed time of finished tasks (one GPU each)"
sacct -u $USER -S $(date -d "-$days days" +%F) -X -n -P -o JobName,State,ElapsedRaw 2>/dev/null | awk -F'|' '{split($2,s," "); k=$1" "s[1]; n[k]++; h[k]+=$3/3600} END{for(k in n) printf "%-26s %4d tasks %8.1f GPU-h\n", k, n[k], h[k]}' | sort
