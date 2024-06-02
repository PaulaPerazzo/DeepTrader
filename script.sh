#!/bin/bash
#SBATCH --job-name=test_job_rl
#SBATCH --ntasks=1
#SBATCH --mem 32G
#SBATCH -c 8
#SBATCH -o job.log
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt

source env/bin/activate

python3 src/run.py -c src/hyper.json
