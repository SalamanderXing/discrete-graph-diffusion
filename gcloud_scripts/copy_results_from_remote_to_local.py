import subprocess
import os

zone = "us-central1-a"
remote_name = "tpu-giulio-dev"
remote_path = "/home/bluesk/Documents/discrete-graph-diffusion/results/hyperparameters/"
# local_path = os.path.expanduser(
#     "~/Documents/discrete-graph-diffusion/results/experiments/ddgd_fo/"
# )
local_path = os.path.expanduser("~/Documents/discrete-graph-diffusion/remote_results/")
if os.path.exists(local_path):
    os.system(f"rm -rf {local_path}")
cmd = f'gcloud alpha compute tpus tpu-vm scp --zone={zone} "{remote_name}:{remote_path}" "{local_path}" --recurse'
print(cmd)
subprocess.run(cmd, shell=True)
