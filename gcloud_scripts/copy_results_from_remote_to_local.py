import subprocess

zone = "us-central1-a"
remote_name = "tpu-giulio-dev"
remote_path = "~/discrete-graph-diffusion/results/experiments/ddgd_fo/"
local_path = "~/Documents/discrete-graph-diffusion/results/experiments/ddgd_fo/"
cmd = f"gcloud compute scp --zone={zone} --recurse {remote_name}:{remote_path} {local_path}"
print(cmd)
subprocess.run(cmd.split(), shell=True)
