import os
import ipdb
import subprocess


def main():
    # name = "tpu-giulio-dev3"
    name = "tpu-giulio-dev2"
    #zone = "europe-west4-a"
    zone = "us-central1-a"
    # zone = "us-central1-b"
    # zone = "us-central1-f"
    success = False
    command = f"gcloud compute tpus tpu-vm create {name} --accelerator-type v3-8 --version tpu-ubuntu2204-base --zone {zone}"
    print(f"{command}")
    while not success:
        output = ""
        try:
            result = subprocess.run(
                command.split(),
                check=True,
                capture_output=True,
                text=True,
            )
            output = result.stdout
            print(output)
            success = True
        except subprocess.CalledProcessError as e:
            print(f"Exception")
            output = e.stderr
            print(output)
            success = False
        print(f"{output}")


if __name__ == "__main__":
    main()
