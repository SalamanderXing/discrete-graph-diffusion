import os
import ipdb
import subprocess


def main():
    # name = "tpu-giulio-dev3"
    name = "tpu-giulio-dev2"
    # zone = "europe-west4-a"
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
        except subprocess.CalledProcessError as e:
            print(f"Exception")
            output = e.stderr
            print(output)
            success = False
        print(f"{output}")

    # Check the outcome of the command
    if not "ERROR" in output:
        # Replace 'desired_output' with the specific string you're checking for
        print("Command succeeded and found the desired output!")
        # Add any additional actions you want to perform here
    else:
        print("Command did not produce the desired output or failed.")
        # Add any additional actions you want to perform here


if __name__ == "__main__":
    main()
