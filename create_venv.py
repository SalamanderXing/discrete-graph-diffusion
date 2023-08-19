import subprocess
import os


def run_command(command, cwd=None):
    """
    Executes a command and returns its output.
    """
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, shell=True, cwd=cwd
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return None


def is_python_311_installed():
    """
    Checks if Python 3.11 is installed.
    """
    return run_command("python3.11 --version") is not None


def install_python_311():
    """
    Installs Python 3.11 and related packages.
    """
    run_command("sudo apt update -y && sudo apt upgrade -y")
    run_command(
        "sudo apt install -y wget build-essential libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev"
    )
    run_command("sudo add-apt-repository ppa:deadsnakes/ppa")
    run_command("sudo apt install -y python3.11")
    run_command("wget https://bootstrap.pypa.io/get-pip.py")
    run_command("python3.11 get-pip.py")
    run_command("python3.11 -m pip install virtualenv")


def create_virtualenv():
    run_command("python3.11 -m virtualenv ~/venv")

    # Add activation line to .bashrc
    with open(os.path.expanduser("~/.bashrc"), "a") as bashrc:
        bashrc.write("\n# Activate Python 3.11 virtual environment\n")
        bashrc.write("source ~/venv/bin/activate\n")


def install_dependencies():
    forbidden_packages = ["jax"]
    forbidden_keywords = ["nvidia"]
    with open("requirements.txt", "r") as lines:
        requirements = lines.readlines()

    for i, line in enumerate(requirements):
        package_name = line.split("==")[0]
        if package_name in forbidden_packages:
            continue
        if any(keyword in line for keyword in forbidden_keywords):
            continue
        run_command(f"pip install {line}")
        print(f"Installed {i+1}/{len(requirements)} packages.")
    run_command(
        "pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
    )
    run_command("git clone https://github.com/salamanderXing/mate")
    run_command("pip install -e .", cwd="mate")


def activate_virtualenv():
    """
    Activates the virtual environment.
    """
    activation_script = os.path.join("~", "venv", "bin", "activate_this.py")
    with open(activation_script) as f:
        exec(f.read(), {"__file__": activation_script})


def main():
    if not is_python_311_installed():
        print("Python 3.11 is not installed.")
        install_python_311()
    if not os.path.exists("~/venv"):
        create_virtualenv()
    activate_virtualenv()
    install_dependencies()


if __name__ == "__main__":
    main()
