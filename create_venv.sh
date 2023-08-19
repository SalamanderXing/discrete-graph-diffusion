if ! python3.11 --version &>/dev/null; then
  echo "Python 3.11 is not installed."
  sudo apt update -y && sudo apt upgrade -y
  sudo apt install -y wget build-essential libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt install -y python3.11
  wget https://bootstrap.pypa.io/get-pip.py
  python3.11 get-pip.py
  python3.11 -m pip install virtualenv
  python3.11 -m virtualenv venv
fi
source venv/bin/activate
