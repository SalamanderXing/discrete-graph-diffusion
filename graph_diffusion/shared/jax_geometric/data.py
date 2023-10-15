import os
import sys
import urllib.request
import ssl
from typing import Optional
import sys
import tarfile
import zipfile


def maybe_log(path, log=True):
    """Logs the extraction process."""
    if log and "pytest" not in sys.modules:
        print(f"Extracting {path}", file=sys.stderr)


def extract_tar(path: str, folder: str, mode: str = "r:gz", log: bool = True):
    """Extracts a tar archive to a specific folder."""
    maybe_log(path, log)
    with tarfile.open(path, mode) as f:
        f.extractall(folder)


def extract_zip(path: str, folder: str, log: bool = True):
    """Extracts a zip archive to a specific folder."""
    maybe_log(path, log)
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)


def download_url(
    url: str, folder: str, log: bool = True, filename: Optional[str] = None
):
    """Downloads the content of an URL to a specific folder."""

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = os.path.join(folder, filename)

    if os.path.exists(path):
        if log and "pytest" not in sys.modules:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log and "pytest" not in sys.modules:
        print(f"Downloading {url}", file=sys.stderr)

    os.makedirs(folder, exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path
