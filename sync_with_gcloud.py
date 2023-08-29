from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os
import subprocess
import threading
import queue


class Watcher:
    def __init__(
        self, directory_to_watch: str, machine_name: str, remote_path: str, zone: str
    ):
        self.local_directory_to_watch = directory_to_watch
        self.machine_name = machine_name
        self.remote_path = remote_path
        self.observer = Observer()
        self.batch_queue = queue.Queue()
        self.batch_processing_thread = threading.Thread(target=self.batch_process)
        self.batch_processing_thread.start()
        self.zone = zone

    def batch_process(self):
        pending_files = []
        while True:
            try:
                file_to_copy = self.batch_queue.get(timeout=10)  # 10-second timeout
                pending_files.append(file_to_copy)
            except queue.Empty:
                if pending_files:
                    self.process_batch(pending_files)
                pending_files = []

    def process_batch(self, pending_files):
        print(f"Batch processing {len(pending_files)} files.")

        tar_paths = set()

        for event in pending_files:
            local_path = event.src_path

            if ".git" in local_path or not os.path.exists(local_path):
                continue

            relative_path = os.path.relpath(local_path, self.local_directory_to_watch)
            tar_paths.add(relative_path)

        tar_paths = list(tar_paths)
        # Create and transfer the tar file over SSH
        if tar_paths:
            # Change to the directory you're watching
            os.chdir(self.local_directory_to_watch)

            # Create tar and pipe it over SSH
            tar_command = ["tar", "-cz"] + tar_paths
            ssh_command = [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                f"--zone={self.zone}",
                self.machine_name,
                "--command",
                f"tar -xz -C {self.remote_path}",
            ]

            tar_process = subprocess.Popen(tar_command, stdout=subprocess.PIPE)
            subprocess.run(ssh_command, stdin=tar_process.stdout)
            tar_process.wait()

    def run(self):
        event_handler = Handler(
            self.machine_name,
            self.remote_path,
            self.local_directory_to_watch,
            self.batch_queue,
        )
        self.observer.schedule(
            event_handler, self.local_directory_to_watch, recursive=True
        )
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer stopped")
        self.observer.join()


class Handler(FileSystemEventHandler):
    def __init__(
        self,
        machine_name: str,
        remote_path: str,
        directory_to_watch: str,
        batch_queue: queue.Queue,
    ):
        self.machine_name = machine_name
        self.remote_path = remote_path
        self.directory_to_watch = directory_to_watch
        self.batch_queue = batch_queue

    def process(self, event):
        self.batch_queue.put(event)

    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)


if __name__ == "__main__":
    directory_to_watch = (
        "/home/bluesk/Documents/discrete-graph-diffusion/graph_diffusion"
    )
    machine_name = "tpu-giulio-dev"
    remote_path = "/home/bluesk/discrete-graph-diffusion/graph_diffusion"
    zone = "us-central1-a"
    w = Watcher(directory_to_watch, machine_name, remote_path, zone)
    w.run()
