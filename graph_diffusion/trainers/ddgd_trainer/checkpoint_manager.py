import pickle
import os


class CheckpointManager:
    def __init__(self, directory: str, **args):
        self.directory = directory
        self.__step = 0

    def latest_step(self) -> int:
        """Returns the latest step saved in the checkpoint folder"""
        return self.__step

    def save(self, step: int, model: object):
        """Saves the model in the checkpoint folder"""
        # self.__step = step
        # with open(os.path.join(self.directory, "model.pkl"), "wb") as file:
        #     pickle.dump({"step": step, "model": model}, file)
        pass

    def restore(self, step: int) -> object:
        """Restores the model from the checkpoint folder"""
        with open(os.path.join(self.directory, "model.pkl"), "rb") as file:
            data = pickle.load(file)
            self.__step = data["step"]
            return data["model"]
