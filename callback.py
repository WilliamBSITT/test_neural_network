import os
from abc import ABC, abstractmethod


class Callback(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch: int, loss: float): ...


class CSVLogger(Callback):
    def __init__(self, file_path: str, overwrite: bool = False):
        self._file_path = file_path
        if os.path.isfile(file_path) and not overwrite:
            raise FileExistsError(f"Log file already exists at {file_path}")
        else:
            with open(self._file_path, "w") as f:
                f.write("Epoch,Loss\n")

    def on_epoch_end(self, epoch: int, loss: float):
        with open(self._file_path, "a") as f:
            f.write(f"{epoch},{loss}\n")


class EarlyStopping(Callback):
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, loss: float):
        if self.best is None or loss < self.best - self.min_delta:
            self.best = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True


class ModelCheckpoint(Callback):
    def __init__(self, file_path: str, save_best_only: bool = True):
        self.file_path = file_path
        self.save_best_only = save_best_only
        self.best = None
        self.model = None

    def on_epoch_end(self, epoch: int, loss: float):
        import json

        if self.model is None:
            return

        save = True
        if self.save_best_only:
            if self.best is None or loss < self.best:
                self.best = loss
                save = True
            else:
                save = False

        if save:
            # save model (model.save handles backend arrays)
            self.model.save(self.file_path)
            # write metadata with last saved epoch
            meta = {"last_epoch": epoch}
            with open(self.file_path + '.meta', 'w') as mf:
                json.dump(meta, mf)
