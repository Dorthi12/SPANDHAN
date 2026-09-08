"""
Asynchronous Worker module for Spandhan DSP calculations.
Prevents GUI freezing during long computations (FFT, MUSIC, ESPRIT, ML inference).
"""

import sys
import traceback
from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    """
    Defines signals available from a running worker thread.
    """
    started = Signal()
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int, str)


class Worker(QRunnable):
    """
    Worker thread for running background tasks.
    """
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        """
        Execute the function with passed args and kwargs.
        """
        try:
            self.signals.started.emit()
            res = self.fn(*self.args, **self.kwargs)
        except Exception:
            exctype, value = sys.exc_info()[:2]
            tb = traceback.format_exc()
            self.signals.error.emit((exctype, value, tb))
        else:
            self.signals.result.emit(res)
        finally:
            self.signals.finished.emit()
