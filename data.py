import numpy as np
from dataclasses import dataclass

@dataclass
class DataStorage:
    fs: np.ndarray = None
    Efield: np.ndarray = None
    Efield_list: np.ndarray = None
    time: np.ndarray = None
