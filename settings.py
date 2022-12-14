import argparse
import numpy as np


class Settings(object):
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Settings, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._filename = None
        self.ParseArguments()

    def ParseArguments(self):
        parser = argparse.ArgumentParser(
            "Compute first and second PCA components for a given dataset.")
        parser.add_argument("input", type=str,
                            help="input file containing the dataset to analyze")

        args = parser.parse_args()
        self._filename = args.input

        return None

    @property
    def filename(self) -> str:
        """
        Filename property.
  
        Returns
        -------
        self._filename : str
            Filename.
        """
        return self._filename

