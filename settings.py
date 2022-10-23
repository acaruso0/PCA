import argparse


class Settings(object):
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Settings, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._filename = None
        self._nats = False
        self.ParseArguments()

    def ParseArguments(self):
        parser = argparse.ArgumentParser(
            "Compute first and second PCA components for a given dataset.")
        parser.add_argument("input", type=str,
                            help="input file containing the dataset to analyze")
        parser.add_argument("-n", "--nats",
                            action="store_true",
                            help="use natural logarithms")

        args = parser.parse_args()
        self._filename = args.input
        self._nats = args.nats

        return None

    @property
    def filename(self):
        return self._filename

