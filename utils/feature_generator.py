from .normalizer import Normalizer

class FeatureGenerator:
    def __init__(self):
        self.normalizer = Normalizer()

    def generate(self, feature_name_args, df):
        """generates a feature specified in the config.
        features are encoded in the naming of the column
        """
        common = {"append": True}
        # copy OHLCV and normalize them
        if feature_name_args == "O":
            df["O"] = df["Open"].copy()
            self.normalizer.normalize(df, "O", ["fd", "rfi", "mmn"])
        elif feature_name_args == "H":
            df["H"] = df["High"].copy()
            self.normalizer.normalize(df, "H", ["fd", "rfi", "mmn"])
        elif feature_name_args == "L":
            df["L"] = df["Low"].copy()
            self.normalizer.normalize(df, "L", ["fd", "rfi", "mmn"])
        elif feature_name_args == "C":
            df["C"] = df["Close"].copy()
            self.normalizer.normalize(df, "C", ["fd", "rfi", "mmn"])
        elif feature_name_args == "V":
            df["V"] = df["Volume"].copy()
            # make it numerically stable
            df["V"] += 100_000
            self.normalizer.normalize(df, "V", ["fd", "rfi", "mmn"])
        elif feature_name_args.startswith("SMA"):
            length = int(feature_name_args.split("_")[-1])
            df.ta.sma(length=length, **common)
            self.normalizer.normalize(df, feature_name_args, ["fd", "mmn"])
        elif feature_name_args.startwith("RSI"):
            length = int(feature_name_args.split("_")[-1])
            df.ta.rsi(length=length, **common)
            # rsi is between 0 and 1 anyways, no need for stationarity
            self.normalizer.normalize(df, "RSI14", ["mmn"], norm=(0, 100))
        else:
            raise ValueError(f"Unknown feature: {feature_name_args}")

