class Normalizer:
    def __init__(self):
        # saves the min and max from the training dataset to apply the same
        # normalization on the test dataset
        self.norms = {}

    def normalize(self, df, name, operations, norm=None):
        for op in operations:
            # apply fractional diff to the last column of the df
            if op == "fd":
                df.iloc[1:, -1] = self.fractional_diff(df.iloc[:, -1])
            # fractional diff works only on len - 1 elements, the first element remains the
            # same so set it to 0 before normalizing else it explodes
            elif op == "rfi":
                df.iloc[0, -1] = 0
            # normalize the last column with max(abs(min), max)
            elif op == "mmn":
                # get the normalizers, if we are in training None will be returned
                # in test we will use the train min, max for normalization
                if norm is None:
                    norm = self.norms.get(name, None)

                df.iloc[:, -1], norm = self.min_max_norm(df.iloc[:, -1], norm)
                # store the min and max
                self.norms[name] = norm



    def fractional_diff(self, col):
        """Calculates the fractional difference of a column to make
        the data stationary.
        """
        col_begin = col.iloc[:-1]
        col_end = col.iloc[1:]

        col_new = (col_begin.to_numpy() / col_end.to_numpy()) - 1.0

        return col_new

    def min_max_norm(self, col, norm=None):
        """Normalizes the data based on the max(abs(min), max))"""
        if norm is None:
            min_ = abs(col.min())
            max_ = abs(col.max())
        else:
            min_, max_ = norm

        normalizer = max(min_, max_)
        normalized = col.to_numpy() / normalizer

        return normalized, (min_, max_)
