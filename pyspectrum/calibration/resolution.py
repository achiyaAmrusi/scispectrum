class ResolutionCalibration:
    """
    Detector resolution calibration.

    Parameters
    ----------
    func : Callable[[float], float]
        Function returning resolution for a given axis value.
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, axis_value: float) -> float:
        """Return the resolution at a given axis value."""
        return self.func(axis_value)

    def apply(self, axis_value):
        return self.func(axis_value)