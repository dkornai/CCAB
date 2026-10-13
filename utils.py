import numpy as np

def niceprint(x : np.ndarray, decimals : int) -> str:
    """
    Format a numpy array to a string with the given number of decimal places for each element
    """
    return np.array2string(np.asarray(x), formatter={'float_kind':lambda x: f'{x: .{decimals}f}'})
