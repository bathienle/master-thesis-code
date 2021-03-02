"""
Utility functions
"""


def convert_time(milliseconds):
    """Convert milliseconds to minutes and seconds.
    Parameters
    ----------
    milliseconds : int
        The time expressed in milliseconds.
    Return
    ------
    minutes : int
        The minutes.
    seconds : int
        The seconds.
    """

    minutes = milliseconds // 60
    seconds = milliseconds % 60

    return minutes, seconds
