import numpy as np
import cython


def cython_sim_ident(unicode char1, unicode char2):
    return 1 if char1 == char2 else 0


def int_max_two(int a, int b):
    """Finds the maximum integer of the given two integers.
        Args:
            integer1, integer2 (int): Input integers.
        Returns:
            Maximum integer (int).
    """
    if a > b : return a
    else: return b


def int_max_three(int a, int b, int c):
    """Finds the maximum integer of the given three integers.
        Args:
            integer1, integer2, integer3 (int): Input integers.
        Returns:
            Maximum integer (int).
    """
    cdef int max_int = a
    if b > max_int:
        max_int = b
    if c > max_int:
        max_int = c
    return max_int


def float_max_two(float a, float b):
    """Finds the maximum float of the given two floats.
        Args:
            float1, float2 (float): Input floats.
        Returns:
            Maximum float (float).
    """
    if a > b : return a
    else: return b


def float_max_three(float a, float b, float c):
    """Finds the maximum float of the given two float.
        Args:
            float1, float2, float3 (float): Input floats.
        Returns:
            Maximum float (float).
    """
    cdef float max_float = a
    if b > max_float:
        max_float = b
    if c > max_float:
        max_float = c
    return max_float


def int_min_two(int a, int b):
    """Finds the minimum integer of the given two integers.
    Args:
        integer a,integer b (int): Input integers.
    Returns:
        Minimum integer (int).
    """
    if a > b : return b
    else: return a


def int_min_three(int a, int b, int c):
    """Finds the minimum integer of the given two integers.
    Args:
        integer a, integer b, integer c (int): Input integers.
    Returns:
        Minimum integer (int).
    """
    cdef int min_int = a
    if b < min_int:
        min_int = b
    if c < min_int:
        min_int = c
    return min_int

def affine(unicode string1, unicode string2, float main_gap_start, float main_gap_continuation, sim_func ):

    cdef float gap_start = - main_gap_start
    cdef float gap_continuation = - main_gap_continuation
    cdef int len_str1 = len(string1)
    cdef int len_str2 = len(string2)
    cdef int i=0, j=0
    cdef double[:, :] m = np.zeros((len_str1 + 1, len_str2 + 1), dtype=np.double)
    cdef double[:, :] x = np.zeros((len_str1 + 1, len_str2 + 1), dtype=np.double)
    cdef double[:, :] y = np.zeros((len_str1 + 1, len_str2 + 1), dtype=np.double)

    # DP initialization
    for i from 1 <= i < (len_str1+1):
        m[i, 0] = -float(np.inf)
        x[i, 0] = gap_start + (i-1) * gap_continuation
        y[i, 0] = -float(np.inf)
    #
    # # DP initialization
    for j from 1 <= j < (len_str2+1):
        m[0, j] = -float(np.inf)
        x[0, j] = -float(np.inf)
        y[0, j] = gap_start + (j-1) * gap_continuation


    # affine gap calculation using DP
    for i from 1 <= i < (len_str1 + 1):
        for j from 1 <= j < (len_str2 + 1):
            # best score between x_1....x_i and y_1....y_j
                # given that x_i is aligned to y_j
            m[i, j] = (sim_func(string1[i-1], string2[j-1]) + float_max_three(m[i-1][j-1],
                                                                       x[i-1][j-1], y[i-1][j-1]))
            # the best score given that x_i is aligned to a gap
            x[i, j] = float_max_two((gap_start + m[i-1, j]), (gap_continuation+ x[i-1, j]))
            # the best score given that y_j is aligned to a gap
            y[i, j] = float_max_two((gap_start+ m[i, j-1]), (gap_continuation + y[i, j-1]))

    return float_max_three(m[len_str1, len_str2], x[len_str1, len_str2], y[len_str1, len_str2])