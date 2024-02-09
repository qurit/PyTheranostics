import math

import numpy
from numpy import exp


# Function definitions
def monoexp_fun(x: numpy.ndarray, a: float, b: float) -> numpy.ndarray:
    return a * exp(-b * x)

def biexp_fun(x: numpy.ndarray, a: float, b: float, c: float, d: float) -> numpy.ndarray:
    return a * exp(-b * x) + c * exp(-d * x)

def triexp_fun(x: numpy.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> numpy.ndarray:
    return a * exp(-b * x) + c * exp(-d * x) + e * exp(-f * x)

def find_a_initial(f: numpy.ndarray, b: numpy.ndarray, t: numpy.ndarray) -> numpy.ndarray:
    return f * exp(b * t)

# TODO: Review Hanscheid inputs/outputs.
def Hanscheid(a, t):
    return a * ((2 * t) / (math.log(2)))