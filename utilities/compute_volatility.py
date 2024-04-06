import pandas as pd
import sys
import numpy
import math

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = numpy.average(values, weights=weights)
    # Fast and numerically precise:
    variance = numpy.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


filename = sys.argv[1]

df = pd.read_csv(filename)

std = weighted_avg_and_std(df['price'], df['quantity'])

print(f'Weighted average: {std[0]}')
print(f'Weighted standard deviation: {std[1]}')
