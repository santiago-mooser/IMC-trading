import pandas as pd
import sys
import numpy
import math

#######
# This file expects the formal to be the same as the activities.csv except just with one ticker
####


def weighted_avg_and_std(values, weights):
    print(values)
    average = numpy.average(values, weights=weights)
    # Fast and numerically precise:
    variance = numpy.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

filename = sys.argv[1]

df = pd.read_csv(filename)
df["spread"] = df["ask_price_1"] - df["bid_price_1"]

std_from_spread = df["spread"].std()
avg_from_spread = df["spread"].mean()

print(f'Standard deviation from mid price: {std_from_spread}')
print(f'Mean spread: {avg_from_spread}')

