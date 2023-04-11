import numpy as np
import random as rand
import math


def number_of_certain_probability(sequence, probability):
    x = rand.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(sequence, probability):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


num0 = 0
num1 = 0
for i in range(0, 10000):
    if number_of_certain_probability([1, 0], [0.6, 0.4]) == 1:
        num1 += 1
    else:
        num0 += 1
print(num1 / num0)
print(math.exp(-(5 - 3) / 4))
print(3*5)
