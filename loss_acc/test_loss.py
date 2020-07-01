import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import Model
# from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import *
from tensorflow.keras.utils import *

cce = CategoricalCrossentropy()

# https://zhuanlan.zhihu.com/p/35709485
y_true = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
y_pred = [[0.1, 0.2, 0.7], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]]
r = cce(y_true, y_pred).numpy()
print("预期lose:", r)
print("实际lose:", r)

print("==============================")

y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0, 1, 0], [0, 0.4, 0.6]]  # ln0.6=-0.5
print("y_true:")
print(y_true)
print("y_pred:")
print(y_pred)
r = cce(y_true, y_pred).numpy()
print("预想lose:0.25")
print("实际lose:", r)

print("==============================")

y_true = \
    [[
        [1, 0, 0],
        [0, 1, 0]
    ], [
        [0, 1, 0],
        [0, 0, 1]
    ]]

y_pred = \
    [[
        [1, 0, 0],
        [0, 1, 0]
    ], [
        [0, 1, 0],
        [0, 0.4, 0.6]  # ln0.6=-0.5
    ]]
print("y_pred:")
print(y_pred)
r = cce(y_true, y_pred).numpy()
print("预想lose:0.125")
print("实际lose:", r)
