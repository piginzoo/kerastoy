import numpy as np
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import Model
# from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import *
from tensorflow.keras.utils import *

y_true = [[0,1],
          [1,2]]
print("y_true:",y_true)
y_true = to_categorical(y_true,num_classes=3)
print("y_true:",y_true)
print("======================================")

print("== 测试accuracy ==")
ca = CategoricalAccuracy()

# case1
y_pred = [[0,1],
          [1,0]]
print("y_pred:",y_pred)
y_pred = to_categorical(y_pred,num_classes=3)
print("y_pred:",y_pred.shape)
print(y_pred)
ca.update_state(y_true,y_pred)
r = ca.result()
print("预期acc:0.75")
print("实际acc:",r)

# case2
y_pred = [[
            [0.5,0.2,0.3], # 0
            [0.2,0.5,0.3], # 1
          ],[
            [0.1, 0.8, 0.1],  # 1
            [0.6, 0.2, 0.2],  # 0
          ]]
y_pred = np.array(y_pred)
print("y_pred:",y_pred)
print(y_pred)
ca.update_state(y_true,y_pred)
r = ca.result()
print("预期acc:0.75")
print("实际acc:",r)
