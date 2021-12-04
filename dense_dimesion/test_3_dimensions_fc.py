from keras import layers
from keras.layers import Input
inputs = Input(name='the_input', batch_shape=(3,8,10), dtype='float32')  # (None, 128, 64, 1)
print(inputs)
dense = layers.Dense(16, name="predictions")(inputs)
print(dense.shape) #结果是 (?, 3, 8, 16)
