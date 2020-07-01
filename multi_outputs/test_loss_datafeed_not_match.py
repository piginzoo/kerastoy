import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import Sequence

class SequenceData(Sequence):
    def __getitem__(self, idx):
        x_train = np.random.random((3, 3))
        y_train_1 = np.random.random((3, 3))
        y_train_2 = np.random.random((3, 3))
        y_train_3 = np.random.random((3, 10))
        return x_train,{"y1": y_train_1, "y2": y_train_2, "y3":y_train_3}

    def __len__(self):
        return 10


class MyModel(Model):

    layers.GRUCell()
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense_1 = layers.Dense(10, name="predictions")
        self.dense_2 = layers.Dense(3, activation="softmax", name="class1")
        self.dense_3 = layers.Dense(3, activation="softmax", name="class2")

    def call(self, inputs, training=None):
        x = self.dense_1(inputs)
        y1 = self.dense_2(x)
        y2 = self.dense_3(x)
        return {"y1": y1, "y2": y2}


model = MyModel()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss={'y1': 'categorical_crossentropy', 'y2': 'categorical_crossentropy'}
)

model.fit(SequenceData(), batch_size=3, epochs=1)
