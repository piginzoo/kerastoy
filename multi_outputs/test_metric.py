import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
import tensorflow.keras.backend as K
"""
这个例子主要想看看metrics到底是怎么工作的。

我这个模型有2个输出：y1,y2
我当然可以算y1的，和y2的，然后给个weight啥的，算一个加权平均的，
但是我想y2,y2的结果可以不用分开算，而是通过一个回调一起给我，我把他们合起来算，
也就是给我(y1_true,y1_pred,y2_true,y2_pred),
可是自定义Metrics只能一个一个分别给我，

我的解决办法是，直接加一个输出，算好y=y1+y2,
然后，在验证函数里，就可以验证2个了，
这其实是把metrics不能结合y1,y2的问题，转移到了model里。

"""


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense_1 = layers.Dense(10, name="predictions")
        self.dense_2 = layers.Dense(3, activation="softmax", name="class1")
        self.dense_3 = layers.Dense(3, activation="softmax", name="class2")

    def call(self, inputs, training=None):
        x = self.dense_1(inputs)
        y1 = self.dense_2(x)
        y2 = self.dense_3(x)
        y = y1+y2
        return {"y1": y1, "y2": y2, "y": y }


class MyMetrics(tf.keras.metrics.Metric):
    def __init__(self, name='my_metrics', **kwargs):
        super(MyMetrics, self).__init__(name=name, **kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None):
        print("y_true/y_pred:",y_true.shape, y_pred.shape)
        return 0

    def result(self):
        return 0

    def reset_states(self):
        return 0

inputs = keras.Input(shape=(3,), name="inputs")
model = MyModel()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss={'y1': 'categorical_crossentropy', 'y2': 'categorical_crossentropy'},
    metrics={"y1":"categorical_accuracy","y2":"categorical_accuracy","y":[MyMetrics()]}
)

# model.build(inputs.shape)

x_train = np.random.random((3, 3))
y_train_1 = np.random.random((3, 3))
y_train_2 = np.random.random((3, 3))
y_train = y_train_2+y_train_1

result = model(x_train)

model.fit(x_train, {"y1":y_train_1, "y2":y_train_2}, batch_size=3, epochs=1)
