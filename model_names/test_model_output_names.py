import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

"""
报错：
ValueError: Unknown entries in loss dictionary: ['my_output']. Only expected following keys: ['output_1']

参考：
https://github.com/tensorflow/tensorflow/issues/35750
本来这种写法是很地道的：这写法，官方也是推荐的。看上面的例子。

而且官方的教程里也是推荐用dict来表示loss：https://keras.io/guides/training_with_built_in_methods/#handling-losses-and-metrics-that-dont-fit-the-standard-signature
虽然他这个例子不是自定model，而是fucntional的。


```
    def call():
        return {"xxx": x}
    losses = {"xxx": 'categorical_crossentropy'}
```
可是，罪恶的报错：ValueError: Unknown entries in loss dictionary: ['my_output']. Only expected following keys: ['output_1']

然后我安装了tf.2.2.0，我靠，错误消失了！
看来这种用法没啥问题啊，就是2.1.0的bug啊。

我不死心，又试验了tf2.1.1，也是有同样的问题。
"""


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense_1 = layers.Dense(10, name="predictions")
        self.dense_2 = layers.Dense(3, activation="softmax", name="class_output123")

    def call(self, inputs, training=None):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return {"my_output": x}


inputs = keras.Input(shape=(3,), name="inputs")
model = MyModel()

model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss={'my_output': 'categorical_crossentropy'}
    # run_eagerly=True
)

# model.build(inputs.shape)

x_train = np.random.random((3, 3))
y_train = np.random.random((3, 3))
result = model(x_train)
model.summary()
print("Result:", result)
print("Output layer name:", model.layers[-1].name)
# print("模型output name：", model.output_names)
model.fit(x_train, y_train, batch_size=3, epochs=1)
