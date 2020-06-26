import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

"""
这个例子，完全没有x,y的概念，就是一个输入，
也没有用generator/sequence输入，
就是一个dict，
然后通过名字跟输入张量绑定了：
inputs = keras.Input(shape=(3,), name="inputs")
和
data = {"inputs": np.random.random((3, 3)),...}
绑定了，
然后，
model.fit(data)，
这样，loss和metric，就变成一个层，
有意思。

之前我一直不知道，如何在keras框架下，来自己告诉框架我的y是谁，
fit(x,y...)也好，fit(genentor/sequence)也好，都是固定，第二个返回的就是y，
而且，没法想这样自由的在第二参数"target"一样传入，
赞！学习了

"""

class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, logits, targets, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy as a metric and add it
        # to the layer using `self.add_metric()`.
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)

inputs = keras.Input(shape=(3,), name="inputs")
targets = keras.Input(shape=(10,), name="targets")
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name="predictions")(logits,targets)

model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer="adam")  # No loss argument!

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}
model.fit(data)