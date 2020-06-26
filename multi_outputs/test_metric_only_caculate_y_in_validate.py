import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf

"""
这个例子主要想看看metrics到底是怎么工作的。

我这个模型有2个输出：y1,y2
我当然可以算y1的，和y2的，然后给个weight啥的，算一个加权平均的，
但是我想y2,y2的结果可以不用分开算，而是通过一个回调一起给我，我把他们合起来算，
也就是给我(y1_true,y1_pred,y2_true,y2_pred),
可是自定义Metrics只能一个一个分别给我，

这都是不是我想要的，test_metric.py虽然解决了，
其实是把计算转移到了模型定义里，
这就会增加模型的计算量，如果我只想在validate的时候瞅瞅怎么办？

我的解决办法，是写一个callback，
然后在里面自己通过调用模型，
算出结果，然后用numpy函数算，
把在计算从GPU里转移到了CPU，
这个主要是想节省点显存。
我的textscanner显存占用特别大，这样可以省。

"""

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense_1 = layers.Dense(10, name="predictions")
        self.dense_2 = layers.Dense(3, activation="softmax", name="class1")
        self.dense_3 = layers.Dense(3, activation="softmax", name="class2")

    def call(self, inputs, training=None):
        x = self.dense_1(inputs,training=training)
        y1 = self.dense_2(x,training=training)
        y2 = self.dense_3(x,training=training)
        return {"y1": y1, "y2": y2}


class MyMetrics(tf.keras.callbacks.Callback):
    def __init__(self,data):
        self.data = data

    def on_epoch_end(self, epoch, logs=None):
        inputs,val = self.data
        result = self.model(inputs,training=False)
        y1 = result['y1']
        y2 = result['y2']
        y = y1+y2
        # print(type(y))
        # print("y1:", y1.shape)
        # print("y:",y.shape)
        y = y.numpy()
        y = np.argmax(y,axis=-1)
        val = np.argmax(val,axis=-1)
        acc = np.mean(np.equal(y,val))
        print(",accuarcy=",acc)


inputs = keras.Input(shape=(3,), name="inputs")
model = MyModel()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss={'y1': 'categorical_crossentropy', 'y2': 'categorical_crossentropy'},
)

# model.build(inputs.shape)

x_train = np.random.random((9, 3))
y_train_1 = np.random.random((9, 3))
y_train_2 = np.random.random((9, 3))
y_train = y_train_2+y_train_1
val = (x_train[:3,:],y_train[:3,:])

result = model(x_train)

model.fit(x=x_train,
          y={"y1":y_train_1, "y2":y_train_2},
          validation_data=val,
          callbacks=[MyMetrics(val)],
          batch_size=3,
          epochs=3)
