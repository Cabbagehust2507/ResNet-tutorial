import tensorflow as tf 
from examples.residual_block_tf import Residual

b1 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
])

class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))
    
    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X

b2 = ResnetBlock(num_channels=64, num_residuals=2, first_block=True)
b3 = ResnetBlock(128, 2)
b4 = ResnetBlock(256, 2)
b5 = ResnetBlock(512, 2)

def net():
    return tf.keras.Sequential([
        b1, b2, b3, b4, b5,
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units = 10)
    ])