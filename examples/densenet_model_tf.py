import tensorflow as tf 
from examples.dense_block_tf import DenseBlock, TransitionBlock

def block_1():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')]
    )

def block_2():
    net = block_1()
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_block = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_block):
        net.add(DenseBlock(num_convs, growth_rate))
        #
        num_channels += num_convs * growth_rate
        #
        if i != (len(num_convs_in_dense_block)-1):
            num_channels //=2
            net.add(TransitionBlock(num_channels))
    
    return net

def net():
    net = block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10))
    return net
