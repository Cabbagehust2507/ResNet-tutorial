import tensorflow as tf 



class Residual(tf.keras.Model):
    """
    Residual block architecture:
                    X
                    |----------------
                  Conv2d            |
                    |               |   
                 BatchNorm          |
                    |               |
                  RelU              |
                    |               |
                  Conv2d            |
                    |               |
                BatchNorm           |
                    |----------------
                   ReLU
    """
    def __init__(self, num_channels, use_1x1conv = False, strides =1):
        super().__init__()
        self.conv1 = tf.keras.layer.Conv2D(num_channels, padding = 'same', 
                                            kernel_size = 3, stride = strides)
        self.conv2 = tf.keras.layer.Conv2D(num_channels, padding = 'same',
                                            kernel_size=3)
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, stride=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
    
    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)