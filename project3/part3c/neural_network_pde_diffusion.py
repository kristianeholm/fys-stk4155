import tensorflow as tf
import numpy as np
from progress.bar import Bar

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from tensorflow.keras import losses                 #This allows using whichever loss function we want (MSE, ...)

seed = 1234
tf.random.set_seed(seed)
np.random.seed(seed)

class NeuralNetworkPDEDiffusion(tf.keras.Sequential):
    def __init__(self, layers, activation_function, learning_rate):
        super(NeuralNetworkPDEDiffusion, self).__init__()
        # First hidden layer, connected to input, the 2 in input shape since we always have x and t
        self.add(Dense(layers[0], input_shape=(2,), activation=activation_function))

        # Rest of hidden layers
        for layer in layers[1:-1]:
            self.add(Dense(layer, activation="relu"))

        # Output layer
        self.add(Dense(layers[-1], activation="linear"))

        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = losses.MeanSquaredError()


    def train_model(self, x, t, epochs):
        x = x.reshape(-1,1)
        t = t.reshape(-1,1)
        self.x = tf.convert_to_tensor(x, dtype=tf.float32)
        self.t = tf.convert_to_tensor(t, dtype=tf.float32)

        losses = np.zeros(epochs)

        bar = Bar("Epochs", max = epochs)
        for epoch in range(epochs):
            loss, gradients = self.compute_gradients()
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            losses[epoch] = loss
            bar.next()

        bar.finish()
        return losses


    @tf.function
    def compute_gradients(self):
        with tf.GradientTape() as tape:
            loss = self.compute_loss()
        gradients = tape.gradient(loss, self.trainable_variables)
        return loss, gradients

    @tf.function
    def trial_function(self):
        x, t = self.x, self.t
        X = tf.concat([x,t], 1)
        N = self(X, training=True)
        f_trial = tf.sin(np.pi*x) + t*x*(1-x)*N
        return f_trial

    @tf.function
    def trial_function_output(self, x, t):
        X = tf.concat([x,t], 1)
        N = self(X, training=False)
        return tf.sin(np.pi*x) + t*x*(1-x)*N

    @tf.function
    def predict(self, x, t):
        return self.trial_function_output(x, t)

    @tf.function
    def compute_loss(self):
        #Using automatic differentiation to get gradients for G = d^2f/dx^2 - df/dt = 0
        x, t = self.x, self.t
        with tf.GradientTape() as gg:
            gg.watch(x)
            with tf.GradientTape(persistent=True) as g:
                g.watch([x,t])
                f_trial = self.trial_function()

            df_dt = g.gradient(f_trial, t)
            df_dx = g.gradient(f_trial, x)

        d2f_dx2 = gg.gradient(df_dx, x)

        #Free up memory resources
        del g
        del gg

        #Compute cost function using the actual differential equation, to get zero when equation is satisfied.
        y_pred = df_dt - d2f_dx2
        loss = self.loss_fn(0., y_pred)
        return loss
