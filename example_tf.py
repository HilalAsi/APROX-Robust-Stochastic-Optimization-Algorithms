import tensorflow as tf
import optimizers
import numpy as np
import math
import matplotlib.pyplot as plt




def run_model(opt='sgd',stepsize=0.1,max_iter=5):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    sgd = tf.keras.optimizers.SGD(lr=stepsize, decay=0, momentum=0)
    trunc = optimizers.Truncated(lr=stepsize)
    trunc_adagrad = optimizers.TruncatedAdagrad(lr=stepsize)
    model_optimizer = sgd
    if opt=='trunc':
        model_optimizer = trunc
    elif opt=='trunc_adagrad':
        model_optimizer = trunc_adagrad

    model.compile(optimizer=model_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=max_iter)
    acc_model = model.evaluate(x_test, y_test,verbose=0)



run_model(opt='sgd',stepsize=50,max_iter=2)
run_model(opt='trunc',stepsize=50,max_iter=2)
run_model(opt='trunc_adagrad',stepsize=50,max_iter=2)


