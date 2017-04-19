import lstm
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.callbacks as cb

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

#Main Run Thread
if __name__=='__main__':
    global_start_time = time.time()
    epochs  = 5
    time_steps = 10

    np.random.seed(777)
    tf.set_random_seed(777)

    print('> Loading data... ')

    #X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', time_steps, True)
    X_train, y_train, X_test, y_test = lstm.load_data('sinwave.csv', time_steps, False)
    #X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', time_steps, True)

    print('> Data Loaded. Compiling...')

    model = lstm.build_model([1, 64, 128, 1])

    tbCallBack = cb.TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(
        X_train,
        y_train,
        batch_size=128,
        nb_epoch=epochs,
        validation_split=0.05,
        callbacks=[tbCallBack]
        )

    #predictions = lstm.predict_sequences_multiple(model, X_test, time_steps, 50)
    #predicted = lstm.predict_sequence_full(model, X_test, time_steps)
    predicted = lstm.predict_point_by_point(model, X_test)

    print('Training duration (s) : ', time.time() - global_start_time)
    #plot_results_multiple(predictions, y_test, 50)
    plot_results(predicted, y_test)
