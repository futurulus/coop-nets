import random

options = [['RecurrentContextSpeaker', 'Speaker'],  #--learner
           ['LSTM', 'GRU', 'Recurrent'],  #--speaker_cell
           ['4', '10', '20', '40', '100', '200'],  #--speaker_cell_size
           ['0.0', '0.01', '0.02', '0.04', '0.1', '0.2', '0.4'],  #--speaker_dropout
           ['0.0', '1.0', '2.0', '5.0', '10.0'],  #--speaker_forget_bias
           ['0.0', '0.1', '1.0', '5.0', '10.0', '100.0'],  #--true_grad_clipping
           ['true', 'false'],  #--speaker_hsv
           ['0.001', '0.002', '0.004', '0.01', '0.02', '0.04', '0.1', '0.2', '0.4', '1.0'],  #--speaker_learning_rate
           ['linear', 'very_leaky_rectify', 'elu', 'leaky_rectify', 'rectify', 'softplus', 'tanh'],  #--speaker_nonlinearity
           ['adagrad', 'rmsprop', 'adam', 'adamax', 'momentum', 'nesterov_momentum', 'adadelta', 'sgd'],  #--speaker_optimizer
           ['1', '2'],  #--speaker_recurrent_layers
           ['raw', 'fourier']]  #--speaker_color_repr


if __name__ == '__main__':
    for i in range(200):
        print(' '.join([random.choice(a) for a in options] + [str(i % 4)]))
