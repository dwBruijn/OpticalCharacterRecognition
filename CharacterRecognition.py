from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt;
import CropAndNormalize as cn
import LineExtraction as le


# letters to learn
letters = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]


# randomly initialize weights and biases for layers (weights are between -0.5 and 0.5)
def initialize_weights(width, height, number_of_hidden_neurons):
    Wi_h = np.random.random(size=(number_of_hidden_neurons, height, width)) - 0.5  # input to hidden layer weights
    Wh_o = np.random.random(size=(26, number_of_hidden_neurons)) - 0.5  # hidden to output weights
    Bh = np.random.random(number_of_hidden_neurons) - 0.5  # hidden layer biases
    Bo = np.random.random(26) - 0.5  # output layer biases

    return Wi_h, Wh_o, Bh, Bo


# activation function
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


# forward propagation through network
# normalized is the normalized input image
def feed_forward(normalized, Wi_h, Wh_o, Bh, Bo):
    n_h = 0
    [number_of_hidden_neurons, height, width] = Wi_h.shape
    # feed Forward: rom input layer to hidden
    output_of_hidden_neurons = []
    net_input_for_hidden_neurons = []

    for hidden_neuron in range(0, number_of_hidden_neurons):  # forward pass from input to hidden
        for i in range(0, height):  # calculating activation input
            for j in range(0, width):
                WxP = Wi_h[hidden_neuron, i, j] * normalized[i, j]  # Weight X Input
                n_h = n_h + WxP  # The overall sum of W's X P's

        n_h = n_h + Bh[hidden_neuron]  # total input = WP+Bias
        output_of_hidden_neurons.append(sigmoid(n_h))  # calculate and store hidden neurons output
        net_input_for_hidden_neurons.append(n_h)  # store total network input
        n_h = 0  # reset n

    # feed forward: from hidden to output
    out_hidden_x_weights_h_O = output_of_hidden_neurons * Wh_o  # out of hidden layer multiplied by weights from hidden to output layer
    # this is a 26 X 10 matrix, each row contains the weights connecting hidden neurons to specific output neuron.
    net_input_for_out_neurons = np.sum(out_hidden_x_weights_h_O, axis=1)  # sum of all (Wh_o weights X hidden neuron outputs), each row is the total input for each output neuron
    output_of_out_neurons = []

    for output_neuron in range(0, 26):
        # Find and Calculate output of output neurons
        total_input_for_neuron = net_input_for_out_neurons[output_neuron] + Bo[output_neuron]  # the input to the kth output neuron
        output_of_out_neurons.append(sigmoid(total_input_for_neuron))  # get the output and save it

    return output_of_out_neurons, output_of_hidden_neurons  # return outputs


# calculate error at output neurons
def calculate_error_at_output(output_of_out_neurons, target_output):
    output_error =[]
    # Calculating the error at the output
    for output_neuron in range(0, 26):
        outputNeuronError = output_of_out_neurons[output_neuron] - target_output[output_neuron]  # error = out - target
        output_error.append(outputNeuronError)  # store error for all outputs

    return output_error

# backpropagation and weights adjustment
def back_propagate(Wi_h, Wh_o, Bh, Bo, normalized, output_error, output_of_out_neurons, output_of_hidden_neurons, learning_rate, momentum):
    old_Wh_o = np.array(Wh_o[:, :])  # save old weights for backpropagation from hidden to input layer
    old_Wi_h = np.array(Wi_h[:, :])

    [number_of_hidden_neurons, height, width] = Wi_h.shape
    # back propagating: from output to hidden and adjusting weights
    for output_neuron in range(0, 26):
        for hidden_neuron in range(0, number_of_hidden_neurons):
            # calculating the adjustment which is learning rate * error at current output neuron * sigmoid derivative *  output of current hidden neuron)
            adjustment = (learning_rate * output_error[output_neuron] * output_of_out_neurons[output_neuron] * (1 - output_of_out_neurons[output_neuron]) * output_of_hidden_neurons[hidden_neuron])
            Wh_o[output_neuron, hidden_neuron] = (momentum * Wh_o[output_neuron, hidden_neuron]) - adjustment  # adjusting weights per this formula, Wnew = momentum* Wold - adjustment for fast convergence

    # back propagating: from hidden to input and adjusting weights
    for hidden_neuron in range(0, number_of_hidden_neurons):
        delta_total_error_hidden_neuron = 0
        for output_neuron in range(0, 26):
            # calculate delta error at each output neuron with respect to current hidden neuron
            delta_error_output_neuron_hidden_neuron = output_error[output_neuron] * output_of_out_neurons[output_neuron] * (1 - output_of_out_neurons[output_neuron]) * old_Wh_o[output_neuron, hidden_neuron]
            delta_total_error_hidden_neuron = delta_total_error_hidden_neuron + delta_error_output_neuron_hidden_neuron  # delta total error with respect to current hidden neuron

        # loop over all input weights connecting to current hidden neuron
        for i in range(0, height):
            for j in range(0, width):
                # delta Total Error with respect to weight to be adjusted. this weight is connecting input to current hidden layer
                delta_total_error_input_to_hidden_neuron_weight = delta_total_error_hidden_neuron * output_of_hidden_neurons[hidden_neuron] * (1 - output_of_hidden_neurons[hidden_neuron]) * normalized[i, j]
                Wi_h[hidden_neuron, i, j] = (momentum * Wi_h[hidden_neuron, i, j]) - (learning_rate * delta_total_error_input_to_hidden_neuron_weight)

    return Wi_h, Wh_o


# train network
def train_net(Wi_h, Wh_o, Bh, Bo, height, width, number_of_training_samples, learning_rate, momentum, target_error):
    epoch = 0
    total_error = 1
    error_list = []  # to save all total error generated
    y_axis = []  # for plotting the error minimization at the end

    while total_error > target_error:  # loop until criteria is met
        for letter_to_train in range(0, 25):  # loop for all letters to be trained
            target_output = np.zeros(26)  # target output is all zeros
            target_output[letter_to_train] = 1  # except the one to be trained

            for n in range(0, number_of_training_samples):  # number of training samples
                # preprocessing phase
                # cropping and normalizing the image to have a uniform input to the neural network
                training_sample = 'samples/%s%d.png' % (letters[letter_to_train], n)  # training sample image file name

                character_in = im.open(training_sample)  # load Image
                black_and_white = cn.convert_to_black_and_white(character_in)  # convert image to black and white
                toggled_black_and_white = cn.toggle_ones_and_zeros(black_and_white)
                cropped_black_and_white = cn.crop(toggled_black_and_white)  # crop image to get character only
                normalized = cn.normalize(cropped_black_and_white, width, height)  # mormalize (resize)

                # training phase
                output_of_out_neurons, output_of_hidden_neurons = feed_forward(normalized, Wi_h, Wh_o, Bh, Bo)  # feed forward
                output_error = calculate_error_at_output(output_of_out_neurons, target_output)  # calculate error at output neurons
                Wi_h, Wh_o = back_propagate(Wi_h, Wh_o, Bh, Bo, normalized, output_error, output_of_out_neurons, output_of_hidden_neurons, learning_rate, momentum)  # backpropage and adjust weights

        # calculate the mean squared error
        total_error = 0
        for x in range(0, 26):
            squared = 0.5 * output_error[x] ** 2
            total_error = total_error + squared

        print('Total Error = %f' % total_error)
        epoch = epoch + 1
        error_list.append(total_error)
        y_axis.append(epoch)

    # plot Total Error vs epoch
    print('Total Number of epochs %d' % epoch)
    plt.plot(y_axis, error_list)
    plt.ylabel('Total Error')
    plt.xlabel('Epoch')
    plt.show()

    return (Wi_h, Wh_o, Bh, Bo)


# make prediction
def recognize_character(input_normalized, Wi_h, Wh_o, Bh, Bo):
    output_of_out_neurons, output_of_hidden_neurons = feed_forward(input_normalized, Wi_h, Wh_o, Bh, Bo)  # feed forward
    # character recognized is neuron with highest output
    # ex: if 1st output neuron is 1 then the character is A
    max_out = np.argmax(output_of_out_neurons)
    return letters[max_out]