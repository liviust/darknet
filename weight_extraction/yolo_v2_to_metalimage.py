import argparse
import numpy as np
import os
import struct

def dat_writeformat(data_count):
    return '<' + str(data_count) + 'f'

class WeightExtractorMPS:
    '''weight extraction for darknet to mpsCNNconvolution
    (apple's metal convs)'''

    def __init__(self, weights_dir):
        self.weights_dir = weights_dir


    def conv_layer_weights(self, conv_name,
                           filter_height, filter_width,
                           input_depth, output_depth,
                           conv_output_name=None):
        conv_output_name = conv_output_name or conv_name
        # order output is
        # inputFeatureChannels * outputFeatureChannels
        # * kernelHeight * kernelWidth.
        ## Kernel weights ##
        f_w = open(os.path.join(
            self.weights_dir, conv_name + '_conv_weights.txt'), 'r')
        total_slots = filter_height * filter_width * input_depth * output_depth
        l_w = np.array(f_w.readlines()).astype('float32')
        assert(l_w.shape[0] == total_slots)
        f_w.close()
        w = np.zeros((input_depth, output_depth, filter_height, filter_width),
                     dtype='float32')
        channel_step = filter_width * filter_height
        filter_step = input_depth * channel_step
        # k is filter_height
        # l is filter_width
        # j is input_depth
        # i is output_depth
        for i in range(output_depth):
            for j in range(input_depth):
                for k in range(filter_height):
                    for l in range(filter_width):
                        w[j,i,k,l] = l_w[i*filter_step + j*channel_step
                                         + k*filter_width + l]
        w = w.reshape((-1))
        weights_filename = 'weights_conv_{}.dat'.format(conv_output_name)
        print("Writing weights in {}, shape: {}".format(
            weights_filename, w.shape))
        weights_filepath = os.path.join(self.weights_dir, weights_filename)
        with open(weights_filepath, 'wb') as f:
            f.write(struct.pack(dat_writeformat(len(w)), *w))
        ## Biases ##
	f_b = open(os.path.join(
            self.weights_dir, conv_name + '_conv_biases.txt'), 'r')
	b = np.array(f_b.readlines()).astype('float32')
        f_b.close()
        biases_filename = 'bias_conv_{}.dat'.format(conv_output_name)
        print("Writing biases in {}, shape: {}".format(
            biases_filename, b.shape))
        biases_filepath = os.path.join(self.weights_dir, biases_filename)
        with open(biases_filepath, 'wb') as f:
            f.write(struct.pack(dat_writeformat(len(b)), *b))
        self.check_conv_weights(conv_output_name,
                                w, b, weights_filename, biases_filename)

    def check_conv_weights(self, name, weights, biases,
                           weights_filename, biases_filename):
        # check
        dat_dir = self.weights_dir
        weights_dat_filepath = os.path.join(dat_dir, weights_filename)
        biases_dat_filepath  = os.path.join(dat_dir, biases_filename)
        if not os.path.exists(weights_dat_filepath) or \
           not os.path.exists(biases_dat_filepath):
            print '%-40s' % (name,)
            return

        weights_maxdelta = '?'
        with open(weights_dat_filepath, 'rb') as f:
            weights_dat = np.fromstring(f.read(), dtype='<f4')
            weights_maxdelta = max(map(abs, weights - weights_dat))

        biases_maxdelta = '?'
        with open(biases_dat_filepath) as f:
            biases_dat = np.fromstring(f.read(), dtype='<f4')
            biases_maxdelta = max(map(abs, biases - biases_dat))

        print '%-40s [max delta: w=%-8f b=%-8f]' % (name, weights_maxdelta, biases_maxdelta,)

def main():
    parser = argparse.ArgumentParser(description='darknet to apple MPS')
    parser.add_argument('weights_dir', type=str,
                        help='path to darknet weights')

    args = parser.parse_args()

    extr = WeightExtractorMPS(args.weights_dir)
    extr.conv_layer_weights('0', 3, 3, 3, 16)
    extr.conv_layer_weights('2', 3, 3, 16, 32)
    extr.conv_layer_weights('4', 3, 3, 32, 64)
    extr.conv_layer_weights('6', 3, 3, 64, 128)
    extr.conv_layer_weights('8', 3, 3, 128, 256)
    extr.conv_layer_weights('10', 3, 3, 256, 512)
    extr.conv_layer_weights('12', 3, 3, 512, 1024)
    extr.conv_layer_weights('13', 3, 3, 1024, 1024)
    extr.conv_layer_weights('14', 1, 1, 1024, 425)

if __name__ == '__main__':
    main()
