from utils.utils import *
import numpy as np

if __name__ == "__main__":
    
    # initializatin parameters for chaotic sequence    
    U = 3.9
    X1 = 0.75
    # sequence to encrypt
    input_to_enc = [11, 23, 37, 45, 68, 25, 236, 58, 59, 90]
    print("Signal to encrypt {}.".format(input_to_enc))
    output_enc = []
    # chaotic seq. generation
    l = len(input_to_enc)
    x = init_chaotic_seq(U, X1, l)
    # initialization of the model with 8 inputs, 8 neurons and the hardlim activation fun. _|-
    model = init_model()

    # encryption loop
    for i, v in enumerate(input_to_enc):
        byte_ = dec_to_byte(x[i])
        byte_arr = np.array([int(i) for i in byte_])
        weights, bias = init_weights(byte_arr)
        weights_0 = [ np.array( weights, ndmin=2, dtype='float32'), np.array(bias, ndmin=1, dtype='float32') ]
        # setting the precalculated weights
        model.layers[0].set_weights(weights_0)
        # calculate predictions
        input_ = [ int(i) for i in dec_to_byte(v) ]
        pred = model.predict(np.array(input_, ndmin = 2))[0]
        value = int(sum([j*(2**i) for i,j in list(enumerate(reversed(pred)))]))
        output_enc.append(value)


    print("Encrypted signal with the usage of chaotic sequence {}.".format(output_enc))

