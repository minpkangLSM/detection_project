import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from segmentations.UNet import UNet
from matplotlib import pyplot as plt
from PIL import Image
"""
2021.01.01. Kangmin Park
This code is for checking results(outputs) of each layer in the model.
"""

def output_checker(model_instance, temp_data_dir=None):
    """
    :param model_instance: model instance.
    :param temp_data_dir: directory of data for testing outputs of the model.

    :return:
    """
    model = model_instance
    data = cv2.imread(temp_data_dir, cv2.IMREAD_UNCHANGED)
    data = np.expand_dims(data, axis=0)
    inp = model.input
    """
    model inner lib : input
    shape of model.input ex1) the num of input is 1 : 
    (<tf.Tensor 'input_1:0' shape=(None, height, width, channels) dtype=?>)
    shape of model.input ex2) the num of input is 2 : 
    (<tf.Tensor 'input_1:0' shape=(None, height, width, channels) dtype=?>), <tf.Tensor 'input_2:0' shape=(None, height, width, channel) dtype=?>)
    """
    outputs__ = [layer.output for layer in model.layers]
    outputs_ = [K.function([inp], [out]) for out in outputs__]
    output = [out(data) for out in outputs_]

    return output

def visualize_output(output):
    """
    :param output: def output_checker returning value, dtype : list
    :return:
    """
    output_results = output
    layer_num = len(output)
    #plot_length = layer_num
    plot_length = 10
    fig = plt.figure()
    axes = []
    for i in range(plot_length):
        img_raw = np.array(output_results[i])
        img_raw = np.squeeze(img_raw, axis=(0,1))
        img = img_raw[:, :, 0]
        img = (img * 255 / (np.max(img) - np.min(img)+1e-4)).astype(np.uint8)
        sub_title = "Layer "+str(i)+" #1 channel"
        axes.append(fig.add_subplot(2, plot_length/2, i+1))
        axes[-1].set_title(sub_title)
        plt.imshow(img)
    plt.show()

if __name__ == "__main__" :

    INPUT_SHAPE = (512, 512, 3)
    NUM_CLASSES = 2
    LR = 1e-3
    DIR = "myeonmok19_0_0.png"
    UNet = UNet(INPUT_SHAPE, NUM_CLASSES, LR)
    model = UNet.build_net()

    test_results = output_checker(model, DIR)
    visualize_output(test_results)
    print("length : ", len(test_results))
    img = np.array(test_results[-7])
    print("before squeeze : ", img.shape)
    print("img max : ", np.max(img))
    print("img min : ", np.min(img))
    img = np.squeeze(img, axis=(0,1))
    print("after squeeze : ", img.shape)
    img = img[:,:,0:3]
    img = (img*255/(np.max(img)-np.min(img))).astype(np.uint8)
    print("img max : ", np.max(img))
    print("img min : ", np.min(img))
    print("extact a channel : ", img.shape)
    plt.imshow(img)
    plt.show()
    #img = Image.fromarray(img)
    #img.show()