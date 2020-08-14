import time

import skimage
import matplotlib
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import PIL
from PIL import Image
from scipy import ndimage
from scipy import misc
from dnn_app_utils_v2 import *
from skimage import transform



plt.rcParams['figure.figsize'] = (5.0,4.0) #default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
index = 10
plt.imshow(train_x_orig[index])
print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") + " picture")
plt.show()

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))

#reshape testing and training examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

#standardize data to have features values between 0 and 1
train_x = train_x_flatten/255
test_x = test_x_flatten/255

print("train_x shape: " + str(train_x.shape))
print("test_x shape: " + str(test_x.shape))

n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x,n_h,n_y)

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    grads = {}
    costs = [] #to keep track of the cost
    m = X.shape[1] #number of examples
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x,n_h,n_y)

    w1 = parameters["W1"]
    b1 = parameters["b1"]
    w2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X,w1,b1, 'relu')
        A2, cache2 = linear_activation_forward((A1,w2,b2,'sigmoid'))

        cost = compute_cost(A2, Y)

        dA2 = - (np.divide(Y,A2) - np.divide(1-Y,1-A2))

        dA1, dW2,db2 = linear_activation_backward(dA2,cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if print_cost and i%100 == 0:
            print(f"Cost after iteration {i, np.squeeze(cost)}")
        if print_cost and i%100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per tens)')
    plt.title("Learning Rate = " + str(learning_rate))
    plt.show()

    return parameters

layers_dims = [12288,20,7,5,1]

def L_layer_model(X,Y, layers_dims, learning_rate = 0.008, num_iterations = 3000, print_cost = False):
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0,num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL,Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i%100 == 0:
            print("Cost after iteration %i: %f"%(i, cost))
        if print_cost and i%100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning Rate =" + str(learning_rate))
    plt.show()

    return parameters

parameters = L_layer_model(train_x,train_y,layers_dims,num_iterations=3000,print_cost=True)

pred_train = predict(train_x,train_y,parameters)

pred_test = predict(test_x, test_y, parameters)

my_image = "test-images/dog_test.jpg"
my_label_y = ["Luna"]

fname = my_image
image = np.array(plt.imread(fname))
my_image = skimage.transform.resize(image, (num_px,num_px)).reshape((num_px*num_px*3,1))

my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-Layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
plt.show()
