import sys, numpy as np
from keras.datasets import mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

images, labels = x_train.reshape(60000, 28 * 28)/255, y_train

one_hot_labels = np.zeros((len(labels), np.max(labels) + 1))

for i, n in enumerate(labels):
    one_hot_labels[i][n] = 1

input_size = 784
hidden_size = 100
output_size = 10

np.random.seed(2)

relu = lambda x: (x > 0) * x
relu_deriv = lambda x: x > 0

weights_0_1 = 0.2 * np.random.random((input_size, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, output_size)) - 0.1

batch_size = 1000

for j in range(700):
    error = 0.0
    correct = 0.0
    for i in range(60000 // batch_size):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
    
        input = images[batch_start:batch_end] # (200, 784) / list
        
        hidden_layer = relu(np.dot(input, weights_0_1)) # (200, 784) . (784, 40) = (200, 40)
        
        dropout_mask = np.random.randint(2, size=hidden_layer.shape)

        hidden_layer *= dropout_mask * 2        
        output = np.dot(hidden_layer, weights_1_2) # (200, 40) . (40, 10) = (200, 10)

        goal = one_hot_labels[batch_start: batch_end] # (200, 10)
        error += np.sum((output - goal) ** 2) # ((200, 10) - (200, 10)) ** 2 == (200, 10) / sum(200, 10) = (1)
        correct += np.sum(np.argmax(output - goal, axis=1) == np.argmax(output - goal, axis=1))
        
        delta = (output - goal) / batch_size # (200, 10)
            
        hl_delta = np.dot(delta, weights_1_2.T) * relu_deriv(hidden_layer) * dropout_mask # (200, 10) . (10, 40) = (200, 40)
    
        wd_1_2 = np.dot(np.transpose(hidden_layer), delta) # (40, 200) . (200, 10) = (40, 10)
        wd_0_1 = np.dot(np.transpose(input), hl_delta) # (784, 200) . (200, 40) = (784, 40)
    
        weights_1_2 -= wd_1_2 * 0.2
        weights_0_1 -= wd_0_1 * 0.2
    sys.stdout.write("\rError: " + str(error/60000) + "\\ Correct: " + str(correct/60000))

testx = x_test.reshape(len(x_test), 28 * 28)/225

test_error = 0.0
test_corrects = 0.0
for i in range(len(testx)):
        input = testx[i] # (784,) / list
        hidden_layer = relu(np.dot(input, weights_0_1)) # (784,) . (784, 40) = (40,)
        output = np.dot(hidden_layer, weights_1_2) # (40,) . (40, 10) = (10,)

        goaln = y_test[i]
        goal = np.zeros((10,))
        goal[goaln] = 1
        test_error += np.sum((output - goal) ** 2)
        test_corrects += (np.argmax(output) == np.argmax(goal))
print("Error: " + str(test_error/len(testx)) + " / Correct: " + str(test_corrects/len(testx)))
