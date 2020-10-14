# NEURAL NETWORKS (lecture 5)

AI that is able to identify which traffic sign appears in a photograph.         
[Project page](https://cs50.harvard.edu/ai/2020/projects/5/traffic/) (there you can download the gtsrb folder containing the traffic signs images).

&nbsp;

```
$ python traffic.py gtsrb
Epoch 1/10
500/500 [==============================] - 26s 51ms/step - loss: 2.6076 - accuracy: 0.4018
Epoch 2/10
500/500 [==============================] - 27s 53ms/step - loss: 0.6410 - accuracy: 0.8207
Epoch 3/10
500/500 [==============================] - 26s 51ms/step - loss: 0.2817 - accuracy: 0.9227
Epoch 4/10
500/500 [==============================] - 26s 52ms/step - loss: 0.1947 - accuracy: 0.9466
Epoch 5/10
500/500 [==============================] - 25s 50ms/step - loss: 0.1563 - accuracy: 0.9590
Epoch 6/10
500/500 [==============================] - 26s 51ms/step - loss: 0.1313 - accuracy: 0.9650
Epoch 7/10
500/500 [==============================] - 26s 51ms/step - loss: 0.1434 - accuracy: 0.9630
Epoch 8/10
500/500 [==============================] - 27s 54ms/step - loss: 0.0956 - accuracy: 0.9755
Epoch 9/10
500/500 [==============================] - 24s 49ms/step - loss: 0.0750 - accuracy: 0.9804
Epoch 10/10
500/500 [==============================] - 23s 47ms/step - loss: 0.0913 - accuracy: 0.9767
333/333 - 4s - loss: 0.1607 - accuracy: 0.9703
```

For the above results I used the following neural network:
- Convolutional layer --> 32 filter, 3x3 kernel matrix, activation relu
- Max-pooling layer --> 2x2 pool size
- Convolutional layer --> 32 filter, 3x3 kernel matrix, activation relu
- Max-pooling layer --> 2x2 pool size
- Dense layer --> 64 units, activation relu
- Output layer --> activation softmax

&nbsp;

#### REQUIREMENTS
```
pip install opencv-python
pip install scikit-learn
pip install tensorflow
```