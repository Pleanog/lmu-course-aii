
## Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a class of deep learning models primarily designed for analyzing visual imagery. They are inspired by the organization of the animal visual cortex and are particularly effective at identifying **patterns in images.**

> [!CITE] Convolutional Neural Network - Wikipedia
> "A convolutional neural network (CNN, or ConvNet) is a class of artificial neural network (ANN), most commonly applied to analyze visual imagery."

### The Convolutional Process

At the core of a CNN is the **convolution operation**. This involves a `filter` (also known as a kernel) that slides over the input image. For each position, the filter performs a dot product with the underlying image pixels, producing a single value in the output feature map. This process helps in extracting features such as edges, textures, and patterns.

The filter itself is a small matrix of weights that detects specific features.
See in the image below the `filter`:

![[Pasted image 20250718144305.png]]
> This filter here enhances certain features or patterns within the image by giving more weight to the central pixel.

The transformed output on the right is called a feature map. This transformation highlights relevant features from the input.

![[Pasted image 20250718144245.png]]
> Example for edge detection filter applied to the same input image


---
### Convolutional Layer (with Code Examples)

The `Conv2D` layer in libraries like TensorFlow/Keras is fundamental to CNNs. It learns features by convolving filters over the input.

*Based on the lecture and on [Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network)*

```python

tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', …)
```

During the training process, the model learns the optimal weights (`w`) for each filter. A filter can be visualized as a small matrix of weights:

$$filter = \begin{pmatrix} w_{0,0} & \cdots & w_{0,m} \\ \vdots & \ddots & \vdots \\ w_{n,0} & \cdots & w_{n,m} \end{pmatrix}$$

**Key parameters** (*see code above*) of a convolutional layer include:

- **Filter Count (`filters`)**: Determines the number of feature maps produced by the layer. Each filter learns to detect a specific pattern (e.g., edges, textures, corners, stripes, ...).
- **Kernel Size (`kernel_size`)**: Defines the dimensions of the sliding window (filter). Common sizes include (3,3) or (5,5).
- **Step Length (`strides`)**: Specifies how many pixels the filter moves across the input image. A stride of (1,1) means the filter moves one pixel at a time, resulting in a larger output feature map. Larger strides reduce the output size.
- **Padding (`padding`)**: Controls how the borders of the input are handled.
    - `'valid'`: No padding, meaning the filter only moves across valid input regions. This reduces the output size. This means the border of an image is not 'scanned'
    - `'same'`: Pads the input so that the output feature map has the same dimensions as the input, assuming a stride of (1,1). Meaning: it pretends there are extra Pixels around the border so the output can keep the the same size as the input and no information is lost.

> [!note]  Constraints
> Convolutional layers require and produce data with specific shapes (e.g., `(height, width, channels)` for images).

---
## Pooling Layer

Pooling layers are used to reduce the spatial dimensions (height and width) of the feature maps, which helps in reducing the computational complexity and controlling overfitting.

> [!tip]  Eli5
> **Pooling** is like zooming out on a very detailed photo. We're not trying to keep every little pixel but just the most important parts. And **max-pooling** looks at a section of the given image and asks: "what is the most important pixel (e.g. highest value) here?" and keeps just that one (*see image below*)

![[Pasted image 20250718205304.png]]
> Max pooling with a 2x2 filter and stride = 2
### Max Pooling Layer

The Max Pooling layer is a common type of pooling.
- **Kernel Size (`kernel_size`)**: Similar to convolutional filters, a kernel (e.g., (2,2)) slides over the input.
- **Max Value within the Filter**: For each window defined by the kernel, the layer outputs only the maximum value, discarding the rest.  
    This process downsamples the feature map, making the model more robust to small translations in the input.

![[Pasted image 20250718205927.png]]

> [!question]  Why do we do pooling, if we just reduce the given information?
>- It makes the data **smaller**, which means faster and cheaper computation.
>- It helps the model **focus on the strongest signals**, like corners, edges, or patterns.
>- It adds some **translation robustness** — meaning, if a cat’s ear moves a tiny bit, the network still recognizes it.

---

## Batch Normalization

Batch Normalization is a technique used to improve the training stability and performance of deep neural networks.

Good read: [Batch Norm Explained Visually – How it works, and why neural networks need it](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739/)

``` python
tf.keras.layers.BatchNormalization()
```

![[Pasted image 20250718211503.png]]
> In the example (left) with two features we can sense that they are on drastically different scales. Since the network output is a linear combination of each feature vector, this means that the network learns weights for each feature that are also on different scales. This could lead to the large feature might simply drowning out the small feature.

**Usecase:**
- **Improves Training**: It normalizes the inputs to each layer, effectively standardizing the mean and variance of the activations within a mini-batch.
- **Often Speeds up Training**: By stabilizing the input distribution for each layer, it allows for higher learning rates and faster convergence during training.

> [!tip]
> 
> ### Effect of Batch Normalization
> 
> Batch Normalization addresses the problem of **Internal Covariate Shift**, where the distribution of activations in a network changes as the parameters of previous layers are updated.  
> Imagine your training data points are spread out, forming two distinct clusters (like many dots spread across, some clustered close, forming two groups).  
> Batch Normalization works by rescaling and re-centering these activation distributions, effectively moving all these points closer together within their respective clusters.  
> This makes the optimization landscape smoother and easier for the model to learn.

---

## Dropout

Dropout is a regularization technique primarily used to **reduce overfitting** in neural networks.

Good read: [Unveiling the Dropout Layer: An Essential Tool for Enhancing Neural Networks](https://towardsdatascience.com/unveiling-the-dropout-layer-an-essential-tool-for-enhancing-neural-networks-e090b726561e/)

With dropout, certain nodes in a NN are set to the value zero in a training run, i.e. removed from the network. Thus, they have no influence on the prediction and also in the [backpropagation](https://databasecamp.de/en/ml/backpropagation-basics). Thus, a new, slightly modified network architecture is built in each run and the network learns to produce good predictions without certain inputs.

When installing the dropout layer, a so-called dropout probability must also be specified. This determines how many of the nodes in the layer will be set equal to 0. In the example below we set the dropout probability or also called dropout-rate to $25\%$, so every $4^{th}$ node is set to $0$!


![[Pasted image 20250718212456.png]]

```python
tf.keras.layers.Dropout(rate=0.25)
```

- **Reduces Overfitting**: During training, a specified percentage (`rate`, e.g., 0.25 for 25%) of neurons in a layer are randomly "dropped out" (i.e., their outputs are temporarily ignored), along with their connections.  
    - This forces the network to learn more robust features because it cannot rely on any single neuron or a small set of neurons always being active.
    - In this way, the danger of overfitting can be reduced in deep neural networks, since the neurons do not form a so-called adaptation among themselves, but recognize deeper structures in the data.
    - The dropout layer can be used in the input layer as well as in the hidden layers.
    - However, once the training has been trained out, the dropout layer is no longer used for predictions. However, in order for the model to continue to produce good results, the weights are scaled using the dropout rate.


> [!tip] Why Dropout Works
> Considering a neural network with 8 input and 2 output nodes.  
> If, for instance, 2 input nodes are "crossed out" (*see image above*), the network is forced to find alternative paths and feature combinations to make predictions.  
> This prevents the model from becoming overly dependent on specific features or co-adaptations between neurons, thus improving its generalization capability to unseen data.

---

### What Can We Do Now?

By combining these powerful components...
1. **Convolutional Layers**: for feature extraction - like shapes and edges
2. **Pooling Layers** for dimensionality reduction - to only focus on the hardest edges (and reduce image size, making everything faster) 
3. **Batch Normalization** for stable training -  to keep the learning process stable and fast by standardizing activations
4. and **Dropout** for regularization and preventing overfitting - to make sure all edges are equally important
... we can construct sophisticated deep learning models capable of solving complex tasks, particularly in computer vision.

---

### Example Model Explanation

This `tf.keras.Sequential` model demonstrates a typical Convolutional Neural Network architecture, combining the layers discussed above for a classification task.

```python
model = Sequential()
model.add(InputLayer((28,28,1), name="InputLayer"))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(rate=0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128,
name="HiddenLayer1", activation='relu'))
model.add(Dense(64, name="HiddenLayer2", activation='relu'))
model.add(Dense(2, name="OutputLayer", activation='softmax'))
```

Let’s break down each line:

- `model = Sequential()`: Initializes a sequential model, which is a linear stack of layers.
- `model.add(InputLayer((28,28,1), name="InputLayer"))`: Defines the input shape for the model. Here, it expects input images of size 28×28 pixels with 1 channel (e.g., grayscale images).
- `Conv2D(...)`:
    - Adds a 2D convolutional layer with 128 filters.
    - Each filter has a (3,3) kernel size.
    - `activation='relu'`: Applies the Rectified Linear Unit (ReLU) activation function.
    - `padding='same'`: Keeps the spatial dimensions unchanged.
- `MaxPool2D()`: Adds a Max Pooling layer (default (2,2) kernel and stride).
- `Dropout(rate=0.25)`: Randomly disables 25% of neurons during training to prevent overfitting.
- Another `Conv2D(...)`: Adds a second convolutional layer with 64 filters.
- Another `MaxPool2D()`: Further reduces spatial dimensions.
- `BatchNormalization()`: Normalizes activations for more stable training.
- `Flatten()`: Converts the multi-dimensional output to 1D for the dense layers.
- `Dense(128, ...)`: First fully connected layer with 128 neurons and ReLU.
- `Dense(64, ...)`: Second dense layer with 64 neurons.
- `Dense(2, activation='softmax')`: Output layer with 2 neurons for binary classification.


---

# Optimizer and Hyperparameter

### #TODO


---
## Pre-Trained Models

Pre-trained models are neural networks that have been trained on a very large dataset for a specific task. Instead of building a model from scratch and training it with your own dataset, you can use a pre-trained model as a starting point. This approach is highly beneficial, especially when dealing with limited data or computational resources.

### VGG16 Example

VGG16 is a popular pre-trained Convolutional Neural Network model developed by the Visual Geometry Group at the University of Oxford. It was trained on the ImageNet dataset, which contains millions of images across 1000 different classes (e.g., milk, traffic light, printer).

The architecture of VGG16 (Image 74) typically consists of multiple convolutional layers followed by max-pooling layers, and then a few dense (fully connected) layers at the end. The final dense layer usually has 1000 output classes, corresponding to the 1000 categories in the ImageNet dataset. The input to the VGG16 model is an image of size 224x224x3 (height, width, color channels).

### Leveraging Pre-Trained Models for New Tasks

A powerful technique in deep learning is **transfer learning**, which involves using a pre-trained model as a feature extractor. This means you can take a pre-trained model like VGG16, remove its original classification head (the final dense layers that predict 1000 classes), and then add new layers on top that are specifically designed for your new classification task (Image 75).

The pre-trained convolutional layers of VGG16 are often "frozen" (their weights are not updated during training on the new dataset). This allows the model to leverage the rich feature representations learned from the massive ImageNet dataset. Only the newly added layers are trained, making the process much faster and requiring less data than training a CNN from scratch. The new structure can consist of additional CNN layers or dense layers, depending on the complexity of the new problem.

An example model implementation (Image 76) would typically involve:

- Defining a `Sequential` model in TensorFlow/Keras.
    
- Adding an `Input` layer with the expected image dimensions (224, 224, 3).
    
- Incorporating the pre-trained VGG16 model, setting `weights='imagenet'` to load the pre-trained weights and `include_top=False` to exclude the original classification head.
    
- Adding a `Flatten()` layer to convert the 3D output of the convolutional layers into a 1D vector.
    
- Adding new `Dense` layers for classification, often with `relu` activation for hidden layers and `softmax` activation for the final output layer to predict probabilities across the new number of classes. A `Dropout` layer can be included to prevent overfitting.
    

For instance, if you have 5 new classes, the final `Dense` layer would have 5 units with a `softmax` activation.

> [!example] Klausurrelevant: Transfer Learning Understanding how to use pre-trained models like VGG16 for transfer learning is a crucial concept. Specifically, know why `include_top=False` is used and how new layers are added for a new classification task. This technique is highly effective for tasks with limited datasets.

### Available Pre-Trained Models

TensorFlow provides a wide range of pre-trained models within its `tf.keras.applications` module. These models include popular architectures like ResNet, Inception, MobileNet, and more (Image 77). The ability to easily integrate these models simplifies the process of building powerful deep learning applications.

Furthermore, the concept of using pre-trained models extends beyond those provided by TensorFlow. You can integrate models from your own previous work, models from prior research that have been open-sourced, or any other publicly available model. This fosters collaboration and rapid development in the field of AI.

---

## Important Concepts in Neural Networks

### Feature Extraction

Feature extraction is the process of reducing the number of features in a dataset by creating new, more informative features from the original ones. In the context of CNNs and pre-trained models, the convolutional layers act as powerful feature extractors, learning hierarchical representations of the input data. These extracted features are then fed into the subsequent layers for classification or regression. When using a pre-trained model, the initial layers act as feature extractors for generic visual features, which can then be fine-tuned or used as-is for specific tasks.

### Fitting the Model to New Classes

When using a pre-trained model for a new task, "fitting the model to new classes" refers to the process of training the newly added layers (or fine-tuning some of the pre-trained layers) on your specific dataset. The goal is for the model to learn to map the extracted features to your desired output classes. This is typically done through standard supervised learning techniques, where the model learns from labeled data and adjusts its weights to minimize the prediction error.

### #TODO

