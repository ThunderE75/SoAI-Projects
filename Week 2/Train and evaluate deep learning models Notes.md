# Train and evaluate deep learning models Notes

## Intro [1]

> *Deep learning* is an advanced form of machine learning that tries to emulate the way the human brain learns.
> 

### Nomenclature

x → The incoming numeric inputs 

- When there's more than one input value,
- ***x*** is considered a vector with elements named ***x1***, ***x2***, and so on.

w → Associated with each ***x*** value is a *weight* (***w***), 

- which is used to strengthen or weaken the effect of the ***x*** value to simulate learning.

*b → bias* (***b***) input is added to enable fine-grained control over the network. 

![image.png](../Resorces/Images/Week%202/Nomenclature.png)

---

> During the training process, the ***w*** and ***b*** values will be adjusted to tune the network so that it "learns" to produce correct outputs.
> 
- The neuron itself encapsulates a **function that calculates a weighted sum of *x*, *w*, and *b*.**
- This function is in turn enclosed in an ***activation function*** that constrains the result 
(often to a value between 0 and 1) to determine **whether or not the neuron passes an output onto the next layer** of neurons in the network.

## **Deep neural network (DNN) concepts** [2]

- A machine learning model is a function that calculates ***y*** (the label) from ***x*** (the features): ***f(x)=y***.
    - y → Answer (the training set has the correct answer which is used to calculate the accuracy of model)
    - x → The inputs that the user provides
    - $f(x)$ → The machine learning model

### S**imple classification example**

The input side 

Specifically, the measurements are: (as inputs)

- The length of the penguin's bill.
- The depth of the penguin's bill.
- The length of the penguin's flipper.
- The penguin's weight.

In this case, the features (***x***) are a vector of four values, or mathematically, ***x***=[x1,x2,x3,x4].

The output (prediction) side 

Let's suppose that the label we're trying to predict (***y***) is the species of the penguin, and that there are three possible species it could be:

*0 → Adelie*

*1 → Gentoo*

*2 → Chinstrap*

---

- In this classification ML model , it must predict the most probable class to which the observation belongs.
- In other words, ***y*** is a vector (enum) of three probability values;
    - one for each of the possible classes: ***y***=[P(0),P(1),P(2)].
- We train the ML model by using test data.
- A perfect classification function should result in a label that indicates a 100% probability for class 0, and a 0% probability for classes 1 and 2:
    - ***y***=[1, 0, 0]

---

### DNN model (aka *Multi-layer Perceptron*)

![image.png](../Resorces/Images/Week%202/DNN%20model%20image.png)

- Example of *fully connected network: Because all the input & hidden layers are connect with EACH node, in the model*

> The deep neural network model for the classifier consists of multiple layers of artificial neurons. ( here 4)
> 
> - An *input* layer with a neuron for each expected input (***x***) value. (thus, 4 inputs)
> - Two so-called *hidden* layers, each containing five neurons. (arbitrary)
> - An *output* layer containing three neurons - one for each class probability (***y***) value to be predicted by the model. (The final class that needs to be determined (enum))
- You can decide the number of hidden layers & number of neurons in them;
- but you have no control over the input and output values for these layers
- these are determined by the model training process.

### **Training a deep neural network**

- Each iteration while training a DNN is called *epochs*
- For the 1st epoch, we assigning random values for the weight (***w***) and bias ***b*** values.
- Then the process is as follows:
    1. Test input data (observations) with known answer (labels) are submitted to the input layer. Generally, these observations are grouped into *batches* (often referred to as *mini-batches*).
    2. The neurons then apply their function, and if activated, pass the result onto the next layer until the output layer produces a prediction.
    3. The prediction is compared to the actual known value, and the amount of variance between the predicted and true values (which we call the *loss*) is calculated.
        
        > Loss = Accurate Answer (from test data set) - Predicted answers (during testing)
        > 
    4. Based on the results, revised values for the weights and bias values are calculated to reduce the loss, and these adjustments are ***backpropagated***   to the neurons in the network layers.
    5. The next epoch repeats the batch training forward pass with the revised weight and bias values, hopefully improving the accuracy of the model (by reducing the loss).

> ❇️ Batch processing of training features makes the training process more efficient by handling multiple observations at once, using matrices of features with weight and bias vectors. Since linear algebra operations are common in 3D graphics, computers with GPUs perform much better than those with only CPUs for deep learning model training.

---

### **A closer look at loss functions and backpropagation**

How do we adjust the values of weight & bias, based on calculated loss & backpropagation

**Calculating loss →** 

- Suppose one of the samples passed through the training process contains features of an *Adelie* specimen (class 0).
- The correct output from the network would be [1, 0, 0].
- Now suppose that the output produced by the network is [0.4, 0.3, 0.3].
- Comparing these, we can calculate an absolute variance for each element
- (in other words, how far is each predicted value away from what it should be) as [0.6, 0.3, 0.3].
- In reality, since we're actually dealing with multiple observations, we typically aggregate the variance -
- for example by squaring the individual variance values and calculating the mean, so we end up with a single, average loss value, like 0.18.

**Optimizers →** 

- Now, here's the clever bit. The loss is calculated using a function, which operates on the results from the final layer of the network, which is also a function.
- The final layer of network operates on the outputs from the previous layers, which are also functions.
- So in effect, the entire model from the input layer right through to the loss calculation is just one big nested function.
- Functions have a few really useful characteristics, including:
    - You can conceptualize a function as a plotted line comparing its output with each of its variables.**
    - You can use differential calculus to calculate the *derivative* of the function at any point with respect to its variables.

We can plot the line of the function to show how an individual weight value compares to loss, and mark on that line the point where the current weight value matches the current loss value.

- The derivative of a function for a given point indicates whether the slope (or *gradient*) of the function output (in this case, loss) is increasing or decreasing with respect to a function variable (in this case, the weight value).

> A positive derivative indicates that the function is increasing, and a negative derivative indicates that it is decreasing.
> 
- In this case, increasing the weight will have the effect of decreasing the loss.

![image.png](../Resorces/Images/Week%202/Loss%20image.png)

- We use an *optimizer* to apply this same trick for all of the weight and bias variables in the model and determine in which direction we need to adjust them (up or down) to reduce the overall amount of loss in the model.
- There are multiple commonly used optimization algorithms, including -
    - *stochastic gradient descent (SGD)*,
    - *Adaptive Learning Rate (ADADELTA)*,
    - *Adaptive Momentum Estimation (Adam)*,
        
        and others; 
        

**Learning rate →** 

- The size of the adjustment is controlled by a parameter that you set for training called the *learning rate*.

> A low learning rate results in small adjustments (so it can take more epochs to minimize the loss), while a high learning rate results in large adjustments (so you might miss the minimum altogether).
> 

---

## **Exercise - Train a deep neural network [3]**

https://learn.microsoft.com/en-us/training/modules/train-evaluate-deep-learn-models/3-exercise-train-deep-neural-network

> 🔵 In Azure AI Studio

---

## **Convolutional neural networks [4]**

- While you can use deep learning models for any kind of machine learning, they're particularly useful for dealing with data that consists of large arrays of numeric values - such as images.
- ML models that work with images are called *computer vision.*
- A CNN typically works by extracting features from images, and then feeding those features into a fully connected neural network to generate a prediction.
- The **feature extraction layers** shrink the huge array of pixel values into a smaller set of features that help predict labels.

### Layers in a CNN

CNNs have multiple layers, with each one focused on extracting features or predicting labels.

1. Convolution layers → 
    - A convolutional layer works by applying a filter to images.
    - The filter is defined by a *kernel* / matrix of weight values.
    
    An image is a matrix of pixel values. To apply a filter, you overlay it on the image and compute the weighted sum of the corresponding pixels. This result is placed in the center of a new 3x3 patch in a matrix the same size as the image.
    
    ```
     1  -1   1
    -1   0  -1
     1  -1   1
    ```
    
    ```
    255 255 255 255 255 255 
    255 255 100 255 255 255
    255 100 100 100 255 255
    100 100 100 100 100 255
    255 255 255 255 255 255
    255 255 255 255 255 255
    ```
    
    ![image.png](../Resorces/Images/Week%202/CNN%20Filters.png)
    
    - Now the **filter is moved along (*convolved*)**, typically using a *step* size of 1 (so moving along one pixel to the right), and the value for the next pixel is calculated
    - The process repeats until we've applied the filter across all of the 3x3 patches of the image to produce a new matrix of values like this:
    - Because of the size of the filter kernel, we can't calculate values for the pixels at the edge; so we typically just apply a *padding* value (often 0):
    
    ```
    ?   ?   ?    ?    ?   ?
    ?  155 -155 155 -155  ?
    ? -155 310 -155  155  ?
    ?  310 155  310   0   ?
    ? -155 -155 -155  0   ?
    ?   ?   ?    ?    ?   ?
    ```
    
    ```
    0   0   0    0    0   0
    0  155 -155 155 -155  0
    0 -155 310 -155  155  0
    0  310 155  310   0   0
    0 -155 -155 -155  0   0
    0   0   0    0    0   0
    ```
    
    > Typically, a convolutional layer applies multiple filter kernels. Each filter produces a different feature map, and all of the feature maps are passed onto the next layer of the network.
    > 

1. Pooling layers →
    - After extracting features from images, pooling layers reduce the number of feature values while keeping the important ones.
    - Max pooling is a common pooling method where a filter is applied to the image, retaining only the maximum pixel value within the filter area. For instance, applying a 2x2 pooling kernel to a patch of an image would yield the result 155.
    - Note that the effect of the 2x2 pooling filter is to reduce the number of values from 4 to 1.

```
0   0
0  155
```

1. **Dropping layers →**
    - One of the most difficult challenges in a CNN is the avoidance of ***overfitting***, where the resulting model performs well with the training data but doesn't generalize well to new data on which it wasn't trained.
    - One technique you can use to mitigate overfitting is to include layers in which the training process randomly eliminates (or "drops") feature maps.
    - Other techniques you can use to mitigate overfitting include randomly flipping, mirroring, or skewing the training images to generate data that varies between training epochs.
2. **Flattening layers →**
    - After using convolutional and pooling layers to extract the important features in the images, the resulting feature maps are multidimensional arrays of pixel values.
    - A flattening layer is used to flatten the feature maps into a vector (array) of values that can be used as input to a fully connected layer.
3. **Fully connected layers →** 
    
    ![image.png](../Resorces/Images/Week%202/Basic%20CNN%20Architecture.png)
    
    > A basic CNN architecture might look similar to this:
    > 
    > 1. Images are fed into a convolutional layer. In this case, there are two filters, so each image produces two feature maps.
    > 2. The feature maps are passed to a pooling layer, where a 2x2 pooling kernel reduces the size of the feature maps.
    > 3. A dropping layer randomly drops some of the feature maps to help prevent overfitting.
    > 4. A flattening layer takes the remaining feature map arrays and flattens them into a vector.
    > 5. The vector elements are fed into a fully connected network, which generates the predictions. In this case, the network is a classification model that predicts probabilities for three possible image classes (triangle, square, and circle).

---

> 🔵 In the case of a CNN, backpropagation of adjusted weights includes filter kernel weights used in convolutional layers as well as the weights used in fully connected layers.

---

## **Exercise - Train a convolutional neural network**

> 🔵 In Azure AI Studio

## **Transfer learning**

![image.png](../Resorces/Images/Week%202/Transfer%20Learning.png)

Conceptually, this neural network consists of two distinct sets of layers:

1. A set of layers from the base model that perform *feature extraction*.
2. A fully connected layer that takes the extracted features and uses them for class *prediction*.

---

- The feature extraction layers apply convolutional filters and pooling to emphasize edges, corners, and other patterns in the images that can be used to differentiate them, and in theory should work for any set of images with the same dimensions as the input layer of the network.
- The prediction layer maps the features to a set of outputs that represent probabilities for each class label you want to use to classify the images.
- By splitting the network into different layers, you can use the feature extraction layers from a pre-trained model and add new layers to predict class labels for your images. This lets you keep the pre-trained weights, so you only need to train the new prediction layers you add.

---

## **Exercise - Use transfer learning**

> 🔵 In Azure AI Studio