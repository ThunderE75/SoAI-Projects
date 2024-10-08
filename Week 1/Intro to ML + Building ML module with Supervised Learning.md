# Intro to ML + Building ML module with Supervised Learning

> The only difference between supervised and unsupervised learning is how the objective function works.

## **What is unsupervised learning?**

In unsupervised learning, we train a model to solve a problem without us knowing the correct answer. In fact, unsupervised learning is typically used for problems where there isn't one correct answer, but instead, better and worse solutions.

In unsupervised learning, the objective function makes its judgment purely on the model's estimate. That means the objective function often needs to be relatively sophisticated.

The only data that we need for unsupervised learning is about features that we provide to the model.

![image.png](../Resorces/Images/Week%201/unsupervised%20learning%20image.png)

## **What is supervised learning?**

Think of supervised learning as learning by example. In supervised learning, we assess the model's performance by comparing its estimates to the correct answer. Although we can have simple objective functions, we need both:

- Features that are provided as inputs to the model
- Labels, which are the correct answers that we want the model to be able to produce

![image.png](../Resorces/Images/Week%201/supervised%20learning%20image.png)

---

## **Error, cost, and loss**

> Supervised learning, error, cost, and loss all refer to the number of mistakes that a model makes in predicting one or more labels.
> 

## **Gradient descent**

Gradient descent uses calculus to estimate how changing each parameter changes the cost.

It calculates the gradient (slope) of the relationship between each model parameter and the cost. 

The parameters are then altered to move down this slope.

> The two main sources of error are local minima and instability.
> 

## **Fitting our model with gradient descent**

The automatic method used the *ordinary least squares* (OLS) method, which is the standard way to fit a line. OLS uses the mean (or sum) of square differences as a cost function. 

# **Summary**

Well done for getting through all of that! Let's recap what we covered:

- Supervised learning is a kind of learning by example. A model makes predictions, the predictions are compared to expected labels, and the model is then updated to produce better results.
- A cost function is a mathematical way to describe what we want a model to learn. Cost functions calculate large numbers when a model isn't making good predictions, and small numbers when it's performing well.
- Gradient descent is an optimization algorithm. It's way of calculating how to improve a model, given a cost function and some data.
- Step size (learning rate) changes how quickly and how well gradient descent performs.