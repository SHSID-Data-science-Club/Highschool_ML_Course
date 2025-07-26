<div align="center">
  
# Tensors
##### **Jerry Zhang** | SHSID Data Science Club

<div align="left">

## Hmm?
Recalling from the previous chapters, a key concept in Machine Learning is Backpropagation--calculating derivatives of individual elements in tensors. Backpropagation is essential for calculating the gradients used in gradient descent. 

<div align="center">
  <img src="4.2 Autograd/gd.png" width="430px" style="border-radius:8px;">
</div>

Calculating the gradients by hand is inefficient and unnecessary. Which is where Autograd comes in.

*Note that a gradient is a value derived from the derivatives; it's how you USE the derivatives. Changing how you calculate the gradient results in different optimizing behavior; more on this in the Optimizers chapter*

## Usage
Autograd is a Pytorch's auto-differentiation engine, that is, it traces a tensor's contribution to some result and calculates the gradient accordingly.

```
# 'requires_grad' turns on Autograd
x = torch.tensor([1., 2., 3.], requires_grad=True)

y = x**2 + 3*x + 5

# Backpropagation
# The tensor passed in is the gradient tensor
# Since each individual element in the tensor contributes to many output values, a weighted sum is taken
# The gradient tensor holds the weights of the weighted sum
y.backward(torch.tensor[1., 1., 1.])

# The resulting gradients are stored their respective tensors
x.grad # -> torch.tensor([5., 7., 9.])

# Resetting the gradients
x.grad.zero_()
# Usually though, you would not call grad.zero_(), more on this in the optimizer section.
```

Sometimes, for example when you are simply evaluating your model, you don't want to track the gradients as it affects performance.
There are two ways to achieve this.
```
x = torch.tensor([1., 2., 3.,], requires_grad=True)

x.detach() # Not recommended

with torch.no_grad():
	# Recommended
	# Only the code within this block won't be tracked
	# In other words, it's a temporary detach
	y = x**10 + 114
```

## True Usage

In most cases though, you will be using a optimizer to do the backpropagation; more on this in the Optimizer chapter.  In this case, you code would resemble something like this.

```
# This line runs your model and returns the loss
loss = model(input)
# Backpropagating the contributions
loss.backward()
# Calculating the gradients and updating the weights
optimizer.step()
# Clearning the gradients
optimizer.zero_grad()
```
