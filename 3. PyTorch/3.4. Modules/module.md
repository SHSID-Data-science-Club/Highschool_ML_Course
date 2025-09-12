<div align="center">

# Modules
##### **Jerry Zhang** | SHSID Data Science Club

<div align="left">

## What?

Pytorch follows the python norm of OOP to let you construct neural networks—specifically they let you inherit a prebuilt module, `torch.nn.Module` which is essentially the skeleton of a neural network.

The benefits of inheriting from a base class rather than assembling functions yourself is that it unifies code and hides unnecessary complexity while still giving you the freedom of modification to any degree.

## Usage

`torch.nn` contains most functions / objects needed to construct a neural network
```python
import torch.nn as nn
```

The architecture
```python
class Your_nn(nn.Module):
	def __init__(self, ...):
		# super() refer to the father class, in this case nn.Module
		# __init__ is Module's init, calling it initiallizes Module's features
		# most sources will tell you to write super().__init__() but it's equivilent to super().__init__() in this case
		super().__init__()
		
		# your layers and functions
		...
	
	def forward(self, x):
		"""
		This is the forward feeding function
		You define the structure of your network using the components defined in __init__
		x is the input tensor
		return the output
		"""
		...
```

A sample to better illustrate this
```python
class MNIST_nn(nn.Module):
	def __init__(self):
		super().__init__()
		
		# these will be explained in section 4
		
		# the convolution layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # the fc layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
		# dropout
        self.dropout = nn.Dropout(0.25)
		# normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        # activation functions
        self.relu = nn.ReLU()
	
	def forward(self, x):
		# one layer
		x = self.conv1(x)
		x = self.relu(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        
        # you can also write it in a more compact way
        x = self.bn2(self.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)

		# flattening
        x = x.view(-1, 64 * 7 * 7)
        
        # here is another way to call relu
        # F is from nn.functional
        # this relu is a function while nn.ReLU() is an object
        # they are basically equivilent in terms of computation
        # F.relu is a bit more simplistic
        # nn.ReLU() is more organized and can be integrated into nn.Sequential
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
```

main function
```python
if __name__ == "__main__":
	model = Your_nn(...)

	for epoch in range(epochs):
		loss = model(input)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
```

## Side facts about the hidden complexity

you might wonder how backpropagation is done

### Auto param registration

there is a hidden attribute called `self._parameters` which is an `OrderedDict`. Whenever a learnable module is defined, it is auto added to it.
This is done via the `__setattr__` dunder method

```python
# rough idea
# whenever an attribute is added to the object this is invoked
def __setattr__(self, name, value):
	# all learnable modules are a child class of nn.Parameter
    if isinstance(value, nn.Parameter):
	    # all learnable modules are a child class of nn.Parameter
        self.register_parameter(name, value)
    elif isinstance(value, nn.Module):
	    # this is the case when, for example, nn.Sequential is added
	    # allowing for the formation of an organized tree architecture
        self._modules[name] = value
    ...
    # finally, the original purose of self.a = b must be fulfilled
    object.__setattr__(self, name, value)
```

for backpropagation
```python
class Module:
	# these two methods expose the parameters to the optimizer
    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            yield param
    
    def named_parameters(self, prefix='', recurse=True):
        # Recursively yields all parameters with names
        # Enables optimizer access: optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

## Hooks

hooks are functions that give you access points to data throughout the entire training process: the inputs, outputs, and the gradients of all modules

### types of hooks

```python
# Forward pre-hook: called before forward()
def forward_pre_hook(module, input):
    # input is a tuple of inputs to the module
    print(f"Module {module.__class__.__name__} received input: {[i.shape for i in input]}")
    # You can modify the input here
    return input  # Or modified input

# Forward hook: called after forward()
def forward_hook(module, input, output):
    # output is the result of forward()
    print(f"Module {module.__class__.__name__} produced output: {output.shape}")
    # You can modify the output here
    return output  # Or modified output

# called during backprop
def backward_hook(module, grad_input, grad_output):
    # grad_input: gradients flowing INTO the module
    # grad_output: gradients flowing OUT OF the module
    print(f"Gradients flowing out: {[g.shape for g in grad_output if g is not None]}")
    # Can be used for gradient clipping or monitoring
    return grad_input
```

### registering hooks

```python
module.register_forward_pre_hook(hook)
module.register_forward_hook(hook)
module.register_full_backwards_hook(hook)

# you can manually assign
# this is adding a hook to the initial input and final output
model.register_forward_prehook(...)
model.register_forward_hook(...)

# or add hooks at mass
for name, module in model.named_modules():
	# for all linear modules within model
    if isinstance(module, nn.Linear):
        handle = module.register_forward_hook(...)
```

with these you can monitor values, clip gradients, etc.

### common hooks

```python
# Gradient clipping using backward hook
def gradient_clipping_hook(module, grad_input, grad_output, max_norm=1.0):
    # Clip gradients to prevent explosion
    total_norm = 0
    for g in grad_output:
        if g is not None:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grad_output:
            if g is not None:
                g.data.mul_(clip_coef)
    
    return grad_input

# Register to specific layers
for module in model.modules():
    if isinstance(module, nn.Linear):
        module.register_full_backward_hook(gradient_clipping_hook)
```

```python
# Monitor activation statistics during training
activation_stats = {}

def activation_stats_hook(name):
    def hook(module, input, output):
        if name not in activation_stats:
            activation_stats[name] = {
                'mean': [], 'std': [], 'min': [], 'max': []
            }
        
        activation_stats[name]['mean'].append(output.mean().item())
        activation_stats[name]['std'].append(output.std().item())
        activation_stats[name]['min'].append(output.min().item())
        activation_stats[name]['max'].append(output.max().item())
    return hook

# Register to all convolutional layers
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        module.register_forward_hook(activation_stats_hook(name))
```

```python
# Identify dead ReLU units
dead_relus = {}

def relu_monitor_hook(name):
    def hook(module, input, output):
        # Count how many outputs are exactly zero
        dead_ratio = (output == 0).float().mean().item()
        if name not in dead_relus:
            dead_relus[name] = []
        dead_relus[name].append(dead_ratio)
    return hook

# Monitor all ReLU layers
for name, module in model.named_modules():
    if isinstance(module, nn.ReLU):
        module.register_forward_hook(relu_monitor_hook(name))
```
