<div align="center">

# Tensors
##### **Jerry Zhang** | SHSID Data Science Club

<div align="left">

## What??

According to *Wikipedia*, a tensor is an **algebraic object** that describes a **multilinear relationship** between sets of **algebraic objects associated with a vector space**--which is a mouthful. 
Lets break it down.

**Algebraic object**, in this scenario, is simply a matrix.

$$
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix}
$$

**Multilinear relationship**, is a type of linear relationship, in which, multiple independent variables determine a dependent variable. In other words, your operating with a matrix.

$$
\begin{bmatrix}
a & b\\
\end{bmatrix}
\begin{bmatrix}
c\\
d\\
\end{bmatrix}
\\ =
\begin{bmatrix}
ac + bd\\
\end{bmatrix}
$$

**Algebraic objects associated with a vector space**, are literally just vectors.

$$
\begin{bmatrix}
1\\
2\\
3\\
\end{bmatrix}
$$

So a simplified, scenario specific, version of the *Wikipedia* definition would be:
A tensor is a **matrix** which describes a **linear operation** a **vector**.

**Which is literally what a matrix does.**

Therefore, the simplest explanation is...
A Tensor is a Matrix.

$$
\begin{bmatrix}
a & b\\
c & d\\
e & f\\
\end{bmatrix}
\begin{bmatrix}
0.1\\
0.2\\
\end{bmatrix}
\\ =
\begin{bmatrix}
0.1a + 0.2b\\
0.1c + 0.2d\\
0.1e + 0.2f\\
\end{bmatrix}
$$

*note that you may occasionally be using tensors to operate on tensors instead of vectors if you are doing a Computer Vision Task; more on this in the CNN section*

## Pytorch Tensor

Now, that we know what the mathematical definition of a Tensor is, lets look at the Pytorch Definition:
A `torch.Tensor` is a multi-dimensional matrix containing elements of a single data type.

This is basically the same as the mathematical explanation, but with a touch of Computer Science.
Pytorch is partially written using the C language, a low-level programming language which gives Pytorch it's efficiency. Unlike python, C is statically typed, which means variables can only host one datatype during it's lifetime; this is why the matrix may only contain a single data type.

### Creating Tensors

First we have to import the Pytorch module
```
import torch
```

The most direct way would be to create a tensor is from a list
```
data = [[1,2], [3, 4]]
t = torch.tensor(data)
```

It can also be initialized from a NumPy array (another popular matrix object)
```
t = torch.from_numpy(np_array)
```

Sometimes we only want to customize the shape of the matrix
```
t = torch.empty(2, 3)
# note that the values in here are going to be garabge memory 
# you need to overwrite all of them for the tensor to be properly used
```

And there is a way to automatically do that
```
# Fill with zeros
t = torch.zeros(2, 3)

# Fill with ones
t = torch.ones(2, 3)

# Identity matrix ((IdentityMatrix)(Matrix) = Matrix)
t = torch.eye(3)

# Random values
t = torch.rand(2, 3)

# Values sampled from a normal distribution (mean=0, std=1)
t = torch.randn(2, 3)
```

### Datatypes

When creating a tensor, you can also pass in a datatype
```
t = torch.rand(..., dtype=torch.float32)
```
Common datatypes in include `torch.float32`,` torch.float64`, and `torch.float16`
The number behind 'float' indicates how many bits the number takes up.

| Large number of bits                                        | Small number of bits                                                   |
| ----------------------------------------------------------- | ---------------------------------------------------------------------- |
| High precision, which improves potential model capabilities | Less precision, theoretically could impair the potentials of the model |
| Slow to operate with                                        | Fast to operate with                                                   |
| Takes up more space                                         | Takes up less space                                                    |

### Tensor attributes

Useful attributes of a tensor
```
t = torch.eye(3)

t.shape # Shape of the tensor -> (3, 3)
t.size() # Shape of the tensor -> (3, 3)

t.ndim # Number of dimensions -> 2

t.reshape(shape) -> reshaped_tensor # Attempts to return a reshaped version of t
t.squeeze() # Remove dimensions of size 1
t.unsqueeze(dim) # Adds a dimension at index dim
t.transpose(dim0, dim1) # Swaps dim0 and dim1
t.T # Transposes the tensor (only for 2-dimensional tensors)
t.flatten() # Flattens the tensor to 1D
```

### Tensor Indexing

#### Tuple indexing

Unlike python native indexing, where, when addressing multiple dimensions, you do recursive indexing, in Pytorch, you do tuple indexing.
```
m = [[1, 2], [3, 4]]

# Recursive indexing
x = m[1][0]

t = torch.eye(3)

# tuple indexing
x = t[1, 0]
```
Tuple indexing is more efficient as only one function call is necessary to convey all the indices.

#### Slicing and cool stuff

A slice is a compact syntax for representing ranges.

`start:end:step` starting from index `start`, all the way to index `end - 1`, with a step size of `step`

```
t = torch.tensor([0,1,2,3,4,5,6,7,8,9])

t[2:8:1] # -> torch.tensor([2,3,4,5,6,7])
t[2:8:3] # -> torch.tensor([2,5])
```

But you don't have to write out all of the parameters of a slice. They have defaults.

| Syntax  | Meaning            |
| ------- | ------------------ |
| `start` | 0                  |
| `end`   | the last index + 1 |
| `step`  | 1                  |

Some examples
```
t = torch.eye(4)

t[0, -1] # this would be the element at (0, 3), colored in RED
t[:, 0] # -> [1,0,0,0], colored in BLUE
t[1:3, 2:4] # -> torch.tensor([[0, 0], [1, 0]]), colored in GREEN
```

$$
\begin{bmatrix}
	\color{blue}1 & \color{blue}0 & \color{blue}0 & \color{blue}0\\
	0 & 1 & \color{green}0 & \color{green}0\\
	0 & 0 & \color{green}1 & \color{green}0\\
	\color{red}0 & 0 & 0 & 1\\
\end{bmatrix}
$$
