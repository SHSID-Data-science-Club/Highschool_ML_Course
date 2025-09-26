# Dataset

Datasets are undoubtably the fuel of machine learning————regardless of what algorithm you have, you learn from data.
However, it's not just a simple drag and drop task.
Data needs to be preprocessed: converted into the right format, normalized, split into train and validation, etc.
To facilitate this, Pytorch integrates many functions and modules, for example the `Dataset` class.

```python
from torch.utils.data import Dataset
```

`Dataset` is a protocol that behaves similarly as an array. 
There are two types: Map-style and Iterable-style

## Initializing

### Map-style

for random access
```python
class Custom_DS(Dataset):
	def __init__(self, ...): ...
	def __len__(self) -> int: ...
	def __getitem__(self, idx): ...
```

### Iterable-style

for serial access
```python
class Custom_DS(Dataset):
	def __init__(self, ...): ...
	def __len__(self) -> int: ...
	def __iter__(self): ...
```

However, you won't have to manually define all of that.
For example if you are loading a dataset from *hugging face*, it will automatically return you a `Dataset` instance.

There are also some predefined Datasets, the common ones being:

```python
from torch.utils.data import TensorDataset, ConcatDataset, Subset, ChainDataset
```

### TensorDataset

```python
features = torch.randn(1000, 10)      # 1000 samples of 10 features each
labels = torch.randint(0, 2, (1000))  # 1000 0/1 s

dataset = TensorDataset(features, labels)

x = dataset[0]
y = dataset[1]
```

### ConcatDataset

```python
# combines map-style datasets
combined = ConcatDataset([dataset1, dataset2, dataset3])
```

### ChainDataset

```python
# combines iterable-style datasets
combined = ChainDataset([dataset1, dataset2, dataset3])
```

### Subset

```python
# makes a subset of dataset with samples of index ∈ idxs
idxs = [1, 2, 3, 5, 8, 13]
dataset_subset = Subset(dataset, idxs)
```

## Dataset wrapper

If you want to apply transformations to your dataset dynamically, you might write a wrapper for an existing dataset

```python
class Custom_DS(Dataset):
	def __init__(self, dataset, transformation):
		self._dataset = dataset
		self._transformation = transformation
	def __len__(self):
		return len(self._dataset)
	def __getitem__(self, idx):
		item = self._dataset[idx]
		x, y = item["image"], item["label"]
		
		x_transformed = self._transformation(x)
		
		return x_transformed, y
```

## Utilities

### Random_split

Datasets might not come presplit into training, validation, and test sets. By using the `random_split` function you can split one dataset randomly into subsets.

```python
from torch.utils.data import random_split

dataset = ...

train_size = int(0.7 * len(dataset))              # 70% train
val_size = int(0.15 * len(dataset))               # 15% validation
test_size = len(dataset) - train_size - val_size  # the rest are test

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)
```

For reproducibility you can fix the seed. (Computers cannot generate true random numbers, they are just seemingly random numbers that follow a certain distribution (usually uniform). The seed is a parameter that creates variance between different "random" generations attempts; in other words, by fixing the seed you will get a reproducible sequence of "random" numbers)

```python
generator = torch.Generator().manual_seed(42)  # seed=42
train_dataset, val_dataset = random_split(
    dataset, [0.8, 0.2], generator=generator
)
```

the seed is set at 42, because 42 is the answer to the universe.
