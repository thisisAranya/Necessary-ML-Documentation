# NumPy Complete Reference for Machine Learning and Scientific Computing

A comprehensive guide with code examples, descriptions, and outputs for scientific computing and machine learning with NumPy.

## Table of Contents
- [1. Creating Arrays](#1-creating-arrays)
- [2. Array Attributes](#2-array-attributes)
- [3. Reshaping and Flattening](#3-reshaping-and-flattening)
- [4. Indexing and Slicing](#4-indexing-and-slicing)
- [5. Mathematical Operations](#5-mathematical-operations)
- [6. Aggregations](#6-aggregations)
- [7. Broadcasting](#7-broadcasting)
- [8. Boolean Masking](#8-boolean-masking)
- [9. Random Module](#9-random-module)
- [10. Linear Algebra](#10-linear-algebra)
- [11. Saving and Loading](#11-saving-and-loading)
- [12. Stacking Arrays](#12-stacking-arrays)
- [13. Splitting Arrays](#13-splitting-arrays)
- [14. Copy vs View](#14-copy-vs-view)
- [15. Advanced Indexing](#15-advanced-indexing)
- [16. Where and Conditionals](#16-where-and-conditionals)
- [17. Unique Elements](#17-unique-elements)
- [18. Sorting](#18-sorting)
- [19. Tiling and Repeating](#19-tiling-and-repeating)
- [20. Meshgrid](#20-meshgrid)
- [21. Datetime](#21-datetime)
- [22. Masked Arrays and NaNs](#22-masked-arrays-and-nans)
- [23. Performance Tools](#23-performance-tools)
- [24. Index Tricks](#24-index-tricks)

## 1. Creating Arrays
```python
import numpy as np

# Basic array creation
arr1 = np.array([1, 2, 3])
arr2 = np.array([[1, 2], [3, 4]])

# Arrays with predefined values
zeros = np.zeros((2, 3))
ones = np.ones((3, 2))
empty = np.empty((2, 2))  # Uninitialized (values depend on memory state)
range_arr = np.arange(0, 10, 2)  # Start, stop, step
linspace_arr = np.linspace(0, 1, 5)  # Start, stop, num-points
identity = np.eye(3)  # Identity matrix
full_arr = np.full((2, 2), 7)  # Fill with specific value
```

**Output:**
```
arr1: [1 2 3]
arr2: [[1 2]
       [3 4]]
zeros: [[0. 0. 0.]
        [0. 0. 0.]]
ones: [[1. 1.]
       [1. 1.]
       [1. 1.]]
empty: [[...]] # Values are unpredictable
range_arr: [0 2 4 6 8]
linspace_arr: [0.   0.25 0.5  0.75 1.  ]
identity: [[1. 0. 0.]
           [0. 1. 0.]
           [0. 0. 1.]]
full_arr: [[7 7]
           [7 7]]
```

## 2. Array Attributes
```python
# Finding array properties
shape = arr2.shape  # Dimensions of the array
dim = arr2.ndim     # Number of dimensions
dtype = arr2.dtype  # Data type of elements
size = arr2.size    # Total number of elements
```

**Output:**
```
shape: (2, 2)
dim: 2
dtype: int64 (may vary based on system)
size: 4
```

## 3. Reshaping and Flattening
```python
# Change array shape while preserving data
reshaped = arr2.reshape((4, 1))  # New shape with same total elements
flattened = arr2.flatten()       # Create a copy as 1D array
transposed = arr2.T              # Transpose dimensions
raveled = arr2.ravel()           # Return a view as 1D array when possible
```

**Output:**
```
reshaped: [[1]
           [2]
           [3]
           [4]]
flattened: [1 2 3 4]
transposed: [[1 3]
             [2 4]]
raveled: [1 2 3 4]
```

## 4. Indexing and Slicing
```python
# Access elements or subarrays
item = arr2[0, 1]  # Element at row 0, col 1
row = arr2[0, :]   # First row
col = arr2[:, 1]   # Second column
```

**Output:**
```
item: 2
row: [1 2]
col: [2 4]
```

## 5. Mathematical Operations
```python
# Element-wise operations
add = arr1 + 2        # [3, 4, 5]
sub = arr1 - 1        # [0, 1, 2]
mul = arr1 * 3        # [3, 6, 9]
div = arr1 / 2        # [0.5, 1.0, 1.5]
mod = arr1 % 2        # [1, 0, 1]

# Math functions (all element-wise)
sqrt = np.sqrt(arr1)  # Square root
power = np.power(arr1, 2)  # Raise to power
exp = np.exp(arr1)    # e^x
log = np.log(arr1)    # Natural logarithm
```

**Output:**
```
add: [3 4 5]
sub: [0 1 2]
mul: [3 6 9]
div: [0.5 1.  1.5]
mod: [1 0 1]
sqrt: [1.         1.41421356 1.73205081]
power: [1 4 9]
exp: [ 2.71828183  7.3890561  20.08553692]
log: [0.         0.69314718 1.09861229]
```

## 6. Aggregations
```python
# Statistical operations
sum_all = arr2.sum()     # Sum of all elements
mean_all = arr2.mean()   # Mean of all elements
std_all = arr2.std()     # Standard deviation
var_all = arr2.var()     # Variance
min_val = arr2.min()     # Minimum value
max_val = arr2.max()     # Maximum value
argmin = arr2.argmin()   # Index of minimum value (flattened)
argmax = arr2.argmax()   # Index of maximum value (flattened)
```

**Output:**
```
sum_all: 10
mean_all: 2.5
std_all: 1.118033988749895
var_all: 1.25
min_val: 1
max_val: 4
argmin: 0
argmax: 3
```

## 7. Broadcasting
```python
# Automatic size matching for operations
arr3 = np.array([[1], [2], [3]])  # Shape (3, 1)
broadcasted = arr3 + arr1         # arr1 is (3,), result is (3, 3)
```

**Output:**
```
arr3: [[1]
       [2]
       [3]]
broadcasted: [[2 3 4]
              [3 4 5]
              [4 5 6]]
```

## 8. Boolean Masking
```python
# Filtering with boolean conditions
bool_mask = arr1 > 1      # Create boolean mask
filtered = arr1[bool_mask] # Select where True
```

**Output:**
```
bool_mask: [False True True]
filtered: [2 3]
```

## 9. Random Module
```python
# Random number generation
rand_uniform = np.random.rand(3, 3)         # Uniform [0,1)
rand_normal = np.random.randn(3, 3)         # Standard normal
rand_int = np.random.randint(0, 10, (2, 3)) # Random integers
np.random.seed(42)                          # Set random seed
choice_arr = np.random.choice([10, 20, 30], size=5)  # Random choices
```

**Output:**
```
rand_uniform: [[0.37454012 0.95071431 0.73199394]
               [0.59865848 0.15601864 0.15599452]
               [0.05808361 0.86617615 0.60111501]]
               
rand_normal: [[ 1.76405235  0.40015721  0.97873798]
              [ 2.2408932   1.86755799 -0.97727788]
              [ 0.95008842 -0.15135721 -0.10321885]]
              
rand_int: [[3 0 9]
           [3 5 2]]
           
choice_arr: [10 30 30 30 10]
```

## 10. Linear Algebra
```python
# Matrix operations
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])
dot_product = np.dot(mat1, mat2)     # Matrix multiplication
matmul = np.matmul(mat1, mat2)       # Also matrix multiplication
inv = np.linalg.inv(mat1)            # Matrix inverse
det = np.linalg.det(mat1)            # Determinant
eigvals, eigvecs = np.linalg.eig(mat1)  # Eigenvalues and eigenvectors
solve = np.linalg.solve(mat1, np.array([1, 2]))  # Solve linear equation
```

**Output:**
```
dot_product: [[19 22]
              [43 50]]
matmul: [[19 22]
         [43 50]]
inv: [[-2.   1. ]
      [ 1.5 -0.5]]
det: -2.0000000000000004
eigvals: [-0.37228132  5.37228132]
eigvecs: [[-0.82456484 -0.41597356]
          [ 0.56576746 -0.90937671]]
solve: [-1.  1.]
```

## 11. Saving and Loading
```python
# Persistent storage
np.save('array.npy', arr1)                 # Binary format
loaded = np.load('array.npy')              # Load binary
np.savetxt('array.txt', arr1)              # Text format
loaded_txt = np.loadtxt('array.txt')       # Load text
```

**Output:**
```
loaded: [1 2 3]
loaded_txt: [1. 2. 3.]
```

## 12. Stacking Arrays
```python
# Combining arrays
stacked_v = np.vstack([arr1, arr1])      # Vertical stack
stacked_h = np.hstack([arr1, arr1])      # Horizontal stack
concatenated = np.concatenate([arr1, arr1])  # Generic concatenation
```

**Output:**
```
stacked_v: [[1 2 3]
            [1 2 3]]
stacked_h: [1 2 3 1 2 3]
concatenated: [1 2 3 1 2 3]
```

## 13. Splitting Arrays
```python
# Dividing arrays
split = np.array_split(arr1, 3)    # Split into sections (handles uneven)
hsplit = np.hsplit(arr2, 2)        # Horizontal split
vsplit = np.vsplit(arr2, 2)        # Vertical split
```

**Output:**
```
split: [array([1]), array([2]), array([3])]
hsplit: [array([[1],
               [3]]), array([[2],
                            [4]])]
vsplit: [array([[1, 2]]), array([[3, 4]])]
```

## 14. Copy vs View
```python
# Memory behavior
copy_arr = arr1.copy()  # Deep copy (new memory)
view_arr = arr1.view()  # View (shares memory)
```

**Output:**
```
copy_arr: [1 2 3]
view_arr: [1 2 3]
# If arr1 changes:
# - copy_arr stays the same
# - view_arr changes with arr1
```

## 15. Advanced Indexing
```python
# Complex selection patterns
indices = [0, 2]
selected = arr1[indices]  # Select elements by index list
```

**Output:**
```
selected: [1 3]
```

## 16. Where and Conditionals
```python
# Conditional operations
where_arr = np.where(arr1 > 1, arr1, 0)  # Choose based on condition
condition = np.logical_and(arr1 > 0, arr1 < 3)  # Combine conditions
```

**Output:**
```
where_arr: [0 2 3]
condition: [ True  True False]
```

## 17. Unique Elements
```python
# Set operations
unique_vals = np.unique(arr1)    # Unique values
in1d = np.in1d(arr1, [1, 3])     # Test membership
intersect = np.intersect1d(arr1, [1, 3])  # Common elements
union = np.union1d(arr1, [4, 5])  # Combined unique elements
diff = np.setdiff1d(arr1, [1, 3])  # Elements in first but not second
```

**Output:**
```
unique_vals: [1 2 3]
in1d: [ True False  True]
intersect: [1 3]
union: [1 2 3 4 5]
diff: [2]
```

## 18. Sorting
```python
# Ordering elements
sorted_arr = np.sort(arr1)      # Sort values
argsorted = np.argsort(arr1)    # Get sorting indices
```

**Output:**
```
sorted_arr: [1 2 3]
argsorted: [0 1 2]
```

## 19. Tiling and Repeating
```python
# Replicating arrays
tiled = np.tile(arr1, (2, 1))   # Repeat whole array
repeated = np.repeat(arr1, 2)   # Repeat each element
```

**Output:**
```
tiled: [[1 2 3]
        [1 2 3]]
repeated: [1 1 2 2 3 3]
```

## 20. Meshgrid
```python
# Create coordinate matrices (used in plotting/contour)
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
xv, yv = np.meshgrid(x, y)  # Create coordinate matrices
```

**Output:**
```
x: [-5.         -3.88888889 -2.77777778 -1.66666667 -0.55555556  0.55555556
  1.66666667  2.77777778  3.88888889  5.        ]
y: [-5.         -3.88888889 -2.77777778 -1.66666667 -0.55555556  0.55555556
  1.66666667  2.77777778  3.88888889  5.        ]
xv: [[-5.         -3.88888889 -2.77777778 ... 2.77777778  3.88888889  5.        ]
     [-5.         -3.88888889 -2.77777778 ... 2.77777778  3.88888889  5.        ]
     ...
     [-5.         -3.88888889 -2.77777778 ... 2.77777778  3.88888889  5.        ]]
yv: [[-5.         -5.         -5.         ... -5.         -5.         -5.        ]
     [-3.88888889 -3.88888889 -3.88888889 ... -3.88888889 -3.88888889 -3.88888889]
     ...
     [ 5.          5.          5.         ...  5.          5.          5.        ]]
```

## 21. Datetime
```python
# Date and time handling
date1 = np.datetime64('2023-01-01')  # Create datetime
date_range = date1 + np.arange(10)   # Date series
duration = np.timedelta64(5, 'D')    # Time duration (5 days)
```

**Output:**
```
date1: 2023-01-01
date_range: ['2023-01-01' '2023-01-02' '2023-01-03' '2023-01-04' '2023-01-05'
 '2023-01-06' '2023-01-07' '2023-01-08' '2023-01-09' '2023-01-10']
duration: 5 days
```

## 22. Masked Arrays and NaNs
```python
# Handling missing data
nan_arr = np.array([1, np.nan, 3])  # Array with NaN
clean_mean = np.nanmean(nan_arr)    # Mean ignoring NaNs
mask = np.ma.masked_array(arr1, mask=[0, 1, 0])  # Mask specific values
```

**Output:**
```
nan_arr: [ 1. nan  3.]
clean_mean: 2.0
mask: [1 -- 3]
```

## 23. Performance Tools
```python
# Optimizing operations
vec_func = np.vectorize(lambda x: x**2)  # Vectorize a scalar function
from_func = np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)  # Generate from function
```

**Output:**
```
vec_func(arr1): [1 4 9]
from_func: [[0 1 2]
            [1 2 3]
            [2 3 4]]
```

## 24. Index Tricks
```python
# Advanced indexing techniques
row_indices = np.array([0, 1])
col_indices = np.array([1, 0])
advanced_selected = arr2[row_indices, col_indices]  # Corresponding pairs

gridded = np.ix_([0, 1], [0, 1])  # Mesh indices for selection
r_concat = np.r_[1:4, 0, 4]       # Row-wise concatenate
c_concat = np.c_[np.array([1, 2]), np.array([3, 4])]  # Column-wise concatenate
```

**Output:**
```
advanced_selected: [2 3]
gridded: (array([[0],
                [1]]), array([[0, 1]]))
r_concat: [1 2 3 0 4]
c_concat: [[1 3]
           [2 4]]
```

## Additional Notes

- NumPy operations are optimized for performance on large arrays
- For many operations, you can specify `axis` parameter to apply along rows or columns
- Use `np.newaxis` or `None` to add dimensions for broadcasting
- Most NumPy functions have a corresponding method on ndarray objects
- Use `astype()` to convert arrays to different data types
- NumPy's random module provides extensive random sampling capabilities

---

This reference document covers the essential NumPy operations for scientific computing and machine learning, with practical examples and expected outputs. Use it as a quick reference guide for your data science projects.
