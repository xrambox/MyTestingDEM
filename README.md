# DeepEnergyMethods

Repository for Deep Learning Methods for solving Partial Differential Equations

Companion paper: https://arxiv.org/abs/1908.10407 or https://doi.org/10.1016/j.cma.2019.112790

Folder tf1 contains the original Tensorflow 1 codes (works with Tensorflow versions up to 1.15).

Folder tf2 contains some examples which are converted to run on Tensorflow 2 (tested with version 2.11).

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# README tf2:

#

This repository contains the implementation of various mathematical and physical simulations using TensorFlow 2.0. Below is a detailed description of each module and the corresponding classes.

## Overview

This repository contains implementations of deep learning methods for solving partial differential equations (PDEs), focusing on various problems including the Helmholtz 2D problem, Poisson equation, and more. The methods are based on TensorFlow 2 and have been tested with version 2.11.

Companion paper: [arXiv:1908.10407](https://arxiv.org/abs/1908.10407) or [DOI:10.1016/j.cma.2019.112790](https://doi.org/10.1016/j.cma.2019.112790).

## Module Descriptions

### Helmholtz2D_Acoustic_Duct.py

### Helmholtz 2D Problem for Acoustic Duct

This section provides a detailed description of the script used to solve a Helmholtz 2D problem for an acoustic duct using Physics-Informed Neural Networks (PINNs). The problem is solved for both the real and imaginary parts of the wave equation with Neumann and Robin boundary conditions.

Problem Statement
We solve the Helmholtz equation:

Δ
𝑤
(
𝑥
,
𝑦
)

- 𝑘
  2
  𝑤
  (
  𝑥
  ,
  𝑦
  )
  =
  0
  for
  (
  𝑥
  ,
  𝑦
  )
  ∈
  Ω
  :
  =
  (
  0
  ,
  2
  )
  ×
  (
  0
  ,
  1
  )
  Δw(x,y)+k
  2
  w(x,y)=0for(x,y)∈Ω:=(0,2)×(0,1)

with the following boundary conditions:

Neumann boundary condition at
𝑥
=
0
x=0:

∂
𝑢
∂
𝑛
=
cos
⁡
(
𝑚
𝜋
𝑥
)
∂n
∂u
​
=cos(mπx)

Robin boundary condition at
𝑥
=
2
x=2:

∂
𝑢
∂
𝑛
=
−
𝑖
𝑘
𝑢
∂n
∂u
​
=−iku

Neumann boundary conditions at
𝑦
=
0
y=0 and
𝑦
=
1
y=1:

∂
𝑢
∂
𝑛
=
0
∂n
∂u
​
=0

The exact solution is given by:

𝑢
(
𝑥
,
𝑦
)
=
cos
⁡
(
𝑚
𝜋
𝑦
)
(
𝐴
1
𝑒
−
𝑖
𝑘
𝑥
𝑥

- 𝐴
  2
  𝑒
  𝑖
  𝑘
  𝑥
  𝑥
  )
  u(x,y)=cos(mπy)(A
  1
  ​
  e
  −ik
  x
  ​
  x
  +A
  2
  ​
  e
  ik
  x
  ​
  x
  )

where
𝐴
1
A
1
​
and
𝐴
2
A
2
​
are constants obtained by solving a linear system.

Code Description
The code is structured as follows:

Define Model Parameters: Sets up the mode number
𝑚
m and wave number
𝑘
k, and computes the constants
𝐴
1
A
1
​
and
𝐴
2
A
2
​
for the exact solution.< br / >

Exact Solution Functions: Defines the exact solution and its derivatives for comparison with the predicted solution.

Boundary Conditions: Defines the boundary conditions for the problem.< br / >

Generate Training Data: Creates the interior points and boundary points for training the model.< br / >

Model Definition: Defines a neural network model using TensorFlow's Keras API with dense layers.< br / >

Training: Trains the model using the Adam optimizer and then fine-tunes it using the L-BFGS optimizer.< br / >

Testing: Tests the trained model and compares it with the exact solution.< br / >

Visualization: Plots the real and imaginary parts of the exact and computed solutions, as well as the error between them. Also plots the convergence of the loss function.

Graphs Generated< br / >
The following graphs are generated during the execution:< br / >

Exact Solution (Real Part):< br / >

Computed Solution (Real Part):< br / >

Error (Real Part):< br / >

Exact Solution (Imaginary Part):< br / >

Computed Solution (Imaginary Part):< br / >

Error (Imaginary Part):

Loss Convergence:< br / >

Results
L2-error norm (Real Part): The L2 error norm for the real part of the solution.< br / >
L2-error norm (Imaginary Part): The L2 error norm for the imaginary part of the solution.

#### Overview

### [Interpolate.py](http://interpolate.py/)

This script demonstrates an interpolation example. It interpolates the function given by `exact_sol(x)` at a discrete set of points.

### PlateWithHole_DEM.py

This module solves the problem of a plate with a hole using the Discrete Element Method (DEM).

### [PlateWithHole.py](http://platewithhole.py/)

This module solves the problem of a plate with a hole using finite element methods.

### Poisson_DEM_adaptive.py

This script solves the Poisson equation using the Discrete Element Method with an adaptive mesh.

### Poisson_DEM.py

This script solves the Poisson equation using the Discrete Element Method.

### Poisson_mixed.py

This module solves the Poisson equation using mixed finite element methods.

### Poisson_Neumann_DEM.py

This module solves the Poisson equation with Neumann boundary conditions using the Discrete Element Method.

### Poisson_Neumann.py

This module solves the Poisson equation with Neumann boundary conditions using finite element methods.

### [Poisson.py](http://poisson.py/)

This module solves the standard Poisson equation using finite element methods.

### Poisson2D_Dirichlet_Circle.py

This module solves the 2D Poisson equation with Dirichlet boundary conditions in a circular domain.

### Poisson2D_Dirichlet_DEM.py

This module solves the 2D Poisson equation with Dirichlet boundary conditions using the Discrete Element Method.

### Poisson2D_Dirichlet_SinCos.py

This module solves the 2D Poisson equation with Dirichlet boundary conditions where the solution is a sine-cosine function.

### Poisson2D_Dirichlet.py

This module solves the 2D Poisson equation with Dirichlet boundary conditions using finite element methods.

### ThickCylinder_DEM.py

This module solves the problem of a thick cylinder using the Discrete Element Method.

### [ThickCylinder.py](http://thickcylinder.py/)

This module solves the problem of a thick cylinder using finite element methods.

### TimonshenkoBeam_DEM.py

This module solves the Timoshenko beam problem using the Discrete Element Method.

### [TimonshenkoBeam.py](http://timonshenkobeam.py/)

This module solves the Timoshenko beam problem using finite element methods.

### [Wave1D.py](http://wave1d.py/)

This script solves the 1D wave equation with Neumann boundary conditions. It includes the exact solution for comparison.

## Installation

### Prerequisites

Ensure you have the following software installed:

- Python 3.8 or higher
- TensorFlow 2.11
- TensorFlow Probability
- NumPy
- Matplotlib
- SciPy

### Steps

1. **Clone the repository:**

   ```
   git clone <https://github.com/yourusername/DeepLearningPDE.git>
   cd DeepLearningPDE

   ```

2. **Create a virtual environment:**

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`

   ```

3. **Install the dependencies:**

   ```
   pip install -r requirements.txt

   ```

If the `requirements.txt` file is not provided, you can manually install the dependencies:

```
pip install tensorflow==2.11 tensorflow-probability numpy matplotlib scipy

```

## Running the Code

To run any of the scripts, navigate to the directory containing the script and run it using Python. For example:

```
cd tf2
python Helmholtz2D_Acoustic_Duct.py

```

### Example: Running `Helmholtz2D_Acoustic_Duct.py`

This script implements the Helmholtz 2D problem for an acoustic duct. The main steps are:

1. Define the model parameters and exact solutions.
2. Generate training data for the Neumann and Robin boundary conditions.
3. Define the neural network model.
4. Train the model using Adam and BFGS optimizers.
5. Test the model and plot the results.

To run the script:

```
python Helmholtz2D_Acoustic_Duct.py

```

### Example: Running `Interpolate.py`

This script provides an example of interpolation. It interpolates the function given by `exact_sol(x)` at a discrete set of points.

To run the script:

```
python Interpolate.py

```

## Detailed Description of `Helmholtz2D_Acoustic_Duct.py`

### Import Statements

```python
import tensorflow as tf
import numpy as np
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from utils.tfp_loss import tfp_function_factory
from utils.Geom_examples import Quadrilateral
from utils.Solvers import Helmholtz2D_coll
from utils.Plotting import plot_convergence_semilog

```

### Model Parameters

- **m**: Mode number
- **k**: Wave number
- **alpha**: Coefficient for Robin boundary conditions

### Exact Solution

```python
def exact_sol(x, y):
    # Implementation of the exact solution

```

### Boundary Conditions

```python
def u_bound_left(x, y):
    # Implementation for left boundary

def u_bound_right(x, y):
    # Implementation for right boundary

def u_bound_up_down(x, y):
    # Implementation for top and bottom boundaries

```

### Data Generation

```python
xmin = 0
xmax = 2
ymin = 0
ymax = 1
domainCorners = np.array([[xmin,ymin], [xmin,ymax], [xmax,ymin], [xmax,ymax]])
myQuad = Quadrilateral(domainCorners)
# Generate points for the domain and boundary conditions

```

### Neural Network Model

```python
l1 = tf.keras.layers.Dense(20, "tanh")
l2 = tf.keras.layers.Dense(20, "tanh")
l3 = tf.keras.layers.Dense(20, "tanh")
l4 = tf.keras.layers.Dense(2, None)
train_op = tf.keras.optimizers.Adam()
pred_model = Helmholtz2D_coll([l1, l2, l3, l4], train_op, num_epoch, print_epoch, k, alpha_real, alpha_imag)

```

### Training and Testing

```python
# Train with Adam optimizer
pred_model.network_learn(Xint_tf, Yint_tf, XbndNeu_tf, YbndNeu_tf, XbndRobin_tf, YbndRobin_tf)

# Train with BFGS optimizer
results = tfp.optimizer.bfgs_minimize(value_and_gradients_function=loss_func, initial_position=init_params, max_iterations=3000, tolerance=1e-14)
loss_func.assign_new_model_parameters(results.position)

# Testing
YTest = pred_model(XTest_tf).numpy()
# Plot results

```

### Plotting

```python
plt.contourf(xPhysTest2D, yPhysTest2D, YExact2D_real, 255, cmap=plt.cm.jet)
plt.colorbar()
plt.title("Exact solution (real)")
plt.show()

```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for

any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Special thanks to all the contributors and the open-source community for their valuable work and resources.
