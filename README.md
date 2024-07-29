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

## Code Description

The code is structured as follows:

1. **Define Model Parameters**:

   - Sets up the mode number \( m \) and wave number \( k \).
   - Computes the constants \( A_1 \) and \( A_2 \) for the exact solution.

2. **Exact Solution Functions**:

   - Defines the exact solution and its derivatives for comparison with the predicted solution.

3. **Boundary Conditions**:

   - Defines the boundary conditions for the problem.

4. **Generate Training Data**:

   - Creates the interior points and boundary points for training the model.

5. **Model Definition**:

   - Defines a neural network model using TensorFlow's Keras API with dense layers.

6. **Training**:

   - Trains the model using the Adam optimizer and then fine-tunes it using the L-BFGS optimizer.

7. **Testing**:

   - Tests the trained model and compares it with the exact solution.

8. **Visualization**:
   - Plots the real and imaginary parts of the exact and computed solutions, as well as the error between them.
   - Also plots the convergence of the loss function.

#### Overview

### [Interpolate.py](http://interpolate.py/)

## Interpolation Example

### Description

This module demonstrates how to interpolate a function using a neural network. The code uses Physics-Informed Neural Networks (PINNs) for this purpose. The structure of the code is as follows:

1. **Define the Function to be Interpolated**:

   - Defines the function \( \text{exact_sol}(x) = \sin(k \pi x) \) where \( k = 4 \).

2. **Generate Training Data**:

   - Generates a set of input points \( X \) and their corresponding function values \( Y \) for training.

3. **Model Definition**:

   - Defines a neural network model using TensorFlow's Keras API with dense layers.

4. **Training**:

   - Trains the model using the Adam optimizer and then fine-tunes it using the L-BFGS optimizer.

5. **Testing**:

   - Tests the trained model and compares it with the exact function values.

6. **Visualization**:
   - Plots the interpolated function against the exact function.
   - Plots the error between the interpolated and exact values.
   - Plots the convergence of the loss function.

### Instructions

1. **Function to be Interpolated**:

   - `exact_sol`: Defines the function \( \sin(k \pi x) \) to be interpolated.

2. **Generate Training Data**:

   - Generates 201 points between -1 and 1 for training.

3. **Model Definition**:

   - Defines a neural network with 3 dense layers using the "tanh" activation function.

4. **Training**:

   - Trains the model using the Adam optimizer for 10,000 epochs.
   - Fine-tunes the model using the L-BFGS optimizer.

5. **Testing**:

   - Tests the trained model on 402 points.
   - Compares the model output with the exact function values.

6. **Visualization**:
   - Plots the interpolated function vs. the exact function.
   - Plots the error between the interpolated and exact values.
   - Displays the convergence of the loss function.

### Usage

This module utilizes Physics-Informed Neural Networks (PINNs) to interpolate the given function. PINNs are neural networks that are trained to satisfy physical laws described by differential equations, making them suitable for solving various scientific and engineering problems.

### PlateWithHole_DEM.py

### 2D Linear Elasticity Example with Deep Energy Method

#### Overview

This script demonstrates a 2D linear elasticity problem using the Deep Energy Method (DEM). The goal is to solve the equilibrium equation for a plate with a hole, applying both Dirichlet and Neumann boundary conditions. The plate is situated in the second quadrant and has symmetry boundary conditions on the x and y axes, with pressure boundary conditions on the hole's interior and exterior.

#### Problem Description

- **Domain**: A square plate with a circular hole centered at the origin.
- **Boundary Conditions**:
  - **Dirichlet Conditions**:
    - \( u_x = 0 \) on the x-axis (left boundary).
    - \( u_y = 0 \) on the y-axis (bottom boundary).
  - **Neumann Conditions**:
    - Pressure \( P\_{int} = 10 \text{ MPa} \) on the interior boundary of the hole.
    - Pressure \( P\_{ext} = 0 \text{ MPa} \) on the exterior boundary of the plate.

#### Features

- **Elasticity Model**: Uses Lame constants (\(\mu\) and \(\lambda\)) in the constitutive law.
- **Deep Energy Method (DEM)**: Applies neural networks to approximate the solution of the elasticity problem.

#### Code Description

1. **Imports and Initialization**:

   - Libraries are imported, and random seeds are set for reproducibility.

2. **Elast_PlateWithHole Class**:

   - Inherits from `Elasticity2D_DEM_dist`.
   - Implements Dirichlet boundary conditions.

3. **Exact Solutions**:

   - Functions to compute exact stresses and displacements are defined.

4. **Geometry and Boundary Points**:

   - `PlateWHole` class is used to generate integration points for the interior and boundary of the plate.
   - Exact traction values on the boundaries are computed.

5. **Model Definition and Training**:

   - A neural network model is defined using TensorFlow's `keras` API with three hidden layers.
   - Training is performed using both Adam optimizer and TensorFlow Probability's BFGS optimizer.

6. **Testing and Evaluation**:
   - The model is tested against exact solutions.
   - Displacement and stress fields are plotted, along with errors and convergence plots.

### Method Used

The script utilizes the **Deep Energy Method (DEM)** for solving the problem. This is evident from the usage of neural networks (implemented via TensorFlow) to approximate the solution to the elasticity problem, and the loss function incorporates energy-based terms typical of DEM approaches.

This module solves the problem of a plate with a hole using finite element methods.

### [PlateWithHole.py](http://platewithhole.py/)

### 2D Linear Elasticity with PINNs

## Overview

This script solves a 2D linear elasticity problem using a **Physics-Informed Neural Network (PINN)** approach. The problem is defined by the equilibrium equation of linear elasticity, strain-displacement relations, and the constitutive law. The script uses TensorFlow to train a neural network to approximate the solution and applies both ADAM and BFGS optimizers to achieve this.

## Problem Description

The problem involves a plate with a hole, modeled in the second quadrant. The governing equations are:

1. **Equilibrium Equation**: \(-\nabla \cdot \sigma(x) = f(x)\)
2. **Strain-Displacement Relation**: \(\epsilon = \frac{1}{2}(\nabla u + \nabla u^T)\)
3. **Constitutive Law**: \(\sigma = 2 \mu \epsilon + \lambda (\nabla \cdot u)I\)

Where:

- \(\sigma\) is the stress tensor,
- \(\epsilon\) is the strain tensor,
- \(\mu\) and \(\lambda\) are the Lamé constants,
- \(I\) is the identity tensor.

**Boundary Conditions**:

- **Dirichlet**: \(u(x) = \hat{u}\) on \(\Gamma_D\) (Symmetry conditions on x and y axes)
- **Neumann**: \(\sigma n = \hat{t}\) on \(\Gamma_N\) (Traction conditions on the left and top edges)

## Components

1. **Imports**: Essential libraries include TensorFlow, NumPy, SciPy, and TensorFlow Probability.
2. **Classes and Functions**:

   - `Elast_PlateWithHole`: A subclass of `Elasticity2D_coll_dist` that defines the specific problem and boundary conditions.
   - `cart2pol`, `exact_stresses`, `getExactTraction`, and `exact_disp`: Utility functions to handle geometric transformations, compute exact stresses, and define exact displacements.

3. **Model Setup**:

   - Define material properties, geometry, and boundary conditions.
   - Prepare collocation points and boundary data.
   - Define the neural network model using TensorFlow.

4. **Training**:

   - Use the ADAM optimizer for initial training.
   - Optionally, switch to SciPy's L-BFGS-B or TensorFlow Probability's BFGS for fine-tuning.

5. **Testing and Evaluation**:
   - Compute exact displacements and stresses.
   - Compare computed results with exact solutions.
   - Plot results, including computed displacements, stresses, and errors.

## Results

The script generates:

- Plots of computed displacements and stresses.
- Error plots comparing the neural network's results with exact solutions.
- Loss convergence plots showing the optimization process.

## Conclusion

This script demonstrates the use of Physics-Informed Neural Networks (PINNs) for solving a 2D linear elasticity problem. The approach ensures that the neural network's predictions are consistent with the underlying physical laws and boundary conditions.

---

### Method Used

The provided code uses **Physics-Informed Neural Networks (PINNs)**. This is evident from:

- The incorporation of governing equations (e.g., equilibrium equations and constitutive laws) into the loss function.
- The explicit handling of boundary conditions within the network’s training process.
- The use of TensorFlow for defining and training the neural network, along with an optimization strategy that includes both ADAM and BFGS methods.

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
