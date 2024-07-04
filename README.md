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

## File Structure

s

```
.
├── README.md
├── tf2
│   ├── Helmholtz2D_Acoustic_Duct.py
│   ├── Interpolate.py
│   ├── PlateWithHole_DEM.py
│   ├── PlateWithHole.py
│   ├── Poisson_DEM_adaptive.py
│   ├── Poisson_DEM.py
│   ├── Poisson_mixed.py
│   ├── Poisson_Neumann_DEM.py
│   ├── Poisson_Neumann.py
│   ├── Poisson.py
│   ├── Poisson2D_Dirichlet_Circle.py
│   ├── Poisson2D_Dirichlet_DEM.py
│   ├── Poisson2D_Dirichlet_SinCos.py
│   ├── Poisson2D_Dirichlet.py
│   ├── ThickCylinder_DEM.py
│   ├── ThickCylinder.py
│   ├── TimonshenkoBeam_DEM.py
│   ├── TimonshenkoBeam.py
│   ├── Wave1D.py
│   └── utils
│       ├── tfp_loss.py
│       ├── scipy_loss.py
│       ├── Geom_examples.py
│       ├── Plotting.py
│       ├── Solvers.py
│       └── Geom.py

```

## Module Descriptions

### Helmholtz2D_Acoustic_Duct.py

### Helmholtz 2D Problem for Acoustic Duct

This section provides a detailed description of the script used to solve the Helmholtz 2D problem for an acoustic duct with specific boundary conditions using a neural network.

#### Overview

The script implements a solution for the 2D Helmholtz equation:

\[ \Delta w(x,y) + k^2 w(x,y) = 0 \quad \text{for} \quad (x,y) \in \Omega := (0,2) \times (0,1) \]

with Neumann and Robin boundary conditions:

- \(\partial u / \partial n = \cos(m\pi x)\), for \(x = 0\)
- \(\partial u / \partial n = -iku\), for \(x = 2\)
- \(\partial u / \partial n = 0\), for \(y=0\) and \(y=1\)

The exact solution is:

\[ u(x,y) = \cos(m\pi y) \left( A_1 \exp(-ik_x x) + A_2 \exp(ik_x x) \right) \]

where \(A_1\) and \(A_2\) are obtained by solving a 2x2 linear system.

#### Steps and Components

1. **Importing Libraries**:

   - Uses `tensorflow`, `numpy`, `time`, `tensorflow_probability`, and `matplotlib.pyplot`.
   - Custom utility modules: `tfp_function_factory`, `Quadrilateral`, `Helmholtz2D_coll`, and `plot_convergence_semilog`.

2. **Setting Random Seed**:

   - Ensures reproducibility by setting the TensorFlow random seed to 42.

3. **Defining Model Parameters**:

   - `m`: mode number, set to 1.
   - `k`: wave number, set to 4.
   - `alpha`: complex coefficient for Robin boundary conditions, calculated as `1j*k`.

4. **Solving for Constants \( A_1 \) and \( A_2 \)**:

   - Computes \( k_x \) and solves a 2x2 linear system to determine \( A_1 \) and \( A_2 \).

5. **Exact Solution and Derivatives**:

   - Defines `exact_sol` and `deriv_exact_sol` functions to compute the exact solution and its derivatives for given \( x \) and \( y \).

6. **Boundary Conditions**:

   - Functions `u_bound_left`, `u_bound_right`, and `u_bound_up_down` define the Neumann and Robin boundary conditions.

7. **Defining Input and Output Data Sets**:

   - Creates a computational domain as a quadrilateral with corners `(0, 0)`, `(0, 1)`, `(2, 0)`, and `(2, 1)`.
   - Generates uniform interior and boundary points for training.

8. **Preparing Training Data**:

   - Generates interior points (`Xint`, `Yint`), Neumann boundary points (`XbndNeu`, `YbndNeu`), and Robin boundary points (`XbndRobin`, `YbndRobin`).

9. **Defining the Model**:

   - Constructs a neural network with four dense layers using TensorFlow, setting the float type to `float64`.
   - Sets up the optimizer (`Adam`) and training parameters.
   - Creates an instance of `Helmholtz2D_coll` to handle the 2D Helmholtz problem.

10. **Training the Model**:

    - Converts the training data to TensorFlow tensors.
    - Trains the model using the Adam optimizer and records the training time.
    - Further refines the model using the BFGS optimizer from TensorFlow Probability and records the training time.

11. **Testing the Model**:

    - Generates test points and computes the predicted solution.
    - Compares the predicted solution with the exact solution, calculating the real and imaginary parts separately.

12. **Plotting Results**:

    - Plots contour maps for the exact solution, computed solution, and the error for both real and imaginary parts.

13. **Computing Errors**:

    - Calculates and prints the L2-error norm for the real and imaginary parts of the solution.

14. **Plotting Loss Convergence**:
    - Plots the loss convergence during training using the Adam optimizer and BFGS optimizer.

This script sets up and trains a neural network to solve the Helmholtz equation in a 2D domain with specific boundary conditions, compares the neural network solution with the exact solution, and visualizes the results and errors.

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
