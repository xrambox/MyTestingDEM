# DeepEnergyMethods

Repository for Deep Learning Methods for solving Partial Differential Equations

Companion paper: https://arxiv.org/abs/1908.10407 or https://doi.org/10.1016/j.cma.2019.112790

Folder tf1 contains the original Tensorflow 1 codes (works with Tensorflow versions up to 1.15).

Folder tf2 contains some examples which are converted to run on Tensorflow 2 (tested with version 2.11).

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# README tf2:

This repository contains the implementation of various mathematical and physical simulations using TensorFlow 2.0. Below is a detailed description of each module and the corresponding classes.

## Overview

This repository contains implementations of deep learning methods for solving partial differential equations (PDEs), focusing on various problems including the Helmholtz 2D problem, Poisson equation, and more. The methods are based on TensorFlow 2 and have been tested with version 2.11.

Companion paper: [arXiv:1908.10407](https://arxiv.org/abs/1908.10407) or [DOI:10.1016/j.cma.2019.112790](https://doi.org/10.1016/j.cma.2019.112790).

## Module Descriptions

# Helmholtz2D_Acoustic_Duct.py

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

# Interpolate.py

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

# PlateWithHole_DEM.py

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

# PlateWithHole.py

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

# Poisson_DEM_adaptive.py

## Overview

This script solves the Poisson equation \(-u''(x) = f(x)\) for \(x \in (a, b)\) with Dirichlet boundary conditions \(u(a)=u0\), \(u(b)=1\). The implementation uses the **Deep Energy Method (DEM)** with an adaptive activation function, as described in the paper [Deep Energy Method](https://doi.org/10.1016/j.jcp.2019.109136).

## Problem Description

The problem involves solving the Poisson equation on a one-dimensional domain with specified boundary conditions. The right-hand side function \(f(x)\) and the exact solution are defined for verification purposes.

## Components

1. **Imports**: Essential libraries include TensorFlow, NumPy, SciPy, and TensorFlow Probability.
2. **Classes and Functions**:

   - `model`: Defines the neural network model, including custom layers, loss functions, and training methods.
   - `generate_quad_pts_weights_1d`: Generates Gauss points and weights for numerical integration.
   - `rhs_fun`, `exact_sol`, and `deriv_exact_sol`: Define the right-hand side function and exact solutions for the Poisson equation.

3. **Model Setup**:

   - Define the neural network architecture using TensorFlow.
   - Implement an adaptive activation function.
   - Define custom loss functions to incorporate the physics of the problem.
   - Use ADAM and BFGS optimizers for training the model.

4. **Training**:

   - Perform initial training with the ADAM optimizer.
   - Fine-tune the model using SciPy's L-BFGS-B optimizer.

5. **Testing and Evaluation**:
   - Compare the neural network's predictions with exact solutions.
   - Plot results, including predicted solutions, exact solutions, and errors.
   - Evaluate the model's performance using error norms.

## Method Used

The provided code uses the **Deep Energy Method (DEM)**, incorporating:

- The energy functional into the loss function.
- Gauss quadrature points and weights for numerical integration.
- An adaptive activation function to improve training efficiency and accuracy.
- TensorFlow for defining and training the neural network, along with optimization strategies using both ADAM and BFGS methods.

## References

- **Deep Energy Method with Adaptive Activation Function**: [https://doi.org/10.1016/j.jcp.2019.109136](https://doi.org/10.1016/j.jcp.2019.109136)

# Poisson_DEM.py

## Overview

This script solves the Poisson equation \(-u''(x) = f(x)\) for \(x \in (a, b)\) with Dirichlet boundary conditions \(u(a)=u0\), \(u(b)=1\). The implementation uses the **Deep Energy Method (DEM)**.

## Problem Description

The problem involves solving the Poisson equation on a one-dimensional domain with specified boundary conditions. The right-hand side function \(f(x)\) and the exact solution are defined for verification purposes.

## Components

1. **Imports**: Essential libraries include TensorFlow, NumPy, SciPy, and TensorFlow Probability.
2. **Classes and Functions**:

   - `model`: Defines the neural network model, including custom layers, loss functions, and training methods.
   - `generate_quad_pts_weights_1d`: Generates Gauss points and weights for numerical integration.
   - `rhs_fun`, `exact_sol`, and `deriv_exact_sol`: Define the right-hand side function and exact solutions for the Poisson equation.

3. **Model Setup**:

   - Define the neural network architecture using TensorFlow.
   - Define custom loss functions to incorporate the physics of the problem.
   - Use ADAM and BFGS optimizers for training the model.

4. **Training**:

   - Perform initial training with the ADAM optimizer.
   - Fine-tune the model using SciPy's L-BFGS-B optimizer.

5. **Testing and Evaluation**:
   - Compare the neural network's predictions with exact solutions.
   - Plot results, including predicted solutions, exact solutions, and errors.
   - Evaluate the model's performance using error norms.

## Method Used

The provided code uses the **Deep Energy Method (DEM)**, incorporating:

- The energy functional into the loss function.
- Gauss quadrature points and weights for numerical integration.
- TensorFlow for defining and training the neural network, along with optimization strategies using both ADAM and BFGS methods.

# Poisson_mixed.py

## Overview

This script solves the Poisson equation \(-u''(x) = f(x)\) for \(x \in (a, b)\) with Dirichlet boundary conditions \(u(a)=u0\), \(u(b)=1\). The implementation uses a **Mixed Physics-Informed Neural Network (PINN)** approach.

## Problem Description

The problem involves solving the Poisson equation on a one-dimensional domain with specified boundary conditions. The right-hand side function \(f(x)\) and the exact solution are defined for verification purposes.

## Components

1. **Imports**: Essential libraries include TensorFlow, NumPy, SciPy, and TensorFlow Probability.
2. **Classes and Functions**:

   - `model`: Defines the neural network model, including custom layers, loss functions, and training methods.
   - `rhs_fun`, `exact_sol`, and `deriv_exact_sol`: Define the right-hand side function and exact solutions for the Poisson equation.

3. **Model Setup**:

   - Define the neural network architecture using TensorFlow.
   - Define custom loss functions to incorporate the physics of the problem.
   - Use ADAM and BFGS optimizers for training the model.

4. **Training**:

   - Perform initial training with the ADAM optimizer.
   - Fine-tune the model using SciPy's L-BFGS-B optimizer.

5. **Testing and Evaluation**:
   - Compare the neural network's predictions with exact solutions.
   - Plot results, including predicted solutions, exact solutions, and errors.
   - Evaluate the model's performance using error norms.

## Method Used

The provided code uses the **Mixed Physics-Informed Neural Network (PINN)** approach, incorporating:

- The physics of the Poisson equation into the loss function.
- TensorFlow for defining and training the neural network, along with optimization strategies using both ADAM and BFGS methods.

# Poisson_Neumann_DEM.py

## Overview

This script solves the Poisson equation \(u''(x) = f(x)\) for \(x \in (a, b)\) with Dirichlet boundary conditions \(u(a)=u0\) and Neumann boundary condition \(u'(b) = u1\). The implementation uses a **Deep Energy Method (DEM)**.

## Problem Description

The problem involves solving the Poisson equation on a one-dimensional domain with specified boundary conditions. The right-hand side function \(f(x)\) and the exact solution are defined for verification purposes.

## Components

1. **Imports**: Essential libraries include TensorFlow, NumPy, TensorFlow Probability, and Matplotlib.
2. **Classes and Functions**:

   - `model`: Defines the neural network model, including custom layers, loss functions, and training methods.
   - `generate_quad_pts_weights_1d`: Generates the Gauss points and weights for numerical integration.
   - `rhs_fun`, `exact_sol`, and `deriv_exact_sol`: Define the right-hand side function and exact solutions for the Poisson equation.

3. **Model Setup**:

   - Define the neural network architecture using TensorFlow.
   - Define custom loss functions to incorporate the physics of the problem using the Deep Energy Method.
   - Use ADAM and BFGS optimizers for training the model.

4. **Training**:

   - Perform initial training with the ADAM optimizer.
   - Fine-tune the model using the BFGS optimizer provided by TensorFlow Probability.

5. **Testing and Evaluation**:
   - Compare the neural network's predictions with exact solutions.
   - Plot results, including predicted solutions, exact solutions, and errors.
   - Evaluate the model's performance using error norms.

## Method Used

The provided code uses the **Deep Energy Method (DEM)** approach, incorporating:

- The energy principles of the Poisson equation into the loss function.
- TensorFlow for defining and training the neural network, along with optimization strategies using both ADAM and BFGS methods.

# Poisson_Neumann.py

## Overview

This script solves the Poisson equation \(-u''(x) = f(x)\) for \(x \in (a, b)\) with Dirichlet boundary conditions \(u(a)=u0\) and Neumann boundary condition \(u'(b) = u1\). The implementation uses **Physics-Informed Neural Networks (PINNs)**.

## Problem Description

The problem involves solving the Poisson equation on a one-dimensional domain with specified boundary conditions. The right-hand side function \(f(x)\) and the exact solution are defined for verification purposes.

## Components

1. **Imports**: Essential libraries include TensorFlow, NumPy, TensorFlow Probability, and Matplotlib.

2. **Classes and Functions**:

   - `model`: Defines the neural network model, including custom layers, loss functions, and training methods. This class handles:
     - The forward pass of the network.
     - Computing derivatives.
     - Custom loss functions that incorporate the Poisson equation and boundary conditions.
     - Training using gradient descent and L-BFGS optimization.
   - `rhs_fun`, `exact_sol`, and `deriv_exact_sol`: Define the right-hand side function and exact solutions for the Poisson equation.

3. **Model Setup**:

   - Define the neural network architecture using TensorFlow with three layers.
   - Set up custom loss functions to include the Poisson equation and boundary conditions.
   - Use ADAM optimizer for initial training and L-BFGS optimizer for fine-tuning.

4. **Training**:

   - Perform initial training using the ADAM optimizer.
   - Fine-tune the model using the L-BFGS optimizer provided by TensorFlow Probability.

5. **Testing and Evaluation**:
   - Compare the neural network's predictions with exact solutions.
   - Plot results, including predicted solutions, exact solutions, and errors.
   - Evaluate the model's performance using error norms for both the solution and its derivative.

## Method Used

The provided code uses **Physics-Informed Neural Networks (PINNs)** approach, incorporating:

- The physics of the Poisson equation directly into the loss function.
- TensorFlow for defining and training the neural network, alongside optimization strategies using both ADAM and L-BFGS methods.

# Poisson2D_Dirichlet_Circle.py

## Overview

This script solves the Poisson equation \(-\Delta u(x, y) = f(x, y)\) for \((x, y) \in \Omega\) with Dirichlet boundary conditions \(u(x, y)=u0\) for \((x, y) \in \partial \Omega\). In this example:

- The domain \(\Omega\) is the unit disk.
- The exact solution is \(u(x, y) = 1 - x^2 - y^2\), corresponding to a constant source term \(f(x, y) = 4\).

The implementation uses **Physics-Informed Neural Networks (PINNs)** to find the solution.

## Problem Description

The problem involves solving the Poisson equation on a circular domain with Dirichlet boundary conditions. The source term \(f(x, y)\) and the exact solution are defined for validation.

## Components

1. **Imports**: Essential libraries include TensorFlow, NumPy, TensorFlow Probability, and Matplotlib.

2. **Classes and Functions**:

   - `rhs_fun`: Defines the source term \(f(x, y) = 4\).
   - `exact_sol`: Provides the exact solution \(u(x, y) = 1 - x^2 - y^2\).
   - `Disk`: Utility class to handle operations related to the circular domain (disk).
   - `Poisson2D_coll`: Class that defines the PINN model for solving the Poisson equation in 2D, including custom layers, loss functions, and training methods.

3. **Model Setup**:

   - Define the neural network architecture using TensorFlow with four layers.
   - Define custom loss functions that incorporate the Poisson equation and boundary conditions.
   - Use ADAM optimizer for initial training and L-BFGS optimizer for fine-tuning.

4. **Training**:

   - Perform initial training using the ADAM optimizer.
   - Fine-tune the model using the L-BFGS optimizer provided by TensorFlow Probability.

5. **Testing and Evaluation**:
   - Compare the neural network's predictions with exact solutions.
   - Plot results, including exact solutions, computed solutions, and errors.
   - Evaluate the model's performance using error norms.

## Method Used

The provided code uses **Physics-Informed Neural Networks (PINNs)** approach, incorporating:

- The physics of the Poisson equation directly into the loss function.
- TensorFlow for defining and training the neural network, alongside optimization strategies using both ADAM and L-BFGS methods.

# Poisson2D_Dirichlet_DEM.py

## Overview

This script solves the Poisson equation \(-\Delta u(x, y) = f(x, y)\) for \((x, y) \in \Omega\) with Dirichlet boundary conditions \(u(x, y) = 0\) for \((x, y) \in \partial \Omega\). For this example:

- The domain \(\Omega\) is a unit square \([0,1] \times [0,1]\).
- The exact solution is \(u(x, y) = x(1-x)y(1-y)\), with the corresponding source term \(f(x, y) = 2(x-x^2 + y-y^2)\).

The implementation uses the **Deep Energy Method (DEM)** for solving the equation.

## Problem Description

The problem involves solving the Poisson equation on a 2D rectangular domain with Dirichlet boundary conditions. The source term \(f(x, y)\) and the exact solution are defined for verification purposes.

## Components

1. **Imports**: Essential libraries include TensorFlow, NumPy, TensorFlow Probability, and Matplotlib.

2. **Classes and Functions**:

   - `rhs_fun`: Defines the source term \(f(x, y) = 2(x-x^2 + y-y^2)\).
   - `exact_sol`: Provides the exact solution \(u(x, y) = x(1-x)y(1-y)\).
   - `deriv_exact_sol`: Computes the derivatives of the exact solution for error analysis.
   - `Quadrilateral`: Utility class to handle operations related to the rectangular domain.
   - `Poisson2D_DEM`: Class that defines the DEM-based model for solving the Poisson equation in 2D, including custom layers, loss functions, and training methods.

3. **Model Setup**:

   - Define the neural network architecture using TensorFlow with four layers.
   - Define custom loss functions that incorporate the Poisson equation and boundary conditions using the Deep Energy Method.
   - Use ADAM optimizer for initial training and BFGS optimizer for fine-tuning.

4. **Training**:

   - Perform initial training using the ADAM optimizer.
   - Fine-tune the model using the BFGS optimizer provided by TensorFlow Probability.

5. **Testing and Evaluation**:
   - Compare the neural network's predictions with exact solutions.
   - Plot results, including exact solutions, computed solutions, and errors.
   - Compute and display relative L2-error and H1-error norms with integration for validation.

## Method Used

The provided code uses the **Deep Energy Method (DEM)** approach, incorporating:

- The energy principles of the Poisson equation into the loss function.
- TensorFlow for defining and training the neural network, with optimization strategies using both ADAM and BFGS methods.

# Poisson2D_Dirichlet_SinCos.py

## Overview

This script solves the Poisson equation \(-\Delta u(x, y) = f(x, y)\) for \((x, y) \in \Omega\) with Dirichlet boundary conditions \(u(x, y) = 0\) for \((x, y) \in \partial \Omega\). Specifically:

- The domain \(\Omega\) is a unit square \([0,1] \times [0,1]\).
- The exact solution is \(u(x, y) = \sin(kx \pi x) \sin(ky \pi y)\) with \(k_x = 1\) and \(k_y = 1\).
- The source term is \(f(x, y) = (k_x^2 + k_y^2) \pi^2 \sin(kx \pi x) \sin(ky \pi y)\).

The implementation uses a **Collocation Method** for solving the equation.

## Problem Description

The problem involves solving the Poisson equation on a 2D rectangular domain with Dirichlet boundary conditions. The source term \(f(x, y)\) and the exact solution are defined for verification purposes.

## Components

1. **Imports**: Essential libraries include TensorFlow, NumPy, TensorFlow Probability, and Matplotlib.

2. **Classes and Functions**:

   - `rhs_fun`: Defines the source term \(f(x, y) = (k_x^2 + k_y^2) \pi^2 \sin(kx \pi x) \sin(ky \pi y)\).
   - `exact_sol`: Provides the exact solution \(u(x, y) = \sin(kx \pi x) \sin(ky \pi y)\).
   - `deriv_exact_sol`: Computes the derivatives of the exact solution for error analysis.
   - `Quadrilateral`: Utility class to handle operations related to the rectangular domain.
   - `Poisson2D_coll`: Class that defines the collocation-based model for solving the Poisson equation in 2D, including custom layers, loss functions, and training methods.

3. **Model Setup**:

   - Define the neural network architecture using TensorFlow with four layers.
   - Define custom loss functions that incorporate the Poisson equation and boundary conditions using the Collocation Method.
   - Use ADAM optimizer for initial training and LBFGS optimizer for fine-tuning.

4. **Training**:

   - Perform initial training using the ADAM optimizer.
   - Fine-tune the model using the LBFGS optimizer provided by TensorFlow Probability.

5. **Testing and Evaluation**:
   - Compare the neural network's predictions with exact solutions.
   - Plot results, including exact solutions, computed solutions, and errors.
   - Compute and display L2-error norm for validation.

## Method Used

The provided code uses the **Collocation Method** approach, incorporating:

- The principles of the Poisson equation into the loss function.
- TensorFlow for defining and training the neural network, with optimization strategies using both ADAM and LBFGS methods.

# Poisson2D_Dirichlet.py

## Overview

This script solves the Poisson equation \(-\Delta u(x, y) = f(x, y)\) for \((x, y) \in \Omega\) with Dirichlet boundary conditions \(u(x, y) = 0\) for \((x, y) \in \partial \Omega\). Specifically:

- The domain \(\Omega\) is a unit square \([0,1] \times [0,1]\).
- The exact solution is \(u(x, y) = x(1-x) y(1-y)\).
- The source term is \(f(x, y) = 2(x - x^2 + y - y^2)\).

The implementation uses a **Collocation Method** for solving the equation.

## Components

1. **Imports**: Essential libraries include TensorFlow, NumPy, TensorFlow Probability, and Matplotlib.

2. **Functions**:

   - `rhs_fun(x, y)`: Defines the source term \(f(x, y) = 2(x - x^2 + y - y^2)\).
   - `exact_sol(x, y)`: Provides the exact solution \(u(x, y) = x(1-x) y(1-y)\).

3. **Data Preparation**:

   - Define the domain and generate the interior and boundary collocation points.
   - Convert these points into TensorFlow tensors for training.

4. **Model Definition**:

   - Define a neural network with four layers using TensorFlow.
   - Utilize the Collocation Method to construct a custom model (`Poisson2D_coll`) that integrates the Poisson equation and boundary conditions into the loss function.

5. **Training**:

   - Perform initial training using the ADAM optimizer.
   - Fine-tune the model using the LBFGS optimizer provided by TensorFlow Probability.

6. **Testing and Evaluation**:
   - Compare the neural network's predictions with exact solutions.
   - Plot results, including exact solutions, computed solutions, and errors.
   - Compute and display the L2-error norm for validation.

## Method Used

The provided code uses the **Collocation Method** approach, incorporating:

- The principles of the Poisson equation into the loss function.
- TensorFlow for defining and training the neural network, with optimization strategies using both ADAM and LBFGS methods.

# ThickCylinder_DEM.py

## Overview

This repository contains a Python script that solves a 2D linear elasticity problem using the Deep Energy Method (DEM). The problem is defined in a quarter annulus domain and involves solving the equilibrium equations, strain-displacement relationships, and constitutive laws for elastic materials.

### Problem Definition

The script addresses the equilibrium equation:
\[ -\nabla \cdot \sigma(x) = f(x) \text{ for } x \in \Omega \]
where \(\Omega\) is a quarter annulus in the first quadrant with:

- Inner radius: 1
- Outer radius: 4

**Boundary Conditions:**

- **Dirichlet Boundary Conditions:**
  - \( u_x(x,y) = 0 \) for \( x = 0 \)
  - \( u_y(x,y) = 0 \) for \( y = 0 \)
- **Neumann Boundary Conditions:**
  - \( \sigma \cdot n = P*{int} \cdot n \) on the interior boundary with \( P*{int} = 10 \text{ MPa} \)
  - \( \sigma \cdot n = P*{ext} \cdot n \) on the exterior boundary with \( P*{ext} = 0 \text{ MPa} \)

### Script Details

1. **Imports and Initialization:**

   - Imports necessary libraries and sets random seeds for reproducibility.

2. **Model Definition:**

   - Defines the `Elast_ThickCylinder` class, inheriting from `Elasticity2D_DEM_dist`, which implements the DEM approach.
   - Uses TensorFlow layers to construct a neural network for solving the elasticity problem.

3. **Geometry and Data Preparation:**

   - Generates the geometry of the quarter annulus and prepares integration points for both the interior and boundary.
   - Defines loading conditions for the boundary points.

4. **Training:**

   - Trains the model using both the Adam optimizer and the TFP-BFGS optimizer.
   - Logs training times and processes.

5. **Testing and Validation:**

   - Compares computed displacements and stresses with exact solutions.
   - Computes and plots errors and stresses to validate the model's accuracy.

6. **Visualization:**
   - Plots computed and exact displacements, stresses, and errors.
   - Plots loss convergence to evaluate training performance.

# ThickCylinder.py

## Overview

This repository contains a Python script that solves a 2D linear elasticity problem in a quarter annulus domain using a neural network approach. The problem is defined with specific boundary conditions and material properties, and the script utilizes TensorFlow and TensorFlow Probability for the implementation.

### Problem Definition

The script addresses the equilibrium equation:
\[ -\nabla \cdot \sigma(x) = f(x) \text{ for } x \in \Omega \]
with:

- **Strain-Displacement Relationship:**
  \[ \epsilon = \frac{1}{2} (\nabla u + \nabla u^T) \]
- **Constitutive Law:**
  \[ \sigma = 2 \mu \epsilon + \lambda (\nabla \cdot u) I \]
  where \(\mu\) and \(\lambda\) are Lame constants, and \(I\) is the identity tensor.

**Boundary Conditions:**

- **Dirichlet Boundary Conditions:**
  - \( u_x(x,y) = 0 \) for \( x = 0 \)
  - \( u_y(x,y) = 0 \) for \( y = 0 \)
- **Neumann Boundary Conditions:**
  - \( \sigma \cdot n = P*{int} \cdot n \) on the interior boundary with \( P*{int} = 10 \text{ MPa} \)
  - \( \sigma \cdot n = P*{ext} \cdot n \) on the exterior boundary with \( P*{ext} = 0 \text{ MPa} \)

### Approach

The script employs a collocation-based approach to solving the 2D linear elasticity problem using a neural network. This method is closely related to Physics-Informed Neural Networks (PINNs) but focuses on collocating points within the domain to enforce the governing equations and boundary conditions.

**Key Components:**

- **Collocation-Based Method**: The `Elasticity2D_coll_dist` class uses a collocation-based approach, where the neural network is trained to satisfy the elasticity equations and boundary conditions at specified points.
- **Neural Network**: The `Elast_ThickCylinder` class, inheriting from `Elasticity2D_coll_dist`, defines the neural network architecture and training procedure.

### Script Details

1. **Imports and Initialization:**

   - Imports necessary libraries and sets random seeds for reproducibility.

2. **Model Definition:**

   - Defines the `Elast_ThickCylinder` class, which extends `Elasticity2D_coll_dist` to implement the neural network for solving the elasticity problem.

3. **Geometry and Data Preparation:**

   - Generates the geometry of the quarter annulus and prepares integration points for both the interior and boundary.
   - Defines loading conditions for the boundary points.

4. **Training:**

   - Trains the model using both the Adam optimizer and TensorFlow Probability’s BFGS optimizer.
   - Logs training times and processes.

5. **Testing and Validation:**

   - Compares computed displacements and stresses with exact solutions.
   - Computes and plots errors and stresses to validate the model’s accuracy.

6. **Visualization:**
   - Plots computed and exact displacements, stresses, and errors.
   - Plots loss convergence to evaluate training performance.

# TimonshenkoBeam_DEM.py

## Overview

This script solves a 2D linear elasticity problem for a Timoshenko beam using a deep neural network approach, specifically employing the Deep Energy Method (DEM). The problem is defined with Dirichlet and Neumann boundary conditions, and the script uses TensorFlow and TensorFlow Probability for the implementation.

### Problem Definition

The script solves the equilibrium equation:
\[ -\nabla \cdot \sigma(x) = f(x) \text{ for } x \in \Omega \]
with:

- **Strain-Displacement Relationship:**
  \[ \epsilon = \frac{1}{2} (\nabla u + \nabla u^T) \]
- **Constitutive Law:**
  \[ \sigma = 2 \mu \epsilon + \lambda (\nabla \cdot u) I \]
  where \(\mu\) and \(\lambda\) are Lame constants, and \(I\) is the identity tensor.

**Boundary Conditions:**

- **Dirichlet Boundary Conditions at \(x=0\):**
  \[ u(x,y) = \frac{P}{6EI} y \left[(2 + \nu) \left(y^2 - \frac{W^2}{4}\right)\right] \]
  \[ v(x,y) = -\frac{P}{6EI} \left[3 \nu y^2 L\right] \]
- **Neumann Boundary Conditions at \(x=8\):**
  \[ p(x,y) = \frac{P}{2I} \left[y^2 - yW\right] \]

where \(P = 2\) (maximum traction), \(E = 1e3\) (Young's modulus), \(\nu = 0.25\) (Poisson ratio), and \(I = \frac{W^3}{12}\) (second moment of area).

### Approach

The script uses the Deep Energy Method (DEM) to solve the 2D elasticity problem. This method integrates boundary conditions and physical laws into the training process of a deep neural network.

**Key Components:**

- **Class `Elast_TimoshenkoBeam`**: Inherits from `Elasticity2D_DEM_dist` and includes boundary conditions specific to the Timoshenko beam problem.
- **Geometry and Data Preparation**: Defines the beam geometry, generates integration points, and prepares boundary conditions.
- **Model Definition**: Uses TensorFlow to define and train a neural network model with the Deep Energy Method.
- **Training**: Trains the model using Adam optimizer and TensorFlow Probability’s BFGS optimizer.
- **Testing and Validation**: Compares computed displacements with exact solutions and computes error metrics.
- **Visualization**: Plots computed displacements, exact displacements, and errors, along with loss convergence.

### Script Details

1. **Imports and Initialization:**

   - Imports necessary libraries and sets random seeds for reproducibility.

2. **Model Definition:**

   - Defines the `Elast_TimoshenkoBeam` class with boundary conditions and neural network architecture.

3. **Geometry and Data Preparation:**

   - Defines beam geometry, generates integration points for the domain and boundary, and prepares data for training.

4. **Training:**

   - Trains the model using both the Adam optimizer and TensorFlow Probability’s BFGS optimizer.
   - Logs training times and processes.

5. **Testing and Validation:**

   - Compares computed displacements with exact solutions and calculates relative L2 error.
   - Plots computed displacements, exact displacements, and errors.

6. **Visualization:**
   - Visualizes results including computed and exact displacements, and error plots.
   - Plots loss convergence during training.

# TimonshenkoBeam.py

## Overview

This script solves a 2D linear elasticity problem for a Timoshenko beam using a deep neural network approach with TensorFlow and TensorFlow Probability. The problem is defined with Dirichlet and Neumann boundary conditions, leveraging the Physics-Informed Neural Network (PINN) approach for solving the elasticity equations.

### Problem Definition

The script addresses the equilibrium equation:
\[ -\nabla \cdot \sigma(x) = f(x) \text{ for } x \in \Omega \]
where:

- **Strain-Displacement Relationship:**
  \[ \epsilon = \frac{1}{2} (\nabla u + \nabla u^T) \]
- **Constitutive Law:**
  \[ \sigma = 2 \mu \epsilon + \lambda (\nabla \cdot u) I \]
  with \(\mu\) and \(\lambda\) as Lame constants, and \(I\) being the identity tensor.

**Boundary Conditions:**

- **Dirichlet Boundary Conditions at \(x=0\):**
  \[ u(x,y) = \frac{P}{6EI} y \left[(2 + \nu) \left(y^2 - \frac{W^2}{4}\right)\right] \]
  \[ v(x,y) = -\frac{P}{6EI} \left[3 \nu y^2 L\right] \]
- **Neumann Boundary Conditions at \(x=8\):**
  \[ p(x,y) = \frac{P}{2I} \left[y^2 - yW\right] \]

where \(P = 2\) (maximum traction), \(E = 1e3\) (Young's modulus), \(\nu = 0.25\) (Poisson ratio), and \(I = \frac{W^3}{12}\) (second moment of area).

### Approach

This script employs the Physics-Informed Neural Network (PINN) approach to integrate boundary conditions and physical laws into the training process of a neural network. The approach combines TensorFlow for model building and TensorFlow Probability for advanced optimization.

**Key Components:**

- **Class `Elast_TimoshenkoBeam`:** Implements boundary conditions and neural network architecture specific to the Timoshenko beam problem.
- **Geometry and Data Preparation:** Defines the beam geometry, generates integration points, and prepares boundary conditions for the training process.
- **Model Definition:** Constructs and trains a neural network using TensorFlow and TensorFlow Probability’s BFGS optimizer.
- **Training:** Utilizes both the Adam optimizer and TensorFlow Probability’s BFGS optimizer for model training.
- **Testing and Validation:** Evaluates the model’s performance by comparing computed displacements with exact solutions and calculating error metrics.
- **Visualization:** Plots computed displacements, exact displacements, errors, and loss convergence during training.

### Script Details

1. **Imports and Initialization:**

   - Imports necessary libraries and sets random seeds for reproducibility.

2. **Model Definition:**

   - Defines the `Elast_TimoshenkoBeam` class, including boundary conditions and neural network layers.

3. **Geometry and Data Preparation:**

   - Defines beam geometry, generates integration points, and prepares data for model training.

4. **Training:**

   - Trains the model using the Adam optimizer and TensorFlow Probability’s BFGS optimizer.
   - Logs training durations and processes.

5. **Testing and Validation:**

   - Compares computed results with exact solutions and computes the relative L2 error.
   - Generates plots for computed and exact displacements and error metrics.

6. **Visualization:**
   - Visualizes results through contour plots of computed displacements, exact displacements, and errors.
   - Plots loss convergence during the training process.

# Wave1D.py

## Overview

This script solves a 1D wave equation problem with Neumann boundary conditions at the left end using a deep neural network approach with TensorFlow and TensorFlow Probability. The problem is addressed using the Physics-Informed Neural Network (PINN) methodology, integrating the governing equations and boundary conditions into the training of the neural network.

## Problem Definition

The script addresses the 1D wave equation:
\[ u*{tt}(x,t) = u*{xx}(x,t) \text{ for } (x,t) \in \Omega \times (0,T) \]
with the exact solution defined as:
\[ u(x,t) = \begin{cases}
\frac{2\alpha}{\pi} & \text{for } x < \alpha (t-1) \\
\frac{\alpha}{\pi} \left[1 - \cos(\pi (t - \frac{x}{\alpha}))\right] & \text{for } \alpha (t-1) \leq x \leq \alpha t \\
0 & \text{for } x > \alpha t
\end{cases} \]
where \(\alpha\) is a constant.

**Boundary Conditions:**

- **Neumann Boundary Conditions at \(x=0\):**
  \[ u_x(0,t) = \begin{cases}
  -\sin(\pi t) & \text{for } 0 \leq t \leq 1 \\
  0 & \text{for } t > 1
  \end{cases} \]

**Initial Conditions:**

- \( u(x,0) = 0 \)
- \( u_t(x,0) = 0 \)

## Approach

The script uses the Physics-Informed Neural Network (PINN) approach to integrate the wave equation and boundary conditions into the training process of a neural network. It leverages TensorFlow for model definition and TensorFlow Probability for advanced optimization.

**Key Components:**

- **Class `Wave1D`:** Implements the neural network model specific to the 1D wave equation problem, including boundary and initial conditions.
- **Geometry and Data Preparation:** Defines the spatial and temporal domain, generates collocation points, and prepares data for training.
- **Model Definition:** Constructs and trains a neural network using TensorFlow and TensorFlow Probability’s BFGS optimizer.
- **Training:** Utilizes both the Adam optimizer and TensorFlow Probability’s BFGS optimizer for model training.
- **Testing and Validation:** Evaluates the model’s performance by comparing computed results with exact solutions and calculating error metrics.
- **Visualization:** Plots computed displacements, velocities, errors, and loss convergence during training.

## Script Details

1. **Imports and Initialization:**

   - Imports necessary libraries and sets random seeds for reproducibility.

2. **Exact Solution Computation:**

   - Defines a function to compute the exact displacement and velocity based on the provided formulas.

3. **Geometry and Data Preparation:**

   - Defines the spatial and temporal domain, generates collocation points for interior and boundary conditions, and prepares initial condition data.

4. **Model Definition:**

   - Constructs the neural network model using TensorFlow layers and sets up the training process with optimizers.

5. **Training:**

   - Trains the model using the Adam optimizer and TensorFlow Probability’s BFGS optimizer.
   - Logs training durations and processes.

6. **Testing and Validation:**

   - Compares computed results with exact solutions and computes the relative L2 error.
   - Generates plots for computed and exact displacements, velocities, and error metrics.

7. **Visualization:**
   - Visualizes results through contour plots of computed and exact displacements and velocities, as well as error metrics.
   - Plots loss convergence during the training process.

````markdown
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

   ```bash
   git clone https://github.com/yourusername/DeepLearningPDE.git
   cd DeepLearningPDE
   ```
````

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**

   If a `requirements.txt` file is provided, use:

   ```bash
   pip install -r requirements.txt
   ```

   If the `requirements.txt` file is not available, you can manually install the dependencies:

   ```bash
   pip install tensorflow==2.11 tensorflow-probability numpy matplotlib scipy
   ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Special thanks to all the contributors and the open-source community for their valuable work and resources.

GIT Push Everyday
GIT push TODAY. yoU ARE aWESOMEEEEEEEEEEEEEEEEEEEEEEEEEEEE

I am Aweeeeeesomeeeeeeeeeeeeeeeeeeeeeeee