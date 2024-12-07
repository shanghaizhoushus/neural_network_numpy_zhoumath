# Neural Network Numpy Zhoumath - Zhoushus (v0.1.2)

This repository contains a custom implementation of nerual network for binary classification using Numpy which includes practical examples for applying the model to real-world datasets.

## Features
- **Custom Neural Network Implementation**: A nerual network model implemented with Numpy, allowing for a deeper understanding of machine learning fundamentals.
- **Optimization Techniques**: Supports kaiming initialization, mini-batch gradient descent, early stopping, learning rate decay, and the Adam optimizer for better performance during training.
- **Example Usage**: A comprehensive example script that demonstrates how to use the model on a real-world dataset from `sklearn`.

## Project Structure
```
├── examples
│   └── nerual_network_zhoumath_example.py  # Example usage script for nerual network
├── scripts
│   ├── nerual_network_zhoumath.py         # Neural network model implementation
│   └── neural_network_helpers_zhoumath.py      # Helper functions for optimization (e.g., Early stopper, Adam optimizer)
├── LICENSE                                     # License information
├── requirements.txt                            # Project dependencies
└── .gitignore                                  # Files and directories to be ignored by git
```

## Installation

To get started, clone the repository and install the required dependencies:

```sh
git clone https://github.com/shanghaizhoushus/neural_network_numpy_zhoumath.git
cd neural_network_numpy_zhoumath
pip install -r requirements.txt
```

## Usage

### Example Script
To see how the neural network model works, you can run the example script:

```sh
python examples/neural_network_helpers_zhoumath.py
```

This script demonstrates the following:
- Loading the breast cancer dataset from `sklearn`.
- Splitting the data into training, validation, and test sets.
- Training the logistic regression model with mini-batch gradient descent and evaluating the performance.

### Nerual Network Implementation
The main implementation can be found in the `scripts/logistic_regression_zhoumath.py` file. Key features include:
- **Gradient Descent**: Supports full-batch and mini-batch training.
- **Kaiming Initialization**: Kaiming initialization is used for the initialization of the linear matrices in the model.
- **Learning Rate Decay**: Gradually reduces the learning rate to improve convergence.
- **Early Stopping**: Stops training if validation performance does not improve for a set number of iterations.
- **Adam Optimizer**: An implementation of the Adam optimization algorithm to enhance model training.

## Dependencies
The project requires the following dependencies, which are listed in `requirements.txt`:
- `numpy`
- `pandas`
- `sklearn`

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue for improvements or bug fixes.

## Version
Current version: **0.1.2**

## Author
- **Zhoushus**
  - Email: [zhoushus@foxmail.com](mailto:zhoushus@foxmail.com)
  - GitHub: [https://github.com/shanghaizhoushus](https://github.com/shanghaizhoushus)

---

Enjoy exploring nerual networks and machine learning with Numpy!
