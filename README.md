# Neural Network Numpy Zhoumath - Zhoushus (v0.1.0)

This repository contains a custom implementation of logistic regression and nerual networks for binary classification using Numpy, featuring various optimization techniques like gradient descent, mini-batch training, learning rate decay, and the Adam optimizer. The project includes practical examples for applying the model to real-world datasets.

## Features
- **Custom Logistic Regression Implementation**: A logistic regression model implemented with Numpy, allowing for a deeper understanding of machine learning fundamentals.
- **Optimization Techniques**: Supports mini-batch gradient descent, early stopping, learning rate decay, and the Adam optimizer for better performance during training.
- **Example Usage**: A comprehensive example script that demonstrates how to use the model on a real-world dataset from `sklearn`.

## Project Structure
```
├── examples
│   └── logistic_regression_zhoumath_example.py  # Example usage script for logistic regression
├── scripts
│   ├── logistic_regression_zhoumath.py         # Logistic regression model implementation
│   └── neural_network_helpers_zhoumath.py      # Helper functions for optimization (e.g., Adam optimizer)
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
To see how the logistic regression model works, you can run the example script:

```sh
python examples/logistic_regression_zhoumath_example.py
```

This script demonstrates the following:
- Loading the breast cancer dataset from `sklearn`.
- Splitting the data into training, validation, and test sets.
- Training the logistic regression model with mini-batch gradient descent and evaluating the performance.

### Logistic Regression Implementation
The main implementation can be found in the `scripts/logistic_regression_zhoumath.py` file. Key features include:
- **Gradient Descent**: Supports full-batch and mini-batch training.
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
Current version: **0.1.0**

## Author
- **Zhoushus**
  - Email: [zhoushus@foxmail.com](mailto:zhoushus@foxmail.com)
  - GitHub: [https://github.com/shanghaizhoushus](https://github.com/shanghaizhoushus)

---

Enjoy exploring logistic regression and machine learning with Numpy!

