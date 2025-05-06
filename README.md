# The Evolution of Optimizer Algorithms in ML: From GD to Adam

This repository accompanies a microteaching project on the evolution of optimization algorithms in machine learning, from traditional Gradient Descent to advanced methods like Adam.

It investigates the performance of several gradient descent–based optimization algorithms through a hands-on regression task using Python and NumPy. The goal is to demonstrate how different optimizers behave when minimizing a loss function in machine learning, particularly in noisy data.

---

##  Context

This project was created as part of a **Big Data Algorithms** course microteaching assignment. A short video presentation also accompanies it.

---

##  Objective

To compare and analyze the convergence behavior of 7 different optimization strategies:

- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-batch Gradient Descent
- Momentum
- Adagrad
- RMSProp
- Adam

Each optimizer is implemented from scratch (no ML libraries), trained on a synthetic noisy regression dataset, and evaluated using Mean Squared Error (MSE).

---

##  Installation

This project uses only standard Python libraries. To install them, run:

```bash
pip install numpy matplotlib pandas scikit-learn



## Usage

After installing the libraries, open the notebook by running:

```bash
jupyter notebook optimizer_comparison.ipynb


##  Dataset

A synthetic linear regression dataset is generated using `make_regression()` from `sklearn.datasets`, with added Gaussian noise (`noise=40`) to simulate a more challenging and realistic optimization landscape.

All input features and targets are standardized using `StandardScaler` to ensure fair comparison between optimizers, especially those sensitive to gradient scale like Adam and Adagrad.

Alternatively, you may load the saved dataset:
```python
data = pd.read_csv("noisy_regression_data.csv")


## Results

**Final Mean Squared Error (MSE) after 100 training epochs:**

| Optimizer        | Final MSE |
|------------------|-----------|
| Batch GD         | 0.3116    |
| SGD              | 0.3116    |
| Mini-batch GD    | 0.2023    |
| Momentum         | 0.3131    |
| Adagrad          | 0.7281    |
| RMSProp          | 0.3596    |
| Adam             | 0.2270    |

---

## Key Takeaways

-  **Mini-batch GD** and **Adam** offered the best convergence.
-  **Adagrad** performed worst due to its aggressive learning rate decay.
-  **Momentum** did not significantly outperform standard GD on this simple regression task.

## Acknowledgments
Big Data Algorithms course
Code and results fully implemented from scratch in Python/NumPy
Optimizer formulas adapted from Wikipedia
