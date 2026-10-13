# Reproducing the Perceptron: From Rosenblatt to Modern ML

## Overview

This project implements a classical perceptron classifier from scratch using Python and NumPy, inspired by Frank Rosenblatt's 1958 paper, *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*.

The objective is to understand the fundamentals of learning through adjustable weights, binary threshold activation, and error-driven parameter updates.

**Scope:** This is a reproduction of the classical perceptron learning algorithm, not a complete reconstruction of Rosenblatt's original probabilistic nervous-system model.

## Research questions

1. Can a perceptron learn linearly separable Boolean functions?
2. How does the learning rate affect training?
3. Why can a single perceptron not solve XOR?
4. How does a from-scratch implementation compare with a modern library implementation?

## Implementation

- NumPy implementation of prediction and weight updates
- AND and OR classification experiments
- XOR limitation experiment
- Decision-boundary and learning-curve visualizations
- Unit tests for linearly separable datasets


## Connection to modern deep learning

The perceptron demonstrates weighted inputs, trainable parameters, activation functions, and data-driven learning. Modern deep neural networks extend these ideas with multiple layers, nonlinear representations, differentiable loss functions, and backpropagation.

## Limitations

The implementation uses a single linear decision boundary and cannot learn non-linearly separable functions such as XOR. It is not a full reproduction of the original 1958 theory or its historical experimental setup.

## Reference

Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review, 65*(6), 386–408.

https://doi.org/10.1037/h0042519