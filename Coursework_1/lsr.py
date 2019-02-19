# -*- coding: utf-8 -*-
import sys

import numpy as np
from matplotlib import pyplot as plt

import utilities


def least_squares_calc(x_ls, y_ls):
    return np.linalg.inv(x_ls.T.dot(x_ls)).dot(x_ls.T.dot(y_ls))


def least_squares_poly(x, y, n):
    """
    Will carry out least squares for a polynomial of degree n
    """
    x_ls = np.vander(x, increasing=True, N=n + 1)
    y_ls = y.T
    return least_squares_calc(x_ls, y_ls)


def least_squares_exp(x, y):
    """
    Will carry out least squares for an exponential
    """
    x_ls = np.vstack((np.ones(x.shape), np.exp(x))).T
    y_ls = y.T
    return least_squares_calc(x_ls, y_ls)


def func_fitter(x, y):
    """
    Finds the best fitting function
    Linear, Polynomial, Unknown?(Maybe Exponential)
    """
    # Temp just use degree 1 polynomials
    # for i in range(2, 7):
    lin_A = least_squares_poly(x, y, 1)
    poly_A = least_squares_poly(x, y, 3)  # Do some tests to ensure it is best modelled by cubic
    exp_A = least_squares_exp(x, y)

    # Currently calculate each one and take the lowest error option.
    lin_func = poly_func(lin_A)
    lin_y = apply_funcs(x, [lin_func])
    lin_error = error(y, lin_y)

    polynomial_func = poly_func(poly_A)
    polynomial_y = apply_funcs(x, [polynomial_func])
    polynomial_error = error(y, polynomial_y)

    exponential_func = exp_func(exp_A)
    exponential_y = apply_funcs(x, [exponential_func])
    exponential_error = error(y, exponential_y)

    if lin_error > exponential_error and polynomial_error > exponential_error:
        print('Exponential')
        return exponential_func
    elif polynomial_error > lin_error and exponential_error > lin_error:
        print('Linear')
        return lin_func
    elif lin_error > polynomial_error and exponential_error > polynomial_error:
        print('Polynomial')
        return polynomial_func


def error(y1, y2):
    """
    Calculates the error between the actual y and predicted y
    """
    return np.sum(((y1) - y2) ** 2)


def poly_func(A):
    """
    Creates a partially evaluated function for a polynomial/linear
    """
    def func(xi):
        y = 0
        for i in range(len(A)):
            y += A[i] * (xi**i)
        return y
    return func


def exp_func(A):
    """
    Creates a partially evaluated function for an exponential
    """
    def func(xi):
        return A[0] + A[1] * np.exp(xi)
    return func


def get_model(x, y):
    """
    Given x and y it will genereate a model for the data points
    """
    funcs = []
    segments = len(x) // 20
    for i in range(segments):
        func = func_fitter(x[i * 20: i * 20 + 20], y[i * 20: i * 20 + 20])
        funcs.append(func)
    return funcs


def apply_funcs(x, funcs):
    """
    Will apply the model to x to generate a predicted y value
    """
    y = np.copy(x)
    for i, func in enumerate(funcs):
        y[i * 20: i * 20 + 20] = func(x[i * 20: i * 20 + 20])
    return y


def main(args):
    args_len = len(args)
    if args_len in [1, 2]:
        x, y = utilities.load_points_from_file(args[0])
        model_funcs = get_model(x, y)
        model_y = apply_funcs(x, model_funcs)
        model_error = error(model_y, y)
        print(model_error)
    if args_len == 2:
        if args[1] == '--plot':
            # Print out graph
            segments = len(x) // 20
            for i in range(segments):
                segment_x = x[i * 20: i * 20 + 20]
                min_x = segment_x.min()
                max_x = segment_x.max()
                x_plot = np.linspace(min_x, max_x, 100)
                y_plot = model_funcs[i](x_plot)
                plt.plot(x_plot, y_plot)
            utilities.view_data_segments(x, y)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
