# -*- coding: utf-8 -*-
import sys

import numpy as np
from matplotlib import pyplot as plt

import utilities


def least_squares_poly(x, y, n):
    x_ls = np.vander(x, increasing=True, N=n + 1)
    y_ls = y.T
    A = np.linalg.inv(x_ls.T.dot(x_ls)).dot(x_ls.T.dot(y_ls))
    return A


def poly_fitter(x, y):
    # Temp just use degree 1 polynomials
    # for i in range(2, 7):
    A = least_squares_poly(x, y, 1)
    return A


def error(y1, y2):
    return np.sum(((y1) - y2) ** 2)


def poly_func(A):
    def func(xi):
        y = 0
        for i in range(len(A)):
            y += A[i] * (xi**i)
        return y
    return func


def get_model(x, y):
    funcs = []
    segments = len(x) // 20
    for i in range(segments):
        A = poly_fitter(x[i * 20: i * 20 + 20], y[i * 20: i * 20 + 20])
        funcs.append(poly_func(A))
    return funcs


def apply_funcs(x, funcs):
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
