# tools for plotting and analyzing cavity resonators

import copy
import json
import re

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add missing imports for constants
import scipy.constants as const
import slab.dsfit as dsf
from jax import jacfwd, jacrev, jit
from lmfit import Model
from lmfit.models import LinearModel, LorentzianModel
from matplotlib.colors import to_rgba
from scipy import optimize
from scipy.constants import e, h
from slab import *
from tabulate import tabulate

# from scipy.optimize import curve_fit

Phi0 = h / (2 * e)
hbar = h / (2 * np.pi)


def signal_plot(freq, log_mag, phase, lin_mag, real, imag):
    fig, ax = plt.subplots(figsize=(6, 6), nrows=2, ncols=2, layout="constrained")

    ax[0][0].plot(freq, log_mag, c="b", label="Mag. (dB)")

    ax[0][0].set_xlabel("Freq. (GHz)")
    ax[0][0].set_ylabel("Mag. (dB)")
    ax[0][0].set_title("Magnitude")
    ax[0][0].tick_params(axis="y", labelcolor="b")

    ax2 = ax[0][0].twinx()
    ax2.plot(freq, lin_mag, c="g", label="Mag. (Lin.)")

    ax2.set_ylabel("Mag. (Lin.)")
    ax2.tick_params(axis="y", labelcolor="g")

    h1, l1 = ax[0][0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[0][0].legend(h1 + h2, l1 + l2)

    ax[0][1].plot(freq, phase)

    ax[0][1].set_xlabel("Freq. (GHz)")
    ax[0][1].set_ylabel("Phase (Degree)")
    ax[0][1].set_title("Phase (Degree)")

    ax[1][0].plot(freq, real, label="I")
    ax[1][0].plot(freq, imag, label="Q")

    ax[1][1].set_aspect("equal", adjustable="box")
    ax[1][0].set_xlabel("Freq. (GHz)")
    ax[1][0].set_ylabel("I and Q")
    ax[1][0].set_title("I and Q")
    ax[1][0].legend()

    ax[1][1].plot(real, imag)
    ax[1][1].scatter(real[0], imag[0], label="Start", c="r")

    ax[1][1].set_xlabel("I")
    ax[1][1].set_ylabel("Q")
    ax[1][1].set_title("I and Q")
    ax[1][1].legend()


def get_lin_correction(x, y, ends_only=True, add_start_offset=True):
    if ends_only:
        p = dsf.fitlinear(np.array([x[0], x[-1]]), np.array([y[0], y[-1]]))
        m = p[1]
    else:
        p = dsf.fitlinear(x, y)
        m = p[1]
    if add_start_offset:
        b = y[0] - m * x[0]
    else:
        b = p[0]

    y_corrected = y - (m * x + b)
    p = [b, m]
    return y_corrected, p


def get_mag_phase_lin_correction(
    freq, log_mag, phase, ends_only=True, add_start_offset=True, display=True
):

    log_mag_cor, p_mag = get_lin_correction(freq, log_mag, ends_only, add_start_offset)
    phase_cor, p_phase = get_lin_correction(freq, phase, ends_only, add_start_offset)

    if display:
        fig, ax = plt.subplots(figsize=(6, 6), nrows=2, ncols=2, layout="constrained")

        ax1 = ax[0][0]
        ax1.plot(freq, log_mag, label="Mag.")
        ax1.plot(freq, dsf.linear(p_mag, freq), label="Fit", linestyle="--", c="r")

        ax1.set_xlabel("Freq. (GHz)")
        ax1.set_ylabel("Mag. (dB)")
        ax1.legend()

        ax2 = ax[0][1]
        ax2.plot(freq, log_mag_cor, label="Mag. Corrected")

        ax2.set_xlabel("Freq. (GHz)")
        ax2.set_ylabel("Mag. (dB)")
        ax2.legend()

        ax3 = ax[1][0]
        ax3.plot(freq, phase, label="Phase")
        ax3.plot(freq, dsf.linear(p_phase, freq), label="Fit", linestyle="--", c="r")

        ax3.set_xlabel("Freq. (GHz)")
        ax3.set_ylabel("Phase (Degree)")
        ax3.legend()

        ax4 = ax[1][1]
        ax4.plot(freq, phase_cor, label="Phase Corrected")

        ax4.set_xlabel("Freq. (GHz)")
        ax4.set_ylabel("Phase (Degree)")
        ax4.legend()

    return log_mag_cor, phase_cor


def get_coupling_mag_fit(freq, real, imag, display=True):
    mag_0 = np.sqrt((real - real[0]) ** 2 + (imag - imag[0]) ** 2)
    d = np.max(mag_0) - np.min(mag_0)
    kappa = 1.0 / (2.0 / d - 1.0)  # kappa is Q_internal/Q_coupling, <1 is undercoupled

    if display:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), layout="constrained")
    fit = fitlor(freq, mag_0**2 / max(mag_0) ** 2, showfit=display)
    Qs = fit[2] / fit[3] / 2
    fs = fit[2]
    Qi = (kappa + 1.0) * Qs
    Qc = Qi / kappa

    print("d: %s" % d)
    print("Q_i/Q_c: %s" % kappa)
    print("Loaded Q: %s" % Qs)
    print("Internal Q: %s" % Qi)
    print("Coupling Q: %s" % Qc)

    return {"d": d, "kappa": kappa, "Qs": Qs, "Qi": Qi, "Qc": Qc}


def get_coupling(
    freq, log_mag, phase, ends_only=True, add_start_offset=True, freq_fit=None
):

    log_mag_cor = get_lin_correction(freq, log_mag, ends_only, add_start_offset)[0]
    phase_cor = get_lin_correction(freq, phase, ends_only, add_start_offset)[0]
    lin_mag_cor = 10 ** (log_mag_cor / 20)
    real_cor = lin_mag_cor * np.cos(phase_cor * np.pi / 180)
    imag_cor = lin_mag_cor * np.sin(phase_cor * np.pi / 180)

    #

    coupling_mag_fit = get_coupling_mag_fit(freq, real_cor, imag_cor, display=False)
    guess_params = [
        freq[int(len(freq) / 2)],
        coupling_mag_fit["Qc"],
        coupling_mag_fit["Qi"],
        0,
        0,
        0,
    ]
    coupling_complex_fit_pre = fit_resonator_complex(
        freq,
        real_cor + 1j * imag_cor,
        "oneport",
        guess_params=guess_params,
        showstartfit=0,
    )[0]  # data array: [f0, Qc, Qi, phi, scale, phi_global]
    coupling_complex_fit = {
        "f0": coupling_complex_fit_pre[0],
        "Qc": coupling_complex_fit_pre[1],
        "Qi": coupling_complex_fit_pre[2],
        "phi": coupling_complex_fit_pre[3],
        "scale": coupling_complex_fit_pre[4],
        "phi_global": coupling_complex_fit_pre[0],
    }

    return coupling_mag_fit, coupling_complex_fit


### Fit code from Tanay Roy


def S11_mag_1port_func(x, *p):
    """
    Asymmetric S11 magnitude function (reflection from 1 port resonator), in voltage!
    Source: https://aip.scitation.org/doi/pdf/10.1063/5.0016463

    S11_mag = scale * abs(bwc*exp(i*phi) / (i*(f-f0) + bwt/2) - 1)
    bwc = f0/Qc # Coupled/External linear bandwidth
    bwt = fo/Qt = f0*(1/Qi + 1/Qc) # Total/Loaded linear bandwidth
    :param x: Frequency points
    :param p: [f0, Qc, Qi, phi, scale]
               p0, p1, p2, p3,  p4
    :return: scale * abs(1/Qc*exp(i*phi) / (i*(f/f0-1) + 1/Qt/2) - 1)
    """
    return p[4] * np.abs(
        1
        / p[1]
        * np.exp(1j * p[3])
        / (1j * (x / p[0] - 1) + (1 / p[1] + 1 / p[2]) / 2.0)
        - 1
    )


def S11_complex_1port_func(x, p):
    """
    Asymmetric S11 magnitude function (reflection from 1 port resonator), in voltage!
    Source: https://aip.scitation.org/doi/pdf/10.1063/5.0016463
    S11 = scale * exp(i*phi_global) * (bwc*exp(i*phi) / (i*(f-f0) + bwt/2) - 1)
    bwc = f0/Qc # Coupled/External linear bandwidth
    bwt = f0*(1/Qi + 1/Qc) # Total/Loaded linear bandwidth
    :param x: Frequency points
    :param p: [f0, Qc, Qi, phi, scale, phi_global]
               p0, p1, p2, p3,  p4,    p5
    :return: scale * exp(i*phi_golbal) * (1/Qc*exp(i*phi) / (i*(f/f0-1) + 1/Qt/2) - 1)
    """
    return (
        p[4]
        * np.exp(1j * p[5])
        * (
            1
            / p[1]
            * np.exp(1j * p[3])
            / (1j * (x / p[0] - 1) + (1 / p[1] + 1 / p[2]) / 2.0)
            - 1
        )
    )


def S11_complex_1port_func_sum(x, scale, globalphase, p_list):
    """
    Asymmetric S11 magnitude function (reflection from 1 port resonator), in voltage!
    Source: https://aip.scitation.org/doi/pdf/10.1063/5.0016463
    for a single resonator:
        S11 = scale * exp(i*phi_global) * (bwc*exp(i*phi) / (i*(f-f0) + bwt/2) - 1)
        bwc = f0/Qc # Coupled/External linear bandwidth
        bwt = f0*(1/Qi + 1/Qc) # Total/Loaded linear bandwidth
        :param x: Frequency points
        :param p: [f0, Qc, Qi, phi, scale, phi_global]
                p0, p1, p2, p3,  p4,    p5
        :return: scale * exp(i*phi_golbal) * (1/Qc*exp(i*phi) / (i*(f/f0-1) + 1/Qt/2) - 1)

    We sum over multiple resonators
    p_list = [[f0, Qc, Qi, phi], [f0, Qc, Qi, phi], ...]
    """
    sum = 0
    for p in p_list:
        sum += (
            scale
            * np.exp(1j * globalphase)
            * (
                1
                / p[1]
                * np.exp(1j * p[3])
                / (1j * (x / p[0] - 1) + (1 / p[1] + 1 / p[2]) / 2.0)
            )
        )
    sum -= scale * np.exp(1j * globalphase)
    return sum


def S21_mag_hangar_func(x, *p):
    """
    Asymmetric S21 magnitude function (resonator coupled to a transmission line), in voltage!
    Source: https://aip.scitation.org/doi/10.1063/1.4907935
    S21_mag = scale * abs( 1 - bwc*exp(i*phi) / (2*i*(f-f0) + bwt) )
    bwc = f0/Qc # Coupled/External linear bandwidth
    bwt = f0*(1/Qi + 1/Qc) # Total/Loaded linear bandwidth
    :param x: Frequency points
    :param p: [f0, Qc, Qi, phi, scale]
               p0, p1, p2, p3,  p4
    :return: scale * abs(1 - 1/Qc*exp(i*phi) / (2*i*(f/f0-1) + 1/Qt) )
    """
    return p[4] * np.abs(
        1 - 1 / p[1] * np.exp(1j * p[3]) / (2j * (x / p[0] - 1) + (1 / p[1] + 1 / p[2]))
    )


def S21_complex_hangar_func(x, p):
    """
    Asymmetric S21 magnitude function (resonator coupled to a transmission line), in voltage!
    Source: https://aip.scitation.org/doi/10.1063/1.4907935
    bwc = f0/Qc # Coupled/External linear bandwidth
    bwt = f0*(1/Qi + 1/Qc) # Total/Loaded linear bandwidth
    :param x: Frequency points
    :param p: [f0, Qc, Qi, phi, scale, phi_global]
               p0, p1, p2, p3,  p4
    :return: scale * exp(i*phi_global) * (1 - 1/Qc*exp(i*phi) / (2*i*(f/f0-1) + 1/Qt) )
    """
    return (
        p[4]
        * np.exp(1j * p[5])
        * (
            1
            - 1
            / p[1]
            * np.exp(1j * p[3])
            / (2j * (x / p[0] - 1) + (1 / p[1] + 1 / p[2]))
        )
    )


def S21_complex_transmission_curvefit(x, *p):
    """
    S21 complex function for a transmission curve, from Yanhao Wang's internal Yale note"""
    f0, Qt, Ql, Qr, scale, phi_global = p

    numerator = 2 * Qt / np.sqrt(Ql * Qr)
    denominator = 1 - 2j * Qt * (x / f0 - 1)
    s21 = -(numerator / denominator)

    return scale * np.exp(1j * phi_global) * s21


def S21_lorfunc(x, *p):
    """
    Lorentzian with or without offset
    Qt = f0/BW # Total linear bandwidth
    :param p: [f0, Qt, scale, offset] or [f0, Qt, scale]
               p0, p1, p2,    p3          p0, p1, p2
    :param x: Frequency points
    :return: scale/ (1 + (2*Qt*(1-x/f0))**2)
    """
    if len(p) == 4:
        return p[2] / (1 + (2 * p[1] * (1 - x / p[0])) ** 2) + p[3]
    else:
        return p[2] / (1 + (2 * p[1] * (1 - x / p[0])) ** 2)


def s11_mag_func_asymmetric_2(x, *p):  # From slab kfit (not used)
    """
    Asymmetric S11 magnitude function (reflection from 1 port resonator), in voltage!
    :param x: Frequency points
    :param p: [f0, Qc, Qi, df, scale]
    :return: scale*()
    """
    f0, Qc, Qi, df, scale = p
    x_f0 = x - f0
    kr = f0 / Qc
    eps = f0 / (2 * Qi)
    return scale * np.abs(
        (1j * x_f0 + (eps - kr / 2)) / (1j * x_f0 + 1j * df + (eps + kr / 2))
    )


def selectdomain(xdata, ydata, domain):
    ind = np.searchsorted(xdata, domain)
    return xdata[ind[0] : ind[1]], ydata[ind[0] : ind[1]]


def print_fitresult(xdata, ydata, bestfitparams, fitparam_errors, fitparam_names=None):
    """To print the fit result we use the legend feature from pyplot.
    This is nice because it has an algorithm that can determine the
    best place to print the text within the figure. We plot the first
    few datapoints (not visible) and assign labels to those.
    Then we can use the loc = 0 to let pyplot figure out the best location."""
    if fitparam_names is None:
        fitparam_names = ["par%d" % k for k in range(len(bestfitparams))]

    # Remember the limits of the y-axis so that we don't change it
    ylims = plt.ylim()

    for k in range(len(bestfitparams)):
        plt.plot(
            xdata[k],
            ydata[k],
            label=r"%s = %.3e $\pm$ %.1e"
            % (fitparam_names[k], bestfitparams[k], fitparam_errors[k]),
            alpha=0,
        )

    # plt.legend(loc=0, frameon=False, prop={'size': 9}, title="Fit result")
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
        prop={"size": 9},
        title="Fit result",
    )

    plt.ylim(ylims)


def fit_res(
    xdata,
    ydata,
    mode="oneport",
    fitparams=None,
    domain=None,
    showfit=False,
    showstartfit=False,
    verbose=True,
    **kwarg,
):
    """
    Fit an S11 curve. For mode='oneport' this code uses s11_mag_func_asymmetric. If mode='twoport' the code uses
    s11_mag_twoport. In both cases the fit function can fit asymmetric line shapes, represented by the parameter df.
    NB: fits the voltage signal, not a power (i.e. use this functionto fit |S11| instead of |S11|**2.
    For mode='oneport', Note Qi = f0/(2*eps), Qc = f0/kr.
    :param xdata: Frequency points
    :param ydata: S11 voltage data
    :param fitparams: [f0, kr, eps, df, scale]
    :param domain: Tuple
    :param showfit: True/False
    :param showstartfit: True/False
    :param label: String
    :param verbose: True/False, prints the fitresults
    :return: Fitresult, Fiterror
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata

    if fitparams is None and mode == "oneport":
        f0_guess = fitdatax[np.argmin(fitdatay)]
        Qi_guess = f0_guess / ((fitdatax[-1] - fitdatax[0]) / 5.0)
        Qc_guess = Qi_guess / 2.0
        phi_guess = 0
        scale_guess = np.max(fitdatay)
        fitparams = [f0_guess, Qi_guess, Qc_guess, phi_guess, scale_guess]

    elif fitparams is None and mode == "hangar":
        f0_guess = fitdatax[np.argmin(fitdatay)]
        Qi_guess = f0_guess / ((fitdatax[-1] - fitdatax[0]) / 5.0)
        Qc_guess = Qi_guess / 2.0
        phi_guess = 0
        scale_guess = np.max(fitdatay)
        fitparams = [f0_guess, Qi_guess, Qc_guess, phi_guess, scale_guess]

    elif fitparams is None and mode == "oneport_mag_1":
        f0_guess = fitdatax[np.argmin(fitdatay)]
        Qc_guess = f0_guess / ((fitdatax[-1] - fitdatax[0]) / 5.0)
        Qi_guess = f0_guess / ((fitdatax[-1] - fitdatax[0]) / 5.0)
        df_guess = 0
        scale_guess = np.max(fitdatay)
        fitparams = [f0_guess, Qc_guess, Qi_guess, df_guess, scale_guess]

    if mode == "oneport":
        params, param_errs = fitbetter(
            fitdatax,
            fitdatay,
            S11_mag_1port_func,
            fitparams,
            parambounds=([0, 0, 0, -np.inf, -np.inf], np.inf),
            domain=None,
            showfit=showfit,
            showstartfit=showstartfit,
            **kwarg,
        )
        names = ["f0", "Qc", "Qi", "phi", "scale"]

    elif mode == "hangar":
        params, param_errs = fitbetter(
            fitdatax,
            fitdatay,
            S21_mag_hangar_func,
            fitparams,
            parambounds=([0, 0, 0, -np.inf, -np.inf], np.inf),
            domain=None,
            showfit=showfit,
            showstartfit=showstartfit,
            **kwarg,
        )
        names = ["f0", "Qc", "Qi", "phi", "scale"]

    elif mode == "transmission":
        params, param_errs = fitbetter(
            fitdatax,
            fitdatay,
            S21_lorfunc,
            fitparams,
            parambounds=([0, 0, 0, -np.inf], np.inf),
            domain=None,
            showfit=showfit,
            showstartfit=showstartfit,
            **kwarg,
        )
        names = ["f0", "Qt", "scale", "offset"]

    if verbose:
        print(
            tabulate(
                zip(names, params, param_errs, param_errs / params * 100),
                headers=["Parameter", "Value", "Std", "Std(%)"],
                tablefmt="fancy_grid",
                floatfmt=".3f",
                numalign="center",
                stralign="left",
            )
        )
        print_fitresult(fitdatax, fitdatay, params, param_errs, fitparam_names=names)

    return params, param_errs


def fitbetter(
    xdata,
    ydata,
    fitfunc,
    fitparams,
    parambounds=None,
    domain=None,
    showfit=False,
    showstartfit=False,
    showdata=True,
    mark_data="o",
    mark_fit="r-",
    **kwargs,
):
    """
    Uses curve_fit from scipy.optimize to fit a non-linear least squares function to ydata, xdata
    Note: when applying bounds the fit method used is a different one than with an unconstrained fit. It's good
    practice to not apply bounds to parameters if it's not needed.
    :param xdata: x-axis
    :param ydata: y-axis
    :param fitfunc: One of the fitfunctions below
    :param fitparams: Parameters for the fitfunction
    :param parambounds: Tuple of bounds for each of the parameters: ([par1_min, par2_min, ...], \
                                                                     [par1_max, par2_max, ...])
    :param domain: Domain for the xdata
    :param showfit: Show the fit
    :param showstartfit: Show the curve with initial guesses
    :param showdata: Plot the data.
    :param label: Label for the data
    :param mark_data: Marker format for the data
    :param mark_fit: Marker format for the fit
    :return:
    """
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata

    if parambounds is None:
        parambounds = (-np.inf, +np.inf)

    startparams = fitparams
    bestfitparams, covmatrix = optimize.curve_fit(
        fitfunc, fitdatax, fitdatay, startparams, bounds=parambounds, **kwargs
    )

    try:
        fitparam_errors = np.sqrt(np.diag(covmatrix))
    except:
        print(covmatrix)
        print(
            "Error encountered in calculating errors on fit parameters.\
            This may result from a very flat parameter space"
        )

    if showfit:
        if showdata:
            f = plt.figure()
            f.set_figwidth(6)
            f.set_figheight(4)
            plt.plot(fitdatax, fitdatay, mark_data, ms=2, label="data")
        if showstartfit:
            plt.plot(fitdatax, fitfunc(fitdatax, *startparams), label="startfit")
        plt.plot(
            fitdatax,
            fitfunc(fitdatax, *bestfitparams),
            mark_fit,
            label="fit",
            linewidth=2,
            alpha=0.8,
        )

    return bestfitparams, fitparam_errors


def fit_resonator_complex(
    xdata,
    ydata,
    mode="oneport",
    guess_params=None,
    method="BFGS",
    domain=None,
    showdata=True,
    showReIm=True,
    showstartfit=False,
    verbose=True,
    **kwarg,
):
    """
    Author: Tanay Roy, Feb 2023
    mode = 'oneport' is for simple reflection S11
    mode = 'hangar' is for hangar type transmission S21"""

    # Select function
    if mode == "oneport":
        fitfunc = S11_complex_1port_func  # Simple reflection S11
    elif mode == "hangar":
        fitfunc = S21_complex_hangar_func  # Hangar transmission S21

    # Define cost function
    def cost_fun(params):
        comp_arr = fitfunc(xdata, params)
        val = np.sum(abs(ydata - comp_arr))
        # print(val)
        return val

    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata

    names = ["f0", "Qc", "Qi", "phi", "scale", "phi_global"]  # Fit parameters

    # Guess parameters
    if guess_params is None and mode == "oneport":
        f0_guess = fitdatax[np.argmin(abs(fitdatay))]
        Qi_guess = f0_guess / ((fitdatax[-1] - fitdatax[0]) / 5.0)
        Qc_guess = Qi_guess / 2.0
        phi_guess = 0
        scale_guess = np.max(abs(fitdatay))
        phi_golbal_guess = 0
        guess_params = [
            f0_guess,
            Qi_guess,
            Qc_guess,
            phi_guess,
            scale_guess,
            phi_golbal_guess,
        ]

    if guess_params is None and mode == "hangar":
        f0_guess = fitdatax[np.argmin(abs(fitdatay))]
        Qi_guess = f0_guess / ((fitdatax[-1] - fitdatax[0]) / 5.0)
        Qc_guess = Qi_guess / 2.0
        phi_guess = 0
        scale_guess = np.max(abs(fitdatay))
        phi_golbal_guess = 0
        guess_params = [
            f0_guess,
            Qi_guess,
            Qc_guess,
            phi_guess,
            scale_guess,
            phi_golbal_guess,
        ]

    # Run minimizer
    optvals = optimize.minimize(cost_fun, guess_params, method=method)
    # Only 'BFGS' provides hess_inv used to calculate erro bar
    # methods = ‘Nelder-Mead’ ‘Powell’ ‘CG’ ‘BFGS’ ‘Newton-CG’ ‘L-BFGS-B’ ‘TNC’ ‘COBYLA’
    #           ‘SLSQP’ ‘dogleg’ ‘trust-ncg’
    print("Convergence: ", optvals.success)

    # Get error bar
    params, param_errs = optvals.x, np.sqrt(np.diag(optvals.hess_inv))

    if verbose:
        print(
            tabulate(
                zip(names, params, param_errs, param_errs / params * 100),
                headers=["Parameter", "Value", "Std", "Std(%)"],
                tablefmt="fancy_grid",
                floatfmt=".2f",
                numalign="center",
                stralign="left",
            )
        )

    # Plot data and fit
    if showdata:
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].plot(fitdatax, abs(fitdatay), "o", ms=2, label="data")
        ax[1].plot(np.real(fitdatay), np.imag(fitdatay), "o", ms=2, label="data")
    if showstartfit:
        data = fitfunc(fitdatax, guess_params)
        ax[0].plot(fitdatax, abs(data), label="startfit")
        ax[1].plot(np.real(data), np.imag(data), label="startfit")
    data = fitfunc(fitdatax, params)
    ax[0].plot(fitdatax, abs(data), "r-", label="fit", lw=2, alpha=0.8)
    ax[0].set_xlabel("Freq.")
    ax[0].set_ylabel("Abs")
    ax[1].plot(np.real(data), np.imag(data), "r-", label="fit", lw=2, alpha=0.8)
    ax[1].set_xlabel("Re")
    ax[1].set_ylabel("Im")
    fig.suptitle("Mode Freq: " + str("{:.6e}".format(params[0])))
    plt.tight_layout()

    print_fitresult(
        np.real(data), np.imag(data), params[0:3], param_errs, fitparam_names=names
    )

    if showReIm:
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax[0].plot(fitdatax, np.real(fitdatay), "o", ms=2, label="data")
        ax[1].plot(fitdatax, np.imag(fitdatay), "o", ms=2, label="data")
        ax[2].plot(fitdatax, np.angle(fitdatay) * 180 / pi, "o", ms=2, label="data")
    if showstartfit:
        data = fitfunc(fitdatax, guess_params)
        ax[0].plot(fitdatax, np.real(data), label="startfit")
        ax[1].plot(fitdatax, np.imag(data), label="startfit")
        ax[2].plot(fitdatax, np.angle(data) * 180 / pi, label="startfit")
    data = fitfunc(fitdatax, params)
    ax[0].plot(fitdatax, np.real(data), "r-", label="fit", lw=2, alpha=0.8)
    ax[0].set_xlabel("Freq.")
    ax[0].set_title("Re")
    ax[1].plot(fitdatax, np.imag(data), "r-", label="fit", lw=2, alpha=0.8)
    ax[1].set_xlabel("Freq.")
    ax[1].set_title("Im")
    ax[2].plot(fitdatax, np.angle(data) * 180 / pi, "r-", label="fit", lw=2, alpha=0.8)
    ax[2].set_xlabel("Freq.")
    ax[2].set_title("Phase (deg)")
    fig.suptitle("Mode Freq: " + str("{:.6e}".format(params[0])))

    plt.tight_layout()

    return params, param_errs


def nbar_from_power(P, nu_c, nu_d, Qc, Q_in):
    nu_d = nu_d * 1e9
    nu_c = nu_c * 1e9
    w_c = 2 * np.pi * nu_c
    w_d = 2 * np.pi * nu_d
    Q_tot = 1 / (1 / Qc + 1 / Q_in)
    kappa_in = w_c / Qc
    kappa = w_c / Q_tot
    photon_flux = P / (const.h * nu_d)
    n = (
        kappa_in / (kappa**2 / 4 + (w_d - w_c) ** 2) * (photon_flux)
    )  # RWA has been used
    return n


def T1(freq, Qi):
    return Qi / (2 * np.pi * freq)


def nbar_from_power(P, nu_c, nu_d, Qc, Q_in):
    nu_d = nu_d * 1e9
    nu_c = nu_c * 1e9
    w_c = 2 * np.pi * nu_c
    w_d = 2 * np.pi * nu_d
    Q_tot = 1 / (1 / Qc + 1 / Q_in)
    kappa_in = w_c / Qc
    kappa = w_c / Q_tot
    photon_flux = P / (const.h * nu_d)
    n = (
        kappa_in / (kappa**2 / 4 + (w_d - w_c) ** 2) * (photon_flux)
    )  # RWA has been used
    return n


def nbar_S11(P, Qi, Qc, nu):
    Qt = 1 / (1 / Qi + 1 / Qc)
    nbar = (4 * Qt**2) / Qc * P / (h * nu**2)
    return nbar


def nbar_hangar(Qi, Qc, power, f0):
    power = 10 ** (power / 10) * 1e-3  # convert power from dbm to watts
    Ql = 1 / (1 / Qi + 1 / Qc)
    omega = 2 * np.pi * f0
    nbar = 2 * Ql**2 * power / (omega**2 * hbar * Qc)
    return nbar


def nbar_S21(Qt, Qd, power, f0):
    power = 10 ** (power / 10) * 1e-3  # convert power from dbm to watts
    omega = 2 * np.pi * f0
    nbar = 4 * Qt**2 * power / (omega**2 * hbar * Qd)
    return nbar


def lw_tot(complex_fit):
    Q_tot = 1 / (1 / complex_fit["Qc"] + 1 / complex_fit["Qi"])
    return complex_fit["f0"] * 1e9 / Q_tot


def linewidth_total_est(f, Qc, Qi):
    Qtot = 1 / (1 / Qc + 1 / Qi)
    return f / Qtot


def resolution(span, swp_pts):
    return span / swp_pts


def chi_from_g(g, Ec, nuq, nur):
    delta = nuq - nur
    alpha = -Ec
    return 2 * alpha * g**2 / delta / (delta + alpha)


def g_from_chi(chi, Ec, nuq, nur):
    delta = nuq - nur
    alpha = -Ec
    return np.sqrt(chi * delta * (delta + alpha) / (2 * alpha))


def purcell_T1(g, fr, fq, qr):
    kappa_r = 2 * pi * fr / qr
    kappa_q = (g**2 / (fr - fq) ** 2) * kappa_r
    return 1 / kappa_q


def purcell_T1_inv(g, fr, fq, qr):
    kappa_r = 2 * pi * fr / qr
    T1q = 1 / ((g**2 / (fr - fq) ** 2) / kappa_r)
    return T1q


def purcell_Qi(g, fr, fq, qr):
    """
    Calculate the Purcell limited internal quality factor Qi for a qubit coupled to a resonator.

    Parameters:
    g (float): Coupling strength between the qubit and the resonator.
    fr (float): Resonator frequency in Hz.
    fq (float): Qubit frequency in Hz.
    qr (float): Resonator quality factor.

    Returns:
    float: Purcell limited internal quality factor Qi.
    """
    kappa_r = 2 * pi * fr / qr
    kappa_q = (g**2 / (fr - fq) ** 2) * kappa_r
    return fr / kappa_q


def Ej_from_Lj(Lj):
    return (Phi0 / (2 * pi)) ** 2 / Lj


def add_complex_fit(fitdict, modenum, phase_fit):
    """
    Add a complex fit result to the all_complex_fit dictionary.

    Parameters:
    modenum (int): The mode number to be used as a key.
    phase_fit (dict): A dictionary containing the fit results with keys 'f0', 'Qc', 'Qi', 'phi', 'scale', and 'phi_global'.

    Returns:
    None
    """
    fitdict["mode" + str(modenum)] = {
        "f0": phase_fit["f0"],
        "Qc": phase_fit["Qc"],
        "Qi": phase_fit["Qi"],
        "phi": phase_fit["phi"],
        "scale": phase_fit["scale"],
        "phi_global": phase_fit["phi_global"],
    }
    print("Added complex fit for mode ", modenum, " to the dictionary ", fitdict)


# <KeysViewHDF5 ['avgs', 'avgs_state', 'elec_delay', 'freq', 'ifbw', 'mags', 'phase_offset', 'phases', 'read_freq', 'read_power', 'span', 'sweep_pts']>


def get_MM_pnax_config(mode, freq, file):
    # for putting into a measurement loop: have a dictionary of each mode's freq and pnax config values
    pnax_config = {
        "mode": mode,
        "fitfreq": freq,
        "config": {
            "avgs": file["avgs"][0],
            "avgs_state": file["avgs_state"][0],
            "elec_delay": file["elec_delay"][0],
            # "freq": file["freq"][0], # removed because the np array does not save to json
            "ifbw": file["ifbw"][0],
            "read_freq": file["read_freq"][0],
            "read_power": file["read_power"][0],
            "span": file["span"][0],
            "sweep_pts": file["sweep_pts"][0],
            "current": file["current"][0] if "current" in file else None,
        },
    }
    print("PNAX config for mode {}: {}".format(mode, pnax_config))
    return pnax_config


# fitting multiple lorentzians
def lorfuncsum(p, x, N=2):
    """p[0]+p[1]/(1+(x-p[2])**2/p[3]**2)"""
    y = 0
    for ii in range(N):
        y += p[3 * ii] / (1 + (x - p[3 * ii + 1]) ** 2 / p[3 * ii + 2] ** 2)

    y += p[3 * N]

    return y


def fitlorsum(
    xdata,
    ydata,
    fitparams=None,
    domain=None,
    showfit=False,
    showstartfit=False,
    label="",
    debug=False,
):
    """fit lorentzian:
    returns [offset,amplitude,center,hwhm]"""
    if domain is not None:
        fitdatax, fitdatay = selectdomain(xdata, ydata, domain)
    else:
        fitdatax = xdata
        fitdatay = ydata
    if fitparams is None:
        fitparams = 0 * np.ones(3 * N + 1)
        fitparams[0] = (fitdatay[0] + fitdatay[-1]) / 2.0
        fitparams[1] = max(fitdatay) - min(fitdatay)
        fitparams[2] = fitdatax[np.argmax(fitdatay)]
        fitparams[3] = (max(fitdatax) - min(fitdatax)) / 10.0
    if debug == True:
        print(fitparams)
    p1 = dsf.fitgeneral(
        fitdatax,
        fitdatay,
        lorfuncsum,
        fitparams,
        domain=None,
        showfit=showfit,
        showstartfit=showstartfit,
        label=label,
    )
    p1[3] = abs(p1[3])
    return p1


# for this notebook
def boltz_P(n, f_q, T):
    np.exp(-h * n * f_q / k / T)
    return np.exp(-h * n * f_q / k / T)


def boltz_T(n, f_q, P):
    T = -h * n * f_q / (k * np.log(P))
    return T


def lorentzian(x, amplitude, center, hwhm, offset):
    return amplitude / (1 + (x - center) ** 2 / hwhm**2) + offset


def lorentzian_sum(x, *params):
    # this version is for a shared offset
    # params = [amp1, center1, hwhm1, ..., ampN, centerN, hwhmN, offset]
    n_peaks = (len(params) - 1) // 3
    offset = params[-1]
    y = np.zeros_like(x)
    for i in range(n_peaks):
        amp = params[3 * i]
        center = params[3 * i + 1]
        hwhm = params[3 * i + 2]
        y += amp / (1 + (x - center) ** 2 / hwhm**2)
    return y + offset


# .eig file parser for Ansys data exported via the "Results > Eigenmode Data" window
def parse_ansys_eig(file_path):
    """
    Parse an ANSYS exported .eig file.

    Returns:
        metadata (dict): Design and solution info.
        variations (dict): The swept variables from the Variation line.
        df (pd.DataFrame): The eigenmode data.
    """
    metadata = {}
    variations = {}
    data = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if "Design:" in line:
                    metadata["Design"] = line.split("Design:", 1)[1].strip()
                elif "Solution:" in line:
                    metadata["Solution"] = line.split("Solution:", 1)[1].strip()
                elif "Variation:" in line:
                    var_str = line.split("Variation:", 1)[1].strip().strip('"')
                    matches = re.findall(r"(\w+)='([^']*)'", var_str)
                    variations = {k: v for k, v in matches}
            else:
                data.append(line.split())

    df = pd.DataFrame(data)

    if not df.empty:
        # Complex Frequency Format: Mode | Real | Sign | Imag | 'j' | Q
        if df.shape[1] == 6:
            df.columns = [
                "Mode",
                "Frequency_Re_GHz",
                "Sign",
                "Frequency_Im_GHz",
                "j",
                "Q",
            ]
            df = df.astype(
                {
                    "Mode": int,
                    "Frequency_Re_GHz": float,
                    "Frequency_Im_GHz": float,
                    "Q": float,
                }
            )

            # Apply negative sign to imaginary part where applicable
            df.loc[df["Sign"] == "-", "Frequency_Im_GHz"] *= -1

            # Drop the strings
            df = df.drop(columns=["Sign", "j"])

        # Real Frequency Format: Mode | Real | Q
        elif df.shape[1] == 3:
            df.columns = ["Mode", "Frequency_GHz", "Q"]
            df = df.astype(float)
            df["Mode"] = df["Mode"].astype(int)

    return metadata, variations, df


# Tom's plotting code refactor


def flux_from_current(current, flux_period, current_offset):
    """Convert current to flux in units of flux quantum."""
    return (current - current_offset) / flux_period


def current_from_flux(flux, flux_period, current_offset):
    """Convert flux in units of flux quantum to current."""
    return flux * flux_period + current_offset


def sideband_to_buffer_freq(sideband_freq, ge_freq, ef_freq):
    return ge_freq + ef_freq - sideband_freq


def bs_decay_func(t, A, k1, k2, gbs, B):
    return (A / 2) * np.exp(-t * k1) * (1 + np.exp(-t * k2) * np.cos(2 * gbs * t)) + B

def bs_decay_func_with_phase(t, A, k1, k2, gbs, B, phase):
    return (A / 2) * np.exp(-t * k1) * (1 + np.exp(-t * k2) * np.cos(2 * gbs * t + phase)) + B

def bs_decay_heating_func(t, A, k1, k2, k_heat, heat_pop, gbs, B):
    return (A / 2) * np.exp(-t * k1) * (1 + np.exp(-t * k2) * np.cos(2 * gbs * t) - heat_pop * (1 - np.exp(-t * k_heat))) + A / 2 * heat_pop * (1 - np.exp(-t * k_heat)) + B


def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)


def dbm_to_volts(dbm, Z=50):
    """Convert power in dBm to voltage in volts."""
    watts = 10 ** ((dbm - 30) / 10)
    volts = np.sqrt(watts * Z)
    return volts


def watts_to_dbm(watts):
    return 10 * np.log10(watts) + 30


class LabData:
    """
    Generalized HDF5 data loader.
    Parses JSON configs and exposes data arrays directly as attributes.
    """

    def __init__(self, data_path, filenum=None, suffix="", filename=None):
        # 1. Path resolution: Allows an exact filename, or auto-builds the 5-digit schema
        if filename is None:
            filename = (
                f"{str(filenum).zfill(5)}_{suffix}.h5"
                if suffix
                else f"{str(filenum).zfill(5)}.h5"
            )

        self.filepath = os.path.join(data_path, filename)
        # print(self.filepath)
        with h5py.File(self.filepath, "r") as f:
            # 2. Extract Config
            self.config = json.loads(f.attrs["config"]) if "config" in f.attrs else {}

            # 3. Extract all data arrays dynamically
            self.data_dict = {
                k: f[k][()] for k in f.keys() if isinstance(f[k], h5py.Dataset)
            }

        # 4. Shortcuts: Flattens the nested config architecture
        self.exp = self.config.get("exp_cfg", {})
        self.q0 = self.config.get("qubit_parameters", {}).get("q0", {})

        # 5. Magic Trick: Bind arrays directly to the object for IDE auto-complete
        for key, val in self.data_dict.items():
            setattr(self, key, val)

    """
    usage:

    # Load the file
    alice_or_bob = 'alice'
    mode = 1
    data = LabData(data_path, filenum=62, suffix=f"bs_{alice_or_bob[0]}{mode}_rabi")

    # --- NO DICTIONARY INDEXING REQUIRED FOR ARRAYS ---
    time = data.xpts * 1e6   # Instantly access xpts
    y = data.P_e             # Instantly access P_e
    # If your file has I and Q arrays, data.I and data.Q will also just work.

    # --- CLEAN METADATA ACCESS ---
    # Use .exp and .q0 to avoid writing cfg['qubit_parameters']['q0']...
    current = data.exp.get("flux_current", 0)
    bs_freq = data.q0[f'bs_{alice_or_bob}_freqs'][mode] / 1e9

    # Handling annoying fallback variables takes two lines
    amp = data.exp.get('bs_amplitude')
    if amp in (None, 'None'):
        amp = data.q0[f'bs_{alice_or_bob}_amps'][mode]
    """


# BS Spectroscopy plotter


# =============================================================================
# FIT WRAPPER
# =============================================================================
def fit_spectroscopy_old(x, y, custom_settings=None):
    lor_model = LorentzianModel()
    lin_model = LinearModel()
    model = lor_model + lin_model

    lor_params = lor_model.guess(y, x)
    lin_params = lin_model.guess(y, x)

    params = model.make_params()
    params.add_many(*lor_params.values(), *lin_params.values())

    # Preserving your specific logic to invert the initial amplitude guess
    params["amplitude"].value *= -1

    if custom_settings:
        for param_name, settings in custom_settings.items():
            params[param_name].set(**settings)

    return model.fit(y, params, x=x)


def _bs_rabi_freq(nu, nu_bs, gbs, tau):
    Delta = 2.0 * np.pi * (nu - nu_bs)
    gbs *= 2.0 * np.pi
    Omega = np.sqrt(4.0 * gbs**2 + Delta**2)
    return (4.0 * gbs**2 / Omega**2) * np.sin(0.5 * Omega * tau)**2

def bs_rabi_freq(x, nu_bs, gbs, tau, a):
    return a * _bs_rabi_freq(x, nu_bs, gbs, tau)

def fit_spectroscopy(x, y, pulse_length, custom_settings=None):
    x, y = np.asarray(x, float), np.asarray(y, float)

    lin_model = LinearModel()
    lin_params = lin_model.guess(y, x)

    bs_model = Model(bs_rabi_freq)
    baseline = np.percentile(y, 95)
    tau = pulse_length
    bs_params = bs_model.make_params(
        nu_bs=x[np.argmin(y)],
        gbs=1.0 / (4.0 * tau),
        tau=tau,
        a=(baseline - y.min()),
    )
    bs_params['nu_bs'].set(min=x.min(), max=x.max())
    bs_params['gbs'].set(min=0.0)
    bs_params['tau'].set(vary=False)
    bs_params.add('width', expr='2.0*sqrt((1/tau)**2 - 4.0*gbs**2)')

    model = bs_model + lin_model
    params = model.make_params()
    params.add_many(*bs_params.values(), *lin_params.values())

    if custom_settings:
        for param_name, settings in custom_settings.items():
            params[param_name].set(**settings)
            
    return model.fit(y, params, x=x)



# =============================================================================
# ORCHESTRATOR
# =============================================================================
def analyze_spectroscopy(
    filenums,
    modes,
    data_path,
    alice_or_bob="alice",
    suffix=None,
    global_overrides=None,
    fit_overrides=None,
    plotfits=True,
    plotfills=True,
):
    if fit_overrides is None:
        fit_overrides = {}

    tasks = list(zip(filenums, modes))
    n_files = len(tasks)
    ncols = int(np.ceil(np.sqrt(n_files)))
    nrows = int(np.ceil(n_files / ncols)) if ncols > 0 else 1
    figsize = np.array(plt.rcParams["figure.figsize"]) * np.array([ncols, nrows])
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axs = axs.flatten()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] * (n_files // 5 + 1)

    results = []
    last_current = 0

    for ii, (filenum, mode) in enumerate(tasks):
        ax, c = axs[ii], colors[ii]
        suffix = f"bs_{alice_or_bob[0]}{mode}_spectroscopy" if suffix is None else suffix

        try:
            data = LabData(data_path, filenum=filenum, suffix=suffix)
        except FileNotFoundError:
            ax.text(
                0.5,
                0.5,
                f"File {filenum}\nMode {mode}\nNot Found",
                ha="center",
                va="center",
            )
            ax.axis("off")
            continue

        freq = data.xpts / 1e9
        y = data.P_e

        current = data.exp.get("flux_current", 0)
        last_current = current
        average_exponent = data.exp.get("average_exponent", 0)

        bs_ramp_len = data.q0[f"bs_{alice_or_bob}_ramp_lens"][mode] * 1e6
        bs_len_raw = data.exp.get("bs_length")

        if bs_len_raw in (None, "None"):
            bs_flat_len = data.q0[f"bs_{alice_or_bob}_flat_lens"][mode] * 1e6
        else:
            bs_flat_len = bs_len_raw * 1e6 - 2 * bs_ramp_len

        bs_range = data.q0[f"bs_{alice_or_bob}_dBm_ranges"][mode]
        bs_amp = data.exp.get("bs_amplitude")
        if bs_amp in (None, "None"):
            bs_amp = data.q0[f"bs_{alice_or_bob}_amps"][mode]

        current_settings = copy.deepcopy(global_overrides) if global_overrides else {}
        specific_overrides = fit_overrides.get((filenum, mode), {})

        for param_name, settings in specific_overrides.items():
            if param_name in current_settings:
                current_settings[param_name].update(settings)
            else:
                current_settings[param_name] = settings

        pulse_length = bs_flat_len + 2 * bs_ramp_len
        pulse_length *= 1e3
        result = fit_spectroscopy(freq, y, pulse_length, custom_settings=current_settings)

        bs_freq = result.params["nu_bs"].value
        bs_freq_err = result.params["nu_bs"].stderr or 0.0
        bs_width = result.params["width"].value
        gbs = result.params["gbs"].value * 2 * np.pi
        tbs = 0.5 * np.pi / gbs * 1e-3

        # 1. Calculate flux here so it can be added to the dataframe
        flux = flux_from_current(current, 73.1561e-3, -2.7060e-3)

        results.append(
            {
                "filenum": filenum,
                "mode": mode,
                "current": current,
                "flux": flux,  # Added to output dataframe
                "bs_freq": bs_freq,
                "bs_freq_err": bs_freq_err,
                "width": bs_width,
                "tbs": tbs,
                "bs_amp": bs_amp,
                "bs_flat_len": bs_flat_len,
                "fit_result": result,
                "x_raw": freq,
                "y_raw": y,
            }
        )

        # --- Plotting ---
        fcolor = to_rgba(c, alpha=0.25)
        ax.plot(
            freq,
            y,
            marker="o",
            linestyle="",
            color=c,
            markerfacecolor=fcolor,
            markeredgecolor=c,
            ms=8,
            label="Data",
        )

        freq_fine = np.linspace(freq.min(), freq.max(), 1000)
        fit_fine = result.eval(x=freq_fine)

        # Prediction Band (Confidence + Residual Noise)
        try:
            model_error = result.eval_uncertainty(x=freq_fine, sigma=1)
            residual_noise = np.std(result.residual)
            prediction_error = np.sqrt(model_error**2 + residual_noise**2)
            if plotfills:
                ax.fill_between(
                    freq_fine,
                    fit_fine - prediction_error,
                    fit_fine + prediction_error,
                    color=c,
                    alpha=0.15,
                    edgecolor="none",
                    label="Prediction Band",
                )
        except Exception:
            pass

        if plotfits:
            ax.plot(freq_fine, fit_fine, c=c, linestyle="-", label="Fit")
            ax.axvline(bs_freq, linestyle="--", color=c, label="Center Freq")

        # Center Frequency Highlights
        if bs_freq_err > 0 and plotfills:
            ax.axvspan(
                bs_freq - bs_freq_err,
                bs_freq + bs_freq_err,
                color=c,
                alpha=0.35,
                zorder=1,
            )

        # Text Block
        info_text = (
            f"range = {bs_range} dBm\n"
            f"amp = {bs_amp:.4f}\n"
            rf"$t_{{BSflat}}$ = {bs_flat_len:.3f} $\mu$s" + "\n"
            rf"$t_{{BSramp}}$ = {bs_ramp_len:.3f} $\mu$s" + "\n"
            rf"$\nu_{{bs}} = {format_err(bs_freq, bs_freq_err)}$ GHz" + "\n"
            rf"n_avgs = $2^{{{average_exponent}}}$" + "\n"
            rf"Flux = {flux:.3f} $\Phi_0$"
        )

        props = dict(boxstyle="round", facecolor="white", alpha=0, edgecolor="none")
        ax.text(
            0.05,
            0.05,
            info_text,
            transform=ax.transAxes,
            fontsize="x-small",
            verticalalignment="bottom",
            bbox=props,
        )

        ax.set(xlabel="Frequency (GHz)", ylabel="$P_e$")
        ax.tick_params(axis="x", rotation=30)

        axtitle = (
            f"{alice_or_bob.capitalize()}-Storage {mode}"
            if mode != 0
            else f"{alice_or_bob.capitalize()}-SNAIL"
        )
        ax.set_title(f"{axtitle}, file {filenum}", fontsize="small")

    # Delete unused axes
    for idx in range(len(tasks), len(axs)):
        fig.delaxes(axs[idx])

    fig.suptitle(
        rf"Beamsplitter spectroscopy {alice_or_bob.capitalize()}. Current={last_current * 1e3:.3f} mA",
        y=1.02,
    )
    plt.tight_layout()

    return pd.DataFrame(results), fig


# BS Rabi plotter
def _bs_fidelity(x):
    k1, k2, gbs = x
    tbs = 0.5 * jnp.pi / gbs
    return 0.5 * (jnp.exp(-k1 * tbs) + jnp.exp(-(k1 + k2) * tbs))
def _bs_fidelity_heated(x):
    k1, k2, k_heat, heat_pop, gbs = x
    tbs = 0.5 * jnp.pi / gbs
    return (1 / 2) * jnp.exp(-tbs * k1) * (1 + jnp.exp(-tbs * k2) - heat_pop * (1 - jnp.exp(-tbs * k_heat))) + 1 / 2 * heat_pop * (1 - jnp.exp(-tbs * k_heat))


def format_err(val, err):
    """
    Rounds error to 2 sig figs, matches the value precision, and returns a LaTeX string.
    Gracefully handles missing errors from lmfit.
    """
    if pd.isna(val) or np.isinf(val):
        return r"\infty" if np.isinf(val) else "NaN"

    if pd.isna(err) or err is None or err == 0 or np.isinf(err):
        return f"{val:.4g}"

    err_order = np.floor(np.log10(abs(err)))
    err_rounded = np.round(err, -int(err_order - 1))
    val_rounded = np.round(val, -int(err_order - 1))
    decimals = max(0, -int(err_order - 1))

    return rf"{val_rounded:.{decimals}f} \pm {err_rounded:.{decimals}f}"


# =============================================================================
# 2. THE FIT WRAPPER
# =============================================================================
def fit_rabi(t, y, pi_guess, custom_settings=None, yerr=None):
    model = Model(bs_decay_func)
    params = model.make_params()

    rough_A = np.max(y) - np.min(y)
    rough_B = np.clip(np.mean(y), 0, 1)
    rough_gbs = np.pi / (2 * pi_guess)

    params["A"].set(value=rough_A, min=0.01, max=2.0)
    params["k1"].set(value=1e-3, min=0, max=100)
    params["k2"].set(value=1e-3, min=0, max=100)
    params["gbs"].set(value=rough_gbs, min=0.005 * rough_gbs, max=5.0 * rough_gbs)
    params["B"].set(value=rough_B, min=0.0, max=1.0)

    if custom_settings:
        for param_name, settings in custom_settings.items():
            params[param_name].set(**settings)

    weights = 1 / yerr if yerr is not None else None
    return model.fit(y, params, t=t, weights=weights)

def fit_rabi_with_phase(t, y, pi_guess, custom_settings=None, yerr=None):
    model = Model(bs_decay_func_with_phase)
    params = model.make_params()

    rough_A = np.max(y) - np.min(y)
    rough_B = np.clip(np.mean(y), 0, 1)
    rough_gbs = np.pi / (2 * pi_guess)

    params["A"].set(value=rough_A, min=0.01, max=2.0)
    params["k1"].set(value=1e-3, min=0, max=100)
    params["k2"].set(value=1e-3, min=0, max=100)
    params["gbs"].set(value=rough_gbs, min=0.005 * rough_gbs, max=5.0 * rough_gbs)
    params["B"].set(value=rough_B, min=0.0, max=1.0)
    params["phase"].set(value=0, min=-np.pi, max=np.pi)

    if custom_settings:
        for param_name, settings in custom_settings.items():
            params[param_name].set(**settings)

    weights = 1 / yerr if yerr is not None else None
    return model.fit(y, params, t=t, weights=weights)

def fit_rabi_heated(t, y, pi_guess, custom_settings=None,yerr=None):
    model = Model(bs_decay_heating_func)
    params = model.make_params()

    rough_A = np.max(y) - np.min(y)
    rough_B = np.clip(np.mean(y), 0, 1)
    rough_gbs = np.pi / (2 * pi_guess)

    params["A"].set(value=rough_A, min=0.01, max=2.0)
    params["k1"].set(value=1e-3, min=0, max=100)
    params["k2"].set(value=1e-3, min=0, max=100)
    params["k_heat"].set(value=1e-3, min=0, max=100)
    params["heat_pop"].set(value=0.1, min=0.0, max=1.0)
    params["gbs"].set(value=rough_gbs, min=0.005 * rough_gbs, max=5.0 * rough_gbs)
    params["B"].set(value=rough_B, min=0.0, max=1.0)

    if custom_settings:
        for param_name, settings in custom_settings.items():
            params[param_name].set(**settings)

    weights = 1 / yerr if yerr is not None else None
    return model.fit(y, params, t=t, weights=weights)




# =============================================================================
# 3. THE MAIN ORCHESTRATOR
# =============================================================================
def analyze_rabi(
    file_mode_pairs,
    startfits,
    pi_times_fit,
    data_path,
    alice_or_bob="alice",
    suffix="bs_bob_3_rabi_with_sb",
    oldsuffix=False, #band-aid for now, we should find a more elegant solution
    global_overrides=None,
    fit_overrides=None,
    heated_fit=False,
    plotfits=True,
    plotfills=True,
    plotlines=True,
    yerr=None
):
    if fit_overrides is None:
        fit_overrides = {}


    # Zip the pairs against your standard 1D arrays
    tasks = list(zip(file_mode_pairs, startfits, pi_times_fit))

    n = len(tasks)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols)) if ncols > 0 else 1
    figsize = np.array(plt.rcParams["figure.figsize"]) * np.array([ncols, nrows]) * 1.4
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axs = axs.flatten()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] * (n // 5 + 1)

    results = []

    # Unpack the tuple-of-tuples cleanly
    for ii, ((filenum, mode), startfit, pi_guess) in enumerate(tasks):
        ax, c = axs[ii], colors[ii]

        # --- 1. Load Data via LabData (Assumed defined globally) ---
        if oldsuffix:
            suffix = f"bs_{alice_or_bob[0]}{mode}_rabi"
        
        try:
            data = LabData(data_path, filenum=filenum, suffix=suffix)
        except FileNotFoundError:
            ax.text(
                0.5,
                0.5,
                f"File {filenum}\nMode {mode}\nNot Found",
                ha="center",
                va="center",
            )
            ax.axis("off")
            continue

        time = data.xpts * 1e6
        y = data.P_e

        bs_freq = data.q0[f"bs_{alice_or_bob}_freqs"][mode] / 1e9
        bs_amp = (
            data.exp.get("bs_amplitude")
            if data.exp.get("bs_amplitude") not in (None, "None")
            else data.q0[f"bs_{alice_or_bob}_amps"][mode]
        )
        bs_drive_range = data.q0[f"bs_{alice_or_bob}_dBm_ranges"][mode]

        current = data.exp.get("flux_current", 0)
        flux = flux_from_current(current, 73.1561e-3, -2.7060e-3)
        average_exponent = data.exp.get("average_exponent", 0)

        # --- 2. Hierarchy of Overrides ---
        current_settings = {}
        if global_overrides:
            current_settings = copy.deepcopy(global_overrides)

        specific_overrides = fit_overrides.get((filenum, mode), {})
        for param_name, settings in specific_overrides.items():
            if param_name in current_settings:
                current_settings[param_name].update(settings)
            else:
                current_settings[param_name] = settings

        start_idx = int(startfit)
        t_fit, y_fit = time[start_idx:], y[start_idx:]

        # --- 3. Fit ---
        if heated_fit:
            result = fit_rabi_heated(t_fit, y_fit, pi_guess, custom_settings=current_settings)
        else:
            result = fit_rabi(t_fit, y_fit, pi_guess, custom_settings=current_settings)

        # --- 4. Parameter and Standard Error Extraction ---
        v = result.values
        err_k1 = result.params["k1"].stderr or 0.0
        err_k2 = result.params["k2"].stderr or 0.0
        err_gbs = result.params["gbs"].stderr or 0.0

        bs_t1 = 1 / v["k1"] if v["k1"] > 0 else np.inf
        bs_t2 = 1 / v["k2"] if v["k2"] > 0 else np.inf
        pi_time = np.pi / (2 * v["gbs"]) if v["gbs"] != 0 else 0.0

        t1_err = (bs_t1**2) * err_k1 if bs_t1 != np.inf else 0.0
        t2_err = (bs_t2**2) * err_k2 if bs_t2 != np.inf else 0.0
        pi_time_err = pi_time * (err_gbs / v["gbs"]) if v["gbs"] != 0 else 0.0

        # --- 5. Generalized Fidelity Error Propagation (JAX) ---
        # var_names = ["k1", "k2", "gbs"]
        # cov = np.zeros((3, 3))

        # if getattr(result, "covar", None) is not None:
        #     # Safely build the covariance matrix, allowing for fixed parameters
        #     for r_idx, name_i in enumerate(var_names):
        #         for c_idx, name_j in enumerate(var_names):
        #             if name_i in result.var_names and name_j in result.var_names:
        #                 idx_i = result.var_names.index(name_i)
        #                 idx_j = result.var_names.index(name_j)
        #                 cov[r_idx, c_idx] = result.covar[idx_i, idx_j]
        # else:
        #     cov = np.diag([err_k1**2, err_k2**2, err_gbs**2])

        if heated_fit:
            x_vals = jnp.array([v["k1"], v["k2"], v["k_heat"], v["heat_pop"], v["gbs"]])

            if result.covar is not None:

                if len(result.covar) != len(x_vals):
                    rows = range(1, len(x_vals)+1)
                    cols = rows
                    cov = result.covar[np.ix_(rows, cols)]
                else:
                    cov = result.covar
                
            else:
                cov = np.diag([err_k1**2, err_k2**2, v["k_heat"]**2, v["heat_pop"]**2, err_gbs**2])

            bs_fidelity_jax, cov_f = propagate(_bs_fidelity_heated, x_vals, cov)

        else:
            x_vals = jnp.array([v["k1"], v["k2"], v["gbs"]])

            if result.covar is not None:
                
                if len(result.covar) != len(x_vals):
                    rows = range(1, len(x_vals)+1)
                    cols = rows
                    cov = result.covar[np.ix_(rows, cols)]
                else:
                    cov = result.covar
            else:
                cov = np.diag([err_k1**2, err_k2**2, err_gbs**2])

            bs_fidelity_jax, cov_f = propagate(_bs_fidelity, x_vals, cov)
        

        bs_fidelity = float(bs_fidelity_jax)
        bs_fidelity_err = float(jnp.sqrt(jnp.abs(cov_f)))

        results.append(
            {
                "filenum": filenum,
                "mode": mode,
                "current": current,
                "flux": flux,
                "bs_freq": bs_freq,
                "bs_amp": bs_amp,
                "t1": bs_t1,
                "t1_err": t1_err,
                "t2": bs_t2,
                "t2_err": t2_err,
                "pi_time": pi_time,
                "fidelity": bs_fidelity,
                "fidelity_err": bs_fidelity_err,
                "fit_result": result,
                "x_raw": time,
                "y_raw": y,
            }
        )

        # --- 6. Plotting ---
        fcolor = to_rgba(c, alpha=0.25)

        # Cleaned up the raw data label
        ax.plot(
            time,
            y,
            marker="o",
            linestyle="",
            color=c,
            markerfacecolor=fcolor,
            markeredgecolor=c,
            ms=8,
            label="Data",
        )

        time_fine = np.linspace(time[0], time[-1], 10000)
        fit_fine = result.eval(t=time_fine)

        # --- Prediction Band ---
        try:
            model_error = result.eval_uncertainty(t=time_fine, sigma=1)
            residual_noise = np.std(result.residual)
            prediction_error = np.sqrt(model_error**2 + residual_noise**2)
            if plotfills:
                ax.fill_between(
                    time_fine,
                    fit_fine - prediction_error,
                    fit_fine + prediction_error,
                    color=c,
                    alpha=0.15,
                    edgecolor="none",
                    label="Prediction Band",
                )
        except Exception:
            pass

        if bs_t2 < time.max():
            if plotlines:
                ax.axvline(
                    bs_t2,
                    linestyle=":",
                    color=c,
                    label=rf"$T_{{2,BS}} = {format_err(bs_t2, t2_err)} \mu s$",
                )
        if plotfits:
            ax.plot(
                time_fine,
                fit_fine,
                linestyle="-",
                color=c,
                label=rf"$T_1={format_err(bs_t1, t1_err)} \mu s$",
            )
        if plotlines:
            ax.axvline(
                pi_time,
                linestyle="--",
                color=c,
                label=rf"$t_{{\pi}} = {format_err(pi_time, pi_time_err)} \mu s$"
                + "\n"
                + rf"$\mathscr{{F}}_{{BS}} = {format_err(bs_fidelity, bs_fidelity_err)}$",
            )

        # --- Text Box for Metadata ---
        info_text = (
            rf"$\nu_{{bs}}$ = {bs_freq:.5f} GHz" + "\n"
            f"amp = {bs_amp:.4f}\n"
            f"range = {bs_drive_range} dBm\n"
            f"Current = {current * 1e3:.3f} mA\n"
            rf"Flux = {flux:.3f} $\Phi_0$" + "\n"
            rf"n_avgs = $2^{{{average_exponent}}}$"
        )

        props = dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
            edgecolor="lightgray",
            linewidth=1,
        )
        ax.text(
            0.05,
            0.05,
            info_text,
            transform=ax.transAxes,
            fontsize="x-small",
            verticalalignment="center",
            horizontalalignment="right",
            bbox=props,
        )

        ax.set(xlabel=r"t ($\mu s$)", ylabel="$P_e$")
        ax.set_xlim(-time.max() * 0.1, time.max() * 1.1)
        ax.set_ylim(-0.05, 1.05)

        axtitle = (f"Storage {mode}" if mode != 0 else "SNAIL") + f", file {filenum}"

        # Legend strictly for fit lines, moved out of the way of the text box
        ax.legend(fontsize="small", title=axtitle, loc="upper right")

    for idx in range(len(tasks), len(axs)):
        fig.delaxes(axs[idx])

    # Removed current from suptitle
    fig.suptitle(f"Beamsplitter Rabi {alice_or_bob.capitalize()}", y=1.02, fontsize=32)
    plt.tight_layout()

    return pd.DataFrame(results), fig


# BS bang-bang plotter
# =============================================================================
# 1. MATH & FORMATTING
# =============================================================================
def bangbang_envelope_func(n, A, gamma, B):
    return B + A * np.exp(-gamma * n)


# =============================================================================
# 2. FIT WRAPPER
# =============================================================================
def fit_bangbang(n_fit, y_fit, custom_settings=None):
    model = Model(bangbang_envelope_func)
    params = model.make_params()

    rough_B = np.clip(np.mean(y_fit[-3:]), 0.0, 1.0) if len(y_fit) >= 3 else 0.5
    rough_A = y_fit[0] - rough_B if len(y_fit) > 0 else 0.5
    rough_gamma = 0.02

    params["A"].set(value=rough_A, min=-1.2, max=1.2)
    params["B"].set(value=rough_B, min=0.0, max=1.0)
    params["gamma"].set(value=rough_gamma, min=0.0, max=1.0)

    if custom_settings:
        for param_name, settings in custom_settings.items():
            params[param_name].set(**settings)

    return model.fit(y_fit, params, n=n_fit)


# =============================================================================
# 3. ORCHESTRATOR (Powered by LabData)
# =============================================================================
def analyze_bangbang(
    filenums,
    startfits,
    modes,
    data_path,
    fit_mode="both",
    alice_or_bob="alice",
    suffix="bs_b3_bangbang",
    global_overrides=None,
    fit_overrides=None,
    plot_2d = False,
    sweep_2d_plot = None
    ):
    if fit_overrides is None:
        fit_overrides = {}

    tasks = list(zip(filenums, startfits, modes))
    n_files = len(tasks)
    ncols = int(np.ceil(np.sqrt(n_files)))
    nrows = int(np.ceil(n_files / ncols)) if ncols > 0 else 1
    figsize = np.array(plt.rcParams["figure.figsize"]) * np.array([ncols, nrows]) * 1.4
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axs = axs.flatten()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] * (n_files // 5 + 1)

    envelopes_to_process = ["even", "odd"] if fit_mode == "both" else [fit_mode]
    results = []
    last_current = 0

    xs_even = []
    xs_odd = []
    ys_even = []
    ys_odd = []
    ys = []
    bs_amps = []
    bs_freqs = []


    for ii, (filenum, startfit, mode) in enumerate(tasks):
        ax, c = axs[ii], colors[ii]

        # --- Clean I/O ---
        try:
            data = LabData(data_path, filenum=filenum, suffix=suffix)
        except FileNotFoundError:
            ax.text(
                0.5,
                0.5,
                f"File {filenum}\nMode {mode}\nNot Found",
                ha="center",
                va="center",
            )
            ax.axis("off")
            continue

        n_pulses = data.xpts
        y = data.P_e

        bs_freq = data.exp.get("bs_freq")
        if bs_freq in (None, "None"):
            bs_freq = data.q0[f"bs_{alice_or_bob}_freqs"][mode] / 1e9
        bs_freqs.append(bs_freq)

        bs_amp = data.exp.get("bs_amplitude")
        if bs_amp in (None, "None"):
            bs_amp = data.q0[f"bs_{alice_or_bob}_amps"][mode]
        bs_amps.append(bs_amp)
        
        bs_ramp_len = data.exp.get("bs_ramp")
        if bs_ramp_len in (None, "None"):
            bs_ramp_len = data.q0[f"bs_{alice_or_bob}_ramp_lens"][mode]
        bs_length = data.exp.get("bs_length")        
        if bs_length in (None, "None"):
            bs_pi_time_flat = data.q0[f"bs_{alice_or_bob}_flat_lens"][mode]
            pulse_duration_us = (bs_pi_time_flat + 2 * bs_ramp_len) * 1e6
        else:
            bs_pi_time_flat = bs_length - 2 * bs_ramp_len
            pulse_duration_us = (bs_length) * 1e6
        bs_pi_time_flat *= 1e6

        last_current = data.exp.get("flux_current", 0)

        ax.plot(
            n_pulses,
            y,
            marker="o",
            linestyle=":",
            color=c,
            markerfacecolor="none",
            markeredgecolor=to_rgba(c, alpha=0.2),
            ms=5,
            alpha=0.3,
        )
        ax.plot(
            [], [], " ", label=r"$\nu_{bs}=$" + f"{bs_freq/1e9:.5f} GHz\namp = {bs_amp:.6f}"
        )

        for env_type in envelopes_to_process:
            mask = (n_pulses % 2 == 1) if env_type == "odd" else (n_pulses % 2 == 0)
            line_style = "--" if env_type == "odd" else "-."
            env_color = (
                c
                if env_type == "odd"
                else (to_rgba(c, alpha=0.8) if fit_mode == "both" else c)
            )
            fill_style = to_rgba(c, alpha=0.6) if env_type == "odd" else "none"

            n_env, y_env = n_pulses[mask], y[mask]
            if env_type == "odd":
                xs_odd.append(n_env)
                ys_odd.append(y_env)
            else:
                xs_even.append(n_env)
                ys_even.append(y_env)
            ax.plot(
                n_env,
                y_env,
                marker="o",
                linestyle="",
                color=env_color,
                markerfacecolor=fill_style,
                markeredgecolor=env_color,
                ms=7,
            )

            current_settings = (
                copy.deepcopy(global_overrides) if global_overrides else {}
            )
            specific_overrides = fit_overrides.get(
                (filenum, mode, env_type), fit_overrides.get((filenum, mode), {})
            )
            for param_name, settings in specific_overrides.items():
                if param_name in current_settings:
                    current_settings[param_name].update(settings)
                else:
                    current_settings[param_name] = settings

            result = fit_bangbang(
                n_env[int(startfit) :],
                y_env[int(startfit) :],
                custom_settings=current_settings,
            )

            val_A, val_B, val_gamma = (
                result.params["A"].value,
                result.params["B"].value,
                result.params["gamma"].value,
            )
            err_gamma = result.params["gamma"].stderr or 0.0

            n_fine = np.linspace(n_pulses[0], n_pulses[-1], 1000)

            # Error bands
            if err_gamma > 0:
                fit_upper = bangbang_envelope_func(
                    n_fine, val_A, max(0.0, val_gamma - err_gamma), val_B
                )
                fit_lower = bangbang_envelope_func(
                    n_fine, val_A, val_gamma + err_gamma, val_B
                )
                ax.fill_between(
                    n_fine,
                    np.minimum(fit_lower, fit_upper),
                    np.maximum(fit_lower, fit_upper),
                    color=env_color,
                    alpha=0.15,
                    edgecolor="none",
                )

            if val_gamma > 0:
                t_eff = pulse_duration_us / val_gamma
                pulse_fidelity = np.exp(-val_gamma)
                err_fidelity = pulse_fidelity * err_gamma
            else:
                t_eff, pulse_fidelity, err_fidelity = np.inf, 1.0, 0.0

            # --- UPDATED LEGEND FORMATTING ---
            label_text = (
                f"{env_type.capitalize()} Env:\n"
                f"$\\mathcal{{F}}_{{BS}} = {format_err(pulse_fidelity, err_fidelity)}$\n"
                f"$T_{{eff}} = {format_err(t_eff, 0) if t_eff != np.inf else 'NaN'} \ \mu s$"
            )

            ax.plot(
                n_fine,
                result.eval(n=n_fine),
                linestyle=line_style,
                color=env_color,
                lw=2,
                label=label_text,
            )

            results.append(
                {
                    "filenum": filenum,
                    "mode": mode,
                    "env_type": env_type,
                    "bs_freq": bs_freq,
                    "bs_amp": bs_amp,
                    "gamma": val_gamma,
                    "gamma_err": err_gamma,
                    "t_eff": t_eff,
                    "fidelity": pulse_fidelity,
                    "fidelity_err": err_fidelity,  # Explicit fidelity error captured
                    "bs_pi_time_flat": bs_pi_time_flat,
                    "bs_ramp_len": bs_ramp_len,
                    "fit_result": result,  # The raw lmfit object for debugging
                    "x_raw": n_env,
                    "y_raw": y_env,
                }
            )
        
        n_fine2 = np.linspace(n_pulses[0], n_pulses[-1]*10, 10000)
        upper_fit = results[-2]['fit_result'].eval(n=n_fine2)
        lower_fit = results[-1]['fit_result'].eval(n=n_fine2)
        intersection_idx = np.argmin(np.abs(upper_fit - lower_fit))
        intersection_n = n_fine2[intersection_idx]
        results[-1].update(
            {"intersection_n": intersection_n}
            )

        ax.set(
            xlabel="Number of $\pi$ pulses (n)",
            ylabel="$P_e$",
            xlim=(-n_pulses.max() * 0.1, n_pulses.max() * 1.1),
            ylim=(-0.05, 1.05),
        )
        axtitle = (f"Storage {mode}" if mode != 0 else "SNAIL") + f", file {filenum}"
        ax.legend(fontsize="small", title=axtitle, loc="upper right")

    for idx in range(len(tasks), len(axs)):
        fig.delaxes(axs[idx])
    fig.suptitle(
        rf"Beamsplitter Bang-Bang {alice_or_bob.capitalize()}, Current={last_current * 1e3:.3f} mA",
        y=1.02,
        fontsize=32,
    )
    plt.tight_layout()

    if plot_2d:
        xs_even = np.array(xs_even)
        xs_odd = np.array(xs_odd)
        ys_even = np.array(ys_even)
        ys_odd = np.array(ys_odd)
        sweep = bs_amps if sweep_2d_plot == "amp" else bs_freqs
        fig2d, ax2d = plt.subplots(1,2, figsize=(13, 4))
        ax2d[0].pcolormesh(xs_even, sweep, ys_even, shading='auto', cmap='viridis')
        ax2d[0].set_xlabel("Number of $\pi$ pulses (n)")
        ax2d[0].set_ylabel(f"Beamsplitter {'Amplitude' if sweep_2d_plot == 'amp' else 'Frequency'}")
        ax2d[0].set_title("Even Pulses")
        ax2d[1].pcolormesh(xs_odd, sweep, ys_odd, shading='auto', cmap='viridis')
        ax2d[1].set_xlabel("Number of $\pi$ pulses (n)")
        ax2d[1].set_ylabel(f"Beamsplitter {'Amplitude' if sweep_2d_plot == 'amp' else 'Frequency'}")
        ax2d[1].set_title("Odd Pulses")
        # todo make colorbar
        # plt.subplots_adjust(top=0.88)
        # cbar_ax = fig2d.add_axes([0.25, 0.92, 0.5, 0.02])
        # fig2d.colorbar(ax2d[0].collections[0], cax=cbar_ax, orientation='horizontal', label='$P_e$', pad = 0.5)

    return pd.DataFrame(results), fig


# Andre's error propagation function
def propagate(f, x, cov):
    f_x = f(x)
    dims = tuple(map(jnp.ndim, (f_x, x, cov)))
    cov = jnp.asarray(cov) if dims[2] >= 1 else cov

    n = jnp.shape(x)[0] if dims[1] >= 1 else 1
    m = jnp.shape(f_x)[0] if dims[0] >= 1 else 1
    if m < n:
        jac = jacrev(f)(x)
    else:
        jac = jacfwd(f)(x)

    # single variable scalar function
    if dims == (0, 0, 0):
        cov_f = jac * cov * jac

    # single variable scalar function (with batch dim)
    # and multivariable (scalar & vector) function
    elif dims in ((1, 1, 1), (0, 1, 2), (1, 1, 2)):
        cov_f = jac @ cov @ jac.T

    # single variable vector function
    elif dims == (1, 0, 0):
        cov_f = cov * jnp.outer(jac, jac)

    # multivariable scalar function (with batch dim)
    elif dims == (1, 2, 3):
        cov_f = jnp.einsum("bib,bij,bjb->b", jac, cov, jac)

    # single variable vector function (with batch dim)
    elif dims == (2, 1, 1):
        cov_f = jnp.einsum("ibb,b,jbb->bij", jac, cov, jac)

    # multivariable vector function (with batch dim)
    elif dims == (2, 2, 3):
        cov_f = jnp.einsum("ibjb,bjk,lbkb->bil", jac, cov, jac)

    else:
        raise ValueError(f"Unsupported input dimensions: {dims}")

    return f_x, cov_f

def diag_square(x):
    x2 = np.square(x)

    if np.ndim(x2) <= 1:
        return np.diag(x2)

    n = x2.shape[0]
    return np.einsum("ji,jk->ijk", x2, np.eye(n))


# T1 plotter
# =============================================================================
# 1. MATH & FORMATTING
# =============================================================================
def t1_decay_func(t, A, T1, B):
    return A * np.exp(-t / T1) + B


# =============================================================================
# 2. FIT WRAPPER
# =============================================================================
def fit_t1(t, y, custom_settings=None):
    model = Model(t1_decay_func)
    params = model.make_params()

    rough_A = np.max(y) - np.min(y)
    if len(y) > 0 and y[0] < y[-1]:
        rough_A = -rough_A
    rough_B = np.mean(y[-3:]) if len(y) > 3 else np.min(y)
    rough_T1 = np.mean(t) if len(t) > 0 else 1.0

    params["A"].set(value=rough_A, min=-2.0, max=2.0)
    params["T1"].set(value=rough_T1, min=0.001, max=1000)
    params["B"].set(value=rough_B, min=-1.0, max=1.0)

    if custom_settings:
        for param_name, settings in custom_settings.items():
            params[param_name].set(**settings)

    return model.fit(y, params, t=t)


def analyze_t1(
    filenums,
    modes,
    data_path,
    alice_or_bob="alice",
    suffix=None,
    global_overrides=None,
    fit_overrides=None,
):
    if fit_overrides is None:
        fit_overrides = {}

    tasks = list(zip(filenums, modes))
    n = len(tasks)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols)) if ncols > 0 else 1
    figsize = np.array(plt.rcParams["figure.figsize"]) * np.array([ncols, nrows])
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axs = axs.flatten()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] * (n // 5 + 1)

    results = []

    for ii, (filenum, mode) in enumerate(tasks):
        ax, c = axs[ii], colors[ii]

        if suffix is None:
            current_suffix = f"bs_{alice_or_bob[0]}{mode}_t1"
        elif "{" in str(suffix):
            current_suffix = str(suffix).format(alice_or_bob=alice_or_bob, mode=mode, a_or_b=alice_or_bob[0])
        else:
            current_suffix = str(suffix)

        try:
            data = LabData(data_path, filenum=filenum, suffix=current_suffix)
        except FileNotFoundError:
            ax.text(
                0.5,
                0.5,
                f"File {filenum}\nMode {mode}\nNot Found",
                ha="center",
                va="center",
            )
            ax.axis("off")
            continue

        time = data.xpts * 1e6
        y = data.P_e

        current = data.exp.get("flux_current", 0)
        flux = data.q0.get("flux", 0.0)

        current_settings = copy.deepcopy(global_overrides) if global_overrides else {}
        specific_overrides = fit_overrides.get((filenum, mode), {})

        for param_name, settings in specific_overrides.items():
            if param_name in current_settings:
                current_settings[param_name].update(settings)
            else:
                current_settings[param_name] = settings

        result = fit_t1(time, y, custom_settings=current_settings)
        
        t1_val = result.params["T1"].value
        t1_err = result.params["T1"].stderr or 0.0

        results.append(
            {
                "filenum": filenum,
                "mode": mode,
                "current": current,
                "flux": flux,
                "t1": t1_val,
                "t1_err": t1_err,
                "fit_result": result,
                "x_raw": time,
                "y_raw": y,
            }
        )

        fcolor = to_rgba(c, alpha=0.25)
        ax.plot(
            time,
            y,
            marker="o",
            linestyle="",
            color=c,
            markerfacecolor=fcolor,
            markeredgecolor=c,
            ms=8,
            label="Data",
        )

        time_fine = np.linspace(time.min(), time.max(), 1000)
        fit_fine = result.eval(t=time_fine)

        try:
            model_error = result.eval_uncertainty(t=time_fine, sigma=1)
            residual_noise = np.std(result.residual)
            prediction_error = np.sqrt(model_error**2 + residual_noise**2)
            ax.fill_between(
                time_fine,
                fit_fine - prediction_error,
                fit_fine + prediction_error,
                color=c,
                alpha=0.15,
                edgecolor="none",
                label="Prediction Band",
            )
        except Exception:
            pass

        ax.plot(
            time_fine, 
            fit_fine, 
            c=c, 
            linestyle="-", 
            label=rf"$T_1 = {format_err(t1_val, t1_err)}\ \mu s$"
        )

        ax.axvline(t1_val, linestyle="--", color=c)

        info_text = (
            f"Current = {current * 1e3:.3f} mA\n"
            rf"Flux = {flux:.3f} $\Phi_0$"
        )

        props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="lightgray", linewidth=1)
        ax.text(
            0.95,
            0.95,
            info_text,
            transform=ax.transAxes,
            fontsize="x-small",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        ax.set(xlabel=r"t ($\mu s$)", ylabel="$P_e$")
        ax.set_xlim(-time.max() * 0.1, time.max() * 1.1)
        
        axtitle = (f"Storage {mode}" if mode != 0 else "SNAIL") + f", file {filenum}"
        ax.legend(fontsize="small", title=axtitle, loc="center right")

    for idx in range(len(tasks), len(axs)):
        fig.delaxes(axs[idx])

    fig.suptitle(f"T1 {alice_or_bob.capitalize()}", y=1.02, fontsize=32)
    plt.tight_layout()

    return pd.DataFrame(results), fig

# =============================================================================
# Ramsey / Echo Plotter
# =============================================================================
def ramsey_decay_func(t, A, T2, f, phi, B):
    return A * np.exp(-t / T2) * np.cos(2 * np.pi * f * t + phi) + B

def fit_ramsey(t, y, is_echo=False, custom_settings=None):
    model = Model(ramsey_decay_func)
    params = model.make_params()
    
    rough_A = (np.max(y) - np.min(y)) / 2.0
    if len(y) > 0 and y[0] < np.mean(y):
        rough_A = -rough_A
    rough_B = np.mean(y)
    rough_T2 = np.mean(t) if len(t) > 0 else 1.0
    
    params["A"].set(value=rough_A, min=-2.0, max=2.0)
    params["T2"].set(value=rough_T2, min=0.001, max=1000)
    params["B"].set(value=rough_B, min=-1.0, max=1.0)
    
    # guess f from FFT or just set a standard bound
    n = len(t)
    if n > 3:
        dt = t[1] - t[0] if n > 1 else 1.0
        freqs = np.fft.rfftfreq(n, d=dt)
        fft_vals = np.abs(np.fft.rfft(y - np.mean(y)))
        guess_f = freqs[np.argmax(fft_vals)]
        if guess_f == 0:
            guess_f = 1 / (t[-1] - t[0])
    else:
        guess_f = 0.1
    params["f"].set(value=guess_f, min=0.0)
    params["phi"].set(value=0.0, min=-np.pi, max=np.pi)

    if custom_settings:
        for param_name, settings in custom_settings.items():
            params[param_name].set(**settings)

    return model.fit(y, params, t=t)

def analyze_ramsey(
    filenums,
    modes,
    data_path,
    alice_or_bob="alice",
    suffix=None,
    global_overrides=None,
    fit_overrides=None,
    is_echo=False,
):
    if fit_overrides is None:
        fit_overrides = {}

    tasks = list(zip(filenums, modes))
    n = len(tasks)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols)) if ncols > 0 else 1
    figsize = np.array(plt.rcParams["figure.figsize"]) * np.array([ncols, nrows])
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axs = axs.flatten()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] * (n // 5 + 1)

    results = []

    for ii, (filenum, mode) in enumerate(tasks):
        ax, c = axs[ii], colors[ii]

        if suffix is None:
            seq_str = "echo" if is_echo else "ramsey"
            current_suffix = f"bs_{alice_or_bob[0]}{mode}_{seq_str}"
        elif "{" in str(suffix):
            current_suffix = str(suffix).format(alice_or_bob=alice_or_bob, mode=mode, a_or_b=alice_or_bob[0])
        else:
            current_suffix = str(suffix)

        try:
            data = LabData(data_path, filenum=filenum, suffix=current_suffix)
        except FileNotFoundError:
            expected_file = f"{str(filenum).zfill(5)}_{current_suffix}.h5" if current_suffix else f"{str(filenum).zfill(5)}.h5"
            ax.text(
                0.5,
                0.5,
                f"File {filenum}\nMode {mode}\nNot Found\n(Expected: {expected_file})",
                ha="center",
                va="center",
                fontsize=8,
                color="red"
            )
            ax.axis("off")
            continue

        time = data.xpts
        if time.ndim > 1:
            time = time[0]
        time = time * 1e6
        
        y = data.P_e

        current = data.exp.get("flux_current", 0)
        flux = data.q0.get("flux", 0.0)

        current_settings = copy.deepcopy(global_overrides) if global_overrides else {}
        specific_overrides = fit_overrides.get((filenum, mode), {})

        for param_name, settings in specific_overrides.items():
            if param_name in current_settings:
                current_settings[param_name].update(settings)
            else:
                current_settings[param_name] = settings

        result = fit_ramsey(time, y, is_echo=is_echo, custom_settings=current_settings)
        
        t2_val = result.params["T2"].value
        t2_err = result.params["T2"].stderr or 0.0
        
        f_val = result.params["f"].value
        f_err = result.params["f"].stderr or 0.0

        results.append(
            {
                "filenum": filenum,
                "mode": mode,
                "current": current,
                "flux": flux,
                "t2": t2_val,
                "t2_err": t2_err,
                "f": f_val,
                "f_err": f_err,
                "fit_result": result,
                "x_raw": time,
                "y_raw": y,
            }
        )

        fcolor = to_rgba(c, alpha=0.25)
        ax.plot(
            time,
            y,
            marker="o",
            linestyle="",
            color=c,
            markerfacecolor=fcolor,
            markeredgecolor=c,
            ms=8,
            label="Data",
        )

        time_fine = np.linspace(time.min(), time.max(), 1000)
        fit_fine = result.eval(t=time_fine)

        try:
            model_error = result.eval_uncertainty(t=time_fine, sigma=1)
            residual_noise = np.std(result.residual)
            prediction_error = np.sqrt(model_error**2 + residual_noise**2)
            ax.fill_between(
                time_fine,
                fit_fine - prediction_error,
                fit_fine + prediction_error,
                color=c,
                alpha=0.15,
                edgecolor="none",
                label="Prediction Band",
            )
        except Exception:
            pass

        label_str = rf"$T_2^* = {format_err(t2_val, t2_err)}\ \mu s$" if not is_echo else rf"$T_{{2,E}} = {format_err(t2_val, t2_err)}\ \mu s$"
        if not is_echo:
            label_str += "\n" + rf"$f = {format_err(f_val, f_err)}$ MHz"
            
        ax.plot(
            time_fine, 
            fit_fine, 
            c=c, 
            linestyle="-", 
            label=label_str
        )

        ax.axvline(t2_val, linestyle="--", color=c)

        info_text = (
            f"Current = {current * 1e3:.3f} mA\n"
            rf"Flux = {flux:.3f} $\Phi_0$"
        )

        props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="lightgray", linewidth=1)
        ax.text(
            0.95,
            0.95,
            info_text,
            transform=ax.transAxes,
            fontsize="x-small",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        ax.set(xlabel=r"t ($\mu s$)", ylabel="$P_e$")
        ax.set_xlim(-time.max() * 0.1, time.max() * 1.1)
        
        axtitle = (f"Storage {mode}" if mode != 0 else "SNAIL") + f", file {filenum}"
        ax.legend(fontsize="small", title=axtitle, loc="upper right")

    for idx in range(len(tasks), len(axs)):
        fig.delaxes(axs[idx])

    title_prefix = "Echo" if is_echo else "Ramsey"
    fig.suptitle(f"{title_prefix} {alice_or_bob.capitalize()}", y=1.02, fontsize=32)
    plt.tight_layout()

    return pd.DataFrame(results), fig

def print_fit_result(result_list, multiplier, line_break=False):
    results = [float(result) * multiplier for result in result_list]
    if line_break:
        if results:
            results = [f"{result}," for result in results]
            results[0] = "[" + results[0]
            results[-1] = results[-1][:-1] + "]"
        else:
            print("[]")
        for result in results:
            print(result)
    else:
        print(results)


# =============================================================================
# MATH HELPERS
# =============================================================================
def temperature_q(nu, rat):
    Kb = 1.38e-23
    h = 2 * np.pi * 1.054e-34
    return h * nu / (Kb * np.log(1 / rat))

def occupation_r(nu, T):
    Kb = 1.38e-23
    h = 2 * np.pi * 1.054e-34
    return 1 / (np.exp(h * nu / (Kb * T)) - 1)

def nth_from_contrast(contrasts):
    ratio = contrasts[1] / contrasts[0]
    return 1 / (ratio - 1)

def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)