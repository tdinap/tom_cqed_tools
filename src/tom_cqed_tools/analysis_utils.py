# tools for plotting and analyzing cavity resonators

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# Add missing imports for constants
import scipy.constants as const
from scipy import optimize
from scipy.constants import e, h
from tabulate import tabulate

# from scipy.optimize import curve_fit


Phi0 = h / (2 * e)
hbar = h / (2 * np.pi)

import slab.dsfit as dsf
from slab import *


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
