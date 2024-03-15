import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import lmfit
from lmfit.models import GaussianModel, LognormalModel
from scipy.interpolate import CubicSpline, UnivariateSpline, interp1d, splrep, BSpline
from scipy.signal import find_peaks
import pandas as pd

def build_complex_model(input):
    """
    Input is of form: type_i, param_1, ..., param_i, ..., type_n, param_1, ...
    Where
    type: type of function, currently 'log' and 'normal'
    params: required number of comma separated params depending of type:
        - log: amplitude, center, sigma
        - normal: amplitude, center, sigma

    Example: "log, 1000, 3.0, 1.3, normal, 2000, 30.0, 1.5"

    Note that the lognormal is sensitive to input params.
    """
    # Parse input: remove whitespace and split on comma
    input_list = input.replace(" ", "").split(',')
    # Create model
    model = None
    n_log = 1
    n_normal = 1
    
    try:
        for i in range(0, len(input_list), 4):
            if "log" in input_list[i]:
                if i == 0:
                    model = LognormalModel(prefix=f"log_{n_log}_")
                    model.set_param_hint('amplitude', value=float(input_list[i + 1]), vary=True, min=0)
                    model.set_param_hint('center', value=np.log(float(input_list[i + 2])), vary=True, min=0, max=5.3) 
                    model.set_param_hint('sigma', value=np.log(float(input_list[i + 3])), vary=True, min=0.3, max=1.6)
                else:
                    log = LognormalModel(prefix=f"log_{n_log}_")
                    log.set_param_hint('amplitude', value=float(input_list[i + 1]), vary=True, min=0)
                    log.set_param_hint('center', value=np.log(float(input_list[i + 2])), vary=True, min=0, max=5.3) # Max 5.3 is about 200 nm
                    log.set_param_hint('sigma', value=np.log(float(input_list[i + 3])), vary=True, min=0.3, max=1.6) # Max 1.6 is reasonable for GSD
                    model += log
                n_log += 1
            
            if "norm" in input_list[i]:
                if i == 0:
                    model = GaussianModel(prefix=f"norm_{n_normal}_")
                    model.set_param_hint('amplitude', value=float(input_list[i + 1]), vary=True, min=0, max=100000000)
                    model.set_param_hint('center', value=float(input_list[i + 2]), vary=True, min=0, max=150)
                    model.set_param_hint('sigma', value=float(input_list[i + 3]), vary=True, min=0.1, max=30)
                else: 
                    normal = GaussianModel(prefix=f"norm_{n_normal}_")
                    normal.set_param_hint('amplitude', value=float(input_list[i + 1]), vary=True, min=0, max=100000000)
                    normal.set_param_hint('center', value=float(input_list[i + 2]), vary=True, min=0, max=150)
                    normal.set_param_hint('sigma', value=float(input_list[i + 3]), vary=True, min=0.1, max=20)
                    model += normal
                n_normal+=1

            print(input_list)
    
    except Exception as e:
        return model, str(e)

    return model, "Success"

def build_autofit_model(x, y):
    """
    Builds autofit model. Not yet implemented.
    """
    pass


def fit_complex_model(x, y, model):
    """
    Fits lmfit model to x, y data.

    Returns best fit array, dict of components, and dict of best fit params.
    """
    params = model.make_params()
    out = model.fit(y, params=params, x=x)     

    return out.best_fit, out.eval_components(), out.best_values


def gaussian(x, *args):
    """
    Args should be provided as *args in the function call.

    Args is a list of [amp1, center1, sigma1, amp2, center2, ...]

    Example of usage with Scipy.optimize: 
    popt, pcov = curve_fit(gaussian, x, y, p0=args)
    where popt are the best fit parameters.

    Code implemented with help from: https://stackoverflow.com/questions/71227196/most-pythonic-way-to-fit-multiple-gaussians-using-scipy-optimize
    """
    x = x.reshape(-1, 1)
    amp    = np.array(args[0::3]).reshape(1, -1)
    center = np.array(args[1::3]).reshape(1, -1)
    sigma  = np.array(args[2::3]).reshape(1, -1)

    return np.sum(amp * np.exp(-(x - center) ** 2 / (2 * sigma ** 2)), axis=1)


def fit_curve(x, y, func, init_args):
    """
    Fits func with Scipy.optimize curve_fit that takes
    a function, x, y and initial guess (init_args).

    Returns best fit argumets.
    """
    popt, pcov = curve_fit(func, x, y, p0=init_args)
    return popt


def reconstruct_gaussian(x, *args):
    """
    Returns numpy array each individual gaussian component
    from args which is a list of kind
    [amp1, center1, sigma1, amp2, center2, ...]
    """
    x = x.reshape(-1, 1)
    amp    = np.array(args[0::3]).reshape(1, -1)
    center = np.array(args[1::3]).reshape(1, -1)
    sigma  = np.array(args[2::3]).reshape(1, -1)

    return amp * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

if __name__=="__main__":
    """
    Example usage.
    """
    x = np.linspace(0, 10, 100)
    args = [1, 1, 1, 0.5, 5, 0.5]
    y = gaussian(x, *args) + 0.1 * (np.random.rand(len(x)) - 0.5)

    peaks, _ = find_peaks(y, distance=5, width=5)

    fig, ax = plt.subplots(1, 1)
    
    ax.plot(y, 'o')
    ax.plot(peaks, y[peaks], '*')

    plt.show()

