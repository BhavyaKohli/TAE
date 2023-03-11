print("Importing all assests from base.py...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def npy(x,dtype=None):
	return np.array(x,dtype=dtype)

axes_off = np.vectorize(lambda ax:ax.axis('off'))

import inspect

def get_var_name(var,depth=2):
	"""
	Returns the name of the argument variable `var` as a string
	"""
	if depth==1: callers_local_vars = inspect.currentframe().f_back.f_locals.items()
	if depth==2: callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
	else: return
	return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def shapes(*args):
	"""
	Prints the shapes of all input arrays
	"""
	maxlen = 0
	for i in args: maxlen = max(len(get_var_name(i)[0]),maxlen)
	
	for i in args: 
		name = get_var_name(i)[0]
		print(name," "*(maxlen-len(name)),":",i.shape)

def values(*args):
	"""
	Prints the values of all inputs
	"""
	maxlen = 0
	for i in args: maxlen = max(len(get_var_name(i)[0]),maxlen)
	
	for i in args: 
		name = get_var_name(i)[0]
		print(name," "*(maxlen-len(name)),":",i)

def plot_history(history,xlabel="epochs"):
    plt.figure(figsize=(10,7))

    for key in history.history:
        plt.plot(history.history[key],label=key)
    plt.legend()
    plt.xlabel(xlabel)
    plt.show()

def minmax(*args):
	for X in args:
		print(f"{get_var_name(X)[0]} - Minimum : {X.min()} | Maximum : {X.max()}")

print("Imported modules   : numpy, pandas, matplotlib.pyplot")
print("Imported functions : npy, axes_off, get_var_name, shapes, tqdm, plot_history, minmax, values")