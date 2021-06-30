#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.signal import find_peaks 
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)


# Zunächst den Kontrast in Abhängigkeit der Polarisationsrichtung des Laserstrahls bestimmen
# 

# In[49]:

#Formel für die Standardabweichung des Mittelwerts
def stanni(Zahlen, Mittelwert):
    i=0
    s=0
    n=len(Zahlen)
    while i<len(Zahlen):
        s = s + (Zahlen[i] - Mittelwert)**2
        i = i + 1
    return np.sqrt(s/(n*(n-1)))

def linear(m, b, x):
    return m*x+b

def kontrast(U_max, U_min):
    return (U_max-U_min)/(U_max+U_min)

def glasnullen(n, dicke, lam, theta):
    return dicke/lam * ( (n - 1) * theta**2 /(2*n)) 

def n_glas(dicke, lam, theta, theta_0, nullen):
    return (1 - 2 * nullen * lam /(dicke * 3 * theta**2))**(-1)

def n_gas(nullen, lam, laenge, n_0):
    return nullen * lam / laenge + n_0

def mol_Brechung(R, p, T, n):
    return R*T*(n**2 -1)/(3*p)

def n_gas_T(m, b, p, T, T_norm):
    return unp.sqrt(1 + T/T_norm * ((m*p+b)**2 - 1))

def n_gas_A(A, p, R, T_norm):
    return unp.sqrt(1 + 3*A*p/(R*T_norm))


# In[50]:


polwi, U_max, U_min = np.genfromtxt("kontrast_data.txt", unpack = "True")


# In[51]:


plt.plot(polwi, kontrast(U_max, U_min), "x")
plt.xlabel("Polarisationswinkel [°]")
plt.ylabel("Kontrast")
plt.savefig("Kontrast.pdf")
plt.clf()


# Maximaler Kontrast bei Polarisationswinkel von 135° (U_max=3000mV U_min=290mV): 

# In[52]:


print(np.max(kontrast(U_max, U_min)))


# Brechungsindex des Glases bestimmen:
# 

# In[35]:


dicke = 0.001 #in Metern
theta_0 = 10 * np.pi/180 #in Grad
theta = 10 * np.pi/180 #in Grad
lam = 632.99e-9 #in Metern
Messreihe, nullen_raw = np.genfromtxt("null_glas.txt", unpack=True)
nullen_mean = np.sum(nullen_raw)/(np.size(nullen_raw))
nullen_mean_err = stanni(nullen_raw, nullen_mean)
nullen_mean = unp.uarray(nullen_mean, nullen_mean_err)


# In[36]:


n_gl = n_glas(dicke, lam, theta, theta_0, nullen_mean)


# In[37]:


print(nullen_mean)
print("Brechungsindex des vermessenen Glases")
print(n_gl)


# Brechungsindex der Luft bestimmen

# In[38]:


druck, nullen = np.genfromtxt("null_gas.txt", unpack = True)
druck = druck * 10**(-3) #Aus mbar in bar umrechnen
laenge = unp.uarray(0.1, 0.0001)
R = const.gas_constant


# In[43]:


n_gs = n_gas(nullen, lam, laenge, 1)

x = np.linspace(0, 1.013)
n_params, n_cov = scipy.optimize.curve_fit(linear, druck, noms(n_gs))
errors = np.sqrt(np.diag(n_cov))
print(errors)
n_params_unc = unp.uarray(n_params, errors)
print(n_params_unc)


# In[44]:


plt.plot(druck, noms(n_gs), "x")
plt.xlabel("Druck [bar]")
plt.ylabel("Brechungsindex")
plt.plot(x, linear(x, *n_params))
plt.savefig("druck_lin.pdf")


# In[46]:


n_norm = n_gas_T(n_params_unc[0], n_params_unc[1], 1.013, 295.05, 288.15)
print("Brechungsindex von Luft")
print(n_norm)


# Brechungsindex über letzten Wert messen: p=1006 mbar, Nulldurchgänge=42

# In[55]:


n_1006 = 42*lam/laenge + 1
A = mol_Brechung(R, 1.006, 295.05, n_1006)
n_norm = n_gas_A(A, 1.013, R, 288.15)
print("Molrefraktion A bei 1006mbar")
print(A)
print("Brechungsindex von Luft bei Normalbedingungen")
print(n_norm)


# In[ ]:





# In[ ]:




