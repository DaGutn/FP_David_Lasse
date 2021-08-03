import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit
import scipy.constants as c

###########################################################################################################
### Mittelwert der Magnetfeldrichtungen B(z) gegen Ort z von der Mitte des Probenspaltes aus auftragen

#Zu fittende Funktion
def x_hoch4(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d * x + e

#Daten einlesen
df_B = pd.read_csv('data/Magnetfeld.csv', delimiter=';')
z = df_B['z'].to_numpy()
B_1 = df_B['Magnetfeldstaerke1'].to_numpy()
B_2 = df_B['Magnetfeldstaerke2'].to_numpy()
B = (np.abs(B_1) + B_2) / 2
print(B)

#Fitten
params_B, cov_B = curve_fit(x_hoch4, z[1:], B[1:])
errors_B = np.sqrt(np.diag(cov_B))

#Plotten
fig, ax = plt.subplots()

z_lin = np.linspace(z[0], z[-1], 200)
ax.plot(z, B, 'b.', label='Messwerte')
ax.plot(z_lin, x_hoch4(z_lin, *params_B), 'orange', alpha=0.8, label='Fit')
ax.set_xlabel(r'$\Delta z \;$ [mm]')
ax.set_ylabel(r'$B(z) \;$ [mT]')
ax.legend(markerscale=2)

plt.savefig('plots/Magnetfeld.pdf')

B_max = np.max(x_hoch4(z_lin, *params_B)) * 10**(-3)
print(r'B_max =', B_max)
print('\n')

#Parameter
parameter_B = unp.uarray(params_B, errors_B)
for name, param in zip(('a','b','c','d','e'), parameter_B):
    print(r'{0}: {1:.8f}'.format(name, param))
print('\n')


###########################################################################################################
### Messwerte von hochreinem und zwei dotierten Proben GaAs plotten

#Daten einlesen und zu plottende Variablen berechnen
df = pd.read_csv('data/Winkel_Filter.csv', delimiter=';', decimal=',')

lamda = df['Filter_lambda'].to_numpy(dtype='float64')
lamda2 = lamda**2

theta_rein_1 = df['GaAs_hochrein_Winkel_1'].to_numpy() * np.pi / 180
theta_rein_2 = df['GaAs_hochrein_Winkel_2'].to_numpy() * np.pi / 180
theta_rein_nom = (theta_rein_1 - theta_rein_2) / (2 * 5.11)

theta_dot1_1 = df['GaAs_dot1_Winkel_1'].to_numpy() * np.pi / 180
theta_dot1_2 = df['GaAs_dot1_Winkel_2'].to_numpy() * np.pi / 180
theta_dot1_nom = (theta_dot1_1 - theta_dot1_2) / (2 * 1.34)

theta_dot2_1 = df['GaAs_dot2_Winkel_1'].to_numpy() * np.pi / 180
theta_dot2_2 = df['GaAs_dot2_Winkel_2'].to_numpy() * np.pi / 180
theta_dot2_nom = (theta_dot2_1 - theta_dot2_2) / (2 * 1.296)

theta_diff1 = theta_dot1_nom - theta_rein_nom
theta_diff2 = theta_dot2_nom - theta_rein_nom
#theta_diff1 = np.append(theta_diff1[0:6], theta_diff1[8])
#theta_diff2 = np.append(theta_diff2[0:6], theta_diff2[8])

#Plotten
fig, ax = plt.subplots()

ax.scatter(lamda2, theta_rein_nom, s=25, c='blue', marker='x', label='hochrein')
ax.scatter(lamda2, theta_dot1_nom, s=25, c='limegreen', marker='x', label=r'$1,2 \cdot 10^{18} / \mathrm{cm}^3$ n-dotiert')
ax.scatter(lamda2, theta_dot2_nom, s=25, c='crimson', marker='x', label=r'$2,8 \cdot 10^{18} / \mathrm{cm}^3$ n-dotiert')
ax.set_xlabel(r'$\lambda^2 \; [\mu\mathrm{m}^2]$')
ax.set_ylabel(r'$\theta_{\mathrm{norm}} \; [\frac{\mathrm{rad}}{\mathrm{mm}}]$')

#Legende (Labels in umgekehrter Reihenfolge)
handles, labels = ax.get_legend_handles_labels()
#handles, labels = handles[::-1], labels[::-1]
# Folgendes sortiert nach Länge des Labels (kürzestes zuerst)
#handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1], reverse=True))
ax.legend(handles, labels, markerscale=1.5, scatteryoffsets=[0.5])

plt.savefig('plots/Winkel_normiert.pdf')


###########################################################################################################
### Differenz der Messwerte von dotiertem zu hochreinem GaAs plotten, fitten und auswerten

#lamda = np.append(lamda[0:6], lamda[8])
#lamda2 = lamda**2

#Zu fittende Funktion
def lin(x, m):
    return m * x

### Fitten für die 1. dotierte Probe
params1, cov1 = curve_fit(lin, lamda2, theta_diff1)
errors1 = np.sqrt(np.diag(cov1))

#Plotten für die 1. dotierte Probe
fig, ax = plt.subplots()

ax.scatter(lamda2, theta_diff1, s=25, c='limegreen', marker='x', label=r'$1,2 \cdot 10^{18} / \mathrm{cm}^3$ n-dotiert')
ax.plot(lamda2, lin(lamda2, *params1), 'orange', alpha=0.9, label='Ausgleichsgerade')
ax.set_xlabel(r'$\lambda^2 \; [\mu\mathrm{m}^2]$')
ax.set_ylabel(r'$\Delta \theta_{\mathrm{norm}} \; [\frac{\mathrm{rad}}{\mathrm{mm}}]$')

#Legende (Labels in umgekehrter Reihenfolge)
handles, labels = ax.get_legend_handles_labels()
handles, labels = handles[::-1], labels[::-1]
ax.legend(handles, labels, markerscale=2, scatteryoffsets=[0.5])

plt.savefig('plots/Faraday_1.pdf')


### Fitten für die 1. dotierte Probe
params2, cov2 = curve_fit(lin, lamda2, theta_diff2)
errors2 = np.sqrt(np.diag(cov2))

#Plotten für die 2. dotierte Probe
fig, ax = plt.subplots()

ax.scatter(lamda2, theta_diff2, s=25, c='crimson', marker='x', label=r'$2,8 \cdot 10^{18} / \mathrm{cm}^3$ n-dotiert')
ax.plot(lamda2, lin(lamda2, *params2), 'orange', alpha=0.9, label='Ausgleichsgerade')
ax.set_xlabel(r'$\lambda^2 \; [\mu\mathrm{m}^2]$')
ax.set_ylabel(r'$\Delta \theta_{\mathrm{norm}} \; [\frac{\mathrm{rad}}{\mathrm{mm}}]$')

#Legende (Labels in umgekehrter Reihenfolge)
handles, labels = ax.get_legend_handles_labels()
handles, labels = handles[::-1], labels[::-1]
ax.legend(handles, labels, markerscale=2, scatteryoffsets=[0.5])

plt.savefig('plots/Faraday_2.pdf')


#Parameter
parameter1 = unp.uarray(params1, errors1)
parameter2 = unp.uarray(params2, errors2)
for name, param in zip(('m'), parameter1):
    print(r'{0}: {1:.8f}'.format(name, param))
for name, param in zip(('m'), parameter2):
    print(r'{0}: {1:.8f}'.format(name, param))
print('\n')


###########################################################################################################
### Effektive Masse der freien zusätzlichen Elektronen in den beiden dotierten Proben bestimmen

lam = np.sum(lamda) / 9
print(r'lamda_mitt =', lam)
print()

const = c.elementary_charge**3 / (8 * np.pi**2 * c.epsilon_0 * c.speed_of_light**3)

N_1 = 1.2 * 10**(24)
N_2 = 2.8 * 10**(24)
n = 3.34

uparams1 = ufloat(params1[0], errors1[0])
uparams2 = ufloat(params2[0], errors2[0])

m_eff1 = unp.sqrt(const * B_max * N_1 / (n * uparams1 * 10**(15)))
m_eff2 = unp.sqrt(const * B_max * N_1 / (n * uparams2 * 10**(15)))
print('m_eff1 = {:.4}'.format(m_eff1))
print('m_eff2 = {:.4}'.format(m_eff2))
print()

rel1 = m_eff1 / c.electron_mass
rel2 = m_eff2 / c.electron_mass
print('rel1 =', rel1)
print('rel2 =', rel2)
print()

a1 = (rel1 - 0.067) / 0.067
a2 = -(rel2 - 0.067) / 0.067
print('a1 = {:.3}'.format(a1))
print('a2 = {:.3}'.format(a2))





