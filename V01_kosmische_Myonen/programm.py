import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit


###########################################################################################################
### Counts pro 10s gegen Differenz zwischen Verzörgerungsleitungen plotten, fitten

#Zu fittende Funktion
def gauss(x, a, x_0, sig):
    n = 2
    return a * np.exp(-((x-x_0) / sig)**(2*n))

#Daten einlesen
df_1 = pd.read_csv('data/Verzoergerung_data.csv', delimiter='\t')
x = df_1['Zeitdifferenz'].to_numpy()
y = unp.uarray(df_1['Counts'], np.sqrt(df_1['Counts']))

#Fitten
params_1, cov_1 = curve_fit(gauss, x, noms(y), sigma=stds(y))
errors_1 = np.sqrt(np.diag(cov_1))

#Messwerte, Fit und Halbwertsbreite Plotten
fig, ax = plt.subplots()

x_lin = np.linspace(x[0], x[-1], 100)
ax.plot(x, noms(y), 'x', label='Messwerte')
#ax.errorbar(x, noms(y), yerr=stds(y), fmt='none', ecolor='gray', alpha=0.9, capsize=2.5, elinewidth=1.5)
ax.plot(x_lin, gauss(x_lin, *params_1), 'orange', label='Fit')
halbwert = np.max(gauss(x_lin, *params_1)) / 2
ax.hlines(halbwert, -8.2, 8.2, linestyles='dashed', colors='gray', alpha=0.7, label='Halbwertsbreite')
ax.vlines(-8.2, 0, 210, linestyles='dashed', colors='gray', alpha=0.7)
ax.vlines(8.2, 0, 210, linestyles='dashed', colors='gray', alpha=0.7)
ax.set_xlabel(r'$\Delta t$')
ax.set_ylabel(r'Counts pro 10s')

handles, labels = ax.get_legend_handles_labels()
handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1], reverse=True))
#handles, labels = handles[::-1], labels[::-1]

ax.legend(handles, labels, markerscale=1.2)

#plt.show()
plt.savefig('plots/Verzoergerung.pdf')
plt.clf()

#Parameter
parameter_1 = unp.uarray(params_1, errors_1)
for name, param in zip(('a','x_0', 'sig'), parameter_1):
    print(r'{0}:  {1:.8f}'.format(name, param))
print('\n')

###########################################################################################################
### Propfaktor zwischen zeitlicher Pulsabstand und Spannungsamplitude bestimmen

#Zu fittende Funktion
def lin(x, a):
    return a * x

#Daten einlesen
df_2 = pd.read_csv('data/Marker_Faktor_data.csv', delimiter=';')
x = df_2['Marker']
y = df_2['Pulsabstand']

#Fitten
params_2, cov_2 = curve_fit(lin, x, y)
errors_2 = np.sqrt(np.diag(cov_2))

#Plotten
fig, ax = plt.subplots()

ax.scatter(x, y, s=25, c='blue', marker='x', label='Messwerte')
ax.plot(x, lin(x, *params_2), 'orange', alpha=0.9, label='Ausgleichsgerade')
ax.set_xlabel(r'Marker')
ax.set_ylabel(r'$\Delta t_{\mathrm{Puls} \; [\mu s]}$')

#Legende (Labels in umgekehrter Reihenfolge)
handles, labels = ax.get_legend_handles_labels()
handles, labels = handles[::-1], labels[::-1]
# Folgendes sortiert nach Länge des Labels (kürzestes zuerst)
#handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1], reverse=True))
ax.legend(handles, labels, markerscale=1.5, scatteryoffsets=[0.5])

plt.savefig('plots/Marker_Faktor.pdf')
plt.clf()

#Parameter
parameter_2 = unp.uarray(params_2, errors_2)
for name, param in zip(('m','b'), parameter_2):
    print(r'{0}: {1:.8f}'.format(name, param))
print('\n')


###########################################################################################################
### Counts gegen Lebenszeiten plotten, fitten und auswerten

#Zu fittende Funktion
def efkt(x, a, lam):
    return a * np.exp(-lam*x)

#Daten einlesen und x-Werte umformen
df_3 = pd.read_csv('data/Lebenszeit_data.csv', names=['counts'])

kanal = np.linspace(0, 511, 512)
t = kanal * parameter_2
counts = df_3['counts'].to_numpy(dtype=np.float128)

#Fitten
x_min = 3
x_max = 228
counts_rel = counts[x_min:x_max] - 5.81
t_rel = t[x_min:x_max]

params_3, cov_3 = curve_fit(efkt, noms(t_rel), counts_rel)
errors_3 = np.sqrt(np.diag(cov_3))

#Plotten
fig, ax = plt.subplots()

#y = np.concatenate((np.ones(x_min), np.zeros(x_max-x_min), np.ones(len(counts)-x_max)))
ax.scatter(noms(t), counts, s=20, c='limegreen', marker='.', label='Messwerte, ungefittet')
ax.scatter(noms(t_rel), counts_rel, s=20, c='blue', marker='.', label='Messwerte')
ax.plot(noms(t), efkt(noms(t), *params_3), 'orange', linewidth=2, label='Fit')
ax.set_xlabel(r'$\Delta t \; [\mu s]$')
ax.set_ylabel(r'Counts')
ax.legend()

#Legende (Labels in umgekehrter Reihenfolge)
handles, labels = ax.get_legend_handles_labels()
handles, labels = handles[::-1], labels[::-1]
ax.legend(handles, labels, markerscale=3, scatteryoffsets=[0.5])

plt.savefig('plots/Lebenszeit.pdf')
plt.clf()

#Parameter
parameter_3 = unp.uarray(params_3, errors_3)
for name, param in zip(('a','lam'), parameter_3):
    print(r'{0}:  {1:.8f}'.format(name, param))

# Lebensdauer berechnen
tau = 1 / parameter_3[1]
print(r'Lebensdauer: ', tau)
tau_theo = 2.196981
a_tau = (tau_theo - noms(tau)) / tau_theo
print(r'Abweichung Lebensdauer: ', a_tau)
