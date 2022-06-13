# %% Imports
import os
import matplotlib.pyplot as plt
import matplotlib_inline  # .py
import numpy as np
import pyvisa as visa
from matplotlib import rcParams

from instrumental import Agilent34970A

# Para aumentar la resolución de los gráficos de Matplot
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")  # .py
# %config InlineBackend.figure_format='retina'
rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['figure.figsize'] = (8, 4)
rcParams['figure.dpi'] = 100
rcParams['axes.grid'] = True
rcParams['legend.fontsize'] = 10

# %% Inicializamos la comunicación con los multiplexor:
rm = visa.ResourceManager()
channels = [101,102,103,104,105,106,107]
termocuplas = [7, 6, 5, 4, 3, 2, 1]
mux = Agilent34970A(name=rm.list_resources()[1],
                    channelsList=channels)

# %% Inicializar Arrays:
tiempos = np.zeros((len(channels), 1))  # s : Tiempo
temperaturas = np.zeros((len(channels), 1))  # K : Temperatura

# %% Medir V y T(V) con termocupla
fig1 = plt.figure(1, tight_layout=True)
ax1 = plt.subplot()
ax1.set_xlabel(r'$t \mathrm{\ [s]}$')
ax1.set_ylabel(r'$T \mathrm{\ [K]}$')
fig2 = plt.figure(2, tight_layout=True)
ax2 = plt.subplot()
_, T, t, _ = mux.one_scan()
temperaturas[:, 0] = T  # K
t0 = t.min()
tiempos[:, 0] = t - t0
for i, termcup in enumerate(termocuplas):
    ax1.plot(tiempos[i, :], temperaturas[i, :], '.-', mew=0.5, mec='k',
             label=str(termcup), c=f'C{termcup-1}')
ax1.legend(title='Termocupla:', loc='upper left', bbox_to_anchor=(1, 1))
ax2.plot(termocuplas, temperaturas[:, -1], 'o-', mew=0.5, mec='k')
while True:
    _, T, t, _ = mux.one_scan()
    t = t.reshape(len(channels), 1)
    T = T.reshape(len(channels), 1)
    tiempos = np.concatenate((tiempos, t - t0), axis=1)  # s
    temperaturas  = np.concatenate((temperaturas, T), axis=1)  # K
    for i, termcup in enumerate(termocuplas):
        ax1.plot(tiempos[i, :], temperaturas[i, :], '.-', mew=1, mec='k',
                 c=f'C{termcup-1}')
    plt.pause(0.01)
    ax2.clear()
    ax2.set_xlabel(r'$\mathrm{Termocupla}$')
    ax2.set_ylabel(r'$T \mathrm{\ [K]}$')
    ax2.plot(termocuplas, temperaturas[:, -1], 'o-', mew=0.5, mec='k')
    plt.pause(0.01)
plt.show()

# %% Guardar mediciones
os.chdir(r'C:\Users\Publico\Documents\Grupo 1 Labo 4\Código y Datos'
         + r'\Transitorio')
print(f'Los archivos se guardaran en la carpeta actaual:\n{os.getcwd()}')
tag = input('sufijo de los archivos a guardar:')
np.save(f'temp_{tag}.npy', temperaturas)
np.save(f'tiempos_{tag}.npy', tiempos)

# %% Grafico Final

plt.figure(figsize=(8, 4), dpi=150, tight_layout=True)
for i, termcup in enumerate(termocuplas, ):
    plt.plot(tiempos[i, :], temperaturas[i, :], '.-',
             mec='k', mew=0.5, c=f'C{termcup-1}', alpha=0.5,
             label=str(termcup))
plt.legend(title='Termocupla:', loc='upper left', bbox_to_anchor=(1, 1))
# plt.ylim(200, 300)
plt.xlabel(r'$t \mathrm{\ [s]}$', fontsize=13)
plt.ylabel(r'$T \mathrm{\ [K]}$', fontsize=13)
plt.show()





