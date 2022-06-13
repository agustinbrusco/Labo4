# %% Imports y formato:
import os
from numbers import Number
from typing import Union
from collections.abc import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy.special import erfc

# get_ipython().run_line_magic('matplotlib', 'qt5')
# import matplotlib_inline
# matplotlib_inline.backend_inline.set_matplotlib_formats('retina')  # .py
%config InlineBackend.figure_format='retina'

rcParams['figure.figsize'] = (8, 4)
rcParams['font.family'] = 'serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['legend.fontsize'] = 10
rcParams['axes.labelsize'] = 'large'

# %% Definiciones:
channels = [101,102,103,104,105,106,107]
termocuplas = [7, 6, 5, 4, 3, 2, 1]
posiciones = [410.5e-3, 329.8e-3, 249.6e-3,
              211.9e-3, 164.0e-3, 123.1e-3,
              81.4e-3]  # m : Posición de cada termocupla
# Estimación del error:
# os.chdir(r'C:/Users/Usuario/Desktop/Facultad/Labo 4'
#          +r'/Difusividad Térmica/Código y Datos/prueba')
os.chdir(r'/home/agus/Documents/2022 1er Cuatrimestre/Laboratorio 4'
         + r'/Difusividad Térmica/Código y Datos/Prueba')
dT_c = np.load('temp_0.npy').std(ddof=1, axis=1)
dt = np.median(np.diff(np.load('tiempos_0.npy')))


def transitorio(t, x, kappa, A, B):
    kt = np.sqrt(4*kappa*t)
    ter1 = kt/np.sqrt(np.pi)*np.exp(-(x/kt)**2)
    ter2 = -x*erfc(x/kt)
    return A*(ter1 + ter2) + B


def senoidal(tiempo, freq, fase, amplitud, offset):
    return amplitud*np.sin(2*np.pi*freq*tiempo + fase) + offset


def exponencial(posicion, decaimiento, amplitud):
    return amplitud*np.exp(-decaimiento*posicion)


# def lineal_a_trozos(posicion, pendiente, ordenada):
#     y = pendiente*posicion + ordenada
#     while (np.abs(y) > np.pi).any():
#         y -= np.sign(y)*2*np.pi*(np.abs(y)//np.pi)
#     return y
def lineal(posicion, pendiente, ordenada):
    return pendiente*posicion + ordenada


def eliminar_transitorio(
        Temps: np.ndarray,  # K
        t: np.ndarray,  # s
        periodo: Number,  # s
        plot: Union[bool, int]=False
        ) -> tuple[np.ndarray]:
    '''Dada una señal periodica montada sobre otra fución (no períodica
    o con un período distinto), promedia los valores de la señal alrededor
    de cada punto y se lo resta. De esta forma, devuelve la señal que
    con valor medio en el origen.'''
    # Excluyo bordes para evitar un IndexError
    l_b = np.argmin(np.abs(t[0] - 2*periodo))  # lower bound
    u_b = np.argmin(np.abs(t[0]- (t[0].max() - 2*periodo)))  # upper bound
    Temps_est = np.zeros_like(Temps[:, l_b:u_b])
    for i, tc in enumerate(termocuplas):
        for j, temp in enumerate(Temps[i, l_b:u_b], start=l_b):
            # Para cada medición de temperatura, busco un intervalo temporal
            # con la duración de un período centrado en ese valor:
            low = j - 1
            while (t[i, low] > t[i, j] - periodo/2):
                low -= 1
            up = j + 1
            while (t[i, up] < t[i, j] + periodo/2):
                up += 1
            # Calculo el valor medio de la temperatura en ese punto durante
            # el intervalo temporal hallado:
            valor_medio = np.mean(Temps[i, low:up])
            Temps_est[i, j-l_b] = temp - valor_medio  # Se le resta al valor actual.
    t_est = t[:, l_b:u_b]  # Se definen los tiempos asociados al nuevo array de temperaturas.
    if plot is not False:
        plt.figure(facecolor='w', dpi=150)
        for i, tc in list(enumerate(termocuplas, ))[::-1]:
            if plot == tc or plot is True:
                plt.errorbar(t_est[i, :], Temps_est[i, :],
                             yerr=dT_c[i], xerr=dt,
                             fmt='.', ms=3,
                             mew=0.5, mec='k', capsize=2,
                             label=str(tc), c=f'C{tc-1}')
        plt.legend(title='Termocupla:', loc='upper left', bbox_to_anchor=(1, 1))
        # plt.yscale('log')
        # plt.xscale('log')
        plt.grid(True)
        plt.xlabel(r'$t \mathrm{\ [s]}$')
        plt.ylabel(r'$\theta \mathrm{\ [K]}$')
        plt.show()
    return Temps_est, t_est


def estimar_fase(
        signal: np.ndarray,
        time: np.ndarray,
        freq: Number) -> Number:
    '''Dada una señal (signal) y su dominio temporal (time),
    calcula una estimación  de la fase de su componente de
    frecuencia freq.'''
    transformada = np.fft.rfft(signal)
    frecuencias = np.fft.rfftfreq(time.size, np.diff(time).mean())
    freq_id = np.argmin(np.abs(frecuencias - freq))
    return np.angle(transformada[freq_id])


def ajustar_seno(
        Temps: np.ndarray,  # K
        t: np.ndarray,  # s
        freq0: Number,  # Hz
        intervalo: Sequence[Number]=[500, 1000],  # s
        display_res: bool=False,
        plot: bool=False) -> tuple[np.ndarray]:
    '''Dadas las componentes armónicas de la temperatura en
    cada termocupla, devuelve su fase (Phi ± dPhi) y
    su amplitud (A ± dA).'''
    Phi, dPhi = [], []
    A, dA = [], []
    filtro = (t > intervalo[0])*(t < intervalo[1])
    for i, tc in enumerate(termocuplas):
        t_i = t[i, filtro[i]]
        T_i = Temps[i, filtro[i]]
        A0 = (T_i.max() - T_i.min())/2
        B0 = T_i.mean()
        phi0 = estimar_fase(T_i, t_i, freq0)
        param_range = [(-np.pi, A0/3, -2*A0),
                       (np.pi, 2*A0, 2*A0)]
        popt, pcov = curve_fit(
            lambda t, fase, amplitud, offset: senoidal(t, freq0, fase, amplitud, offset),
            t_i, T_i, p0=[phi0, A0, B0], bounds=param_range
            )
        perr = np.sqrt(np.diag(pcov))
        Phi.append(popt[0])
        dPhi.append(perr[0])
        A.append(popt[1])
        dA.append(perr[1])
        if display_res:
            print(f'\nTermocupla {tc}:')
            for param, param_err, tag, uni in zip(popt, perr,
                                                  ['φ', 'A', 'B'],
                                                  ['', 'K', 'K']):
                print(f'{tag} = ({param:.3g} ± {param_err:.1g}) {uni}')
        
        if plot == tc or plot is True:
            # t_dense = np.linspace(t_i[0], t_i[-1], 10000)
            t_dense = np.linspace(t[i, 0], t[i, -1], 10000)
            T_dense = senoidal(t_dense, freq0, *popt)
            plt.figure()
            # plt.errorbar(t_i, Temps[i, filtro[i]],
            plt.errorbar(t[i], Temps[i, :],
                        yerr=dT_c[i], xerr=dt,
                        fmt='.', ms=3, lw=1,
                        mew=0.5, mec='k', capsize=2,
                        label=str(tc), c=f'C{tc-1}',)
            plt.plot(t_dense, T_dense, 'k-', lw=1)
            plt.grid(True)
            plt.xlabel(r'$t \mathrm{\ [s]}$')
            plt.ylabel(r'$\theta \mathrm{\ [K]}$')
            plt.show()

    Phi = np.array(Phi)
    dPhi = np.array(dPhi)
    A = np.array(A)
    dA = np.array(dA)
    return Phi, dPhi, A, dA


def ajustar_amplitud(
        x: Sequence,  # m
        A: Sequence,  # K
        dA: Sequence,  # K
        plot: bool=False) -> tuple:
    '''Ajusta la amplitud de la señal en función de la distancia entre
    las termocuplas, y devuelve la constante de decaimiento (∊ ± Δ∊)'''
    popt, pcov = curve_fit(exponencial, x, A,
                           sigma=dA, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    epsilon = popt[0]  # m⁻¹
    depsilon = perr[0]  # m⁻¹
    
    if plot:
        x_dense = np.linspace(x[-1], x[0], 1000)
        A_dense = exponencial(x_dense, *popt)
        plt.figure()
        plt.errorbar(x, A, xerr=0.05e-3, yerr=dA,
                    fmt='.', ms=10, mec='k', ecolor='k', capsize=2,
                    label='Amplitudes Ajustadas')
        plt.plot(x_dense, A_dense, 'k-', lw=1,
                 label=r'$A = C e^{-\epsilon x}$')
        plt.legend()
        plt.grid(True)
        plt.xlabel(r'$x \mathrm{\quad [m]}$')
        plt.ylabel(r'$A \mathrm{\quad [K]}$')
        plt.show()
    return epsilon, depsilon


def ajustar_fase(
        x: Sequence,  # m
        Phi: Sequence,
        dPhi: Sequence,
        omega: Number,  # s⁻¹
        plot: bool=False) -> tuple:
    '''Ajusta la fase de la señal en función de la distancia entre
    las termocuplas, y devuelve la velocidad de propagación (ν ± Δν)'''
    # m0 = (Phi[1]- Phi[0])/(x[1] - x[0])
    # b0 = (m0*(0+x[0]) + Phi[0])%np.pi
    # param_range = [(2*m0, -np.pi), (m0/2, np.pi)]
    for i in range(5, -1, -1):
        while Phi[i] > Phi[i+1]:
            Phi[i] -= 2*np.pi
    popt, pcov = curve_fit(lineal, x, Phi,
                             sigma=dPhi, absolute_sigma=True,)
                            #  p0=[m0, b0], bounds=param_range)
    perr = np.sqrt(np.diag(pcov))

    nu = -omega/popt[0]  # m/s
    dnu = np.abs(perr[0]*nu/popt[0])  # m/s

    if plot:
        x_dense = np.linspace(0, 0.5, 1000)
        Phi_dense = lineal(x_dense, *popt)
        plt.figure()
        plt.errorbar(x, Phi, xerr=0.05e-3, yerr=dPhi,
                    fmt='.', ms=10, mec='k', ecolor='k', capsize=2,
                    label='Fases Ajustadas')
        plt.plot(x_dense, Phi_dense, 'k-', lw=1,
                 label=r'$\varphi = -\frac{\omega}{\nu}x$')
        plt.legend()
        plt.grid(True)
        plt.xlabel(r'$x \mathrm{\quad [m]}$')
        plt.ylabel(r'$\varphi$')
        plt.show()
    return nu, dnu


# %% Análisis Exitación Escalón
# os.chdir(r'C:/Users/Usuario/Desktop/Facultad/Labo 4'
#          +r'/Difusividad Térmica/Código y Datos/Transitorio')
os.chdir(r'/home/agus/Documents/2022 1er Cuatrimestre/Laboratorio 4'
         + r'/Difusividad Térmica/Código y Datos/Transitorio')
Temps = np.load('temp_escalon.npy')
t = np.load('tiempos_escalon.npy')

kappas = np.zeros_like(termocuplas, dtype=float)
F0_K = np.zeros_like(termocuplas, dtype=float)
X0 = np.zeros_like(termocuplas, dtype=float)



plt.figure(facecolor='w', dpi=150, figsize=(6.7, 3))
for i, (tc, x) in list(enumerate(zip(termocuplas, posiciones)))[::-1]:
    t0_idx = 35
    t0 = t[i, t0_idx]
    func_i = lambda t, kappa, A, x0: transitorio(t-t0, x+x0, kappa, A, Temps[i, 0])
    popt, pcov = curve_fit(func_i, t[i, t0_idx+1:], Temps[i, t0_idx+1:],
                        p0=[111e-6, 400, 5e-3])
    perr = np.sqrt(np.diag(pcov))
    for val, err in zip(popt, perr):
        print(f'errror_relativo = {err/val:.3g}')
    kappas[i] = popt[0]
    F0_K[i] = popt[1]
    X0[i] = popt[2]
    plt.errorbar(t[i, :], Temps[i, :], yerr=dT_c[i], xerr=dt,
                 fmt='.', ms=3, lw=1, mew=0.5, mec='k', capsize=2,
                 label=f'${x*1e3:.1f}'.replace('.',',\!') + r'\ \mathrm{mm}$', c=f'C{tc-1}',
                 )
    plt.plot(t[i, t0_idx+1:], func_i(t[i, t0_idx+1:], *popt), 'r', lw=0.5, zorder=15)
plt.legend(title=r'$\mathrm{Posición:}$', loc='upper left', bbox_to_anchor=(1, 1))
plt.text(2300, 492, 'Ajustes', rotation=13, c='r')
# plt.legend(title='Termocupla:', loc='lower left', bbox_to_anchor=(0, 1), ncol=7)
plt.grid(True)
plt.xlabel(r'$t \quad \mathrm{[s]}$')
plt.ylabel(r'$\theta \quad \mathrm{[K]}$')
plt.show()

# %% Análisis Exitación Cuadrada
# os.chdir(r'C:/Users/Usuario/Desktop/Facultad/Labo 4'
#          +r'/Difusividad Térmica/Código y Datos/Transitorio')
os.chdir(r'/home/agus/Documents/2022 1er Cuatrimestre/Laboratorio 4'
         + r'/Difusividad Térmica/Código y Datos/Transitorio')
Temps = np.load('temp_cuad180.npy')
t = np.load('tiempos_cuad180.npy')

plt.figure(facecolor='w', dpi=150, figsize=(6.7, 3))
for i, (tc, x) in list(enumerate(zip(termocuplas, posiciones)))[::-1]:
    plt.errorbar(t[i, :], Temps[i, :], yerr=dT_c[i], xerr=dt,
                 fmt='.', ms=3, lw=1, mew=0.5, mec='k', capsize=2,
                 label=f'${x*1e3:.1f}'.replace('.',',\!') + r'\ \mathrm{mm}$', c=f'C{tc-1}',
                 )
plt.legend(title=r'$\mathrm{Posición:}$', loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.xlim(0, 3400)
plt.xlabel(r'$t \quad \mathrm{[s]}$')
plt.ylabel(r'$\theta \quad \mathrm{[K]}$')
plt.show()


# se guardan los valores asociados a cada experimento
periodos = np.array([180, 120])
intervalo_temp = np.array([(2000, 2400), (3500, 4000),])
# se inicializan los arrays donde se guardaran los resultados
Ec, dEc = np.zeros(2, dtype=float), np.zeros(2, dtype=float)
Vc, dVc = np.zeros(2, dtype=float), np.zeros(2, dtype=float)
Kc, dKc = np.zeros(2, dtype=float), np.zeros(2, dtype=float)
# se itera sobre los experimentos
for i, (per, inter) in enumerate(zip(periodos, intervalo_temp)):
    print(f'Señal de período ~ {per} s (frecuencia = {1/per*1e3:.3g} mHz):')
    Temps = np.load(f'temp_cuad{per}.npy')
    t = np.load(f'tiempos_cuad{per}.npy')
    frec = 1/per
    Temps_est, t_est = eliminar_transitorio(Temps, t, per)
    Phi, dPhi, A, dA = ajustar_seno(Temps_est, t_est,
                                    freq0=1/per,
                                    intervalo=inter,
                                    display_res=False,
                                    plot=False)
    Ec[i], dEc[i] = ajustar_amplitud(posiciones, A, dA, plot=False)  # m⁻¹
    Vc[i], dVc[i] = ajustar_fase(posiciones, Phi, dPhi,
                                2*np.pi/per, plot=False)  # m·s⁻¹
    Kc[i] = Vc[i]/(2*Ec[i])  # m²·s⁻¹
    dKc[i] = Kc[i]*np.sqrt((dVc[i]/Vc[i])**2 + (dEc[i]/Ec[i])**2)  # m²·s⁻¹
    print(f'∊ = ({Ec[i]:.2f} ± {dEc[i]:.1g}) m⁻¹')
    print(f'ν = ({Vc[i]*1e3:.3f} ± {dVc[i]*1e3:.1g}) mm·s⁻¹')
    print(f'κ = ({Kc[i]*1e6:.1f} ± {dKc[i]*1e6:.1g}) mm²·s⁻¹\n')


# %% Exitación: Senoidal
# os.chdir(r'C:/Users/Usuario/Desktop/Facultad/Labo 4'
#          +r'/Difusividad Térmica/Código y Datos/Transitorio')
os.chdir(r'/home/agus/Documents/2022 1er Cuatrimestre/Laboratorio 4'
         + r'/Difusividad Térmica/Código y Datos/Transitorio')
Temps = np.load('temp_seno180.npy')
t = np.load('tiempos_seno180.npy')

plt.figure(facecolor='w', dpi=150, figsize=(6.7, 3))
for i, (tc, x) in list(enumerate(zip(termocuplas, posiciones)))[::-1]:
    plt.errorbar(t[i, :], Temps[i, :], yerr=dT_c[i], xerr=dt,
                 fmt='.', ms=3, lw=1, mew=0.5, mec='k', capsize=2,
                 label=f'${x*1e3:.1f}'.replace('.',',\!') + r'\ \mathrm{mm}$', c=f'C{tc-1}',
                 )
plt.legend(title=r'$\mathrm{Posición:}$', loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.xlim(0, 3500)
plt.xlabel(r'$t \quad \mathrm{[s]}$')
plt.ylabel(r'$\theta \quad \mathrm{[K]}$')
plt.show()

Temps_est, t_est = eliminar_transitorio(Temps, t, 1/6e-3)
Phi, dPhi, A, dA = ajustar_seno(Temps_est, t_est,
                                freq0=6e-3,
                                intervalo=(400, 1400),
                                display_res=False,
                                plot=False)
for i in range(5, -1, -1):
    while Phi[i] > Phi[i+1]:
        Phi[i] -= 2*np.pi

apopt, _ = curve_fit(exponencial, posiciones, A,
                     sigma=dA, absolute_sigma=True)
ppopt, _ = curve_fit(lineal, posiciones, Phi,
                     sigma=dPhi, absolute_sigma=True,)
x_dense = np.linspace(posiciones[-1], posiciones[0], 1000)
A_dense = exponencial(x_dense, *apopt)
Phi_dense = lineal(x_dense, *ppopt)
fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True,
                        facecolor='w', dpi=150, figsize=(6.7, 3))
axs[0].errorbar(posiciones[0]*1e3, A[0], xerr=1e-5, yerr=1e-5, c=f'C0',
                fmt='.', ms=10, mec='k', ecolor='k', capsize=2,
                label='Valores Ajustados')
for i, (tc, x) in list(enumerate(zip(termocuplas, posiciones)))[::-1]:
    axs[0].errorbar(x*1e3, A[i], xerr=0.05e-3, yerr=dA[i], c=f'C{tc-1}',
                    fmt='.', ms=10, mec='k', ecolor='k', capsize=2)

axs[0].plot(x_dense*1e3, A_dense, 'k-', lw=1,
            label=r'$\mathrm{Ajuste\ según:}\quad A = C e^{-\epsilon x}$')
axs[0].legend()
axs[0].grid(True)
axs[0].set_ylabel(r'$A \mathrm{\quad [K]}$')
axs[0].text(0.015, 0.1, s='a)', fontsize=14, transform=axs[0].transAxes,
            bbox={'facecolor': 'white', 'boxstyle': 'round'}, zorder=15)

for i, (tc, x) in list(enumerate(zip(termocuplas, posiciones)))[::-1]:
    axs[1].errorbar(x*1e3, Phi[i], xerr=0.05e-3, yerr=dPhi[i], c=f'C{tc-1}',
                    fmt='.', ms=10, mec='k', ecolor='k', capsize=2,)
axs[1].plot(x_dense*1e3, Phi_dense, 'k-', lw=1,
            label=r'$\mathrm{Ajuste\ según:}\quad \varphi = -\frac{\omega}{\nu}x$')
axs[1].legend()
axs[1].set_yticks([np.pi, np.pi/2, 0, -np.pi/2],
                  [r'$\pi$', r'$\frac{\pi}{2}$',
                   r'$0$', r'$-\frac{\pi}{2}$'],
                  fontsize=11)
axs[1].grid(True)
axs[1].set_xlabel(r'$x \mathrm{\quad [mm]}$')
axs[1].set_ylabel(r'$\varphi$')
axs[1].text(0.015, 0.1, s='b)', fontsize=14, transform=axs[1].transAxes,
            bbox={'facecolor': 'white', 'boxstyle': 'round'}, zorder=15)
plt.show()

# %% Análisis casos senoidales de distintas frecuencias:
os.chdir(r'/home/agus/Documents/2022 1er Cuatrimestre/Laboratorio 4'
         + r'/Difusividad Térmica/Código y Datos/Transitorio')
# se guardan los valores asociados a cada experimento
periodos = np.array([180, 120, 60, 30])
frecuencias = np.array([6e-3, 8e-3, 16e-3, 34e-3])
intervalo_temp = np.array([(400, 1400), (550, 1000),
                           (700, 1300), (800, 1100)])
# se inicializan los arrays donde se guardaran los resultados
E, dE = np.zeros(4, dtype=float), np.zeros(4, dtype=float)
V, dV = np.zeros(4, dtype=float), np.zeros(4, dtype=float)
K, dK = np.zeros(4, dtype=float), np.zeros(4, dtype=float)
# se itera sobre los experimentos
for i, (per, frec, inter) in enumerate(zip(periodos,
                                           frecuencias,
                                           intervalo_temp)):
    print(f'Señal de período ~ {per} s (frecuencia = {frec*1e3} mHz):')
    Temps = np.load(f'temp_seno{per}.npy')
    t = np.load(f'tiempos_seno{per}.npy')
    Temps_est, t_est = eliminar_transitorio(Temps, t,
                                            periodo=1/frec,
                                            plot=False)
    Phi, dPhi, A, dA = ajustar_seno(Temps_est, t_est,
                                    freq0=frec,
                                    intervalo=inter,
                                    display_res=False,
                                    plot=False)
    E[i], dE[i] = ajustar_amplitud(posiciones, A, dA, plot=False)  # m⁻¹
    V[i], dV[i] = ajustar_fase(posiciones, Phi, dPhi,
                           2*np.pi*frec, plot=False)  # m·s⁻¹

    K[i] = V[i]/(2*E[i])  # m²·s⁻¹
    dK[i] = K[i]*np.sqrt((dV[i]/V[i])**2 + (dE[i]/E[i])**2)  # m²·s⁻¹
    print(f'∊ = ({E[i]:.2f} ± {dE[i]:.1g}) m⁻¹')
    print(f'ν = ({V[i]*1e3:.3f} ± {dV[i]*1e3:.1g}) mm·s⁻¹')
    print(f'κ = ({K[i]*1e6:.1f} ± {dK[i]*1e6:.1g}) mm²·s⁻¹\n')


# %% Comparación ∊, ν, κ vs τ:

fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True,
                        facecolor='w', dpi=150, figsize=(6.7, 4))
for ax, var, dvar, tag in zip(axs, [E, V*1e3, K*1e6], [dE, dV*1e3, dK*1e6],
                              [r'$\epsilon \quad \mathrm{[m^{-1}]}$',
                               r'$\nu \quad \mathrm{[mm\cdot s^{-1}]}$',
                               r'$\kappa \quad \mathrm{[mm^2 \cdot s^{-1}]}$']):
    ax.errorbar(1/frecuencias, var, dvar,
                fmt='o', ms=4, mec='k', mew=1, ecolor='k', capsize=2,)
    ax.grid(True)
    ax.set_ylabel(tag, labelpad=((ax == axs[1])*12 + (ax == axs[0])*7))
for ax, var, dvar, tag in zip(axs, [Ec, Vc*1e3, Kc*1e6], [dEc, dVc*1e3, dKc*1e6], 'abc'):
    ax.errorbar([180, 120], var, dvar,
                fmt='s', ms=4, mec='k', mew=1, ecolor='k', capsize=2,)
    ax.text(0.02, 0.15, s=f'{tag})', fontsize=14, transform=ax.transAxes,
            bbox={'facecolor': 'white', 'boxstyle': 'round'}, zorder=15)
ax.set_xlabel(r'$\tau \quad \mathrm{[s]}$')
axs[0].errorbar([], [], [], fmt='oC0', ms=4, mec='k', mew=1, ecolor='k', capsize=3,
                label=r'$\mathrm{Onda\ Senoidal}$')
axs[0].errorbar([], [], [], fmt='sC1', ms=4, mec='k', mew=1, ecolor='k', capsize=3,
                label=r'$\mathrm{Onda\ Cuadrada}$')

periodos = np.concatenate([1/frecuencias, [180, 120]])
popt, pcov = curve_fit(lambda tau, A: np.sqrt(A/tau),
                       periodos, np.concatenate([E, Ec]))
tau_dense = np.linspace(periodos.min(), periodos.max(), 1000)
axs[0].plot(tau_dense, np.sqrt(popt[0]/tau_dense), '-k', lw=0.5)
popt, pcov = curve_fit(lambda tau, A: np.sqrt(np.abs(A)/tau),
                       periodos, np.concatenate([V, Vc]))
axs[1].plot(tau_dense, 1e3*np.sqrt(popt[0]/tau_dense), '-k', lw=0.5)
axs[0].legend()

# ax.set_xlabel(r'$f \quad \mathrm{[Hz]}$')
plt.show()

