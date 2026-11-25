from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent
lam, flux = np.loadtxt(base/'HD225239_muestreo_original.csv', delimiter=',', skiprows=1, unpack=True)
dlambda = float(np.median(np.diff(lam)))
region = (lam >= 380) & (lam <= 700) & np.isfinite(flux)

def filtro_suave(ventana_nm):
    n = int(round(ventana_nm / dlambda))
    if n % 2 == 0:
        n += 1
    return savgol_filter(flux, window_length=n, polyorder=3, mode='interp')

print('Intervalo elegido: 380–700 nm')
max_crudo_idx = np.where(region)[0][np.argmax(flux[region])]
print(f'Máximo puntual crudo: {lam[max_crudo_idx]:.2f} nm')
for ventana in (20, 30, 40):
    cont = filtro_suave(ventana)
    idx = np.where(region)[0][np.argmax(cont[region])]
    print(f'Savitzky-Golay, ventana {ventana} nm: lambda_max={lam[idx]:.2f} nm')

suavizado = filtro_suave(30)
idx = np.where(region)[0][np.argmax(suavizado[region])]
max_lam, max_flux = lam[idx], suavizado[idx]
fig, ax = plt.subplots(figsize=(10,5.5))
ax.plot(lam[region],flux[region], color='#c783aa', alpha=0.55, lw=0.7, label='Espectro original (MILES)')
ax.plot(lam[region],suavizado[region], color='#b60b76', lw=2.2, label='Ajuste suave preliminar (30 nm)')
ax.scatter([max_lam],[max_flux], color='#30243c', zorder=4)
ax.axvline(max_lam,ls='--',lw=1,color='#30243c',alpha=0.75)
ax.annotate(f'Máximo suavizado ≈ {max_lam:.2f} nm',xy=(max_lam,max_flux),xytext=(520,1.17),
            arrowprops=dict(arrowstyle='->',color='#30243c'),fontsize=10,color='#30243c')
ax.set_xlim(380,700)
ax.set_ylim(bottom=0)
ax.set_xlabel('Longitud de onda en aire (nm)')
ax.set_ylabel('Flujo espectral relativo (escala del archivo MILES)')
ax.set_title('HD 225239: estimación preliminar del máximo espectral')
ax.grid(alpha=0.2)
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(base/'HD225239_maximo_observado_preliminar.png',dpi=200)