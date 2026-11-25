from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

carpeta = Path(__file__).resolve().parent
archivo = carpeta / "s0003.fits"
with fits.open(archivo) as hdul:
    encabezado = hdul[0].header.copy()
    flujo = np.asarray(hdul[0].data, dtype=np.float64).ravel()

pixeles = np.arange(flujo.size) + 1
longitud_onda = (encabezado['CRVAL1'] +
                (pixeles - encabezado['CRPIX1']) * encabezado['CDELT1'])
unidades = encabezado.get('CUNIT1', 'Angstrom').strip().lower()
if unidades in ['angstrom', 'angstroem', 'ang', 'a', 'å']:
    longitud_onda /= 10.0
elif unidades != 'nm':
    raise ValueError(f'Unidad espectral sin verificar: {unidades}')

intervalo = (longitud_onda >= 380) & (longitud_onda <= 700)
validos = np.isfinite(flujo)
analisis = intervalo & validos

print('Objeto:', encabezado.get('OBJECT', 'no especificado'))
print('Puntos espectrales:', flujo.size)
print('Intervalo (nm):', round(longitud_onda[0], 2), 'a', round(longitud_onda[-1], 2))
print('Muestreo (nm/pixel):', round(np.median(np.diff(longitud_onda)), 4))
print('Valores no finitos:', np.count_nonzero(~validos))
print('Valores de flujo negativos:', np.count_nonzero(flujo < 0))
print('Puntos dentro de 380–700 nm:', np.count_nonzero(analisis))

np.savetxt(carpeta / 'HD225239_muestreo_original.csv',
           np.column_stack((longitud_onda, flujo)), delimiter=',',
           comments='', header='longitud_onda_nm,flujo_relativo', fmt='%.9g')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(longitud_onda, flujo, color='#d32286', linewidth=0.8, label='Espectro MILES original')
ax.axvspan(380, 700, color='#a9a9a9', alpha=0.18, label='Intervalo de análisis (380–700 nm)')
ax.set_xlabel('Longitud de onda en aire (nm)')
ax.set_ylabel('Flujo espectral relativo (u. a.)')
ax.set_title('HD 225239: inspección del espectro original')
ax.grid(alpha=0.2)
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(carpeta / 'HD225239_espectro_original.png', dpi=200)
plt.show()