from pathlib import Path
import csv
import numpy as np
from scipy.signal import savgol_filter

CARPETA = Path(__file__).resolve().parent
ARCHIVO_DATOS = CARPETA / 'resultados_simulacion' / 'HD225239_11_espectros_simulados.csv'
ARCHIVO_SALIDA = CARPETA / 'resultados_simulacion' / 'HD225239_temperaturas_errores_R20_R1400.csv'
R_VALORES = (20, 40, 80, 100, 200, 400, 600, 800, 1000, 1200, 1400)
T_REF_K = 5775.0
B_WIEN_NM_K = 2.898e6
VENTANA_NM = 30.0
GRADO_POLINOMICO = 3
INTERVALO_NM = (380.0, 700.0)


def onda_aire_a_vacio_nm(onda_aire):
    """Conversión de longitudes de onda en aire a vacío para región óptica."""
    vacio = onda_aire * 1.00028
    for _ in range(5):
        s2 = (1000.0 / vacio) ** 2
        n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
        vacio = onda_aire * n
    return vacio


def main():
    with ARCHIVO_DATOS.open(newline='', encoding='utf-8') as f:
        filas = list(csv.DictReader(f))
    onda_aire = np.array([float(f['longitud_onda_aire_nm']) for f in filas])
    paso = float(np.median(np.diff(onda_aire)))
    puntos = int(round(VENTANA_NM / paso))
    if puntos % 2 == 0:
        puntos += 1
    region = (onda_aire >= INTERVALO_NM[0]) & (onda_aire <= INTERVALO_NM[1])
    indices_region = np.flatnonzero(region)

    def calcular(columna):
        flujo = np.array([float(f[columna]) for f in filas])
        curva = savgol_filter(flujo, window_length=puntos,
                             polyorder=GRADO_POLINOMICO, mode='interp')
        indice = indices_region[np.argmax(curva[region])]
        lam_aire = float(onda_aire[indice])
        lam_vacio = float(onda_aire_a_vacio_nm(lam_aire))
        T = B_WIEN_NM_K / lam_vacio
        E = abs(T - T_REF_K) / T_REF_K * 100
        return lam_aire, lam_vacio, T, E

    T_original = calcular('flujo_original_relativo')[2]
    tabla = []
    for R in R_VALORES:
        lam_a, lam_v, T, E = calcular(f'flujo_simulado_R{R}')
        tabla.append(dict(R=R, lambda_max_aire_nm=round(lam_a, 5),
                          lambda_max_vacio_nm=round(lam_v, 5),
                          T_Wien_K=round(T, 5), error_pct=round(E, 5),
                          diferencia_T_vs_original_K=round(T - T_original, 5)))
    with ARCHIVO_SALIDA.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(tabla[0]));w.writeheader();w.writerows(tabla)
    print(f'Puntos por ventana: {puntos} (~{puntos*paso:.2f} nm)')
    print(f'Temperatura espectro original: {T_original:.2f} K')
    for dato in tabla:
        print(f"R={dato['R']:4d}  λ aire={dato['lambda_max_aire_nm']:.2f} nm  "
              f"T={dato['T_Wien_K']:.2f} K  error={dato['error_pct']:.2f} %")
    print('Archivo guardado:', ARCHIVO_SALIDA)

if __name__ == '__main__':
    main()