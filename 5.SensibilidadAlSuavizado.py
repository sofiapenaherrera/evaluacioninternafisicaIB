from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress

BASE = Path(__file__).resolve().parent
ENTRADA = BASE / 'resultados_simulacion' / 'HD225239_11_espectros_simulados.csv'
SALIDAS = BASE / 'resultados_simulacion'
SALIDAS.mkdir(exist_ok=True)
R_VALORES = [20, 40, 80, 100, 200, 400, 600, 800, 1000, 1200, 1400]
VENTANAS_NM = [20, 30, 40]
GRADO = 3
T_REFERENCIA_K = 5775.0
B_WIEN_NM_K = 2.898e6


def aire_a_vacio_nm(onda_aire):
    onda_vacio = onda_aire * 1.00028
    for _ in range(5):
        s2 = (1000.0 / onda_vacio) ** 2
        indice = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
        onda_vacio = onda_aire * indice
    return onda_vacio


def main():
    if not ENTRADA.exists():
        raise FileNotFoundError(f'Primero ejecuta el programa 4.4 para crear: {ENTRADA}')
    datos = np.genfromtxt(ENTRADA, names=True, delimiter=',', encoding='utf-8')
    lam = np.asarray(datos['longitud_onda_aire_nm'], dtype=float)
    paso = float(np.median(np.diff(lam)))
    region = (lam >= 380) & (lam <= 700)
    indices = np.flatnonzero(region)

    registros = []
    for R in R_VALORES:
        flujo = np.asarray(datos[f'flujo_simulado_R{R}'], dtype=float)
        registro = {'R': R}
        errores = []
        for ventana in VENTANAS_NM:
            longitud_puntos = round(ventana / paso)
            if longitud_puntos % 2 == 0:
                longitud_puntos += 1
            curva = savgol_filter(flujo, window_length=longitud_puntos, polyorder=GRADO, mode='interp')
            indice = indices[np.argmax(curva[region])]
            lam_aire = float(lam[indice])
            lam_vacio = float(aire_a_vacio_nm(lam_aire))
            temperatura = B_WIEN_NM_K / lam_vacio
            error_pct = abs(temperatura - T_REFERENCIA_K) / T_REFERENCIA_K * 100
            registro[f'lambda_aire_{ventana}_nm'] = lam_aire
            registro[f'lambda_vacio_{ventana}_nm'] = lam_vacio
            registro[f'T_{ventana}_K'] = temperatura
            registro[f'error_{ventana}_pct'] = error_pct
            errores.append(error_pct)
        registro['diferencia_max_min_pp'] = max(errores) - min(errores)
        registros.append(registro)

    ruta_csv = SALIDAS / 'HD225239_sensibilidad_11R_3ventanas.csv'
    with ruta_csv.open('w', newline='', encoding='utf-8') as fd:
        w = csv.DictWriter(fd, fieldnames=list(registros[0].keys()))
        w.writeheader()
        w.writerows(registros)

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    for ventana, color in zip(VENTANAS_NM, ['#2c7896', '#cd2b84', '#8864a8']):
        ax.plot(R_VALORES, [registro[f'error_{ventana}_pct'] for registro in registros],
                marker='o', markersize=5, linewidth=1.9, color=color,
                label=f'Savitzky–Golay: ventana {ventana} nm')
    ax.set_xscale('log')
    ax.set_xticks(R_VALORES)
    ax.set_xticklabels([str(R) for R in R_VALORES])
    ax.set_xlabel('Poder de resolución espectral R (adimensional)')
    ax.set_ylabel('Error porcentual de temperatura (%)')
    ax.set_title('HD 225239 — sensibilidad al ancho de la ventana de suavizado')
    ax.grid(alpha=0.23)
    ax.legend()
    fig.tight_layout()
    ruta_png = SALIDAS / 'HD225239_sensibilidad_11R_3ventanas.png'
    fig.savefig(ruta_png, dpi=220)
    plt.close(fig)

    print('Longitud de la ventana, en puntos:', {v: round(v/paso) + (round(v/paso)%2==0) for v in VENTANAS_NM})
    print('R      lambda20  T20      E20     lambda30 T30      E30     lambda40 T40      E40      rango_pp')
    for a in registros:
        print(f"{a['R']:4d}    {a['lambda_aire_20_nm']:6.2f}   {a['T_20_K']:7.2f} {a['error_20_pct']:7.3f}"
              f"    {a['lambda_aire_30_nm']:6.2f}   {a['T_30_K']:7.2f} {a['error_30_pct']:7.3f}"
              f"    {a['lambda_aire_40_nm']:6.2f}   {a['T_40_K']:7.2f} {a['error_40_pct']:7.3f}"
              f"   {a['diferencia_max_min_pp']:7.3f}")
    print('Mayores diferencias por ventana:')
    for reg in sorted(registros, key=lambda x: x['diferencia_max_min_pp'], reverse=True)[:3]:
        print(reg['R'], reg['diferencia_max_min_pp'])
    print('Datos:', ruta_csv)
    print('Gráfico:', ruta_png)


if __name__ == '__main__':
    main()