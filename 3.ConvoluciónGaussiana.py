from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

CARPETA = Path(__file__).resolve().parent
SALIDAS = CARPETA / "resultados_simulacion"
SALIDAS.mkdir(exist_ok=True)

# Estos valores reemplazan la serie anterior de R=20 a R=700.
RESOLUCIONES = [20, 40, 80, 100, 200, 400, 600, 800, 1000, 1200, 1400]
FWHM_ORIGINAL_NM = 0.250  # resolución instrumental MILES v9.1
LAMBDA_REF_NM = 501.8     # región del máximo TEÓRICO, no se usa para fijar R en todo el espectro
LIMITE_INFERIOR_NM = 380
LIMITE_SUPERIOR_NM = 700


def leer_espectro():
    """Priorizar FITS: usar sus palabras clave WCS para el eje de longitud de onda."""
    fits_path = CARPETA / "s0003.fits"
    csv_path = CARPETA / "HD225239_muestreo_original.csv"
    if fits_path.exists():
        try:
            from astropy.io import fits
        except ImportError as exc:
            if not csv_path.exists():
                raise ImportError(
                    "Para leer s0003.fits, instala Astropy con: python -m pip install astropy"
                ) from exc
            print("AVISO: Astropy no está instalado. Se usará el CSV exportado del FITS.")
        else:
            with fits.open(fits_path) as hdul:
                flujo = np.asarray(hdul[0].data, dtype=float).squeeze()
                h = hdul[0].header
                pixeles_fits = np.arange(flujo.size) + 1
                onda_nm = (h["CRVAL1"] + (pixeles_fits - h["CRPIX1"]) * h["CDELT1"]) / 10.0
            print("Origen: s0003.fits (datos originales MILES)")
            return onda_nm, flujo
    if csv_path.exists():
        onda_nm, flujo = np.loadtxt(csv_path, delimiter=",", skiprows=1, unpack=True)
        print("Origen: HD225239_muestreo_original.csv (extraído del FITS)")
        return onda_nm, flujo
    raise FileNotFoundError("Coloca s0003.fits junto a este archivo .py y vuelve a ejecutar.")


def degradar_espectro(onda_nm, flujo, R, paso_nm):
    """Convolución con kernel gaussiano LOCAL, normalizado, para R constante.

    La FWHM objetivo cambia con lambda: FWHM_obj(lambda) = lambda/R.
    La anchura a aplicar considera la FWHM del espectro de entrada:
    FWHM_conv^2 = FWHM_obj^2 - FWHM_original^2.
    """
    fwhm_objetivo = onda_nm / float(R)
    if np.any(fwhm_objetivo < FWHM_ORIGINAL_NM - 1e-10):
        raise ValueError(f"R={R} excede la resolución original en una zona del FITS")

    fwhm_adicional = np.sqrt(np.maximum(0.0, fwhm_objetivo**2 - FWHM_ORIGINAL_NM**2))
    sigma = fwhm_adicional / (2 * np.sqrt(2 * np.log(2)))
    resultado = np.empty_like(flujo, dtype=float)

    for i, (centro, desv) in enumerate(zip(onda_nm, sigma)):
        if desv < 1e-9:  # sin ensanchamiento adicional en ese punto
            resultado[i] = flujo[i]
            continue
        # Truncar en +/-4.5 sigma (contribución de colas despreciable).
        radio = int(np.ceil(4.5 * desv / paso_nm))
        izquierda = max(0, i - radio)
        derecha = min(len(onda_nm), i + radio + 1)
        distancia = (onda_nm[izquierda:derecha] - centro) / desv
        pesos = np.exp(-0.5 * distancia**2)
        resultado[i] = np.dot(flujo[izquierda:derecha], pesos) / pesos.sum()

    return resultado


def exportar_csv(onda_nm, flujo, resultados):
    archivo = SALIDAS / "HD225239_11_espectros_simulados.csv"
    with archivo.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["longitud_onda_aire_nm", "flujo_original_relativo"] +
                        [f"flujo_simulado_R{r}" for r in RESOLUCIONES])
        for i, longitud in enumerate(onda_nm):
            writer.writerow([f"{longitud:.5f}", f"{flujo[i]:.9g}"] +
                            [f"{resultados[r][i]:.9g}" for r in RESOLUCIONES])
    print("Datos guardados:", archivo)


def grafica_comparativa(onda_nm, flujo, resultados, region):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(onda_nm[region], flujo[region], color="#d82489", alpha=0.55,
            lw=0.8, label="Original MILES")
    for R, color, ancho in [(1400, "#8162ad", 1.15), (400, "#30939a", 1.45),
                            (100, "#c98834", 1.65), (20, "#31234a", 2.2)]:
        ax.plot(onda_nm[region], resultados[R][region], color=color,
                lw=ancho, label=f"Simulado R={R}")
    ax.set(xlabel="Longitud de onda en aire (nm)",
           ylabel="Flujo espectral relativo (escala MILES)",
           title="HD 225239: espectro original y degradaciones de resolución",
           xlim=(LIMITE_INFERIOR_NM, LIMITE_SUPERIOR_NM))
    ax.grid(alpha=0.18)
    ax.legend(fontsize=9)
    fig.tight_layout()
    destino = SALIDAS / "HD225239_comparacion_resoluciones.png"
    fig.savefig(destino, dpi=220)
    plt.close(fig)
    print("Gráfica guardada:", destino)


def ventana_interactiva(onda_nm, flujo, resultados, region):
    fig, ax = plt.subplots(figsize=(11.3, 6.5))
    fig.subplots_adjust(bottom=0.27)
    ax.plot(onda_nm[region], flujo[region], lw=0.85, color="#d82489",
            alpha=0.52, label="Original MILES")
    actual = 1400
    linea, = ax.plot(onda_nm[region], resultados[actual][region],
                     lw=2, color="#40305d", label="Resolución degradada")
    ax.set(xlabel="Longitud de onda en aire (nm)",
           ylabel="Flujo espectral relativo (escala MILES)",
           title="HD 225239: visualizador interactivo de resolución",
           xlim=(LIMITE_INFERIOR_NM, LIMITE_SUPERIOR_NM))
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=9)

    eje_control = fig.add_axes([0.18, 0.13, 0.62, 0.035])
    control = Slider(eje_control, "Índice", 0, len(RESOLUCIONES) - 1,
                     valinit=len(RESOLUCIONES) - 1, valstep=1)
    indicador = fig.text(0.5, 0.06, "", ha="center", fontsize=10)
    eje_boton = fig.add_axes([0.83, 0.115, 0.14, 0.06])
    boton = Button(eje_boton, "Guardar PNG")

    def actualizar(_):
        R = RESOLUCIONES[int(control.val)]
        linea.set_ydata(resultados[R][region])
        fwhm_ref = LAMBDA_REF_NM / R
        indicador.set_text(
            f"R = {R} | FWHM objetivo a 501.8 nm = {fwhm_ref:.3f} nm "
            f"| FWHM original = {FWHM_ORIGINAL_NM:.3f} nm"
        )
        fig.canvas.draw_idle()

    def guardar(_):
        R = RESOLUCIONES[int(control.val)]
        destino = SALIDAS / f"HD225239_R{R}_comparacion.png"
        fig.savefig(destino, dpi=220)
        print("Vista guardada:", destino)

    control.on_changed(actualizar)
    boton.on_clicked(guardar)
    actualizar(None)
    print("Abriendo visualizador: mueve el deslizador; pulsa 'Guardar PNG' si deseas guardar la vista.")
    plt.show()


def main():
    onda, flujo = leer_espectro()
    if onda.ndim != 1 or flujo.ndim != 1 or onda.size != flujo.size:
        raise ValueError("Se espera un espectro unidimensional.")
    if np.any(~np.isfinite(onda)) or np.any(~np.isfinite(flujo)) or np.any(np.diff(onda) <= 0):
        raise ValueError("Revisa el archivo: longitudes de onda o flujos no válidos.")
    paso = float(np.median(np.diff(onda)))
    region = (onda >= LIMITE_INFERIOR_NM) & (onda <= LIMITE_SUPERIOR_NM)
    if region.sum() == 0:
        raise ValueError("El FITS no cubre el intervalo 380-700 nm.")
    print(f"Puntos originales: {onda.size} | Muestreo: {paso:.3f} nm")
    print("R original en 380 nm:", 380 / FWHM_ORIGINAL_NM)
    print("R original en 700 nm:", 700 / FWHM_ORIGINAL_NM)
    print("Procesando resoluciones:", RESOLUCIONES)
    resultados = {}
    for R in RESOLUCIONES:
        resultados[R] = degradar_espectro(onda, flujo, R, paso)
        print(f"  R={R:4d} -> FWHM objetivo en {LAMBDA_REF_NM:.1f} nm: "
              f"{LAMBDA_REF_NM/R:.3f} nm")
    exportar_csv(onda, flujo, resultados)
    grafica_comparativa(onda, flujo, resultados, region)
    ventana_interactiva(onda, flujo, resultados, region)


if __name__ == "__main__":
    main()