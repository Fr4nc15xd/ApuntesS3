# Resumen — Procesamiento Digital de Imágenes (PDI)


## 1. Teoría base

### 1.1 La imagen como función matemática
- Una imagen es una función bidimensional **f(x, y)**, donde x, y son coordenadas espaciales finitas (mundo discreto).
- El mundo real es continuo (energía radiada por un objeto físico); la cámara hace **muestreo y cuantización** para pasar de analógico a digital.
- **Píxel (Picture Element):** no es "un cuadrito", es un **valor de intensidad cuantizado** en una coordenada específica.

### 1.2 Espectro electromagnético
- El ojo humano solo ve **400–700 nm** (luz visible).
- Fuera de ese rango existen: rayos gamma (PET médico), rayos X (control industrial), infrarrojo (satélites), microondas (radar), ondas de radio (MRI).
- Una máquina no solo "ve más rápido": procesa un universo físico distinto e invisible para el humano.

### 1.3 Analogía ojo ↔ cámara
| Función | Ojo humano | Cámara digital |
|---|---|---|
| Enfoque óptico | Córnea / Cristalino | Sistema de lentes de vidrio |
| Control lumínico | Pupila / Iris | Diafragma mecánico |
| Transducción (luz→señal) | Retina | Sensor CCD / CMOS |
| Baja luz | Bastones | Píxeles de luminancia (alto ISO) |
| Percepción de color | Conos (RGB) | Filtro Bayer (capa RGB) |

### 1.4 Tres niveles de decisión (jerarquía computacional)
1. **Procesamiento de Imágenes (PDI):** píxeles → píxeles. Mejora, restauración, filtrado. Salida: imagen visual mejorada. *(Cero inteligencia)*
2. **Análisis de Imágenes:** píxeles → datos. Extracción de atributos, vectores, contornos, segmentación geométrica. Salida: estructuras de datos.
3. **Visión Artificial (IA):** datos → decisión. Comprensión semántica, reconocimiento de objetos. Ej: "Es un tumor", "Frena el vehículo".

Jerarquía por nivel:
- **Nivel bajo (pre-procesamiento):** tratamiento matemático de la matriz. Filtrado, suavizado, ecualización.
- **Nivel medio (geometría):** organización de la información. Detección de bordes, segmentación, separación fondo-forma.
- **Nivel alto (semántica):** modelado y cognición. Integración con IA.

### 1.5 Pipeline clásico del sistema PDI
Adquisición → Mejora y restauración → Procesamiento morfológico y segmentación → Representación → Reconocimiento, todo retroalimentado por una **Base de Conocimiento** (Machine Learning / modelos previos).

### 1.6 Filtrado — manipulación matemática del píxel
- **Filtros de paso bajo (suavizado):** atenúan altas frecuencias (cambios bruscos), promedian píxeles vecinos. Uso: reducción de ruido electrónico en sensores.
- **Filtros de paso alto (bordes):** suprimen bajas frecuencias (zonas uniformes), resaltan picos de intensidad. Uso: extracción geométrica de contornos.

### 1.7 Aplicaciones
- **Medicina:** angiografía por rayos X (contraste para aislar venas/arterias), tomografía PET por rayos gamma (isótopos, mapeo metabólico), resonancia magnética (ondas de radio → volúmenes 3D de tejido blando).
- **Industria/control de calidad:** visión de rayos X para micro-soldaduras (PCB), segmentación y morfología para verificar empaques farmacéuticos y nivel de llenado de líquidos. El algoritmo no "sabe" qué es una pastilla: compara perfiles geométricos contra tolerancias.
- **Teledetección:** análisis multiespectral (satélite, bandas invisibles: infrarrojo térmico/cercano) para segmentar humedad, medir biomasa, crecimiento urbano; microondas/radar para topografía a través de nubes/oscuridad.
- **Robótica/navegación autónoma:** sensores activos (láser) para mapas de profundidad 3D; traducción de matrices de píxeles (x, y) a coordenadas volumétricas (z); tracking de objetos en movimiento.

### 1.8 Paradoja de Moravec
> Lo que es biológicamente instintivo para el humano (reconocer una escena en milisegundos) requiere una complejidad algorítmica y de cómputo inmensa para una máquina.

El PDI no es "mejorar fotos": es la capa de traducción que permite a los sistemas artificiales cruzar del mundo analógico (luz) hacia la toma de decisiones autónoma.

---

## 2. Matrices con NumPy

### 2.1 Crear matrices
```python
import numpy as np

# Matriz manual (lista de listas)
m33 = [[1,2,3],[4,5,6],[7,8,9]]
a33 = np.array(m33)

np.ones((1920, 1080))                    # matriz de puros 1's, dtype float64 por defecto
np.zeros((100, 100), dtype=np.uint8)     # matriz de puros 0's, tipo específico
np.random.random()                        # un float aleatorio entre 0 y 1
np.random.randint(256, size=(1920,1080), dtype=np.uint8)  # matriz aleatoria de enteros
np.random.seed(27)                        # fija la semilla para reproducibilidad
```

### 2.2 Indexado
```python
m33[1][1]     # indexado estilo lista de Python
a33[0, 2]     # indexado estilo numpy (fila, columna)
fila, columna = mat1.shape
len(mat1)     # número de filas
```

### 2.3 Atributos de un ndarray
```python
type(a33)     # numpy.ndarray
a33.shape     # dimensiones (filas, columnas)
a33.size      # número total de elementos
a33.dtype     # tipo de dato (uint8, float64, etc.)
a33.T         # transpuesta
```

### 2.4 Operaciones entre matrices
```python
A + B     # suma elemento a elemento
A - B     # resta elemento a elemento
A * B     # multiplicación elemento a elemento (Hadamard)
A @ B     # producto matricial (dot product)
np.matmul(A, B)  # equivalente a @

A + 5     # broadcasting: suma 5 a cada elemento
B - 10    # broadcasting: resta 10 a cada elemento
```
- Para que `A @ B` funcione, las dimensiones internas deben coincidir (columnas de A = filas de B).
- Se puede cambiar el tipo con `.astype()`, por ejemplo `A.astype(np.float32)`, útil antes de multiplicar matrices grandes para evitar overflow.

### 2.5 Recorrer una matriz manualmente
```python
fila, columna = mat1.shape
for i in range(fila):       # filas
    for j in range(columna):  # columnas
        mat1[i, j] += 10     # ej: sumar 10 a cada elemento
```
(Esto es equivalente, pero más lento, que `mat1 + 10` con broadcasting.)

### 2.6 Copia vs. referencia (¡importante!)
```python
mat3 = mat2         # NO copia: mat3 apunta al mismo objeto que mat2
mat3[0,0] = 100      # esto también modifica mat2

mat4 = mat2.copy()   # SÍ copia (deep copy): son objetos independientes
```

---

## 3. Imágenes como matrices (con Matplotlib)

### 3.1 Mostrar una matriz como imagen
```python
from matplotlib import pyplot as plt

img = np.random.randint(256, size=(8,8), dtype=np.uint8)
plt.imshow(img, cmap="gray")
plt.show()
```
- `cmap="gray"` interpreta la matriz como escala de grises (0 = negro, 255 = blanco).
- `vmin=0` fuerza el mínimo de la escala de color.

### 3.2 Imágenes básicas
```python
img_blanca = np.ones((100,100), dtype=np.uint8) * 255   # imagen blanca
img_negra  = np.zeros((100,100), dtype=np.uint8)          # imagen negra
```

### 3.3 Slicing para crear patrones
```python
img_negra = np.zeros((99,99), dtype=np.uint8)
img_negra[:34] = 127     # primeras 34 filas en gris medio
img_negra[66:] = 255     # últimas filas en blanco
plt.imshow(img_negra.T, cmap="gray")   # .T transpone (invierte filas/columnas)
plt.show()
```

### 3.4 Cargar una imagen real
```python
imagen = plt.imread(r"/ruta/a/la/imagen.jpg")
type(imagen)     # numpy.ndarray
imagen.shape     # (alto, ancho, canales) -> imagen a color = 3 canales (R,G,B)
imagen.size      # alto * ancho * canales

plt.imshow(imagen)
plt.title('Título')
plt.show()
```
> Nota de rutas: en Mac/Linux se usa `/`, en Windows `\`. Para evitar problemas con los `\` en Windows se usa un raw string: `r"C:\Users\..."`.

---

## 4. Canales de color e imágenes RGB

### 4.1 Separar canales
```python
canal_R = imagen[:, :, 0]
canal_G = imagen[:, :, 1]
canal_B = imagen[:, :, 2]

plt.imshow(canal_R, cmap='gray')   # cada canal por separado se ve en escala de grises
plt.show()

plt.imshow(canal_R[100:280, 590:740], cmap='gray')  # recortar (slicing) una región
plt.show()
```

### 4.2 Estadísticas de una imagen/canal
```python
canal_R.min()      # valor mínimo
canal_R.max()      # valor máximo
canal_R.mean()      # promedio
canal_R.std()       # desviación estándar
np.median(canal_R)  # mediana
np.sum(canal_R)     # suma de todos los valores

# Contar cuántas veces aparece un valor específico
(canal_R == 255).astype(np.int32).sum()
len(np.where(canal_R == 0)[0])

canal_R[126, 730]   # valor de un píxel específico
```

### 4.3 Escala de grises a partir de RGB
```python
# Promedio de los 3 canales
imagen_gris = (canal_R + canal_G + canal_B) / 3
imagen_gris = imagen_gris.astype(np.uint8)
plt.imshow(imagen_gris, cmap='gray')
plt.show()

# Equivalente con numpy
np.mean(imagen, axis=2)   # promedio por canal (grises)
```
- `axis=0` → promedio por columna
- `axis=1` → promedio por fila
- `axis=2` → promedio por canal (esto da la escala de grises)

### 4.4 Aislamiento cromático (mostrar solo un color)
```python
imagen_roja = imagen.copy()      # siempre copiar antes de modificar
imagen_roja[:, :, 1] = 0          # apagar canal G
imagen_roja[:, :, 2] = 0          # apagar canal B
plt.imshow(imagen_roja)
plt.show()

imagen_verde = imagen.copy()
imagen_verde[:, :, (0, 2)] = 0    # apagar R y B a la vez (tupla de índices)
plt.imshow(imagen_verde)
plt.show()
```

### 4.5 Mezcla de canales (colores secundarios)
```python
# R + G (sin B) -> Amarillo
imagen_amarilla = imagen.copy()
imagen_amarilla[:, :, 2] = 0

# R + B (sin G) -> Magenta
imagen_magenta = imagen.copy()
imagen_magenta[:, :, 1] = 0

# G + B (sin R) -> Cian
imagen_cian = imagen.copy()
imagen_cian[:, :, 0] = 0
```

---

## 5. Cheatsheet rápida

| Quiero... | Código |
|---|---|
| Crear matriz de ceros | `np.zeros((f,c), dtype=np.uint8)` |
| Crear matriz de unos | `np.ones((f,c))` |
| Matriz aleatoria 0–255 | `np.random.randint(256, size=(f,c), dtype=np.uint8)` |
| Fijar semilla aleatoria | `np.random.seed(N)` |
| Forma / tamaño / tipo | `.shape` / `.size` / `.dtype` |
| Transponer | `.T` |
| Cambiar tipo de dato | `.astype(np.float32)` |
| Suma / resta elemento a elemento | `A + B` / `A - B` |
| Multiplicación Hadamard | `A * B` |
| Producto matricial | `A @ B` o `np.matmul(A,B)` |
| Copiar sin alias | `.copy()` |
| Cargar imagen | `plt.imread(ruta)` |
| Mostrar imagen | `plt.imshow(img, cmap='gray'); plt.show()` |
| Recortar región (slicing) | `img[y1:y2, x1:x2]` |
| Separar canal | `img[:, :, 0]` (R), `1` (G), `2` (B) |
| Apagar un canal | `img_copia[:, :, i] = 0` |
| Escala de grises (promedio) | `np.mean(img, axis=2).astype(np.uint8)` |
| Min/Max/Promedio/Desv./Mediana | `.min() .max() .mean() .std() np.median()` |
| Contar píxeles con cierto valor | `(img == valor).sum()` |
| Ubicar píxeles con cierto valor | `np.where(img == valor)` |

---

## 6. Temas de exposición vistos en clase (por compañeros)
**Matrices:**
- ¿Qué es una matriz?
- Suma de matrices
- Multiplicación de matrices
- Producto punto (de matrices)
- Producto de Hadamard

**Color:**
- Grayscale
- RGB / BGR
- CMY / CMYK
- YCbCr
- HSV
