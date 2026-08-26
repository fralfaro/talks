# Imágenes de fondo · separadores de bloque · Taller 2

Ilustraciones de fondo para las 11 slides separadoras de
`taller_2_ia_investigacion_productividad.qmd` (IA aplicada a la investigación y la productividad).

Carpeta separada de `images/blocks/` (Taller 1) porque varios nombres se repiten
—`portada`, `pausa`, `anexo`, `gracias`— y cada taller usa su propia ilustración.

**Estilo sugerido:** flat minimalista (unDraw / Storyset / Humaaans), paleta cercana a
`#0F2044` (azul profundo), `#75AADB` / `#4A9FD4` (azul claro) y `#C9A84C` (dorado).
Se muestran con `background-opacity` entre 0.15 y 0.22, así que conviene que sean
imágenes de formas amplias y poco detalle: los trazos finos desaparecen al bajar la opacidad.

**Formato:** PNG horizontal, idealmente 1920×1080 (o el mismo aspecto), fondo transparente
o del mismo azul del tema.

| # | Archivo | Slide | Qué ilustrar |
|---|---|---|---|
| 1 | `portada.png` | Portada · "IA Generativa Aplicada a la Investigación y la Productividad" | Escena de apertura con acento en investigación: escritorio de trabajo académico, papers y datos, persona frente al computador. Debe leerse como continuación del Taller 1 pero con foco en investigar, no en enseñar. |
| 2 | `bloque0-encuadre.png` | Bloque 0 · Encuadre y resguardos | Integridad académica: escudo, lupa sobre un documento, firma o sello de autoría. Idea de verificar antes de publicar. |
| 3 | `bloque1-literatura.png` | Bloque 1 · Literatura científica | Búsqueda y síntesis de antecedentes: pila de papers, buscador, mapa o red de referencias conectadas. |
| 4 | `bloque2-datos.png` | Bloque 2 · Análisis exploratorio de datos | Exploración de datos: gráficos, tabla o dashboard, persona interrogando una visualización (el foco es preguntar, no automatizar). |
| 5 | `pausa.png` | Pausa · 5 minutos (15:20) | Descanso: taza de café, reloj, personas conversando de pie; tono liviano y calmado. |
| 6 | `bloque3-redaccion.png` | Bloque 3 · Redacción académica | Escritura de un paper: documento estructurado por secciones, persona redactando, borrador con correcciones. |
| 7 | `bloque4-automatizacion.png` | Bloque 4 · Automatización de tareas recurrentes | Trabajo repetitivo que se sistematiza: engranajes, flujo de pasos encadenados, correos e informes saliendo de una plantilla. |
| 8 | `bloque5-prueba-cruzada.png` | Bloque 5 · Prueba cruzada de plantillas (grupos de 4–5) | Intercambio entre pares: dos personas cambiando documentos, revisión cruzada, grupo probando el trabajo de otro. |
| 9 | `cierre-repositorio.png` | Cierre · Su repositorio de prompts | Organización de lo producido: carpetas o fichero ordenado, biblioteca personal, colección etiquetada de plantillas. |
| 10 | `anexo.png` | Anexo · Material de profundización | Material de consulta: libros, archivador, biblioteca o glosario. Fondo más claro (`#1A3A6B`), opacidad 0.15: usar la ilustración más sobria del set. |
| 11 | `gracias.png` | Cierre · Gracias | Cierre del taller: aplauso, personas despidiéndose, meta alcanzada; tono cálido. |

## Cómo se usan

Cada archivo se referencia desde el encabezado de su slide, por ejemplo:

```
##  {background-color="#0F2044" background-image="images/blocks-t2/portada.png" background-opacity="0.18" data-state="no-logo"}
```

Basta con dejar el PNG en esta carpeta con el nombre exacto de la tabla; no hay que
tocar el `.qmd`. Si alguna imagen resulta demasiado presente o demasiado tenue,
se ajusta solo el `background-opacity` de esa slide (rango recomendado: 0.15–0.25).

Si prefiere reutilizar una ilustración del Taller 1 (por ejemplo la pausa o el anexo),
copie el archivo a esta carpeta en vez de cambiar la ruta en el `.qmd`.
