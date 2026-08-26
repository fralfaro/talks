# Imágenes de fondo · separadores de bloque

Ilustraciones de fondo para las 12 slides separadoras de `taller_1_ia_docencia.qmd`.

**Estilo sugerido:** flat minimalista (unDraw / Storyset / Humaaans), paleta cercana a
`#0F2044` (azul profundo), `#75AADB` / `#4A9FD4` (azul claro) y `#C9A84C` (dorado).
Se muestran con `background-opacity` entre 0.15 y 0.22, así que conviene que sean
imágenes de formas amplias y poco detalle: los trazos finos desaparecen al bajar la opacidad.

**Formato:** PNG horizontal, idealmente 1920×1080 (o el mismo aspecto), fondo transparente
o del mismo azul del tema.

| # | Archivo | Slide | Qué ilustrar |
|---|---|---|---|
| 1 | `portada.png` | Portada · "Fundamentos de IA Generativa y su Aplicación a la Docencia" | Escena de apertura: docente frente a una pantalla o pizarra con elementos de IA; tono institucional y de bienvenida. |
| 2 | `bloque0-encuadre.png` | Bloque 0 · Encuadre y uso responsable | Reglas y acuerdos de trabajo: checklist, escudo o balanza, personas acordando cómo van a trabajar. |
| 3 | `bloque1-fundamentos.png` | Bloque 1 · Fundamentos de IA generativa | Cómo funciona un modelo de lenguaje: red neuronal, texto que se predice, engranajes o cerebro digital. |
| 4 | `bloque2-prompting.png` | Bloque 2 · Prompting efectivo | Escribir instrucciones: persona tecleando en una interfaz de chat, burbujas de conversación, texto que se estructura. |
| 5 | `pausa.png` | Pausa · 5 minutos (15:20) | Descanso: taza de café, reloj, personas conversando de pie; tono liviano y calmado. |
| 6 | `bloque3-diseno.png` | Bloque 3 · Diseño instruccional con IA | Planificación de una unidad: bloques o piezas que se alinean, mapa de ruta, tablero de diseño. |
| 7 | `bloque4-evaluacion.png` | Bloque 4 · Evaluación para el aprendizaje | Instrumentos y rúbricas: tabla de criterios, lista de cotejo, prueba siendo revisada. |
| 8 | `bloque5-retroalimentacion.png` | Bloque 5 · Retroalimentación del aprendizaje | Comentario que orienta: docente devolviendo un trabajo con notas al margen, diálogo uno a uno. |
| 9 | `bloque6-clinica.png` | Bloque 6 · Clínica de alineación (grupos de 4–5) | Trabajo colaborativo: grupo alrededor de una mesa o pizarra, post-its, construcción conjunta. |
| 10 | `anuncio-taller2.png` | Anuncio · Taller 2 (investigación y productividad) | Continuidad hacia lo que viene: literatura científica, gráficos de datos, flecha o camino hacia adelante. |
| 11 | `anexo.png` | Anexo · Material de profundización | Material de consulta: libros, archivador, biblioteca o glosario. Fondo más claro (`#1A3A6B`), opacidad 0.15: usar la ilustración más sobria del set. |
| 12 | `gracias.png` | Cierre · Gracias | Cierre del taller: aplauso, personas despidiéndose, meta alcanzada; tono cálido. |

## Cómo se usan

Cada archivo se referencia desde el encabezado de su slide, por ejemplo:

```
##  {background-color="#0F2044" background-image="images/blocks/portada.png" background-opacity="0.18" data-state="no-logo"}
```

Basta con dejar el PNG en esta carpeta con el nombre exacto de la tabla; no hay que
tocar el `.qmd`. Si alguna imagen resulta demasiado presente o demasiado tenue,
se ajusta solo el `background-opacity` de esa slide (rango recomendado: 0.15–0.25).
