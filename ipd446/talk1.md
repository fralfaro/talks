## Estructura Charala Talk1

**Slide 1 — Título**

Además del título y tu nombre, agregar en un corner pequeño el logo o referencia al seminario de Araya. Esto ancla la charla en el contexto del ramo sin necesitar explicación verbal.

---

**Slide 2 — "¿Dónde estamos?"**

Tomar el diagrama Agent Model que ya conocen (el ciclo clásico) y agregarle *una sola anotación nueva*: una flecha o recuadro que señale "¿dónde vive π?" con dos respuestas posibles destacadas visualmente:

- **Implícita** → emerge de Q(s,a) [capítulos anteriores]
- **Explícita** → es el objeto que aprendemos [Cap. 13]

El auditorio ya conoce el diagrama, así que el único trabajo cognitivo nuevo es esa distinción. No necesitas reexplicar el MDP.

---

**Slide 3 — El mapa del curso hasta aquí**

Reutilizar la lógica visual de los sellos A/B/C/D de Araya pero como línea de tiempo del libro:

```
Caps 2-4   →   Caps 5-7   →   Caps 8-12   →   Cap 13
Bandits       MC / TD        Aprox. func.     Policy Gradient
[base]        [value-based]  [escalabilidad]  [← estamos aquí]
```

Con una anotación explícita: *"todos los capítulos anteriores aprendían Q o V — la política era siempre derivada"*. Y Cap. 13 marcado con algo que indique ruptura, no continuación incremental.

El guiño a la slide de incertidumbre epistémica/aleatoria de Araya puede ir aquí como una línea al pie: *"la política estocástica óptima no es ruido — es variabilidad aleatoria racional"*. Una oración, no un bloque. Pero bien puesta genera conversación.

Aquí el resto de las ideas concretas slide por slide, con el mismo nivel de detalle:

---

**Slide 4 — Parametrización: Softmax en preferencias**

La tabla comparativa que ya tenías (ε-greedy vs softmax) funciona bien, pero agregar una columna implícita visual: un mini-diagrama de dos curvas, una que nunca toca 0 o 1 (ε-greedy) y una que sí puede llegar a los extremos (softmax con preferencias). Una imagen vale más que la explicación verbal aquí.

Conectar explícitamente con la notación del curso de Araya: él usa π(a|s) directamente, así que mostrar que π(a|s,θ) es la misma cosa pero ahora θ es lo que se aprende. Una línea, no un párrafo.

---

**Slide 5 — Punto difícil #1: ¿Por qué no softmax sobre Q?**

La pregunta retórica funciona bien como título. Lo que agregaría es un ejemplo numérico mínimo: supón Q(s,a1)=10, Q(s,a2)=9. Con softmax sobre Q las probabilidades convergen a algo como 0.73/0.27 — nunca a 1/0. Con preferencias h pueden crecer sin límite → probabilidades 0.99/0.01 o más. Dos números concretos hacen el argumento inmediato para un auditorio técnico.

---

**Slide 6 — Ejemplo 13.1: El corredor corto**

Este es el slide más importante de la introducción, dale espacio visual generoso. La estructura que sugiero:

Primero el gridworld dibujado limpio (los 3 estados + terminal, las flechas, la inversión en s2 marcada en rojo). Segundo la gráfica J(θ) vs probabilidad de right. Tercero — y esto es lo que agregaría — una conexión explícita con la slide B de Araya ("Partially Observable Problem"): *"todos los estados se ven iguales bajo la aproximación de función — es como un POMDP de facto"*. El auditorio ya tiene ese marco mental del curso.

El número clave a destacar visualmente: **0.59** y **−11.6** vs **−44**. Esa diferencia es el argumento completo.

---

**Slide 7 — El Teorema del Gradiente de la Política**

La ecuación central en grande, centrada:

$$\nabla J(\theta) \propto \sum_s \mu(s) \sum_a q_\pi(s,a) \nabla\pi(a|s,\theta)$$

Debajo, tres anotaciones con flechas señalando cada componente:
- µ(s): *"cuánto tiempo pasamos en s — viene del comportamiento"*
- q_π(s,a): *"qué tan buena es la acción — viene del valor"*  
- ∇π: *"en qué dirección cambiar θ para favorecer esa acción"*

Y una caja destacada con el insight clave: *"∇J no involucra ∇µ — eso es lo no obvio"*. Esa frase es la que hay que que el auditorio se lleve.

---

**Slide 8 — Punto difícil #2: La prueba**

No mostrar la prueba completa en el slide — está en el capítulo y el auditorio puede leerla. Lo que sí mostrar es la *estructura* de la prueba como diagrama de pasos:

```
∇v_π(s)
  ↓ regla del producto
∇π · q_π + π · ∇q_π
  ↓ Bellman sobre q_π
∇π · q_π + π · Σ p(s'|s,a) · ∇v_π(s')
  ↓ unrolling recursivo
Σ_x Pr(s→x, k, π) · Σ_a ∇π · q_π
  ↓ normalizar por Ση(s')
∝ Σ_s µ(s) Σ_a ∇π · q_π  ✓
```

El punto verbal a enfatizar: el unrolling es la misma idea que Bellman pero aplicada al gradiente, no al valor. Si ya entienden la ecuación de Bellman, la estructura de esta prueba es familiar.

---

**Slide 9 — REINFORCE: el algoritmo**

La ecuación de actualización en grande:

$$\theta_{t+1} \doteq \theta_t + \alpha G_t \nabla \ln \pi(A_t|S_t,\theta_t)$$

Luego el pseudocódigo del libro tal cual — es limpio y no necesita reescribirse. Lo que agregar es una anotación lateral con la analogía directa con SGD en deep learning:

```
θ ← θ + α · loss_gradient
θ ← θ + α · Gt · ∇ln π
```

Para un auditorio de doctorado que ya conoce backprop, esta analogía es inmediata y ancla el algoritmo en algo familiar.

---

**Slide 10 — Punto difícil #3: ∇ ln π**

Tres partes en este slide:

Primera: la identidad matemática simple, sin contexto, solo para que quede clara:
$$\nabla \ln x = \frac{\nabla x}{x} \quad \Rightarrow \quad \frac{\nabla\pi}{\pi} = \nabla\ln\pi$$

Segunda: por qué importa en la práctica — el eligibility vector es lo único que cambia si cambias la parametrización. Todo lo demás del algoritmo es igual. Eso es modularidad.

Tercera: el ejemplo concreto con softmax lineal:
$$\nabla\ln\pi(a|s,\theta) = \mathbf{x}(s,a) - \sum_b \pi(b|s,\theta)\mathbf{x}(s,b)$$

Interpretación verbal: *"feature de la acción tomada menos el promedio ponderado de features — cuánto se desvía esta acción del comportamiento esperado"*. Esa interpretación es la que no da el libro directamente.

---

**Slide 11 — REINFORCE: propiedades y limitaciones**

Aquí meter la Figura 13.1 del libro (las curvas con distintos α). Lo que agregar sobre la figura es una anotación que conecte con algo que el auditorio del seminario ya vio: la varianza alta de Monte Carlo vs TD es el mismo trade-off que aparece en los capítulos anteriores, ahora reapareciendo a nivel de política. No es un problema nuevo, es el mismo problema en otro nivel.

Una frase para el cierre del slide: *"REINFORCE converge, pero lento — la misma tensión de siempre entre sesgo y varianza"*.

---

**Slide 12 — REINFORCE con Baseline**

La ecuación generalizada con (Gt − b(St)) y una visualización intuitiva: imaginar que b(St) es el "promedio esperado" en ese estado. Si Gt > b → la acción fue mejor de lo esperado → reforzar. Si Gt < b → fue peor → debilitar. Esa intuición no la da explícitamente el libro.

La Figura 13.2 del libro (comparación con y sin baseline) es el argumento empírico — mostrarla con la diferencia de convergencia resaltada visualmente. La curva con baseline en verde, sin baseline en gris, la brecha entre ellas señalada con una flecha y el texto *"misma garantía teórica, muy distinta práctica"*.

---

**Slide 13 — Actor-Crítico: la distinción conceptual**

La tabla comparativa que ya tenías funciona, pero agregar un diagrama de ciclo que muestre los dos roles claramente separados:

```
Estado S
   ↓
[ACTOR: π_θ] → Acción A → Entorno → R, S'
      ↑
   δt = R + γv̂(S') - v̂(S)
      |
[CRÍTICO: v̂_w]
```

El punto verbal clave: el crítico no elige acciones, solo califica. El actor no calcula valores, solo actúa. La división de responsabilidades es lo que hace el método escalable.

Conectar con la slide de Araya sobre el Agent Model — el diagrama que ya conocen ahora tiene dos módulos internos donde antes había uno.

---

**Slide 14 — Las tres variantes Actor-Crítico**

En lugar de tres pseudocódigos completos (que son densos), una tabla de una línea por variante:

| Variante | Señal de aprendizaje | Online? | Sesgo |
|---|---|---|---|
| One-step | δt = R + γv̂(S') − v̂(S) | Sí | Bajo |
| Con trazas (episódico) | λ-return | Sí | Medio |
| Continuo | R − r̄ + v̂(S') − v̂(S) | Sí | Bajo |

Y una nota al pie: *"en los tres casos la estructura Actor-Crítico es idéntica — solo cambia qué se usa como señal δ"*. Eso simplifica mucho la lectura del pseudocódigo para quien lo quiera ver después.

---

**Slide 15 — Acciones continuas**

La distribución Gaussiana parametrizada es el contenido central, pero el insight que vale la pena destacar visualmente es la división del vector θ en dos partes:

```
θ = [θ_µ | θ_σ]
     ↓       ↓
  media    desv. estándar
 (lineal)  (exp de lineal)
```

Y una caja con la pregunta que conecta con la realidad del auditorio: *"¿por qué exp para σ?"* — porque garantiza positividad sin restricciones en la optimización. Un detalle de implementación que parece menor pero que sale mal si no se entiende.

Conexión con el estado del arte: SAC, PPO, DDPG todos usan exactamente esta estructura. No como dato de trivia sino como señal de que este slide no es histórico — es actual.

---

**Slide 16 — Punto difícil #4: Caso continuo vs episódico**

El retorno diferencial es el concepto más abstracto del capítulo. La forma de hacerlo concreto:

$$G_t \doteq \sum_{k=1}^{\infty}(R_{t+k} - r(\pi))$$

Visualización: dos líneas de recompensas en el tiempo. Una con r(π) marcado como línea horizontal. El retorno diferencial son las desviaciones sobre y bajo esa línea. *"No descuentas el tiempo — descuentas la tasa promedio"*. Esa frase es la clave.

La tabla episódico vs continuo que ya tenías sirve bien aquí para anclar.

---

**Slide 17 — Mapa del capítulo**

El árbol que ya tenías es bueno. Lo que agregaría es colorear los nodos por el trade-off que representan:

- Rojo: alta varianza, sin sesgo (REINFORCE puro)
- Amarillo: varianza reducida, sin sesgo adicional (baseline)
- Verde: varianza baja, sesgo controlado (Actor-Crítico)

Esa codificación de color hace el mapa legible de un vistazo y conecta con la tensión sesgo-varianza que es el hilo conductor del capítulo.

---

**Slide 18 — Comparación filosófica**

La tabla que ya tenías está bien construida. Lo que agregaría es una fila final que el libro no dice explícitamente:

| Dimensión | Action-value | Policy gradient |
|---|---|---|
| ... | ... | ... |
| **¿Cuándo elegir?** | Espacios discretos pequeños, Q simple | Política simple, acciones continuas, óptimo estocástico |

Y una cita de cierre del propio Sutton en la sección 13.8: *"Today they are less well understood in some respects, but a subject of excitement and ongoing research"* — honestidad académica que el auditorio de doctorado aprecia.

---

**Slide 19 — Conexión con el estado del arte**

No solo listar PPO, SAC, A3C como datos. Para cada uno, una línea que conecte con algo específico del capítulo:

- **PPO** → REINFORCE con baseline + restricción en el paso
- **A3C** → Actor-Crítico one-step con múltiples agentes paralelos
- **SAC** → Actor-Crítico continuo + entropía como baseline adicional
- **RLHF (ChatGPT)** → política parametrizada por LLM + reward model como crítico

Esa última conexión — RLHF — es la que genera conversación en un auditorio de doctorado en 2026. Y es técnicamente correcta: REINFORCE con baseline es exactamente lo que está debajo de PPO que está debajo de RLHF.

---

**Slide 20 — Cierre**

Los tres resultados que ya tenías son buenos. Lo que cambiaría es el tono de la pregunta final — en lugar de una pregunta abierta genérica, dejar una pregunta que conecte con el proyecto del seminario:

> *"¿Cuándo en tu problema tiene más sentido aprender π directamente que aprender Q primero?"*

Eso hace que el cierre sea útil para cada persona en la sala, no solo un remate retórico. Y abre la discusión de 5 minutos que normalmente queda al final de una charla de 45.

## Imagenes


**Slide 2 — "¿Dónde estamos en el libro?"**

Un diagrama de línea de tiempo horizontal mostrando los capítulos del libro, con el Cap. 13 destacado al final como bifurcación. Algo tipo:

```
[2-4 Bandits] → [5-7 MC/TD] → [8-12 Aprox.] → [13 Policy ★]
      ↓               ↓              ↓                ↓
  Q implícita    Q implícita    Q implícita      π explícita
```

Lo haría como SVG inline en el QMD — limpio y sin dependencias externas.

---

**Slide 6 — Ejemplo 13.1 (el corredor corto)**

El gridworld dibujado como SVG: los 3 estados + terminal, las flechas de transición, y la inversión en $s_2$ marcada en rojo. Es el diagrama más importante de la charla y el libro lo tiene pequeño.

---

**Slide 8 — Estructura de la prueba**

Un diagrama de flujo vertical con los 5 pasos de la prueba, cada uno en un recuadro con la operación matemática clave. Más legible que el bloque de código que tenemos ahora.

---

**Slide 13 — Actor–Crítico**

El ciclo Actor–Crítico como diagrama: dos bloques (Actor/Crítico) conectados con flechas etiquetadas. El que tenemos en ASCII funciona pero un SVG se vería mucho más profesional.

---

**Slide 17 — Mapa del capítulo**

El árbol de métodos como diagrama visual con nodos coloreados por el trade-off sesgo/varianza (rojo → amarillo → verde). El texto ASCII actual es difícil de leer en pantalla grande.

---

**Slide 19 — Conexión con el estado del arte**

Una línea de tiempo 1990→2026 mostrando la evolución: REINFORCE (1992) → A3C (2016) → PPO (2017) → SAC (2018) → RLHF (2022). Con los algoritmos modernos conectados visualmente al capítulo.

---

**Mi recomendación:** empezar con las slides 6, 13 y 17 — son las que más se benefician visualmente y las que el auditorio va a mirar más tiempo. Las hago como SVG embebido directamente en el QMD para que no necesites archivos externos.

