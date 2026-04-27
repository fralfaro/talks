# 🎤 Speech Completo — Policy Gradient Methods
### Capítulo 13 · Sutton & Barto · Francisco Alfaro · UTFSM 2026

---

> **Guía de lectura:**
> - `[acción]` → indicación escénica
> - **negrita** → énfasis de voz
> - *cursiva* → término técnico a pronunciar con cuidado
> - `> Nota de ritmo` → recordatorio de tiempo y pausa

---

## BLOQUE 1 — Apertura e Introducción
### Slides: Título → El problema que motiva → ¿Dónde estamos? → Bandit → Baseline
**⏱ Tiempo estimado: 10–12 minutos**

---

### [SLIDE: TÍTULO]

*[Pausa breve. Mirar a la audiencia antes de hablar.]*

Buenas tardes. El capítulo que voy a presentar hoy es el trece de Sutton y Barto — *Policy Gradient Methods*. El título que elegí para la presentación es **Más allá de los valores**, y voy a tratar de justificar ese nombre a lo largo de la sesión.

La idea central es la siguiente: todo lo que vimos en los capítulos anteriores — Q-learning, TD, aproximación de función — tiene una cosa en común. Primero aprendemos los **valores** de las acciones, y después la política **emerge** de esos valores. Hoy vamos a hacer algo distinto: vamos a aprender la política **directamente**, sin pasar por los valores.

El capítulo tiene tres resultados principales que voy a desarrollar en orden: el *Policy Gradient Theorem*, el algoritmo REINFORCE, y los métodos Actor–Crítico.

---

### [SLIDE: El problema que motiva este capítulo]

Para contextualizar por qué esto importa, voy a usar el mapa que hemos ido construyendo en el curso sobre fuentes de incertidumbre en el modelo agente–entorno.

Tenemos cuatro tipos de problema. El problema A — cuando el siguiente estado es difícil de predecir — lo resolvimos con MDPs. El problema B — cuando el estado no es completamente observable — es el mundo de los POMDPs, que es difícil. El problema D — si las recompensas deben ser externas o internas — corresponde al área de *intrinsic motivation*.

Nosotros estamos en el **problema C**: la dinámica del entorno es desconocida. El agente no sabe cómo funciona el mundo. Solo tiene experiencia. Y en ese escenario, aprender directamente la política tiene sentido natural — porque nunca vamos a tener acceso a `p(s', r | s, a)`, la función de transición.

*[Clic para mostrar el callout.]*

Pero hay algo más importante todavía. Cuando la política óptima es **estocástica** — cuando la mejor decisión es asignar probabilidades específicas a distintas acciones — los métodos basados en valores simplemente no pueden llegar ahí. Y voy a mostrarles un ejemplo concreto de eso en unos minutos.

---

### [SLIDE: ¿Dónde estamos en el libro?]

Mirando la hoja de ruta del libro, hay un patrón claro. En los capítulos dos al doce, la política siempre fue **implícita** — algo que se derivaba de los valores aprendidos. Nunca fue el objeto de aprendizaje en sí mismo.

El capítulo trece rompe con eso. La política pasa a ser el **primer objeto**, no un subproducto.

*[Clic para mostrar el fragment.]*

Y hay algo importante que Sutton y Barto mencionan: este no es un tema completamente nuevo en el libro. En la sección 2.8, el *gradient bandit algorithm* ya hacía exactamente esto — pero en el caso más simple posible: un solo estado, sin transiciones, sin futuro. El capítulo trece es la **generalización completa** de esa idea al mundo de los MDPs. Si entendieron bien ese algoritmo, ya tienen la maquinaria conceptual central de todo lo que viene.

---

### [SLIDE: Punto de partida — el Bandit de gradiente]

Entonces empecemos por ahí — por el Bandit — para establecer el vocabulario.

En el Bandit tenemos `k` acciones, un solo estado, sin futuro. No hay `s`, no hay `γ`, no hay episodios. Solo acciones y recompensas inmediatas.

En lugar de estimar los valores de las acciones directamente, definimos **preferencias** `H_t(a)` — números reales que no representan valores, sino puntuaciones relativas entre acciones. Y convertimos esas preferencias en probabilidades mediante *softmax*:

```
π_t(a) = e^{H_t(a)} / Σ_b e^{H_t(b)}
```

¿Por qué softmax y no simplemente normalizar? Porque softmax es **diferenciable**, lo que nos permite calcular gradientes. Y eso es fundamental.

La actualización sigue ascenso de gradiente sobre `J = E[R_t]`:

```
H_{t+1}(a) = H_t(a) + α · (R_t - R̄_t) · (1_{a=A_t} - π_t(a))
```

*[Señalar cada parte de la ecuación.]*

Hay **tres ingredientes** aquí que quiero que tengan presentes porque van a aparecer **idénticos** en REINFORCE y en Actor–Crítico más adelante.

Primero, el **softmax** — política diferenciable.

Segundo, el **baseline** `R̄_t` — el promedio de recompensas que se resta para reducir varianza.

Y tercero, el **truco del log-gradiente**: `∇π/π = ∇ ln π`, que simplifica todos los cálculos.

Solo va a cambiar una cosa entre el Bandit y el capítulo 13: la señal. En el Bandit es `R_t` inmediata. En REINFORCE será `G_t`, el retorno acumulado.

---

### [SLIDE: Por qué el baseline no sesga el gradiente]

Antes de avanzar, quiero dedicar un minuto a este resultado porque es la **base matemática de todo lo que sigue** — y a veces se da por sentado sin demostrar.

La pregunta es: ¿por qué podemos restar un baseline sin cambiar la dirección del gradiente?

La respuesta está en esta identidad:

```
Σ_x ∂π_t(x)/∂H_t(a) = ∂/∂H_t(a) · Σ_x π_t(x) = ∂/∂H_t(a) · 1 = 0
```

Las probabilidades siempre suman uno. Por lo tanto, la derivada de esa suma siempre es cero. Eso significa que si yo agrego cualquier escalar `B_t` dentro de la suma, su contribución al gradiente es **exactamente cero**. El baseline puede ser lo que quiera — no cambia la esperanza del gradiente. Solo cambia su varianza.

*[Señalar el callout de la derecha.]*

Y esta misma identidad, trasladada al capítulo trece, justifica restar `v̂(S_t, w)` en REINFORCE. La demostración es idéntica, solo que ahora la suma es sobre acciones con la política `π_θ`.

---

### [SLIDE: La idea central del Cap. 13]

Bien. Con ese vocabulario establecido, llegamos a la idea central.

Queremos aprender una política `π(a|s,θ)` parametrizada por un vector `θ`. El objetivo es **maximizar** `J(θ)`, que en el caso episódico es el valor del estado inicial — cuánto retorno espera obtener el agente desde donde empieza.

La estrategia es ascenso de gradiente:

```
θ_{t+1} = θ_t + α · ∇̂J(θ_t)
```

La analogía con deep learning es directa. Los parámetros `θ` son como los pesos de la red. La performance `J(θ)` es como el negativo del loss. Y el *Policy Gradient Theorem* cumple el rol de la backpropagation — nos dice cómo calcular el gradiente.

*[Clic para el fragment.]*

Y hay dos familias de métodos que emergen de esta idea. Los métodos *policy-only*, donde `θ` es todo lo que aprendemos. Y los métodos **Actor–Crítico**, donde aprendemos `θ` para la política **y** un vector `w` para una función de valor que asiste al aprendizaje. Vamos a ver los dos.

---

> **📍 Nota de ritmo — Bloque 1:** Aquí van aproximadamente **10–12 minutos**. Buen momento para preguntar si hay dudas sobre el vocabulario antes de entrar al teorema.

---

## BLOQUE 2 — Definiciones, Ejemplo 13.1 y Policy Gradient Theorem
### Slides: Definiciones → Softmax → ¿Por qué no Q? → Ejemplo 13.1 → p* → Bellman → Teorema → Prueba
**⏱ Tiempo estimado: 13–15 minutos**

---

### [SLIDE: Definiciones fundamentales — G_t y ∇]

Antes de entrar al ejemplo y al teorema, quiero anclar dos objetos matemáticos que van a aparecer en **absolutamente todas** las ecuaciones que siguen. Si estos quedan claros, el resto se lee solo.

El primero es el **retorno** `G_t`. Es la suma de todas las recompensas futuras, descontadas exponencialmente:

```
G_t = Σ_{k=0}^∞ γ^k · R_{t+k+1}
```

Con `γ = 0`, el agente es completamente miope — solo le importa la recompensa del próximo paso. Con `γ = 1`, tiene visión infinita — todas las recompensas futuras valen igual que las inmediatas. En el capítulo trece vamos a usar `γ = 1` para el caso episódico, lo que simplifica bastante la notación.

Y hay una recursión clave que vamos a usar en la prueba del teorema: `G_t = R_{t+1} + γ · G_{t+1}`. El retorno desde `t` es la recompensa inmediata más el retorno descontado desde el paso siguiente.

El segundo objeto es el **operador gradiente** `∇_θ`. Es simplemente el vector de todas las derivadas parciales de una función respecto a los parámetros. Lo que importa intuitivamente es que **apunta en la dirección de mayor crecimiento** de la función. Hacer ascenso de gradiente sobre `J(θ)` significa moverse en esa dirección.

---

### [SLIDE: Parametrización — softmax en preferencias]

Ahora la pregunta práctica: ¿cómo parametrizamos la política?

Para espacios de acción discretos, la respuesta natural es la misma que en el Bandit: *softmax* sobre preferencias `h(s,a,θ)`.

Noten la tabla comparativa. Hay una diferencia crítica entre softmax y *ε-greedy*.

Con *ε-greedy*, la política siempre tiene una componente aleatoria fija — epsilon — que no puede desaparecer. La política **nunca** puede volverse completamente determinista. Y en particular, nunca puede asignar una probabilidad intermedia específica, como `p = 0.59`, a una acción.

Con softmax sobre preferencias, las preferencias son **libres de crecer sin límite**. Si la política óptima es determinista, las preferencias de las mejores acciones simplemente crecen hacia infinito, y las probabilidades convergen a cero y uno. Si la política óptima es estocástica, las preferencias convergen a los valores exactos que producen las proporciones correctas.

---

### [SLIDE: ¿Por qué no softmax sobre Q?]

Alguien podría preguntar: ¿por qué no simplemente tomar softmax sobre los valores `Q` aprendidos? Eso también daría una política suave.

El problema es **sutil pero fundamental**. Los valores `Q(s,a)` convergen a números específicos — por ejemplo, `Q(s, a₁) = 10` y `Q(s, a₂) = 9`. Esos son los valores verdaderos y no van a cambiar. El softmax sobre esos valores da probabilidades fijas — `0.73` y `0.27` en este ejemplo — y **ahí se queda**. Nunca llegan a cero ni a uno.

Con preferencias `h(s,a,θ)`, en cambio, no hay un ancla. El gradiente puede empujar `h(s, a*)` hacia `+∞` indefinidamente si esa es la acción óptima. La política converge a lo que el problema **requiere** — sea determinista o estocástica.

---

### [SLIDE: Ejemplo 13.1 — El corredor corto]

Este ejemplo es el **corazón didáctico del capítulo**. Es simple, tiene solución analítica, y demuestra con claridad por qué los métodos de policy gradient son necesarios.

El entorno tiene cuatro estados en línea: S, s₂, s₃ y la meta G. En cada estado hay dos acciones: right y left. Las acciones funcionan como esperan en S y en s₃. Pero en s₂ están **invertidas** — ir a la derecha te lleva de vuelta a S, e ir a la izquierda te avanza hacia s₃.

La recompensa es `-1` por paso. El agente quiere llegar a G lo antes posible.

El problema difícil está en la **representación**. Todos los estados tienen exactamente la misma representación de características:

```
x(s, right) = [1,0]ᵀ   para todo s
x(s, left)  = [0,1]ᵀ   para todo s
```

El agente **no puede distinguir** entre S, s₂ y s₃. Tiene que aplicar la misma política en todos.

*[Señalar el gráfico.]*

Esto es lo que produce esa curva. El eje horizontal es `p`, la probabilidad de elegir right. El eje vertical es `J(p)`, el valor del estado inicial bajo esa política. La **zona rosa** es inaccesible para *ε-greedy* — con `ε = 0.1`, solo puede estar en `p ≈ 0.05` o `p ≈ 0.95`.

El mejor resultado de *ε-greedy* es `J ≈ -44`. El óptimo estocástico en `p* ≈ 0.59` logra `J ≈ -11.7`. Eso es casi **cuatro veces mejor**. Y *ε-greedy* no puede llegar ahí, estructuralmente.

---

### [SLIDE: Ejercicio 13.1 — ¿Por qué p* ≈ 0.59?]

Sutton y Barto dejan esto como ejercicio, pero creo que vale la pena hacer la derivación porque revela por qué el óptimo está **exactamente donde está**.

El setup: como todos los estados se ven idénticos, el agente aplica la misma probabilidad `p` de ir a la derecha en todos. Planteamos el sistema de ecuaciones de Bellman para cada estado:

```
v_S   = -1 + p·v_{s₂} + (1-p)·v_S
v_{s₂} = -1 + p·v_S   + (1-p)·v_{s₃}
v_{s₃} = -1 + (1-p)·v_{s₂}
```

*[Pausar un momento aquí, dejar que lo lean.]*

Noten que en `v_{s₂}`, ir a la derecha con probabilidad `p` te lleva **de vuelta a S** — no a s₃. Ahí está la inversión.

Resolviendo este sistema con `γ = 1` y `v_G = 0`, se obtiene una expresión cerrada:

```
J(p) = v_S = -1/(1-p) - 2/p
```

Esta es **exactamente** la función que vimos en la gráfica — cóncava, con un máximo en el interior.

Para encontrar el máximo, derivamos e igualamos a cero:

```
dJ/dp = -1/(1-p)² + 2/p² = 0
⟹ p² = 2(1-p)²
⟹ p = √2 / (1 + √2) ≈ 0.586
```

El óptimo no está en `p = 1` porque en s₂ ir siempre a la derecha es contraproducente — te manda de vuelta al inicio. Hay un **tradeoff** entre avanzar en s₃ y no retroceder en s₂. El valor `p* ≈ 0.59` es el equilibrio exacto de ese tradeoff.

Y la implicación para Policy Gradient es directa: REINFORCE puede encontrar este punto porque `θ` varía **continuamente** y el gradiente `dJ/dp` apunta exactamente hacia `p*`. *ε-greedy* no puede porque sus únicos puntos evaluables están en los extremos.

---

### [SLIDE: La ecuación de Bellman — fundamento de la prueba]

Antes del teorema, un recordatorio de la herramienta que lo hace posible.

La ecuación de Bellman dice que el valor de un estado es la recompensa inmediata esperada más el valor descontado del estado siguiente, promediado sobre todas las acciones y transiciones:

```
v_π(s) = Σ_a π(a|s) · Σ_{s',r} p(s',r|s,a) · [r + γ·v_π(s')]
```

Y la relación clave entre `v_π` y `q_π`: el valor de un estado es el promedio ponderado por la política de los valores de las acciones disponibles.

Menciono esto porque la prueba del Policy Gradient Theorem empieza **exactamente** tomando el gradiente de `v_π(s) = Σ_a π(a|s) q_π(s,a)`, y luego aplica recursión de Bellman sobre `∇q_π` para ir desenrollando la dependencia en estados futuros. La ecuación de Bellman es literalmente el motor de la demostración.

---

### [SLIDE: El Teorema del Gradiente de la Política]

Este es el **resultado teórico central del capítulo**. El problema que resuelve es el siguiente.

`J(θ) = v_π(s₀)` depende de cómo la política distribuye el tiempo entre los estados — es decir, de `μ(s)`. Pero `μ(s)` depende de `θ` a través de las transiciones del entorno, que son **desconocidas**. Entonces calcular `∇J` parecería requerir `∇μ(s)`, que involucra la dinámica `p(s'|s,a)` — algo que en RL *model-free* no tenemos.

El teorema dice que ese problema **no existe**:

```
∇J(θ) ∝ Σ_s μ(s) · Σ_a q_π(s,a) · ∇_θ π(a|s,θ)
```

*[Señalar la anatomía del lado derecho.]*

Cada término tiene una lectura directa.

`μ(s)` es la fracción del tiempo que el agente pasa en el estado `s` — que podemos estimar simplemente **siguiendo la política**.

`q_π(s,a)` es cuánto vale tomar la acción `a` desde `s` — que podemos aproximar con los **retornos observados**.

`∇_θ π` es la dirección en el espacio de parámetros que más aumenta la probabilidad de esa acción — que podemos calcular **analíticamente** si conocemos la parametrización.

Lo no obvio — y lo que hace el teorema poderoso — es que `∇μ(s)` **desapareció completamente** de la expresión. Podemos hacer ascenso de gradiente sobre el desempeño **sin conocer la dinámica del entorno**.

---

### [SLIDE: Estructura de la prueba]

La prueba tiene cuatro pasos limpios. No voy a hacerla completa, pero sí quiero mostrar la estructura porque es elegante.

El **paso uno** es aplicar la regla del producto al gradiente de `v_π(s)`, separando la contribución de `∇π` y la de `∇q_π`.

El **paso dos** es expandir `∇q_π` usando Bellman — el gradiente del valor acción se expresa en términos del gradiente del valor de estado siguiente.

El **paso tres** es la clave: sustituir esa expansión de vuelta en la expresión del paso uno, y **repetir el proceso**. Es un desenrollado recursivo, exactamente igual a como Bellman acumula recompensas futuras — pero aquí acumula **gradientes futuros**.

Y en el **paso cuatro**, después de ese desenrollado, lo que emerge como coeficiente de cada estado es precisamente `μ(s)` — la probabilidad de visitar ese estado bajo la política `π`. Y `∇μ` **nunca aparece**.

La intuición es bonita: es la misma idea recursiva que la ecuación de Bellman, pero aplicada al gradiente en lugar del valor.

---

> **📍 Nota de ritmo — Bloque 2:** Aquí van aproximadamente **22–25 minutos** acumulados. Buen punto para una pausa de dos minutos o para preguntar dudas sobre el teorema antes de entrar a REINFORCE.

---

## BLOQUE 3 — REINFORCE, All-Actions, Baseline y Actor–Crítico
### Slides: Derivación → Eligibility → All-actions → Pseudocódigo → Baseline → Actor–Crítico → Variantes → Caso continuo
**⏱ Tiempo estimado: 13–15 minutos**

---

### [SLIDE: REINFORCE — derivación paso a paso]

Tenemos el teorema. Ahora necesitamos convertirlo en un algoritmo que podamos ejecutar. La derivación de REINFORCE tiene **cuatro pasos**, y cada uno es una manipulación justificada.

Partimos del teorema:

```
∇J(θ) ∝ Σ_s μ(s) · Σ_a q_π(s,a) · ∇π(a|s,θ)
```

**Paso 1.** La suma sobre `s` ponderada por `μ(s)` es exactamente la definición de una **esperanza bajo la política π**. Si seguimos la política, visitamos los estados con frecuencia proporcional a `μ(s)`. Entonces podemos escribir:

```
= E_π [ Σ_a q_π(S_t, a) · ∇π(a|S_t,θ) ]
```

**Paso 2.** Aquí está el truco clave — el mismo que vimos en el Bandit. Necesitamos convertir la suma sobre `a` en una esperanza sobre `A_t`, para poder **muestrearla**. Para eso multiplicamos y dividimos por `π(a|S_t, θ)`:

```
= E_π [ q_π(S_t, A_t) · ∇π(A_t|S_t,θ) / π(A_t|S_t,θ) ]
```

Ahora `A_t` es una muestra de `π` — podemos obtenerla simplemente **siguiendo la política**.

**Paso 3.** Aplicamos la identidad del log-gradiente: `∇f/f = ∇ ln f`. Esto nos da el *eligibility vector*:

```
= E_π [ q_π(S_t, A_t) · ∇ ln π(A_t|S_t,θ) ]
```

**Paso 4.** El último paso es reemplazar `q_π(S_t, A_t)` por la muestra `G_t`. Esto es válido porque `E[G_t | S_t, A_t] = q_π(S_t, A_t)` por definición. `G_t` es un estimador **sin sesgo** de `q_π`.

Y llegamos a la regla de actualización:

```
θ_{t+1} = θ_t + α · G_t · ∇ ln π(A_t|S_t,θ_t)
```

La lectura intuitiva componente a componente: cada parámetro `θ_i` se mueve en la dirección que más aumenta la probabilidad de la acción tomada, escalado por **qué tan bueno fue el futuro** desde ese punto. Si el retorno fue alto, reforzamos mucho. Si fue bajo, reforzamos poco o incluso debilitamos.

---

### [SLIDE: ¿Por qué aparece el logaritmo?]

Vale la pena detenerse un momento en el *eligibility vector* porque es el **único lugar donde aparece la parametrización específica** de la política.

La identidad `∇ ln π = ∇π/π` es cálculo básico. Pero su consecuencia es importante: el algoritmo REINFORCE tiene una estructura **modular**. El *eligibility vector* `∇ ln π` encapsula todo lo que depende de cómo parametrizamos la política — si es softmax lineal, red neuronal, o distribución gaussiana. El resto del algoritmo no cambia.

Para el softmax lineal con `h(s,a,θ) = θᵀ x(s,a)`, el *eligibility vector* tiene una forma particularmente interpretable:

```
∇ ln π(a|s,θ) = x(s,a) - Σ_b π(b|s,θ) · x(s,b)
```

Es el feature de la acción tomada **menos** el promedio ponderado de features bajo la política actual. Es decir: ¿en qué se distingue la acción `a` del comportamiento **promedio** del agente? Esa es la dirección en que empujamos los parámetros.

---

### [SLIDE: Antes de REINFORCE — el método all-actions]

Antes de pasar al pseudocódigo, quiero mencionar brevemente una alternativa que Sutton y Barto presentan en la ecuación 13.7, porque aparece en la literatura reciente con nombre propio.

Del teorema también podemos construir un estimador que en lugar de muestrear una sola acción `A_t`, usa **directamente todas las acciones disponibles**:

```
θ_{t+1} = θ_t + α · Σ_a q̂(S_t, a, w) · ∇_θ π(a|S_t, θ)
```

La ventaja es clara: al sumar sobre todas las acciones en lugar de muestrear una, la **varianza baja**. No hay ruido de muestreo.

Pero el costo es que necesitamos evaluar todas las acciones en cada paso — `O(|A|)` por paso — y necesitamos un `q̂` aprendido. En espacios de acción grandes o continuos, esto es **inviable**.

REINFORCE resuelve ese problema introduciendo `A_t` como muestra de la esperanza — un estimador con más varianza pero con costo `O(1)` por paso.

En la literatura reciente este método ha resurgido como *expected policy gradients* — Ciosek y Whiteson, 2018 — para casos donde el espacio de acciones sí es manejable.

---

### [SLIDE: REINFORCE — pseudocódigo y convergencia]

El pseudocódigo es directo. Para cada episodio, generamos la trayectoria completa siguiendo la política `π(·|·,θ)`. Luego, al final del episodio, recorremos calculando los retornos `G_t` y actualizando `θ` con la regla que derivamos.

Las propiedades son claras:

- ✅ **Sin sesgo** — la esperanza del update es proporcional al gradiente real
- ✅ **Converge a un óptimo local** con step size apropiado — no global
- ✗ Requiere el **episodio completo** para calcular `G_t` — no es online
- ✗ `G_t` es ruidoso — **alta varianza** — lo que hace el aprendizaje lento

Y aquí hay algo que quiero remarcar especialmente: la **sensibilidad al step size** `α` en REINFORCE es crítica. En el corredor corto, `α = 2⁻¹²` diverge, `α = 2⁻¹³` converge bien, y `α = 2⁻¹⁴` converge pero muy lento. Un **factor dos** en `α` — literalmente el doble o la mitad — cambia completamente el resultado.

Esta fragilidad no es un detalle técnico menor. Es la **motivación directa de TRPO y PPO**. En lugar de restringir el tamaño del paso en el espacio de parámetros `θ`, esos algoritmos restringen el tamaño del paso en el **espacio de políticas** — controlando cuánto cambia la distribución `π(·|s)`, no cuánto cambia `θ`. Eso da estabilidad independiente de la parametrización.

---

### [SLIDE: REINFORCE con Baseline]

REINFORCE funciona, pero aprende lento por la alta varianza de `G_t`. La solución es agregar una *baseline*.

La generalización del teorema permite restar cualquier función `b(s)` que **no dependa de la acción**:

```
∇J(θ) ∝ Σ_s μ(s) · Σ_a (q_π(s,a) - b(s)) · ∇π(a|s,θ)
```

Y ya demostramos antes que esa resta no cambia el gradiente esperado. La elección más natural para `b(s)` es `v̂(S_t, w)` — la estimación del valor del estado actual.

La regla de actualización queda:

```
θ_{t+1} = θ_t + α · (G_t - v̂(S_t, w)) · ∇ ln π(A_t|S_t,θ_t)
                      ─────────────────
                              δ_t
```

La interpretación de `δ_t` es muy intuitiva:

- Si `G_t > v̂(S_t)` → el futuro fue **mejor** de lo esperado → reforzar la acción
- Si `G_t < v̂(S_t)` → el futuro fue **peor** de lo esperado → debilitar la acción

El agente aprende en términos **relativos** a sus expectativas, no en términos absolutos.

*[Señalar el callout experimental.]*

Y la evidencia experimental lo confirma. Si miran la trayectoria de `θ` en el paisaje de `J(p)` para el corredor corto, tanto REINFORCE como REINFORCE con baseline convergen exactamente al **mismo punto** — `p* ≈ 0.59`. El baseline no desplaza hacia dónde converge el algoritmo. Solo **acelera el camino**. Eso es la confirmación experimental directa de la no-introducción de sesgo.

---

### [SLIDE: Actor–Crítico — la distinción conceptual]

REINFORCE con baseline está muy cerca de Actor–Crítico, pero hay una **diferencia conceptual importante** que quiero destacar.

En REINFORCE con baseline, la función de valor `v̂(S_t, w)` estima el valor del estado **antes** de que se tome la acción. Actúa como punto de referencia para el retorno `G_t`, pero no evalúa la acción en sí.

En Actor–Crítico, el critic evalúa **la acción que se acaba de tomar** mediante bootstrapping — usando el valor estimado del estado siguiente:

```
δ_t = R_{t+1} + γ · v̂(S_{t+1}, w) - v̂(S_t, w)
```

Esto es exactamente el **error TD** que conocemos de los capítulos anteriores. La diferencia con REINFORCE es que en lugar de esperar hasta el final del episodio para calcular `G_t`, usamos una **estimación de un solo paso**.

La tabla resume el tradeoff:

| | REINFORCE + baseline | Actor–Crítico |
|---|---|---|
| Baseline | antes de la transición | después (bootstrapping) |
| Retorno | `G_t` (MC) | `G_{t:t+1}` (TD) |
| Sesgo | sin sesgo | con sesgo |
| Varianza | alta | baja |
| Online | no (episodio completo) | **sí** (cada paso) |

La analogía con lo que ya saben es directa y vale la pena repetirla: **MC versus TD en funciones de valor es exactamente REINFORCE versus Actor–Crítico en políticas**. El mismo tradeoff sesgo-varianza, ahora aplicado al aprendizaje de la política.

---

### [SLIDE: Las tres variantes Actor–Crítico]

Actor–Crítico no es un solo algoritmo sino una **familia**. Las tres variantes principales solo difieren en qué usan como señal `δ`.

El **one-step** usa el error TD de un solo paso. Es el más simple — completamente online, sin trazas de elegibilidad, el análogo exacto de TD(0) en el espacio de políticas. Es el punto de entrada natural.

La variante **con trazas** usa el `λ`-return, que interpola entre one-step y Monte Carlo. Da más flexibilidad para controlar el tradeoff sesgo-varianza según el problema.

Y la variante **continua** — para problemas sin episodios — reemplaza el descuento por la desviación respecto a la tasa promedio de recompensa `r̄`. El retorno diferencial `G_t = Σ_k (R_{t+k} - r(π))` acumula cuánto se desvían las recompensas del promedio.

Lo importante es que la **estructura es idéntica** en los tres casos. Actor y critic. Error `δ`. Actualización del critic con gradiente de `v̂`. Actualización del actor con el *eligibility vector* escalado por `δ`. Solo cambia **de qué está hecha** esa señal.

---

### [SLIDE: Caso continuo vs episódico]

Una nota rápida sobre el caso continuo porque cambia sutilmente cómo se define el desempeño.

En el caso episódico, `J(θ) = v_{π_θ}(s₀)` — el valor del estado inicial. Tiene sentido porque hay un comienzo claro.

En el caso continuo, no hay episodios ni estado inicial especial. El desempeño es la **tasa promedio de recompensa** por paso de tiempo:

```
J(θ) = r(π) = lim_{t→∞} E[R_t | A_{0:t} ~ π]
```

Y el retorno diferencial acumula desviaciones sobre esa tasa promedio — recompensas por encima del promedio son positivas, por debajo son negativas. Son esas **desviaciones** las que guían el aprendizaje.

El resultado notable es que el Policy Gradient Theorem tiene la **misma forma** en ambos casos. Todo lo que derivamos para el caso episódico se transfiere directamente.

---

> **📍 Nota de ritmo — Bloque 3:** Aquí van aproximadamente **37–40 minutos** acumulados. Quedan 5–8 minutos para el cierre.

---

## BLOQUE 4 — Cierre
### Slides: Acciones continuas → Tabla analogías → Mapa → Comparación → SOTA → Tres resultados → Final
**⏱ Tiempo estimado: 8–10 minutos**

---

### [SLIDE: Políticas para acciones continuas]

Antes de ir al cierre, una extensión importante del capítulo que abre la puerta al Deep RL moderno: ¿qué hacemos cuando el espacio de acciones es **continuo**?

Con acciones discretas podíamos asignar una probabilidad a cada acción. Pero si las acciones son números reales — un torque, una velocidad, un ángulo — no podemos listar todas las opciones.

La solución es aprender los **estadísticos de una distribución**, no las probabilidades directamente. La elección más natural es una gaussiana parametrizada:

```
π(a|s,θ) = 1/(σ(s,θ)√2π) · exp(-(a - μ(s,θ))² / 2σ(s,θ)²)
```

Lo que el agente aprende son:
- `μ(s,θ)` — la **acción más probable** en el estado `s`
- `σ(s,θ)` — cuánta **exploración** hace alrededor de esa media

Noten que `σ` se parametriza como exponencial de una combinación lineal. Eso garantiza que siempre sea positiva **sin restricciones** en la optimización.

Y el *eligibility vector* gaussiano tiene dos partes: una para los parámetros de la media, y otra para los de la desviación. Si la acción tomada estuvo por encima de la media, el gradiente empuja la media hacia arriba. Si la varianza fue inadecuada respecto a lo que el retorno justifica, se ajusta.

---

### [SLIDE: Tabla de analogías Bandit → Policy Gradient]

Antes del cierre quiero hacer una pausa para consolidar todo lo que hemos visto con esta tabla. Es la que mejor muestra que el capítulo trece **no introduce ideas fundamentalmente nuevas** — sino que generaliza sistemáticamente lo que ya conocíamos del Bandit.

Fila por fila: el parámetro pasa de un escalar `H_t(a)` a un vector `θ ∈ ℝᵈ`. La política pasa de no depender del estado a estar condicionada en `s`. El desempeño pasa de una esperanza simple a depender de la distribución `μ(s)`. La señal pasa de `R_t` inmediato a `G_t` acumulado. La baseline pasa de un promedio escalar a una función del estado. Y el *eligibility* pasa de un escalar a un vector.

La estructura de la actualización es **idéntica** en todos los casos.

*[Clic para el fragment.]*

Y el único ingrediente genuinamente nuevo — el que no estaba en el Bandit — es el **Policy Gradient Theorem**. Que resuelve el único problema genuinamente nuevo: cómo calcular el gradiente de desempeño cuando la distribución de estados depende de `θ` a través de la dinámica desconocida del entorno.

---

### [SLIDE: Mapa del capítulo]

Este diagrama resume la **arquitectura completa** del capítulo. Todo parte del Policy Gradient Theorem. A partir de ahí se desarrolla la familia REINFORCE: primero sin baseline, luego con baseline para reducir varianza. Cuando añadimos bootstrapping al baseline, cruzamos a Actor–Crítico — en sus tres variantes. Y en paralelo, la extensión a acciones continuas con la gaussiana.

El **hilo conductor** es siempre el mismo: ascenso de gradiente en `J(θ)`. Lo que varía es el estimador de la señal de gradiente. Y el eje de esa variación es el tradeoff **sesgo-varianza** que ya conocen de MC versus TD, ahora aplicado al aprendizaje de la política.

---

### [SLIDE: Comparación — Value-based vs Policy Gradient]

La tabla de comparación final. Cada fila es un eje de decisión cuando tienen que elegir entre métodos.

La política en value-based es **implícita** — emerge de los valores. En Policy Gradient es **explícita** — es el objeto central de aprendizaje.

El óptimo estocástico es problemático para *ε-greedy* — lo vimos en el corredor. Es natural para Policy Gradient.

Los espacios continuos son difíciles para value-based porque requieren `max_a Q(s,a)` — inviable con infinitas acciones. Para Policy Gradient son directos con la parametrización gaussiana.

*[Clic para el fragment y el callout.]*

Y sobre convergencia — quiero ser **preciso** aquí. Policy Gradient tiene garantías más sólidas que los métodos value-based en términos de continuidad: pequeños cambios en `θ` producen pequeños cambios en `π`. Pero la garantía es a un **óptimo local**, no global. En problemas con múltiples óptimos locales — y muchos problemas reales los tienen — esto no es trivial. Es la motivación de los métodos de gradiente natural como TRPO, que usan la geometría del espacio de políticas para dar pasos más inteligentes.

---

### [SLIDE: Conexión con el estado del arte]

¿Por qué importa todo esto más allá del capítulo trece? Porque el *eligibility vector* `∇ ln π` está literalmente en el corazón de los algoritmos que **dominan el Deep RL y el entrenamiento de LLMs** hoy.

**A3C y A2C** de DeepMind en 2016 son Actor–Crítico one-step con múltiples agentes paralelos.

**TRPO** de 2015 es Policy Gradient con restricción en el espacio de políticas usando divergencia KL — la respuesta directa al problema de sensibilidad al step size que mencionamos.

**PPO** de 2017 es la versión simplificada de TRPO que **dominó la década**.

**SAC** de 2018 extiende Actor–Crítico continuo agregando la entropía de la política como baseline — lo que maximiza tanto el retorno como la exploración.

Y **RLHF** — el algoritmo detrás de ChatGPT, de Claude, de prácticamente todos los LLMs alineados — es Policy Gradient donde la política es un modelo de lenguaje de miles de millones de parámetros y el critic es un *reward model* entrenado con preferencias humanas.

*[Clic para el callout de PPO.]*

Si miran la actualización de PPO, es exactamente REINFORCE con baseline — el `δ_t` estimado por el critic — con una sola restricción adicional: el ratio entre la política nueva y la vieja se **recorta** para evitar pasos demasiado grandes. Todo lo que derivamos hoy está ahí.

---

### [SLIDE: Tres resultados para llevarse]

Para cerrar, tres resultados concretos que quiero que se lleven de esta presentación.

**Primero, el Policy Gradient Theorem.** Es el resultado teórico que hace posible todo lo demás. Resuelve el problema de calcular `∇J` sin conocer la dinámica del entorno. Sin él, el ascenso de gradiente sobre la política no sería viable.

```
∇J ∝ Σ_s μ(s) · Σ_a q_π · ∇π
```

**Segundo, REINFORCE.** Es el método canónico — simple, sin sesgo, con garantías de convergencia a un óptimo local. La regla de actualización es tres objetos multiplicados:

```
θ ← θ + α · G_t · ∇ ln π
```

Retorno, *eligibility vector*, step size. Nada más. Es la **base de todos los algoritmos modernos** de policy gradient.

**Tercero, Actor–Crítico.** Cierra la brecha entre Monte Carlo y TD aplicada al aprendizaje de políticas. El error TD:

```
δ_t = R + γ·v̂(S') - v̂(S)
```

reemplaza a `G_t` con un estimador de menor varianza al costo de introducir sesgo. Es el **framework dominante** en Deep RL — A3C, PPO, SAC son todos variantes de esta idea.

*[Clic para el callout final.]*

Y la pregunta que los dejo para reflexionar, especialmente si están pensando en sus proyectos: ¿en su problema tiene más sentido aprender `π` directamente que aprender `Q` primero? ¿El óptimo podría ser estocástico? ¿El espacio de acciones es continuo? Si la respuesta a alguna de esas preguntas es sí, Policy Gradient probablemente es el camino.

---

### [SLIDE FINAL — ¿Preguntas?]

*[Pausa larga. Mirar a la audiencia con calma.]*

Eso es todo. Cubrimos el Policy Gradient Theorem, la derivación de REINFORCE, la motivación y el cálculo del baseline, Actor–Crítico en sus variantes, y la conexión con los algoritmos que están transformando el campo hoy.

Quedo disponible para cualquier pregunta — sobre la derivación del teorema, sobre el ejemplo del corredor, o sobre cómo alguno de estos conceptos conecta con los temas que están trabajando ustedes.

---

## Resumen de tiempos

| Bloque | Contenido | Tiempo aprox. |
|--------|-----------|---------------|
| 1 | Apertura, motivación, Bandit, baseline | 10–12 min |
| 2 | Definiciones, Ejemplo 13.1, p*, Teorema | 13–15 min |
| 3 | REINFORCE, all-actions, baseline, Actor–Crítico | 13–15 min |
| 4 | Acciones continuas, cierre, SOTA | 8–10 min |
| **Total** | | **44–52 min** |

---

*Francisco Alfaro · Ingeniero Matemático · UTFSM · Seminario Avanzado TIC · 2026*
