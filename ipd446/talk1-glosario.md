## Glosario de nomenclatura — Capítulo 13

**Agente, entorno y tiempo**

$t$ es el índice de tiempo discreto. En cada paso el agente observa un estado $S_t$, elige una acción $A_t$, recibe una recompensa $R_{t+1}$ y llega al estado $S_{t+1}$.

$T$ es el paso final de un episodio. Un episodio es una secuencia que termina en un estado terminal (como llegar a la meta en el corredor corto).

---

**Estados, acciones y política**

$s \in \mathcal{S}$ es un estado del entorno. $\mathcal{S}$ es el conjunto de todos los estados posibles.

$a \in \mathcal{A}(s)$ es una acción disponible en el estado $s$. $\mathcal{A}$ es el conjunto de todas las acciones posibles.

$\pi(a \mid s)$ es la **política**: la probabilidad de elegir la acción $a$ estando en el estado $s$. Una política estocástica asigna probabilidades a las acciones; una política determinista asigna probabilidad 1 a una sola acción.

$\pi_\theta$ o $\pi(a \mid s, \theta)$ es una **política parametrizada**: la política está definida por un vector de parámetros $\theta$ que se puede optimizar.

$\pi^*$ es la **política óptima**: aquella que maximiza el retorno esperado desde cualquier estado.

---

**Recompensas y retornos**

$R_t$ es la **recompensa** recibida en el paso $t$. Es un escalar que indica qué tan buena fue la transición inmediata.

$G_t$ es el **retorno** desde el paso $t$: la suma descontada de todas las recompensas futuras a partir de $t$:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
Es la cantidad que el agente quiere maximizar en expectativa. En el caso episódico con $\gamma = 1$, es simplemente la suma de todas las recompensas hasta el final del episodio.

$\gamma \in [0,1]$ es el **factor de descuento**. Con $\gamma = 0$ el agente solo le importa la recompensa inmediata; con $\gamma = 1$ considera todo el futuro con igual peso. En el Cap. 13 se usa $\gamma = 1$ para simplificar.

$r(\pi)$ es la **tasa promedio de recompensa** en el caso continuo (sin episodios):
$$r(\pi) = \lim_{t \to \infty} \mathbb{E}[R_t \mid A_{0:t} \sim \pi]$$
Reemplaza a $J(\theta) = v_\pi(s_0)$ cuando no hay estados terminales.

---

**Funciones de valor**

$v_\pi(s)$ es la **función de valor de estado** bajo la política $\pi$: el retorno esperado al estar en el estado $s$ y seguir $\pi$ desde ahí:
$$v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$$

$v^*(s)$ es el **valor óptimo de estado**: el mayor retorno esperado posible desde $s$ bajo cualquier política. $v^*(s) = \max_\pi v_\pi(s)$.

$q_\pi(s, a)$ es la **función de valor acción** (*action-value function*) bajo la política $\pi$: el retorno esperado al estar en $s$, tomar la acción $a$, y luego seguir $\pi$:
$$q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$$

$q^*(s,a)$ es la **función de valor acción óptima**: $q^*(s,a) = \max_\pi q_\pi(s,a)$. También se escribe $Q^*$ en la literatura de Deep RL.

$\hat{v}(s, w)$ es la **aproximación aprendida** de $v_\pi(s)$, parametrizada por un vector de pesos $w$. El critic en Actor–Crítico aprende esta función.

---

**Parámetros y gradientes**

$\theta \in \mathbb{R}^{d'}$ es el **vector de parámetros de la política**. Es lo que REINFORCE y Actor–Crítico actualizan. En una red neuronal puede tener millones de componentes.

$w \in \mathbb{R}^d$ es el **vector de pesos de la función de valor** (el critic en Actor–Crítico). Se aprende por separado de $\theta$.

$\nabla_\theta f(\theta)$ es el **gradiente** de $f$ respecto a $\theta$: el vector de derivadas parciales $(\partial f/\partial \theta_1, \ldots, \partial f/\partial \theta_{d'})^\top$. Apunta en la dirección de mayor crecimiento de $f$.

$\alpha$ es el **tamaño de paso** (*step size* o *learning rate*): controla cuánto se mueven los parámetros en cada actualización. Su elección es crítica — un $\alpha$ demasiado grande diverge, demasiado pequeño converge muy lento.

$\alpha_\theta$, $\alpha_w$ son los step sizes para el actor y el critic respectivamente en Actor–Crítico (pueden ser distintos).

---

**Medidas de desempeño**

$J(\theta)$ es la **medida de desempeño** que se quiere maximizar. En el caso episódico es $J(\theta) = v_{\pi_\theta}(s_0)$: el valor del estado inicial bajo la política $\pi_\theta$. En el caso continuo es $J(\theta) = r(\pi)$.

---

**Distribuciones de estados**

$\mu(s)$ es la **distribución on-policy**: la fracción del tiempo que el agente pasa en el estado $s$ siguiendo la política $\pi$. En el caso episódico es proporcional al número de visitas esperadas a $s$ desde $s_0$. Aparece en el Policy Gradient Theorem como el peso de cada estado.

$p(s', r \mid s, a)$ es la **dinámica del entorno**: la probabilidad de llegar al estado $s'$ con recompensa $r$ al tomar la acción $a$ en el estado $s$. En RL model-free esta función es **desconocida** — y el Policy Gradient Theorem es notable precisamente porque $\nabla J$ no la requiere.

---

**Nomenclatura de la política softmax**

$h(s, a, \theta)$ es la **preferencia** de la acción $a$ en el estado $s$: un número real sin unidades que no representa un valor, sino una "puntuación" relativa. Las preferencias pueden crecer arbitrariamente, lo que permite que la política se vuelva determinista.

La **política softmax** convierte preferencias en probabilidades:
$$\pi(a \mid s, \theta) = \frac{e^{h(s,a,\theta)}}{\sum_b e^{h(s,b,\theta)}}$$

$H_t(a)$ es la notación del Bandit (Cap. 2.8) para la preferencia de la acción $a$ en el paso $t$. Es el análogo escalar de $h(s,a,\theta)$ cuando solo hay un estado.

---

**El eligibility vector y el truco del logaritmo**

$\nabla_\theta \ln \pi(a \mid s, \theta)$ es el **eligibility vector** (vector de elegibilidad): la dirección en el espacio de parámetros que más aumenta la probabilidad de elegir la acción $a$ en el estado $s$. Es matemáticamente igual a $\nabla_\theta \pi / \pi$ por la identidad $\nabla \ln x = \nabla x / x$. Es el único lugar donde aparece la parametrización específica de la política en el algoritmo REINFORCE.

---

**Error TD y baseline**

$\delta_t$ es el **error TD** (*temporal difference error*) en Actor–Crítico:
$$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)$$
Mide cuánto mejor o peor fue la transición de lo que el critic esperaba. Si $\delta_t > 0$, la acción fue mejor de lo previsto y se refuerza; si $\delta_t < 0$, se debilita.

$b(s)$ es el **baseline**: cualquier función del estado (no de la acción) que se resta a $G_t$ para reducir la varianza sin introducir sesgo. La elección más natural es $\hat{v}(S_t, w)$.

$\bar{R}_t$ es el baseline del Bandit (Cap. 2.8): el promedio de todas las recompensas recibidas hasta $t$. Es el análogo escalar del baseline $\hat{v}(S_t, w)$.

---

**$\varepsilon$-greedy**

$\varepsilon$-greedy es una **estrategia de exploración** para políticas basadas en valores: con probabilidad $1 - \varepsilon$ elige la acción con mayor valor estimado (greedy), y con probabilidad $\varepsilon$ elige una acción aleatoria uniforme. Su limitación fundamental es que solo puede asignar dos niveles de probabilidad a las acciones — $1 - \varepsilon/|\mathcal{A}|$ a la mejor y $\varepsilon/|\mathcal{A}|$ a las demás — lo que le impide representar políticas óptimas estocásticas en el interior del simplex.

---

**Actor y Critic**

El **actor** es la política $\pi_\theta$: decide qué acciones tomar. Se actualiza usando el eligibility vector escalado por la señal del critic.

El **critic** es la función de valor $\hat{v}(s, w)$: evalúa qué tan buenas son las situaciones en las que el actor se encuentra. Proporciona la señal $\delta_t$ que guía al actor sin necesitar el retorno completo $G_t$.

---

**Trazas de elegibilidad**

$z_\theta$ y $z_w$ son los **vectores de trazas de elegibilidad** para el actor y el critic respectivamente. Acumulan los gradientes de pasos anteriores con decaimiento exponencial controlado por $\lambda$, lo que permite propagar señales de error hacia atrás en el tiempo de forma eficiente. Son la implementación online del $\lambda$-return.