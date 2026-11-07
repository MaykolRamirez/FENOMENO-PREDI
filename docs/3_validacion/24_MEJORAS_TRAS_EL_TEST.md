# Mejoras tras el test de realidad

El [test de realidad](23_TEST_DE_REALIDAD.md) mostró que algunas variables usaban información que no se tiene a
tiempo. Este documento cuenta qué se cambió, qué se probó y se descartó, y qué se dejó igual.

**Regla:** solo se adopta un cambio si quita una fuga de información o si mejora el resultado en la validación
realista. Cada candidato se probó **una sola vez**, con la regla de decisión fijada antes de ver el resultado. Si
empeoraba, se descartaba.

## 1. Cambios adoptados en el modelo por plazo (`src/prediccion_semanal.py`)

| Cambio | Por qué |
|---|---|
| Se quitó la variable «¿hubo emergencia en la semana t?» | El SINPAD publica cada seis meses: en tiempo real no existe. Queda solo como regla de referencia. |
| La fila de la semana t lleva la lluvia, el caudal y el Niño 1+2 de la semana t−1 | Se midió el retraso real: el 6/11/2026 la lluvia de NASA POWER estaba completa solo hasta el 31/10, unos 6 días de retraso. |
| Umbrales p95/p99 de lluvia, p95 de caudal y mediana del caudal calculados solo con los años anteriores al año de prueba | Antes usaban 2001–2020, es decir, años futuros al predecir 2008–2019. |
| SINPAD unido por distrito **y provincia**, con clave única para los dos Salitral | Antes las emergencias de Salitral (Morropón) y Salitral (Sullana) se sumaban y sus semanas se mezclaban. |
| El aviso muestra la semana de los datos (`semana_datos`) además de la semana base | Así queda claro de qué semana son los datos que usa el modelo. |
| Aviso automático cuando una variable de hoy supera el máximo del entrenamiento | Ver la sección 3. |

### Resultado después de los cambios (validación por años 2008–2025, PR-AUC)

| Plazo | Antes (con fugas) | **Ahora (realista)** | Climatología | Años que gana | BSS del modelo | BSS de la climatología |
|---|---|---|---|---|---|---|
| 1 semana | 0,227 | **0,198** | 0,161 | 12 de 18 | 0,088 | 0,064 |
| 2 semanas | 0,207 | **0,174** | 0,161 | 10 de 18 | 0,072 | 0,064 |
| 3 semanas | 0,191 | **0,160** | 0,161 | 9 de 18 | 0,064 | 0,064 |
| 4 semanas | 0,176 | **0,163** | 0,161 | 7 de 18 | 0,065 | 0,064 |

Las cifras bajan porque ya no hay fugas: **ahora son creíbles**. Para saber si la diferencia con la climatología es
real, se hizo un bootstrap por años completos (1 000 réplicas):

- **A 1 semana:** +0,033, IC 95 % [+0,007; +0,058]. La mejora es real, con un 99,7 % de las réplicas a favor del
  modelo.
- **A 2 semanas:** +0,011, IC 95 % [−0,011; +0,031]. No se distingue de la climatología.
- **A 3 y 4 semanas:** empata.

## 2. Probado y descartado

| Candidato | Resultado | Decisión |
|---|---|---|
| Modelo por plazo «anclado» a la climatología: la logística recibe el logit de la climatología del distrito y el mes | PR-AUC 0,189 a 1 semana, frente a 0,198 sin anclar; también peor a 2–4 semanas | **Descartado** (empeora) |
| Modelo de 4 semanas (el mapa de alerta) anclado a la climatología, escenario realista | PR-AUC 0,358, frente a 0,355 sin anclar y 0,395 de la climatología; gana 5 de 18 años | **Descartado** (no supera a la climatología) |

LightGBM con las mismas variables saca 0,204 a 1 semana. Es un poco mejor que la logística, pero **no se cambió el
modelo del tablero**: elegirlo ahora, después de ver el resultado, sería ajustar el modelo a la validación. La
logística es más simple de explicar y la diferencia es pequeña.

## 3. Hallazgo: El Niño de 2026 está fuera de lo que el modelo conoce

El Niño 1+2 está hoy en **+5,4 °C**. ENFEN emitió el 30/10/2026 una alerta de El Niño Costero de magnitud
"extraordinaria". **El máximo del periodo de entrenamiento (2003–2025) es +3,5 °C.** La logística extrapola en línea
recta, así que las probabilidades de ahora pueden estar sobre o subestimadas.

Ejemplo: Tumbes tiene 37 % de probabilidad a 1 semana para la semana del 9/11/2026, cuando la climatología de
noviembre es 0,2 %. **No se recortó el valor de El Niño**, porque tampoco hay forma de validar esa decisión. En cambio:

- `prediccion_semanal.json` trae la lista `fuera_de_rango`;
- el tablero muestra el aviso de que esas probabilidades son una extrapolación no validada.

Pendiente: el mapa de alerta a 4 semanas (`riesgo.json`) también usa el Niño 1+2 y todavía no muestra este aviso.

## 4. Lo que se dejó igual

- **El modelo de alerta a 4 semanas** (`src/modelo_inundacion.py`, el mapa del tablero). Ningún cambio probado lo hizo
  superar a la climatología, así que no se tocó. Su límite está documentado en el informe 23.
- **La calibración del modelo a 4 semanas.** El tablero lo muestra como índice de 0 a 100 (percentil), no como
  probabilidad, así que la sobreestimación no afecta lo que se ve.

## 5. Prueba prospectiva 2027 (ya en marcha)

`src/registro_prospectivo.py` corre cada lunes en el flujo "Base para el modelo". Agrega a
`data/prospectivo/predicciones.csv`, por cada distrito y plazo:

- la probabilidad del modelo y la climatología del mismo momento;
- el índice de 4 semanas;
- la fecha de registro y la **huella SHA-256 del código del modelo**.

Como queda en git, nadie puede cambiar las predicciones después. El 6/11/2026 se registraron las primeras 312: 78
distritos por 4 plazos.

Para que la prueba valga:

1. **No cambiar `src/prediccion_semanal.py` ni `src/modelo_inundacion.py` de enero a abril de 2027.** Si cambian, la
   huella cambia y la evaluación lo separa.
2. Cuando el SINPAD publique el primer semestre de 2027, correr:

```bash
python src/registro_prospectivo.py --evaluar
```

El script compara el modelo con la climatología, solo en semanas que todavía no habían ocurrido cuando se registró
la predicción.
