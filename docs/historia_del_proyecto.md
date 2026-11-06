# Historia del proyecto

Registro de cómo evolucionó la propuesta, qué se pidió en cada revisión y qué decidimos.
Sirve para que el asesor entienda el proyecto desde cero.

## Versión 1 · «Radar Macroeconómico Perú» (setiembre de 2026)

- **Equipo:** cinco integrantes.
- **Idea central:** cuantificar cuánto responde cada sector del PBI a shocks climáticos, sociales y externos.
- **Métodos:** proyecciones locales en panel (Jordà), errores de Driscoll–Kraay, corrección de Romano–Wolf,
  nowcasting con redes neuronales (LSTM y TFT), detector de anomalías (Isolation Forest),
  capa de lenguaje con noticias (RAG) y mapa departamental.
- **Archivo:** `propuesta_v1_corregida_por_profesor.pdf` (con las anotaciones del profesor).

### Comentarios recibidos

| Comentario | Qué significaba |
|---|---|
| Centrarse en lo que le compete al INEI: medición, pronóstico y exactitud | Medir cuánto responde un sector a un shock no es tarea del INEI; su tarea es producir un mejor indicador |
| Sacrificar los modelos econométricos; la comparación con la econometría «ya es mucho» | Demasiados frentes para cinco semanas |
| Le gustó El Niño: comparar con 2017 y hacer nowcasting del evento actual | El Niño pasa a ser el eje |
| Usar temperatura del mar si no hay mucho Niño | Variable continua, útil aunque el evento sea débil |
| Las noticias de forma predictiva serían un plus, pero son complejas | Quedan como extensión |
| «Muy difícil de leer», «¿cómo converge con lo anterior?» | Redacción y coherencia |

## Versión 2 · «Indicador Adelantado de la Producción Nacional (IAPN)» (octubre de 2026)

- **Equipo:** cuatro integrantes.
- **Idea central:** estimar la producción nacional y sus nueve sectores unos 30 días antes de la cifra oficial,
  con su error medido, también durante El Niño.
- **Qué se retiró:** proyecciones locales, Driscoll–Kraay, Romano–Wolf, Isolation Forest, RAG, shock social y mapa.
- **Qué se agregó:** temperatura del mar como predictora, índice diario de electricidad, rezagos según la fecha
  de publicación (t−2 o t−3, nunca t−1), calibración conforme del intervalo, primeras cifras publicadas por el INEI.
- **Revisiones externas:** dos revisiones adicionales del documento (rezagos, anticipación real, validación de la
  regla de control, plan B para cada fuente).
- **Archivo:** `propuesta_v2_corregida_por_profesor.pdf` (con las anotaciones del profesor).

### Comentarios recibidos

| Comentario | Respuesta |
|---|---|
| «No hay datos INEI» | La variable objetivo y las primeras cifras salen de los informes del INEI; el Avance Coyuntural del INEI entra como predictora. BCRPData (que combina datos del INEI y de los ministerios) solo se usa tras verificar que coincide con el INEI |
| Redacción telegráfica, mucho texto, título largo | Documento reescrito en prosa, dentro de los límites de página, con subtítulo corto |
| Objetivos confusos (P4, relación con 2017) | Tres objetivos; El Niño de 2017, 2023 y 2026 agrupado en el OE2 |
| Definiciones faltantes (*T*, ganancia, periodo de entrenamiento) | Definidas en el texto y en el glosario, con fórmulas |
| No quedaba claro cómo opera el sistema en el tiempo | Tabla de operación: diaria, días 12, 15 y 30, y trimestral |
| Integrantes sin tareas en algunas semanas | Cronograma con tareas para todos cada semana |

- **Archivo de la versión final:** `propuesta_final.pdf` y `presentacion.pptx`.

## Revisión en clase (noviembre de 2026)

Comentarios del docente:

1. Definiciones más específicas (operativas: cómo se mide cada cosa).
2. Le gustan los gráficos: incluir más.
3. Mostrar toda la historia desde cero al asesor → **este archivo**.
4. Modelar todos los sectores a la vez, como un sistema de ecuaciones (modelos multivariables o conjuntos).
5. Considerar un componente espacial: series por región y temperatura del mar por zonas costeras
   (por ejemplo, hexágonos agregados por región); para sierra y selva, nubosidad o humedad.
6. Prever un proxy o una imputación para que el modelo no falle si una fuente no llega.

Comentario del grupo revisor (I like / I wish / What if): correspondencia entre los 9 sectores del BCRP y los
14 del INEI, señal de confianza por sector en el tablero, gestión de datos faltantes y precipitación satelital
de libre acceso (por ejemplo, CHIRPS).

### Decisiones para la etapa de ejecución

| Tema | Decisión |
|---|---|
| Sistema de ecuaciones | Agregar un modelo de factores dinámicos (los 9 sectores comparten factores comunes) y compararlo con el modelo global |
| Coherencia sectores–total | Reconciliación jerárquica si el tiempo alcanza |
| Datos faltantes | Variable de respaldo definida para cada fuente; el filtro de Kalman del modelo de factores imputa de forma natural |
| Componente espacial | Piloto con la temperatura del mar frente a la costa norte; desagregación regional completa como extensión |
| Precipitación | CHIRPS como fuente de la primera extensión |

## Parte 1 · Ejecución (02/11/2026)

**Hecho:** diez series objetivo verificadas contra el INEI (julio 2026 = 3,56 %), predictoras con código verificado,
temperatura del mar (Niño 1+2) hasta octubre, tres referencias evaluadas con origen móvil desde 2010 para k = 1 y k = 2,
y dos estimaciones de setiembre registradas antes de la cifra oficial (v1 y v2). Detalle: `resultados/README.md`.

### Hallazgos y cambios de procedimiento

| Hallazgo | Cambio |
|---|---|
| La API del BCRP devuelve las series en otro orden y sin códigos si se piden varias juntas | Descarga de una serie por vez, verificando el nombre |
| La electricidad mensual del BCRP sale casi junto con la cifra oficial (el 02/11 llegaba a agosto) | Se usa el COES, que adelanta 2 meses (resuelto el 03/11). El plan B de la propuesta (electricidad del BCRP) se corrige: si falla el COES, la electricidad sale del modelo |
| El AR supera a la persistencia en 8 de 10 series (k = 2); en Comercio y Construcción no | Los modelos de la parte 2 deben superar a la mejor referencia **de cada sector** |
| El Niño 2026 es extremo: Niño 1+2 en +5,3 °C (oct-2026), sobre el máximo de 1997–98 | Se declara como limitación (fuera del rango histórico; LightGBM no extrapola) y se agregan variables de cambio |
| En Pesca, la persistencia gana en los tres episodios de El Niño | Errores reportados por régimen (El Niño / normal) y v2 con selección automática por régimen |
| La selección por régimen no mejora el promedio histórico (solo 34 meses de El Niño) | La temperatura del mar entra como variable continua en los modelos de la parte 2 |

### Parte 1 de R1 (03/11/2026)

- Código del IGV interno verificado en BCRPData: **PN38697FM** (millones de S/ de 2021). Se asigna al mes anterior.
- `src/predictoras.py`: tabla de predictoras en variación % interanual y reporte de cuántos meses adelanta cada fuente.
- Descarga automática diaria en GitHub Actions: el proyecto ya no depende de la computadora de nadie.
- COES: leyendo el JavaScript del portal se identificó la fuente (`Portal/portalinformacion/Generacion`). `src/coes.py`
  descarga la producción mensual del SEIN desde 2013 (SCADA y medidores) y el mes en curso. Adelanta **2 meses** a la cifra
  oficial (el 03/11 ya tenía setiembre y octubre) y su variación interanual tiene correlación 0,993 con la del BCRP.
- Disponibilidad al 03/11: objetivo, electricidad BCRP, cemento e IGV hasta agosto; metales hasta setiembre;
  COES hasta octubre (y noviembre parcial); temperatura del mar hasta octubre (`resultados/disponibilidad_predictoras.csv`).

## Próximos hitos

| Fecha | Hito |
|---|---|
| Mié 4 nov | ✔ Adelantado al 02/11: datos verificados, referencias evaluadas, setiembre v1 y v2 registradas |
| Dom 8 nov | LightGBM y modelo de factores dinámicos con error medido; electricidad diaria del COES |
| Mié 11 nov | Calibración, estudio de El Niño y estimación de setiembre v3 (mejor modelo) registrada antes del 13/11 |
| Dom 15 nov | Entrega final; comparación de v1, v2 y v3 con la cifra oficial de setiembre |
