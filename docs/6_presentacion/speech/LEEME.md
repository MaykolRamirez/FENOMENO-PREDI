# Guion del expositor y presentación del proyecto

- `presentacion_proyecto.tex` → `presentacion_proyecto.pdf`: las 41 láminas que se proyectan. Es la misma versión que `docs/proyecto_completo.pdf`.
- `guion_expositor.tex` → `guion_expositor.pdf`: qué decir en cada lámina (no se proyecta). Es la misma versión que `docs/speech_proyecto.pdf`.
- `estilo.tex`: colores y cajas comunes; `figuras/`: imágenes usadas.

Las cifras salen del informe final (`docs/6_presentacion/informe_final/`) y de los resultados del repositorio al 9 de noviembre de 2026.

Para volver a generarlos (dos veces cada uno, primero la presentación):

```bash
pdflatex presentacion_proyecto.tex && pdflatex presentacion_proyecto.tex
pdflatex guion_expositor.tex && pdflatex guion_expositor.tex
cp presentacion_proyecto.pdf ../../proyecto_completo.pdf
cp guion_expositor.pdf ../../speech_proyecto.pdf
```
