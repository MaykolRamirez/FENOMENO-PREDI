# Guía rápida de GitHub para el equipo

## 1. Crear el repositorio (lo hace una sola persona)

1. Entrar a https://github.com e iniciar sesión.
2. Botón **New** (o el signo **+** arriba a la derecha → *New repository*).
3. Nombre: `iapn-inei`. Descripción: *Indicador Adelantado de la Producción Nacional · PEU-CD INEI*.
4. Elegir **Private** si el profesor no pidió que sea público.
5. **No** marcar «Add a README» (ya tenemos uno). Clic en **Create repository**.

## 2. Subir los archivos

### Opción A · Desde el navegador (más fácil)

1. Descomprimir `iapn-inei.zip` en la computadora.
2. En la página del repositorio vacío, clic en **uploading an existing file**.
3. Abrir la carpeta descomprimida, seleccionar **todo su contenido** (no la carpeta) y arrastrarlo.
4. Mensaje del commit: `Estructura inicial del proyecto`. Clic en **Commit changes**.

> GitHub web no sube carpetas vacías ni archivos ocultos como `.gitignore`.
> Si falta `.gitignore`, crearlo con **Add file → Create new file** y pegar su contenido.

### Opción B · Desde la terminal (conserva el historial)

```bash
cd iapn-inei
git remote add origin https://github.com/MarciaShapiama/iapn-inei.git
git branch -M main
git push -u origin main
```

## 3. Invitar al equipo y al profesor

**Settings → Collaborators → Add people**, y escribir el usuario o correo de cada integrante y del profesor.

## 4. Trabajar en equipo

| Acción | Comando |
|---|---|
| Traer lo último antes de empezar | `git pull` |
| Crear mi rama | `git checkout -b datos` (o `modelos`, `evaluacion`, `clima`) |
| Guardar un avance | `git add .` y `git commit -m "Agrega descarga de BCRPData"` |
| Subir mi rama | `git push -u origin datos` |
| Unir a `main` | En GitHub: **Compare & pull request** → revisar → **Merge** |

En Google Colab: **Archivo → Guardar una copia en GitHub** guarda el cuaderno directamente en el repositorio.

**Repositorio privado en Colab.** `git clone` pide credenciales. Cada integrante crea un token personal en
GitHub (**Settings → Developer settings → Personal access tokens → Fine-grained**, con acceso solo a
`iapn-inei`) y lo guarda en Colab con el ícono de la llave (**Secretos**) con el nombre `GITHUB_TOKEN`.
Luego clona así:

```python
from google.colab import userdata
token = userdata.get("GITHUB_TOKEN")
!git clone https://{token}@github.com/MarciaShapiama/iapn-inei.git
```

Nunca pegar el token directamente en el cuaderno ni compartirlo por chat.

## 5. Descarga automática de datos

GitHub descarga los datos cada día a las 8:20 a. m. (Lima) sin que nadie prenda su computadora. Para lanzarla a mano:
pestaña **Actions → Descarga de datos → Run workflow**. Si una corrida sale en rojo, abrirla para ver qué fuente falló;
las demás fuentes se guardan igual.

## 6. Registrar una estimación en tiempo real

1. Generar la estimación y guardarla con `registrar(...)` de `src/registro.py`.
2. Inmediatamente: `git add registros/nowcasts/` · `git commit -m "Registra nowcast de setiembre 2026 v1"` · `git push`.
3. La fecha del commit en GitHub es la constancia de que la estimación se hizo antes de la cifra oficial.
4. Nunca editar ni borrar un registro. Una nueva estimación es una nueva versión (`v2`).
