GUÍA RÁPIDA — Cómo usar split_stratificado_yolo.py en otros datasets (YOLO)

1) Qué hace el script
- Genera un split train/val/test (copiando imágenes y labels) desde un dataset YOLO.
- Hace estratificación por clases presentes en cada imagen (multi-label).
- Opcionalmente balancea además por “grupo” (ej: escena, subescena, momento) según carpetas bajo images/.

2) Requisitos mínimos del dataset
El dataset debe tener esta estructura base:

DATASET/
  images/   (puede tener subcarpetas)
  labels/   (puede ser plano o en espejo)

- Imágenes: .jpg/.jpeg/.png/.webp/.bmp
- Labels: .txt en formato YOLO (cada línea: class x y w h)

3) Tipos de labels soportados

A) Labels “en espejo” (recomendado)
Ejemplo:
images/escena_1/sub_escena_1/manana/img001.jpg
labels/escena_1/sub_escena_1/manana/img001.txt
=> Funciona directo, sin flags extra.

B) Labels “planos” (todos juntos)
Ejemplo:
images/.../img001.jpg
labels/img001.txt
=> Funciona (tiene fallback automático).
Si quieres forzarlo explícitamente: usar --labels-flat

IMPORTANTE: si hay nombres repetidos (stems) en distintas carpetas y labels planos,
puede haber ambigüedad (img1.jpg en 2 carpetas y solo 1 img1.txt).

4) Cómo decide escena/subescena/momento
El script obtiene:
- escena    = 1er nivel de carpeta bajo images/
- subescena = 2do nivel
- momento   = 3er nivel

Ejemplo:
images/escena_19/sub_escena_1/noche/xxx.jpg
-> escena=escena_19, subescena=sub_escena_1, momento=noche

Si tu dataset no tiene esa estructura, igual corre, pero esos campos pueden quedar como:
unknown_escena / unknown_subescena / unknown_momento o valores no “semánticos”.

5) Plantilla de ejecución (general)
Cambia SOLO --src y --out según tu dataset:

python3 /mnt/work/repos/catu_delivery_tools/scripts/split_stratificado_yolo.py \
  --src /RUTA/AL/DATASET \
  --out /RUTA/DE/SALIDA/split \
  --train 0.8 --val 0.1 --test 0.1 \
  --group-keys escena,momento \
  --keep-structure \
  --seed 42

6) Recomendaciones de group-keys (según estructura)
- Si tu dataset tiene carpetas tipo Camai (escena/sub_escena/momento):
  --group-keys escena,momento    (recomendado)
- Si solo tienes escena:
  --group-keys escena
- Si solo tienes “día/noche” o algo equivalente:
  --group-keys momento
- Si NO quieres grouping (solo estratifica por clases):
  --group-keys ""

7) Flags útiles
- --labels-flat
  Fuerza que labels sean planos: labels/<stem>.txt

- --keep-structure
  Mantiene subcarpetas originales en out/<split>/images/...
  (Los labels se guardan planos en out/<split>/labels/<stem>.txt)

- --min-instances N
  Excluye imágenes con menos de N instancias. Default = 1
  Para incluir labels vacíos: --min-instances 0

- --seed 42
  Reproducibilidad del split.

- --dry
  No copia archivos; solo genera split_manifest.json (para validar rápido).

8) Checklist rápido antes de correr (30s)
A) Ver estructura base:
ls /RUTA/AL/DATASET/images
ls /RUTA/AL/DATASET/labels

B) Contar imágenes y labels:
find /RUTA/AL/DATASET/images -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" -o -iname "*.webp" \) | wc -l
find /RUTA/AL/DATASET/labels -type f -name "*.txt" | wc -l

C) Probar en seco (sin copiar):
python3 /mnt/work/repos/catu_delivery_tools/scripts/split_stratificado_yolo.py \
  --src /RUTA/AL/DATASET \
  --out /mnt/work/outputs/splits/mi_dataset_split_dry \
  --train 0.8 --val 0.1 --test 0.1 \
  --group-keys "" \
  --seed 42 \
  --dry

9) Validación post-split (integridad)
Conteos:
find /RUTA/DE/SALIDA/split/train/images -type f | wc -l
find /RUTA/DE/SALIDA/split/train/labels -type f | wc -l
find /RUTA/DE/SALIDA/split/val/images -type f | wc -l
find /RUTA/DE/SALIDA/split/val/labels -type f | wc -l
find /RUTA/DE/SALIDA/split/test/images -type f | wc -l
find /RUTA/DE/SALIDA/split/test/labels -type f | wc -l

Manifest (auditoría):
/RUTA/DE/SALIDA/split/split_manifest.json

10) Problemas típicos y solución
- “Imágenes sin label” > 0:
  Falta label para algunas imágenes. Decide si excluirlas (default) o corregir dataset.

- “Ambigüedad por stems duplicados” (labels planos):
  Renombrar archivos para que el stem sea único o usar labels en espejo.

- “Labels vacíos y se excluyen”:
  Usa --min-instances 0 si quieres mantenerlas.

Fin.
