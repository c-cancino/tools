#Split estratificado YOLO 

## Objetivo
Generar un split train/val/test para datasets YOLO con estructura tipo Camai:
- images/ en subcarpetas (escena/sub_escena/momento/...)
- labels/ plano (labels/*.txt) o en espejo (auto-detect)

El split es estratificado por:
- Clases presentes por imagen (multi-label)
- Y grupos adicionales (recomendado: escena,momento)

## Requisitos
- Python 3 (Ubuntu 22.04 OK)

## Dataset esperado
Estructura mínima:
- <SRC>/images/...
- <SRC>/labels/...

Ejemplo:
- SRC: /mnt/work/camai_core

## Script
- scripts/split_stratificado_yolo.py

## Ejecución recomendada
Split 80/10/10 balanceado por escena,momento (mantiene estructura de carpetas):

python3 /mnt/work/repos/catu_delivery_tools/scripts/split_stratificado_yolo.py \
  --src /mnt/work/camai_core \
  --out /mnt/work/outputs/entrega_v1/split \
  --train 0.8 --val 0.1 --test 0.1 \
  --group-keys escena,momento \
  --keep-structure \
  --seed 42

## Validación rápida
find /mnt/work/outputs/entrega_v1/split/train/images -type f | wc -l
find /mnt/work/outputs/entrega_v1/split/train/labels -type f | wc -l
find /mnt/work/outputs/entrega_v1/split/val/images -type f | wc -l
find /mnt/work/outputs/entrega_v1/split/val/labels -type f | wc -l
find /mnt/work/outputs/entrega_v1/split/test/images -type f | wc -l
find /mnt/work/outputs/entrega_v1/split/test/labels -type f | wc -l

Manifest:
- /mnt/work/outputs/entrega_v1/split/split_manifest.json

## Notas
- Si hay nombres repetidos de imágenes (mismo stem) y labels planos, puede haber ambigüedad.
  Recomendación: labels en espejo o nombres únicos.
