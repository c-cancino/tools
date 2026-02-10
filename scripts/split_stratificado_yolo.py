#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

@dataclass
class Sample:
    stem: str
    img_path: Path
    lbl_path: Path
    rel_img: Path
    escena: str
    subescena: str
    momento: str
    classes: Set[int]

def parse_args():
    ap = argparse.ArgumentParser(description="Split estratificado YOLO (Camai-style).")
    ap.add_argument("--src", required=True, help="Dataset fuente con images/ y labels/")
    ap.add_argument("--out", required=True, help="Carpeta salida del split")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-structure", action="store_true")
    ap.add_argument("--labels-flat", action="store_true")
    ap.add_argument("--min-instances", type=int, default=1)
    ap.add_argument("--group-keys", default="escena,momento")
    ap.add_argument("--max-classes", type=int, default=10_000)
    ap.add_argument("--dry", action="store_true")
    return ap.parse_args()

def build_image_stem_counts(images_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            counts[p.stem] = counts.get(p.stem, 0) + 1
    return counts

def extract_meta_from_rel(rel: Path) -> Tuple[str, str, str]:
    parts = rel.parts
    escena = parts[0] if len(parts) >= 1 else "unknown_escena"
    subescena = parts[1] if len(parts) >= 2 else "unknown_subescena"
    momento = parts[2] if len(parts) >= 3 else "unknown_momento"
    return escena, subescena, momento

def read_classes(lbl_path: Path, max_classes: int) -> Tuple[Set[int], int]:
    classes: Set[int] = set()
    n = 0
    for ln in lbl_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        try:
            c = int(float(parts[0]))
        except:
            continue
        if 0 <= c <= max_classes:
            classes.add(c)
            n += 1
    return classes, n

def resolve_label_path(labels_dir: Path, images_dir: Path, img_path: Path, labels_flat: bool) -> Optional[Path]:
    if labels_flat:
        cand = labels_dir / f"{img_path.stem}.txt"
        return cand if cand.exists() else None

    rel = img_path.relative_to(images_dir)
    cand = labels_dir / rel.with_suffix(".txt")
    if cand.exists():
        return cand

    cand2 = labels_dir / f"{img_path.stem}.txt"
    if cand2.exists():
        return cand2

    return None

def load_samples(src: Path, labels_flat: bool, min_instances: int, max_classes: int) -> List[Sample]:
    images_dir = src / "images"
    labels_dir = src / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit("src debe tener images/ y labels/")

    stem_counts = build_image_stem_counts(images_dir)
    dups = sum(1 for _, v in stem_counts.items() if v > 1)

    samples: List[Sample] = []
    missing = 0
    empty = 0

    img_files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    for img in img_files:
        lbl = resolve_label_path(labels_dir, images_dir, img, labels_flat=labels_flat)
        if lbl is None:
            missing += 1
            continue
        classes, n_inst = read_classes(lbl, max_classes=max_classes)
        if n_inst < min_instances:
            empty += 1
            continue
        rel = img.relative_to(images_dir)
        escena, subescena, momento = extract_meta_from_rel(rel)
        samples.append(Sample(
            stem=img.stem,
            img_path=img,
            lbl_path=lbl,
            rel_img=rel,
            escena=escena,
            subescena=subescena,
            momento=momento,
            classes=classes,
        ))

    print(f"Samples cargados: {len(samples)}")
    print(f"Imágenes sin label: {missing}")
    print(f"Imágenes excluidas por instancias<{min_instances}: {empty}")
    if dups > 0 and labels_flat:
        print(f"AVISO: stems duplicados en images/: {dups} (labels planos pueden ser ambiguos).")

    return samples

def make_group_key(s: Sample, keys: List[str]) -> str:
    parts: List[str] = []
    for k in keys:
        if k == "escena":
            parts.append(s.escena)
        elif k == "subescena":
            parts.append(s.subescena)
        elif k == "momento":
            parts.append(s.momento)
    return "|".join(parts) if parts else "all"

def greedy_stratified_split(
    samples: List[Sample],
    ratios: Dict[str, float],
    group_keys: List[str],
    seed: int
) -> Dict[str, List[Sample]]:
    random.seed(seed)
    n = len(samples)

    targets_n = {k: int(round(v * n)) for k, v in ratios.items()}
    diff = n - sum(targets_n.values())
    if diff != 0:
        targets_n["train"] = targets_n.get("train", 0) + diff

    class_tot: Dict[int, int] = {}
    for s in samples:
        for c in s.classes:
            class_tot[c] = class_tot.get(c, 0) + 1

    target_class: Dict[str, Dict[int, float]] = {sp: {} for sp in ratios.keys()}
    for sp, r in ratios.items():
        for c, tot in class_tot.items():
            target_class[sp][c] = r * tot

    group_tot: Dict[str, int] = {}
    for s in samples:
        g = make_group_key(s, group_keys)
        group_tot[g] = group_tot.get(g, 0) + 1

    target_group: Dict[str, Dict[str, float]] = {sp: {} for sp in ratios.keys()}
    for sp, r in ratios.items():
        for g, tot in group_tot.items():
            target_group[sp][g] = r * tot

    splits: Dict[str, List[Sample]] = {k: [] for k in ratios.keys()}
    split_counts_n = {k: 0 for k in ratios.keys()}
    cur_class: Dict[str, Dict[int, float]] = {k: {} for k in ratios.keys()}
    cur_group: Dict[str, Dict[str, float]] = {k: {} for k in ratios.keys()}

    def rarity_score(s: Sample) -> float:
        sc = 0.0
        for c in s.classes:
            sc += 1.0 / max(1, class_tot.get(c, 1))
        return -sc

    ordered = sorted(samples, key=rarity_score)

    def cost_if_assign(sp: str, s: Sample) -> float:
        if split_counts_n[sp] >= targets_n[sp]:
            return 1e9

        g = make_group_key(s, group_keys)
        cost = 0.0

        for c in s.classes:
            cur = cur_class[sp].get(c, 0.0)
            tgt = target_class[sp].get(c, 0.0)
            after = cur + 1.0
            cost += (after - tgt) ** 2 - (cur - tgt) ** 2

        curg = cur_group[sp].get(g, 0.0)
        tgtg = target_group[sp].get(g, 0.0)
        afterg = curg + 1.0
        cost += (afterg - tgtg) ** 2 - (curg - tgtg) ** 2

        curN = split_counts_n[sp]
        tgtN = targets_n[sp]
        cost += 0.5 * ((curN + 1 - tgtN) ** 2 - (curN - tgtN) ** 2)

        return cost

    for s in ordered:
        best_sp = None
        best_cost = None
        for sp in ratios.keys():
            c = cost_if_assign(sp, s)
            if best_cost is None or c < best_cost:
                best_cost = c
                best_sp = sp

        assert best_sp is not None
        splits[best_sp].append(s)
        split_counts_n[best_sp] += 1

        g = make_group_key(s, group_keys)
        cur_group[best_sp][g] = cur_group[best_sp].get(g, 0.0) + 1.0
        for cls in s.classes:
            cur_class[best_sp][cls] = cur_class[best_sp].get(cls, 0.0) + 1.0

    for sp in splits:
        random.shuffle(splits[sp])

    return splits

def export_split(out: Path, src: Path, splits: Dict[str, List[Sample]], keep_structure: bool, dry: bool):
    images_dir = src / "images"
    manifest = {"splits": {}}

    for sp, items in splits.items():
        manifest["splits"][sp] = []
        out_img = out / sp / "images"
        out_lbl = out / sp / "labels"
        if not dry:
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

        for s in items:
            if keep_structure:
                rel_img = s.img_path.relative_to(images_dir)
                dst_img = out_img / rel_img
            else:
                dst_img = out_img / s.img_path.name

            dst_lbl = out_lbl / f"{s.stem}.txt"

            manifest["splits"][sp].append({
                "stem": s.stem,
                "img": str(s.img_path),
                "lbl": str(s.lbl_path),
                "escena": s.escena,
                "subescena": s.subescena,
                "momento": s.momento,
                "classes": sorted(list(s.classes)),
                "dst_img": str(dst_img),
                "dst_lbl": str(dst_lbl),
            })

            if not dry:
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s.img_path, dst_img)
                shutil.copy2(s.lbl_path, dst_lbl)

    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

def main():
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise SystemExit("train+val+test debe sumar 1.0")

    ratios = {"train": args.train, "val": args.val, "test": args.test}

    group_keys = [k.strip() for k in args.group_keys.split(",") if k.strip()] if args.group_keys.strip() else []
    for k in group_keys:
        if k not in {"escena", "subescena", "momento"}:
            raise SystemExit(f"group-keys inválido: {k}")

    samples = load_samples(src, labels_flat=args.labels_flat, min_instances=args.min_instances, max_classes=args.max_classes)
    if len(samples) == 0:
        raise SystemExit("No hay samples válidos. Revisa rutas o min-instances.")

    splits = greedy_stratified_split(samples, ratios, group_keys=group_keys, seed=args.seed)

    print("\n=== Resumen split ===")
    for sp, items in splits.items():
        print(f"{sp}: {len(items)}")

    out.mkdir(parents=True, exist_ok=True)
    export_split(out, src, splits, keep_structure=args.keep_structure, dry=args.dry)
    print(f"\nListo. Manifest: {out}/split_manifest.json")
    if args.dry:
        print("DRY RUN: no se copiaron archivos.")

if __name__ == "__main__":
    main()
