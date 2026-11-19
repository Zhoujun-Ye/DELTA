"""Adapt author-published annotations to DELTA's existing dataset schemas."""

import ast
import csv
import json
from pathlib import Path


SUGAR_TYPES = (
    "add_obj", "add_att", "replace_obj", "replace_att", "replace_rel",
    "swap_obj", "swap_att",
)
VALSE_FILES = (
    "actant-swap", "action-replacement", "coreference-hard",
    "coreference-standard", "counting-adversarial", "counting-hard",
    "counting-small-quant", "existence", "foil-it", "plurals", "relations",
)
KARPATHY_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
CREPE_URL = "https://raw.githubusercontent.com/RAIVNLab/CREPE/master/data/prod_hard_negatives.zip"
SWIG_URL = "https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip"
VISUAL7W_URL = "http://vision.stanford.edu/yukezhu/visual7w_images.zip"


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def coco_examples(metadata, image_root):
    for sample in metadata["images"]:
        if sample["split"] not in ("train", "restval"):
            continue
        yield {
            "filename": str(image_root / sample["filepath"] / sample["filename"]),
            "sentences": [sentence["raw"] for sentence in sample["sentences"]],
            "cocoid": str(sample["cocoid"]),
        }


def sugar_examples(metadata, image_root, category):
    if category not in SUGAR_TYPES:
        raise ValueError(f"Unknown SugarCrepe category: {category}")
    for sample in metadata.values():
        yield {
            "image_file_name": str(image_root / sample["filename"]),
            "positive_caption": [sample["caption"]],
            "negative_caption": [sample["negative_caption"]],
            "original_file_name": sample["filename"],
            "category": category,
        }


def crepe_examples(csv_path, image_index):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for sample in csv.DictReader(handle):
            filename = f"{int(sample['image_id'])}.jpg"
            negatives = ast.literal_eval(sample["hard_negs"])
            if not isinstance(negatives, list) or not negatives:
                raise ValueError(f"Invalid hard_negs in {csv_path}")
            yield {
                "image": str(image_index[filename]),
                "positive_caption": [sample["caption"]],
                "negative_caption": negatives,
                "original_file_name": filename,
                **{key: float(sample[key]) for key in ("x", "y", "width", "height")},
            }


def valse_examples(metadata, image_index):
    for key, sample in metadata.items():
        if sample["mturk"]["caption"] < 2:
            continue
        filename = sample["image_file"]
        yield {
            "image_file_name": str(image_index[filename]),
            "positive_caption": [sample["caption"]],
            "negative_caption": [sample["foil"]],
            "original_file_name": filename,
            "key": key,
            **{field: sample[field] for field in (
                "dataset", "linguistic_phenomena", "original_split",
            )},
        }


def _resolve(name, files):
    from src.builders.download_utils import resolve_dataset_files

    return {key: Path(value) for key, value in resolve_dataset_files(
        dataset_name=f"raw/{name}", files=files,
    ).items()}


def _coco_images(split):
    files = _resolve("coco", {
        split: f"http://images.cocodataset.org/zips/{split}.zip",
    })
    return files[split] / split


def load_coco_metadata():
    files = _resolve("coco", {"annotations": KARPATHY_URL})
    return read_json(files["annotations"] / "dataset_coco.json")


def _image_index(roots):
    index = {}
    for root in roots:
        for path in root.rglob("*.jpg"):
            if path.name in index and index[path.name] != path:
                raise ValueError(f"Ambiguous image filename: {path.name}")
            index[path.name] = path.resolve()
    return index


def prepare_dataset(name):
    from src.builders.download_utils import DATASET_BASE_DIR

    root = Path(DATASET_BASE_DIR).resolve() / "raw"
    filename = "examples-v2.jsonl" if name == "sugarcrepe" else "examples.jsonl"
    target = root / name / filename
    marker = target.with_suffix(".complete.json")
    if target.is_file() and marker.is_file():
        return {"examples": str(target), "images": str(root)}

    sources = []
    if name == "coco":
        sources = [KARPATHY_URL] + [
            f"http://images.cocodataset.org/zips/{split}.zip"
            for split in ("train2014", "val2014")
        ]
        metadata = load_coco_metadata()
        train = _coco_images("train2014")
        val = _coco_images("val2014")
        # The resolver adds an extraction directory above the archive's own folder.
        rows = coco_examples(metadata, train.parent)
        rows = ({**row, "filename": str(
            (train if Path(row["filename"]).parent.name == "train2014" else val)
            / Path(row["filename"]).name
        )} for row in rows)
    elif name == "sugarcrepe":
        urls = {kind: f"https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data/{kind}.json"
                for kind in SUGAR_TYPES}
        sources = list(urls.values()) + ["http://images.cocodataset.org/zips/val2017.zip"]
        files = _resolve(name, urls)
        images = _coco_images("val2017")
        rows = (row for kind in SUGAR_TYPES
                for row in sugar_examples(read_json(files[kind]), images, kind))
    elif name == "crepe":
        sources = [CREPE_URL]
        annotations = _resolve(name, {"annotations": CREPE_URL})["annotations"]
        roots = []
        for part, archive in (("1", "images.zip"), ("2", "images2.zip")):
            url = f"https://cs.stanford.edu/people/rak248/VG_100K_2/{archive}"
            sources.append(url)
            roots.append(_resolve(f"visual_genome_{part}", {"images": url})["images"])
        index = _image_index(roots)
        # *_all.csv repeats the same examples as the per-complexity files.
        csvs = sorted(annotations.rglob("prod_vg_hard_negs_*_complexity_*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No official CREPE productivity CSVs in {annotations}")
        rows = (row for path in csvs for row in crepe_examples(path, index))
    elif name == "valse":
        urls = {kind: f"https://raw.githubusercontent.com/Heidelberg-NLP/VALSE/main/data/{kind}.json"
                for kind in VALSE_FILES}
        sources = list(urls.values()) + [SWIG_URL, VISUAL7W_URL] + [
            f"http://images.cocodataset.org/zips/{split}.zip"
            for split in ("train2014", "val2014", "val2017")
        ]
        files = _resolve(name, urls)
        roots = [_coco_images(split) for split in ("train2014", "val2014", "val2017")]
        roots.append(_resolve("swig", {"images": SWIG_URL})["images"])
        roots.append(_resolve("visual7w", {"images": VISUAL7W_URL})["images"])
        index = _image_index(roots)
        rows = (row for kind in VALSE_FILES
                for row in valse_examples(read_json(files[kind]), index))
    else:
        raise ValueError(f"Unknown official dataset: {name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".jsonl.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            image = row.get("filename", row.get("image", row.get("image_file_name")))
            if not Path(image).is_file():
                raise FileNotFoundError(image)
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    if not count:
        raise ValueError(f"Official dataset is empty: {name}")
    temporary.replace(target)
    marker.write_text(json.dumps({"sources": sources, "rows": count}), encoding="utf-8")
    return {"examples": str(target), "images": str(root)}


def caption_pair_examples(examples_file):
    with open(examples_file, encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            image_path = sample.pop("image", sample.pop("image_file_name", None))
            if not image_path or not Path(image_path).is_file():
                raise FileNotFoundError(image_path)
            yield {**sample, "images": [image_path], "image_paths": [image_path]}


def load_caption_pairs(name, features, split):
    from datasets import Dataset

    if split != "test":
        raise ValueError(f"{name} only supports the test split, got {split!r}")
    files = prepare_dataset(name)
    return Dataset.from_generator(
        caption_pair_examples,
        gen_kwargs={"examples_file": files["examples"]},
        features=features,
        split=split,
    )
