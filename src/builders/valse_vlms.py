import os
from datasets import DatasetInfo, GeneratorBasedBuilder, SplitGenerator, Version, Features, Value, Sequence, Image, Split

_CITATION = """\
@inproceedings{parcalabescu-etal-2022-valse,
    title = "{VALSE}: A Task-Independent Benchmark for Vision and Language Models Centered on Linguistic Phenomena",
    author = "Parcalabescu, Letitia  and
      Cafagna, Michele  and
      Muradjan, Lilitta  and
      Frank, Anette  and
      Calixto, Iacer  and
      Gatt, Albert",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.567",
    pages = "8253--8280",
    abstract = "We propose VALSE (Vision And Language Structured Evaluation), a novel benchmark designed for testing general-purpose pretrained vision and language (V{\\&}L) models for their visio-linguistic grounding capabilities on specific linguistic phenomena. VALSE offers a suite of six tests covering various linguistic constructs. Solving these requires models to ground linguistic phenomena in the visual modality, allowing more fine-grained evaluations than hitherto possible. We build VALSE using methods that support the construction of valid foils, and report results from evaluating five widely-used V{\\&}L models. Our experiments suggest that current models have considerable difficulty addressing most phenomena. Hence, we expect VALSE to serve as an important benchmark to measure future progress of pretrained V{\\&}L models from a linguistic perspective, complementing the canonical task-centred V{\\&}L evaluations.",
}
"""

_DESCRIPTION = """\
Code and datasets for "VALSE: A Task-Independent Benchmark for Vision and Language Models Centered on Linguistic Phenomena".
"""

_HOMEPAGE = "https://huggingface.co/datasets/Mayfull/valse_vlms"
_LICENSE = "MIT License"

_DATASET_NAME = "valse_vlms"
_URLS = {
    "images": "https://huggingface.co/datasets/Mayfull/valse_vlms/resolve/main/images.zip",
    "examples": "https://huggingface.co/datasets/Mayfull/valse_vlms/resolve/main/examples.jsonl",
}


class VALSEVLMsDataset(GeneratorBasedBuilder):
    VERSION = Version("1.0.0")

    def _info(self):
        return DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            features=Features(
                {
                    "images": Sequence(Image()),  # List of images
                    "image_paths": Sequence(Value("string")),
                    "positive_caption": Sequence(Value("string")),
                    "negative_caption": Sequence(Value("string")),
                    "original_file_name": Value("string"),
                    "dataset": Value("string"),
                    "key": Value("string"),
                    "linguistic_phenomena": Value("string"),
                    "original_split": Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        from src.builders.download_utils import resolve_dataset_files

        downloaded_files = resolve_dataset_files(
            dataset_name=_DATASET_NAME,
            files=_URLS,
            dl_manager=dl_manager,
        )

        return [
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "examples_file": downloaded_files["examples"],
                    "images_dir": downloaded_files["images"],
                },
            ),
        ]

    def _generate_examples(self, examples_file, images_dir):
        # Read the examples.jsonl file
        with open(examples_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = eval(line.strip())

                # Get image path and wrap it in a list
                image_file_name = data.get("image_file_name")
                image_path = os.path.join(images_dir, image_file_name)
                images = [image_path]  # Wrap single image path in a list

                # Ensure the image file exists
                if not os.path.exists(image_path):
                    continue  # Skip if image not found

                # Prepare the example
                yield idx, {
                    "images": images,  # List of images
                    "image_paths": [image_path],
                    "positive_caption": data.get("positive_caption", []),
                    "negative_caption": data.get("negative_caption", []),
                    "original_file_name": data.get("original_file_name", ""),
                    "dataset": data.get("dataset", ""),
                    "key": data.get("key", ""),
                    "linguistic_phenomena": data.get("linguistic_phenomena", ""),
                    "original_split": data.get("original_split", ""),
                }
