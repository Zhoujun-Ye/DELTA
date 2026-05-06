import os
from datasets import DatasetInfo, GeneratorBasedBuilder, SplitGenerator, Version, Features, Value, Sequence, Image, Split

_CITATION = """\
@inproceedings{hsieh2023sugarcrepe,
  title={SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality},
  author={Hsieh, Cheng-Yu and Zhang, Jieyu and Ma, Zixian and Kembhavi, Aniruddha and Krishna, Ranjay},
  booktitle={Thirty-Seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
"""

_DESCRIPTION = """\
Code and datasets for "SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality".
"""

_HOMEPAGE = "https://huggingface.co/datasets/Mayfull/sugarcrepe_vlms"
_LICENSE = "MIT License"

_DATASET_NAME = "sugarcrepe_vlms"
_URLS = {
    "images": "https://huggingface.co/datasets/Mayfull/sugarcrepe_vlms/resolve/main/images.zip",
    "examples": "https://huggingface.co/datasets/Mayfull/sugarcrepe_vlms/resolve/main/examples.jsonl",
}


class SugarCrepeDataset(GeneratorBasedBuilder):
    VERSION = Version("1.0.0")

    def _info(self):
        return DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            features=Features(
                {
                    "images": Sequence(Image()),
                    "image_paths": Sequence(Value("string")),
                    "positive_caption": Sequence(Value("string")),
                    "negative_caption": Sequence(Value("string")),
                    "original_file_name": Value("string"),
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
        with open(examples_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = eval(line)
                image_path = os.path.join(images_dir, data["image_file_name"])
                yield idx, {
                    "images": [image_path],
                    "image_paths": [image_path],
                    "positive_caption": data["positive_caption"],
                    "negative_caption": data["negative_caption"],
                    "original_file_name": data["original_file_name"],
                }
