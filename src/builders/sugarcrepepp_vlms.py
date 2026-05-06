import os
from datasets import DatasetInfo, GeneratorBasedBuilder, SplitGenerator, Version, Features, Value, Sequence, Image, Split

_CITATION = """\
@article{dumpala2024sugarcrepe++,
  title={SUGARCREPE++ Dataset: Vision-Language Model Sensitivity to Semantic and Lexical Alterations},
  author={Dumpala, Sri Harsha and Jaiswal, Aman and Sastry, Chandramouli and Milios, Evangelos and Oore, Sageev and Sajjad, Hassan},
  journal={arXiv preprint arXiv:2406.11171},
  year={2024}
}
"""

_DESCRIPTION = """\
Code and datasets for "SUGARCREPE++ Dataset: Vision-Language Model Sensitivity to Semantic and Lexical Alterations".
"""

_HOMEPAGE = "https://huggingface.co/datasets/Mayfull/sugarcrepepp_vlms"
_LICENSE = "cc-by-4.0"

_DATASET_NAME = "sugarcrepepp_vlms"
_URLS = {
    "images": "https://huggingface.co/datasets/Mayfull/sugarcrepepp_vlms/resolve/main/images.zip",
    "examples": "https://huggingface.co/datasets/Mayfull/sugarcrepepp_vlms/resolve/main/examples.jsonl",
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
                    "positive_caption_1": Sequence(Value("string")),
                    "positive_caption_2": Sequence(Value("string")),
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
                image_path = os.path.join(images_dir, data["filename"])
                yield idx, {
                    "images": [image_path],
                    "image_paths": [image_path],
                    "positive_caption_1": [data["caption"]],
                    "positive_caption_2": [data["caption2"]],
                    "negative_caption": [data["negative_caption"]],
                    "original_file_name": data["name"],
                }
