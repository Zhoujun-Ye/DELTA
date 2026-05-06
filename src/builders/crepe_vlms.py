import os
from datasets import DatasetInfo, GeneratorBasedBuilder, SplitGenerator, Version, Features, Value, Sequence, Image, Split

_CITATION = """\
@inproceedings{ma2023crepe,
  title={Crepe: Can vision-language foundation models reason compositionally?},
  author={Ma, Zixian and Hong, Jerry and Gul, Mustafa Omer and Gandhi, Mona and Gao, Irena and Krishna, Ranjay},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10910--10921},
  year={2023}
}
"""

_DESCRIPTION = """\
Code and datasets for "CREPE: Can Vision-Language Foundation Models Reason Compositionally?".
"""

_HOMEPAGE = "https://huggingface.co/datasets/Mayfull/crepe_vlms"
_LICENSE = "MIT License"

_DATASET_NAME = "crepe_vlms"
_URLS = {
    "images": "https://huggingface.co/datasets/Mayfull/crepe_vlms/resolve/main/images.zip",
    "examples": "https://huggingface.co/datasets/Mayfull/crepe_vlms/resolve/main/examples.jsonl",
}


class CREPEVLMsDataset(GeneratorBasedBuilder):
    VERSION = Version("1.0.0")

    def _info(self):
        return DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            features=Features(
                {
                    "images": Sequence(Image()),  # Sequence of images
                    "image_paths": Sequence(Value("string")),
                    "positive_caption": Sequence(Value("string")),
                    "negative_caption": Sequence(Value("string")),
                    "x": Value("float"),
                    "y": Value("float"),
                    "width": Value("float"),
                    "height": Value("float"),
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
        # Read the examples.jsonl file
        with open(examples_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = eval(line.strip())

                # Get image file path
                image_file_name = data.get("image")
                image_path = os.path.join(images_dir, image_file_name)

                # Ensure the image file exists
                images = [image_path] if os.path.exists(image_path) else []

                # Convert bounding box values to float
                x = float(data.get("x", 0.0))
                y = float(data.get("y", 0.0))
                width = float(data.get("width", 0.0))
                height = float(data.get("height", 0.0))

                # Prepare the example
                yield idx, {
                    "images": images,
                    "image_paths": [image_path] if os.path.exists(image_path) else [],
                    "positive_caption": data.get("positive_caption", []),
                    "negative_caption": data.get("negative_caption", []),
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "original_file_name": data.get("original_file_name", ""),
                }
