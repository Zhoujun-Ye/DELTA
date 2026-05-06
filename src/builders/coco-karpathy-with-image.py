import os
from datasets import DatasetInfo, GeneratorBasedBuilder, SplitGenerator, Version, Features, Value, Sequence, Image, Split

_CITATION = """\
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={Computer Vision--ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13},
  pages={740--755},
  year={2014},
  organization={Springer}
}
"""

_DESCRIPTION = """\
Code and datasets for "Microsoft COCO: Common Objects in Context".
"""

_HOMEPAGE = "https://huggingface.co/datasets/Mayfull/coco-karpathy-with-image"
_LICENSE = "Null"

_DATASET_NAME = "coco_raw_data"
_URLS = {
    "images": "https://huggingface.co/datasets/Mayfull/coco-karpathy-with-image/resolve/main/images.zip",
    "examples": "https://huggingface.co/datasets/Mayfull/coco-karpathy-with-image/resolve/main/examples.jsonl",
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
                    "images": Image(),
                    "sentences": Sequence(Value("string")),
                    "cocoid": Value("string"),
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
                name=Split.TRAIN,
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
                    "images": image_path,
                    "sentences": data["sentences"],
                    "cocoid": data["cocoid"],
                }
