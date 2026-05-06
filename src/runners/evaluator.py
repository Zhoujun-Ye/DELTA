from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from transformers.utils import logging

from src.collators import BaseCollator
from src.common.registry import registry
from .base import BaseEvaluator

logger = logging.get_logger(__name__)

CollatorType = Type[BaseCollator]

__all__ = [
    "CrepeEvaluator",
    "SugarCrepeEvaluator",
    "SugarCrepePPEvaluator",
    "ValseEvaluator",
    "AROCOCOOrderEvaluator",
    "AROFlickrOrderEvaluator",
    "ColorSwapEvaluator",
]


def _average_result_by_substring(
    result: Dict,
    substring: str,
    suffix: Optional[str] = None,
) -> Optional[float]:
    normalized_substring = substring.lower()
    normalized_suffix = suffix.lower() if suffix is not None else None
    matched_values = []

    for key, value in result.items():
        normalized_key = key.lower()
        if normalized_key.startswith("total"):
            continue
        if normalized_substring not in normalized_key:
            continue
        if normalized_suffix is not None and not normalized_key.endswith(normalized_suffix):
            continue
        matched_values.append(value)

    if not matched_values:
        return None
    return round(sum(matched_values) / len(matched_values), 3)


@dataclass
@registry.register_evaluator("CrepeEvaluator")
class CrepeEvaluator(BaseEvaluator):
    builder_cls_name: Optional[str] = 'CrepeBuilder'
    default_filename: str = 'CREPE'

    def _build_batch_samples(self, batch):
        sample_infos = []
        samples_images = []
        samples_text = []
        for sample in batch:
            images = [
                image.crop(
                    (
                        sample["x"],
                        sample["y"],
                        sample["x"] + sample["width"],
                        sample["y"] + sample["height"],
                    )
                )
                for image in sample["images"]
            ]
            text = [*sample["positive_caption"], *sample["negative_caption"]]
            sample_infos.append({"original_file_name": sample["original_file_name"]})
            samples_images.append(images)
            samples_text.append(text)
        return sample_infos, samples_images, samples_text

    def evaluate(self):
        existing_result = self._load_existing_result()
        if existing_result is not None:
            return existing_result
        dataloader = self._prepare_dataloader()

        result = defaultdict(list)

        for batch in dataloader:
            if self._is_preprocessed_batch(batch):
                sample_infos = batch["sample_infos"]
                batched_results = self.get_image_to_text_score_batched_preprocessed(batch)
            else:
                sample_infos, samples_images, samples_text = self._build_batch_samples(batch)
                batched_results = self.get_image_to_text_score_batched(samples_images, samples_text)

            for sample_info, score in zip(sample_infos, batched_results):
                result[sample_info["original_file_name"]].append(score)

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0 for key, values in result.items()
        }
        average_result.update({"total": round(sum(average_result.values()) / len(average_result), 3)})

        self._save_result(average_result)
        return average_result

    def _save_result(self, result: Dict, filename: str = 'CREPE') -> None:
        super()._save_result(result, filename)

    def get_summary_metrics(self, result: Dict) -> Dict[str, float]:
        return {"CREPE": result["total"]}


@dataclass
@registry.register_evaluator("SugarCrepeEvaluator")
class SugarCrepeEvaluator(BaseEvaluator):
    builder_cls_name: Optional[str] = 'SugarCrepeBuilder'
    default_filename: str = 'SUGARCREPE'

    def _build_batch_samples(self, batch):
        sample_infos = []
        samples_images = []
        samples_text = []
        for sample in batch:
            sample_infos.append({"original_file_name": sample["original_file_name"]})
            samples_images.append(sample["images"])
            samples_text.append([*sample["positive_caption"], *sample["negative_caption"]])
        return sample_infos, samples_images, samples_text

    def evaluate(self):
        existing_result = self._load_existing_result()
        if existing_result is not None:
            return existing_result
        dataloader = self._prepare_dataloader()

        result = defaultdict(list)

        for batch in dataloader:
            if self._is_preprocessed_batch(batch):
                sample_infos = batch["sample_infos"]
                batched_results = self.get_image_to_text_score_batched_preprocessed(batch)
            else:
                sample_infos, samples_images, samples_text = self._build_batch_samples(batch)
                batched_results = self.get_image_to_text_score_batched(samples_images, samples_text)

            for sample_info, score in zip(sample_infos, batched_results):
                result[sample_info["original_file_name"]].append(score)

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0 for key, values in result.items()
        }
        average_result.update({"total": round(sum(average_result.values()) / len(average_result), 3)})

        self._save_result(average_result)
        return average_result

    def _save_result(self, result: Dict, filename: str = 'SUGARCREPE') -> None:
        super()._save_result(result, filename)

    def get_summary_metrics(self, result: Dict) -> Dict[str, float]:
        summary = {"SugarCrepe": result["total"]}

        add_avg = _average_result_by_substring(result, "add")
        swap_avg = _average_result_by_substring(result, "swap")
        replace_avg = _average_result_by_substring(result, "replace")

        if add_avg is not None:
            summary["SugarCrepe_Add"] = add_avg
        if swap_avg is not None:
            summary["SugarCrepe_Swap"] = swap_avg
        if replace_avg is not None:
            summary["SugarCrepe_Replace"] = replace_avg

        return summary


@dataclass
@registry.register_evaluator("SugarCrepePPEvaluator")
class SugarCrepePPEvaluator(BaseEvaluator):
    builder_cls_name: Optional[str] = 'SugarCrepePPBuilder'
    default_filename: str = 'SUGARCREPEPP'

    def _build_batch_samples(self, batch):
        sample_infos = []
        samples_images = []
        samples_text = []
        for sample in batch:
            sample_infos.append({"original_file_name": sample["original_file_name"]})
            samples_images.append(sample["images"])
            samples_text.append([*sample["positive_caption_1"], *sample["positive_caption_2"], *sample["negative_caption"]])
        return sample_infos, samples_images, samples_text

    def evaluate(self):
        existing_result = self._load_existing_result()
        if existing_result is not None:
            return existing_result
        dataloader = self._prepare_dataloader()

        result = defaultdict(list)

        for batch in dataloader:
            if self._is_preprocessed_batch(batch):
                sample_infos = batch["sample_infos"]
                batched_results = self.get_image_to_text_score_batched_preprocessed(
                    batch,
                    return_tot=True,
                )
            else:
                sample_infos, samples_images, samples_text = self._build_batch_samples(batch)
                batched_results = self.get_image_to_text_score_batched(
                    samples_images,
                    samples_text,
                    return_tot=True,
                )

            for sample_info, (itt_score, tot_score) in zip(sample_infos, batched_results):
                result[sample_info["original_file_name"] + "_itt"].append(itt_score)
                result[sample_info["original_file_name"] + "_tot"].append(tot_score)

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0 for key, values in result.items()
        }
        average_result.update(
            {
                "total_itt": round(
                    sum(v for k, v in average_result.items() if k.endswith("itt")) / sum(
                        k.endswith("itt") for k in average_result
                    ), 3
                )
            }
        )
        average_result.update(
            {
                "total_tot": round(
                    sum(v for k, v in average_result.items() if k.endswith("tot")) / sum(
                        k.endswith("tot") for k in average_result
                    ), 3
                )
            }
        )

        self._save_result(average_result)
        return average_result

    def _save_result(self, result: Dict, filename: str = 'SUGARCREPEPP') -> None:
        super()._save_result(result, filename)

    def get_summary_metrics(self, result: Dict) -> Dict[str, float]:
        summary = {
            "SugarCrepePP_ITT": result["total_itt"],
            "SugarCrepePP_TOT": result["total_tot"],
        }

        replace_itt = _average_result_by_substring(result, "replace", suffix="itt")
        replace_tot = _average_result_by_substring(result, "replace", suffix="tot")
        swap_itt = _average_result_by_substring(result, "swap", suffix="itt")
        swap_tot = _average_result_by_substring(result, "swap", suffix="tot")

        if replace_itt is not None:
            summary["SugarCrepePP_Replace_ITT"] = replace_itt
        if replace_tot is not None:
            summary["SugarCrepePP_Replace_TOT"] = replace_tot
        if swap_itt is not None:
            summary["SugarCrepePP_Swap_ITT"] = swap_itt
        if swap_tot is not None:
            summary["SugarCrepePP_Swap_TOT"] = swap_tot

        return summary


@dataclass
@registry.register_evaluator("ValseEvaluator")
class ValseEvaluator(BaseEvaluator):
    builder_cls_name: Optional[str] = 'ValseBuilder'
    default_filename: str = 'VALSE'

    def _build_batch_samples(self, batch):
        sample_infos = []
        samples_images = []
        samples_text = []
        for sample in batch:
            sample_infos.append({"linguistic_phenomena": sample["linguistic_phenomena"]})
            samples_images.append(sample["images"])
            samples_text.append([*sample["positive_caption"], *sample["negative_caption"]])
        return sample_infos, samples_images, samples_text

    def evaluate(self):
        existing_result = self._load_existing_result()
        if existing_result is not None:
            return existing_result
        dataloader = self._prepare_dataloader()

        result = defaultdict(list)

        for batch in dataloader:
            if self._is_preprocessed_batch(batch):
                sample_infos = batch["sample_infos"]
                batched_results = self.get_image_to_text_score_batched_preprocessed(batch)
            else:
                sample_infos, samples_images, samples_text = self._build_batch_samples(batch)
                batched_results = self.get_image_to_text_score_batched(samples_images, samples_text)

            for sample_info, score in zip(sample_infos, batched_results):
                result[sample_info["linguistic_phenomena"]].append(score)
                result["total"].append(score)

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0 for key, values in result.items()
        }

        self._save_result(average_result)
        return average_result

    def _save_result(self, result: Dict, filename: str = 'VALSE') -> None:
        super()._save_result(result, filename)

    def get_summary_metrics(self, result: Dict) -> Dict[str, float]:
        return {"VALSE": result["total"]}


@dataclass
class BaseAROEvaluator(BaseEvaluator):
    group_key: Optional[str] = None

    def _build_batch_samples(self, batch):
        sample_infos = []
        samples_images = []
        samples_text = []
        for sample in batch:
            bbox = sample["bbox"]
            image = sample["image"].crop(
                (
                    bbox["x"],
                    bbox["y"],
                    bbox["x"] + bbox["w"],
                    bbox["y"] + bbox["h"],
                )
            )
            sample_info = {}
            if self.group_key is not None:
                sample_info[self.group_key] = sample[self.group_key]
            sample_infos.append(sample_info)
            samples_images.append([image])
            samples_text.append([sample["true_caption"], sample["false_caption"]])
        return sample_infos, samples_images, samples_text

    def evaluate(self):
        existing_result = self._load_existing_result()
        if existing_result is not None:
            return existing_result

        dataloader = self._prepare_dataloader()
        result = defaultdict(list)

        for batch in dataloader:
            if self._is_preprocessed_batch(batch):
                sample_infos = batch["sample_infos"]
                batched_results = self.get_image_to_text_score_batched_preprocessed(batch)
            else:
                sample_infos, samples_images, samples_text = self._build_batch_samples(batch)
                batched_results = self.get_image_to_text_score_batched(samples_images, samples_text)

            for sample_info, score in zip(sample_infos, batched_results):
                result["total"].append(score)
                if self.group_key is not None:
                    result[sample_info[self.group_key]].append(score)

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0
            for key, values in result.items()
        }

        self._save_result(average_result)
        return average_result


@dataclass
class BaseAROOrderEvaluator(BaseEvaluator):
    def _build_batch_samples(self, batch):
        sample_infos = []
        samples_images = []
        samples_text = []
        for sample in batch:
            sample_infos.append({})
            samples_images.append([sample["images"]])
            samples_text.append(
                [
                    sample["correct_caption"],
                    sample["hard_text_1"],
                    sample["hard_text_2"],
                    sample["hard_text_3"],
                    sample["hard_text_4"],
                ]
            )
        return sample_infos, samples_images, samples_text

    def evaluate(self):
        existing_result = self._load_existing_result()
        if existing_result is not None:
            return existing_result

        dataloader = self._prepare_dataloader()
        result = defaultdict(list)

        for batch in dataloader:
            if self._is_preprocessed_batch(batch):
                batched_results = self.get_image_to_text_score_batched_preprocessed(batch)
            else:
                _, samples_images, samples_text = self._build_batch_samples(batch)
                batched_results = self.get_image_to_text_score_batched(samples_images, samples_text)

            result["total"].extend(batched_results)

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0
            for key, values in result.items()
        }

        self._save_result(average_result)
        return average_result


@dataclass
@registry.register_evaluator("AROCOCOOrderEvaluator")
class AROCOCOOrderEvaluator(BaseAROOrderEvaluator):
    builder_cls_name: Optional[str] = "AROCOCOOrderBuilder"
    default_filename: str = "ARO_COCO_ORDER"

    def _save_result(self, result: Dict, filename: str = "ARO_COCO_ORDER") -> None:
        super()._save_result(result, filename)

    def get_summary_metrics(self, result: Dict) -> Dict[str, float]:
        return {"ARO_COCO_Order": result["total"]}


@dataclass
@registry.register_evaluator("AROFlickrOrderEvaluator")
class AROFlickrOrderEvaluator(BaseAROOrderEvaluator):
    builder_cls_name: Optional[str] = "AROFlickrOrderBuilder"
    default_filename: str = "ARO_FLICKR_ORDER"

    def _save_result(self, result: Dict, filename: str = "ARO_FLICKR_ORDER") -> None:
        super()._save_result(result, filename)

    def get_summary_metrics(self, result: Dict) -> Dict[str, float]:
        return {"ARO_Flickr_Order": result["total"]}


@dataclass
@registry.register_evaluator("ColorSwapEvaluator")
class ColorSwapEvaluator(BaseEvaluator):
    builder_cls_name: Optional[str] = "ColorSwapBuilder"
    default_filename: str = "COLORSWAP"

    @staticmethod
    def _get_image_pair(sample):
        if "image_1" in sample and "image_2" in sample:
            return [sample["image_1"], sample["image_2"]]
        if "image_0" in sample and "image_1" in sample:
            return [sample["image_0"], sample["image_1"]]
        raise KeyError("ColorSwap sample is missing the expected image pair columns.")

    @staticmethod
    def _get_caption_pair(sample):
        if "caption_1" in sample and "caption_2" in sample:
            return [sample["caption_1"], sample["caption_2"]]
        if "caption_0" in sample and "caption_1" in sample:
            return [sample["caption_0"], sample["caption_1"]]
        raise KeyError("ColorSwap sample is missing the expected caption pair columns.")

    def _build_batch_samples(self, batch):
        sample_infos = []
        samples_images = []
        samples_text = []
        for sample in batch:
            sample_infos.append({"id": sample.get("id")})
            samples_images.append(self._get_image_pair(sample))
            samples_text.append(self._get_caption_pair(sample))
        return sample_infos, samples_images, samples_text

    def evaluate(self):
        existing_result = self._load_existing_result()
        if existing_result is not None:
            return existing_result

        dataloader = self._prepare_dataloader()
        result = defaultdict(list)

        for batch in dataloader:
            if self._is_preprocessed_batch(batch):
                batched_results = self.get_group_score_batched_preprocessed(batch)
            else:
                _, samples_images, samples_text = self._build_batch_samples(batch)
                batched_results = self.get_group_score_batched(samples_images, samples_text)

            for score in batched_results:
                result["text"].append(int(score["text_correct"]))
                result["image"].append(int(score["image_correct"]))
                result["group"].append(int(score["group_correct"]))

        average_result = {
            key: 100 * round(sum(values) / len(values), 5) if values else 0
            for key, values in result.items()
        }
        average_result["itt"] = average_result["text"]

        self._save_result(average_result)
        return average_result

    def _save_result(self, result: Dict, filename: str = "COLORSWAP") -> None:
        super()._save_result(result, filename)

    def get_summary_metrics(self, result: Dict) -> Dict[str, float]:
        return {"ColorSwap_ITT": result["itt"]}
