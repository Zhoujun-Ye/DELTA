import json
import os
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
from typing import TypeVar

import torch
from PIL import Image as PILImage
from PIL.Image import Image
from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedModel
from transformers import Trainer
from transformers.utils import logging

from src.common.registry import registry
from src.utils import dummy_collator

# from src.collators import BaseCollator

logger = logging.get_logger(__name__)

# CollatorType = Type[BaseCollator]
CollatorType = TypeVar("CollatorType", bound="BaseCollator")

__all__ = ["BaseTrainer", "BaseEvaluator"]


class BaseTrainer(Trainer):
    """
    A subclass of the Hugging Face `Trainer` class, designed to be extended with additional
    training logic and customized behavior.
    """
    pass


@dataclass
class BaseEvaluator:
    """
    A base class for model evaluation.

    Attributes:
        model (Optional[PreTrainedModel]): The model to be evaluated.
        data_collator (Optional[CollatorType]): Custom collator for preparing the data.
        builder_cls_name (Optional[str]): The builder class name for dataset creation.
        overwrite_results (Optional[bool]): Whether to overwrite existing results in the output directory.
        output_dir (Optional[Union[str, os.PathLike]]): Directory where evaluation results will be saved.
    """
    model: Optional[PreTrainedModel] = None
    data_collator: Optional[CollatorType] = None
    builder_cls_name: Optional[str] = None
    overwrite_results: Optional[bool] = False
    output_dir: Optional[Union[str, os.PathLike]] = None
    default_filename: Optional[str] = None
    batch_size: int = 128
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: Optional[int] = None
    use_feature_getters_for_similarity: bool = False

    def __post_init__(self):
        """
        Validates initialization and prepares the model for evaluation.
        """
        if not self.model:
            raise ValueError("A model must be provided for evaluation.")
        self.model.eval()
        logger.info(f"Preparing {self.builder_cls_name.replace('Builder', '').upper()} Benchmark")

    def _get_eval_dataset(self) -> Dataset:
        """
        Retrieves the dataset for evaluation based on the builder class.

        Returns:
            Dataset: The evaluation dataset.
        """
        if self.builder_cls_name is None:
            raise ValueError("A Builder must be provided for evaluation.")
        builder_cls = registry.get_builder_class(self.builder_cls_name)

        if not builder_cls:
            logger.error(f"Model class {self.builder_cls_name} not registered.")
            raise ValueError(f"Model class {self.builder_cls_name} not registered.")

        return builder_cls().build_dataset()

    def _prepare_dataloader(self):
        """
        Prepares a DataLoader for the evaluation dataset.

        Returns:
            DataLoader: The prepared DataLoader.
        """
        dataset = self._get_eval_dataset()
        collate_fn = self._collate_preprocessed_batch if self._supports_independent_encoding() else dummy_collator

        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.dataloader_num_workers,
            "pin_memory": self.dataloader_pin_memory,
        }
        dataloader_kwargs["collate_fn"] = collate_fn
        if self.dataloader_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.dataloader_persistent_workers
            if self.dataloader_prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = self.dataloader_prefetch_factor

        dataloader = tqdm(
            DataLoader(
                **dataloader_kwargs
            )
        )
        dataloader.set_description(f"Computing {self.builder_cls_name.replace('Builder', '')} scores")

        return dataloader

    def evaluate(self):
        """
        Abstract method for evaluation. Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def _get_result_path(self, filename: Optional[str] = None) -> str:
        if not self.output_dir:
            raise ValueError("Output directory is not set.")
        return os.path.join(self.output_dir, f"{filename or self.default_filename}.json")

    def _load_existing_result(self, filename: Optional[str] = None) -> Optional[Dict]:
        file_path = self._get_result_path(filename)
        if not os.path.isfile(file_path):
            return None

        logger.info(
            f"The file '{os.path.basename(file_path)}' already exists. "
            "Skipping evaluation and loading the existing result."
        )
        with open(file_path, "r") as f:
            return json.load(f)

    def get_summary_metrics(self, result: Dict) -> Dict[str, float]:
        return {}

    def _build_batch_samples(
        self,
        batch: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[List[Image]], List[List[str]]]:
        raise NotImplementedError("Subclasses must implement `_build_batch_samples` for evaluation batching.")

    def _collate_preprocessed_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_infos, samples_images, samples_texts = self._build_batch_samples(batch)

        all_images: List[Image] = []
        all_texts: List[str] = []
        img_spans: List[tuple[int, int]] = []
        txt_spans: List[tuple[int, int]] = []

        for images in samples_images:
            start = len(all_images)
            all_images.extend(images)
            img_spans.append((start, len(all_images)))

        for texts in samples_texts:
            start = len(all_texts)
            all_texts.extend(texts)
            txt_spans.append((start, len(all_texts)))

        image_features = {}
        text_features = {}
        if all_images:
            image_features = dict(self.data_collator.prepare_images(images=all_images))
        if all_texts:
            text_features = dict(self.data_collator.prepare_texts(texts=all_texts))

        return {
            "sample_infos": sample_infos,
            "img_spans": img_spans,
            "txt_spans": txt_spans,
            "image_features": image_features,
            "text_features": text_features,
        }

    @staticmethod
    def _is_preprocessed_batch(batch: Any) -> bool:
        return isinstance(batch, dict) and {
            "sample_infos",
            "img_spans",
            "txt_spans",
            "image_features",
            "text_features",
        }.issubset(batch.keys())

    @staticmethod
    def _spans_to_slices(spans: List[tuple[int, int]]) -> List[slice]:
        return [slice(start, stop) for start, stop in spans]

    def _prepare_inputs(
            self,
            images: Optional[List[Image]] = None,
            text: Optional[List[str]] = None,
    ) -> BatchEncoding:
        """
        Prepares inputs for the model by collating and moving to the correct device.

        Args:
            images (List[Image]): List of image inputs.
            text (List[str]): List of text inputs.

        Returns:
            BatchEncoding: The prepared inputs.
        """
        inputs: BatchEncoding = self.data_collator(
            {"images": images, "text": text}
        )
        return inputs.to(self.model.device)

    @staticmethod
    def _move_features_to_device(features: BatchEncoding, device: torch.device) -> BatchEncoding:
        return BatchEncoding(
            {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in features.items()
            }
        )

    @staticmethod
    def _get_text_feature_kwargs(features: BatchEncoding) -> Dict[str, Any]:
        text_feature_keys = (
            "input_ids",
            "attention_mask",
            "position_ids",
        )
        return {
            feature_name: features[feature_name]
            for feature_name in text_feature_keys
            if feature_name in features and features[feature_name] is not None
        }

    def _supports_independent_encoding(self) -> bool:
        return bool(
            self.use_feature_getters_for_similarity
            and hasattr(self.model, "get_image_features")
            and hasattr(self.model, "get_text_features")
            and hasattr(self.data_collator, "prepare_images")
            and hasattr(self.data_collator, "prepare_texts")
        )

    def _prepare_image_features(
        self,
        images: List[Image],
    ) -> BatchEncoding:
        image_features = self.data_collator.prepare_images(images=images)
        return self._move_features_to_device(image_features, self.model.device)

    def _prepare_text_features(
        self,
        text: List[str],
    ) -> BatchEncoding:
        text_features = self.data_collator.prepare_texts(texts=text)
        return self._move_features_to_device(text_features, self.model.device)

    @staticmethod
    def _dummy_text_inputs(num_images: int) -> List[str]:
        return [""] * max(1, num_images)

    @staticmethod
    def _dummy_image_inputs(num_texts: int) -> List[Image]:
        return [PILImage.new("RGB", (1, 1), color=0) for _ in range(max(1, num_texts))]

    @torch.inference_mode()
    def _encode_images(
        self,
        images: List[Image],
    ) -> Tensor:
        if self._supports_independent_encoding():
            image_features = self._prepare_image_features(images=images)
            return self.model.get_image_features(
                pixel_values=image_features.get("pixel_values"),
                output_attentions=False,
                output_hidden_states=False,
            )

        inputs = self._prepare_inputs(
            images=images,
            text=self._dummy_text_inputs(len(images)),
        )
        outputs = self.model(**inputs)
        return outputs.image_embeds

    @torch.inference_mode()
    def _encode_texts(
        self,
        text: List[str],
    ) -> Tensor:
        if self._supports_independent_encoding():
            text_features = self._prepare_text_features(text=text)
            return self.model.get_text_features(
                **self._get_text_feature_kwargs(text_features),
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        inputs = self._prepare_inputs(
            images=self._dummy_image_inputs(len(text)),
            text=text,
        )
        outputs = self.model(**inputs)
        return outputs.text_embeds

    @staticmethod
    def _score_sample_groups(
        image_embeds: Tensor,
        text_embeds: Tensor,
        img_slices: List[slice],
        txt_slices: List[slice],
    ) -> List[Tensor]:
        results: List[Optional[Tensor]] = [None] * len(img_slices)
        grouped_indices: Dict[tuple[int, int], List[int]] = defaultdict(list)

        for sample_index, (img_slice, txt_slice) in enumerate(zip(img_slices, txt_slices)):
            grouped_indices[(img_slice.stop - img_slice.start, txt_slice.stop - txt_slice.start)].append(sample_index)

        for (num_images, num_texts), sample_indices in grouped_indices.items():
            if len(sample_indices) == 1:
                sample_index = sample_indices[0]
                results[sample_index] = torch.matmul(
                    image_embeds[img_slices[sample_index]],
                    text_embeds[txt_slices[sample_index]].t(),
                )
                continue

            image_batch = torch.stack(
                [image_embeds[img_slices[sample_index]] for sample_index in sample_indices],
                dim=0,
            )
            text_batch = torch.stack(
                [text_embeds[txt_slices[sample_index]] for sample_index in sample_indices],
                dim=0,
            )

            if num_images == 1:
                batch_scores = torch.einsum("bd,bld->bl", image_batch.squeeze(1), text_batch)
                for batch_index, sample_index in enumerate(sample_indices):
                    results[sample_index] = batch_scores[batch_index].unsqueeze(0)
                continue

            if num_texts == 1:
                batch_scores = torch.einsum("bkd,bd->bk", image_batch, text_batch.squeeze(1))
                for batch_index, sample_index in enumerate(sample_indices):
                    results[sample_index] = batch_scores[batch_index].unsqueeze(-1)
                continue

            if num_images == 2 and num_texts == 2:
                batch_scores = torch.einsum("bid,bjd->bij", image_batch, text_batch)
                for batch_index, sample_index in enumerate(sample_indices):
                    results[sample_index] = batch_scores[batch_index]
                continue

            batch_scores = torch.matmul(image_batch, text_batch.transpose(-1, -2))
            for batch_index, sample_index in enumerate(sample_indices):
                results[sample_index] = batch_scores[batch_index]

        if any(score is None for score in results):
            raise RuntimeError("Failed to compute all local similarity score blocks.")
        return results  # type: ignore[return-value]

    @torch.inference_mode()
    def _compute_similarity(
            self,
            images: List[Image],
            text: List[str],
            return_outputs: bool = False
    ) -> Union[Tensor | tuple[Tensor, Any]]:
        """
        Computes similarity scores between images and text.

        Args:
            images (List[Image]): List of image inputs.
            text (List[str]): List of text inputs.
            return_outputs (bool): Whether to return the output of the model (default: 'False').

        Returns:
            torch.Tensor: Similarity scores between image and text embeddings.
            outputs: Output of the model.
        """
        inputs = self._prepare_inputs(images, text)
        if self.use_feature_getters_for_similarity:
            image_embeds = self.model.get_image_features(
                pixel_values=inputs.get("pixel_values"),
                output_attentions=False,
                output_hidden_states=False,
            )
            text_embeds = self.model.get_text_features(
                **self._get_text_feature_kwargs(inputs),
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            outputs = SimpleNamespace(image_embeds=image_embeds, text_embeds=text_embeds)
        else:
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
        if not return_outputs:
            return torch.matmul(image_embeds, text_embeds.t().to(image_embeds.device))
        else:
            return torch.matmul(image_embeds, text_embeds.t().to(image_embeds.device)), outputs

    @torch.inference_mode()
    def get_image_to_text_score(
            self,
            images: List[Image],
            text: List[str],
            return_tot: bool = False,
            return_raw_scores: bool = False
    ) -> tuple[int, Any, int, list[int | float | bool]] | tuple[int, Any] | int | tuple[int, int]:
        """
        Determines if the first text matches the single image.

        Args:
            images (List[Image]): A single image wrapped in a list.
            text (List[str]): Multiple text inputs.
            return_tot (bool): if return tot score (default: 'False').
            return_raw_scores (bool): if return raw similarity scores (default: 'False').

        Returns:
            int: 1 if the first text matches the image, otherwise 0.
            or
            tuple: (score, similarity_scores, text_similarity_scores)
        """
        if len(images) != 1:
            raise ValueError("`get_image_to_text_score` requires exactly one image.")
        if len(text) <= 1:
            raise ValueError("`get_image_to_text_score` requires more than one text input.")

        similarity_scores, outputs = self._compute_similarity(images, text, return_outputs=True)

        if return_raw_scores:
            if not return_tot:
                return int(similarity_scores.argmax() == 0), similarity_scores
            else:
                i0_c0 = similarity_scores[0, 0].item()
                i0_c1 = similarity_scores[0, 1].item()
                i0_c2 = similarity_scores[0, 2].item()

                image_correct = i0_c0 > i0_c2 and i0_c1 > i0_c2

                text_similarity_scores = torch.matmul(outputs.text_embeds, outputs.text_embeds.t())
                c0_c1 = text_similarity_scores[0, 1].item()
                c0_c2 = text_similarity_scores[0, 2].item()
                c1_c2 = text_similarity_scores[1, 2].item()

                text_correct = c0_c1 > c0_c2 and c0_c1 > c1_c2

                return int(image_correct), similarity_scores, int(text_correct), [c0_c1, c0_c2, c1_c2]
        else:
            if not return_tot:
                return int(similarity_scores.argmax() == 0)
            else:
                i0_c0 = similarity_scores[0, 0].item()
                i0_c1 = similarity_scores[0, 1].item()
                i0_c2 = similarity_scores[0, 2].item()

                image_correct = i0_c0 > i0_c2 and i0_c1 > i0_c2

                text_similarity_scores = torch.matmul(outputs.text_embeds, outputs.text_embeds.t())

                c0_c1 = text_similarity_scores[0, 1].item()
                c0_c2 = text_similarity_scores[0, 2].item()
                c1_c2 = text_similarity_scores[1, 2].item()

                text_correct = c0_c1 > c0_c2 and c0_c1 > c1_c2

                return int(image_correct), int(text_correct)

    @torch.inference_mode()
    def get_text_to_image_score(self, images: List[Image], text: List[str]) -> int:
        """
        Determines if the first image matches the single text.

        Args:
            images (List[Image]): Multiple images.
            text (List[str]): A single text input wrapped in a list.

        Returns:
            int: 1 if the first image matches the text, otherwise 0.
        """
        if len(images) <= 1:
            raise ValueError("`get_text_to_image_score` requires more than one image.")
        if len(text) != 1:
            raise ValueError("`get_text_to_image_score` requires exactly one text input.")

        similarity_scores = self._compute_similarity(images, text)
        return int(similarity_scores.argmax() == 0)

    @torch.inference_mode()
    def get_group_score(self, images: List[Image], text: List[str]) -> Dict[str, bool]:
        """
        Computes correctness scores for a group of 2 images and 2 text inputs.

        Args:
            images (List[Image]): A list of exactly 2 images.
            text (List[str]): A list of exactly 2 text inputs.

        Returns:
            Dict[str, bool]: A dictionary containing correctness scores.
        """
        if len(images) != 2:
            raise ValueError("`get_group_score` requires exactly 2 images.")
        if len(text) != 2:
            raise ValueError("`get_group_score` requires exactly 2 text inputs.")

        similarity_scores = self._compute_similarity(images, text)

        # Extract scores
        c0_i0 = similarity_scores[0, 0].item()
        c0_i1 = similarity_scores[0, 1].item()
        c1_i0 = similarity_scores[1, 0].item()
        c1_i1 = similarity_scores[1, 1].item()

        # `text_correct` matches the ColorSwap/Winoground convention:
        # given an image, the aligned caption should score higher.
        text_correct = c0_i0 > c0_i1 and c1_i1 > c1_i0
        image_correct = c0_i0 > c1_i0 and c1_i1 > c0_i1
        group_correct = text_correct and image_correct

        return {
            "text_correct": text_correct,
            "image_correct": image_correct,
            "group_correct": group_correct,
        }

    @torch.inference_mode()
    def _compute_similarity_batched(
            self,
            samples_images: List[List[Image]],
            samples_texts: List[List[str]],
            return_outputs: bool = False
    ) -> Union[List[Tensor], tuple[List[Tensor], List[Any]]]:
        if self._supports_independent_encoding():
            return self._compute_similarity_batched_fast(
                samples_images=samples_images,
                samples_texts=samples_texts,
                return_outputs=return_outputs,
            )
        return self._compute_similarity_batched_legacy(
            samples_images=samples_images,
            samples_texts=samples_texts,
            return_outputs=return_outputs,
        )

    @torch.inference_mode()
    def _compute_similarity_batched_fast(
            self,
            samples_images: List[List[Image]],
            samples_texts: List[List[str]],
            return_outputs: bool = False
    ) -> Union[List[Tensor], tuple[List[Tensor], List[Any]]]:
        if not self._supports_independent_encoding():
            return self._compute_similarity_batched_legacy(
                samples_images=samples_images,
                samples_texts=samples_texts,
                return_outputs=return_outputs,
            )

        all_images = []
        all_texts = []
        img_slices = []
        txt_slices = []

        for images in samples_images:
            img_slices.append(slice(len(all_images), len(all_images) + len(images)))
            all_images.extend(images)

        for texts in samples_texts:
            txt_slices.append(slice(len(all_texts), len(all_texts) + len(texts)))
            all_texts.extend(texts)

        if len(all_images) == 0 or len(all_texts) == 0:
            if not return_outputs:
                return []
            else:
                return [], []

        image_embeds = self._encode_images(all_images)
        text_embeds = self._encode_texts(all_texts)
        results = self._score_sample_groups(image_embeds, text_embeds, img_slices, txt_slices)

        if not return_outputs:
            return results

        outputs_list = [
            SimpleNamespace(text_embeds=text_embeds[txt_slice])
            for txt_slice in txt_slices
        ]
        return results, outputs_list

    @torch.inference_mode()
    def _compute_similarity_batched_legacy(
            self,
            samples_images: List[List[Image]],
            samples_texts: List[List[str]],
            return_outputs: bool = False
    ) -> Union[List[Tensor], tuple[List[Tensor], List[Any]]]:
        all_images = []
        all_texts = []
        img_slices = []
        txt_slices = []

        for images in samples_images:
            img_slices.append(slice(len(all_images), len(all_images) + len(images)))
            all_images.extend(images)

        for texts in samples_texts:
            txt_slices.append(slice(len(all_texts), len(all_texts) + len(texts)))
            all_texts.extend(texts)

        if len(all_images) == 0 or len(all_texts) == 0:
            if not return_outputs:
                return []
            return [], []

        if return_outputs:
            similarity_scores, outputs = self._compute_similarity(all_images, all_texts, return_outputs=True)
            results = []
            text_embeds = outputs.text_embeds
            outputs_list = []

            for islice, tslice in zip(img_slices, txt_slices):
                sample_sim = similarity_scores[islice, tslice]
                sample_text_embeds = text_embeds[tslice]
                results.append(sample_sim)
                outputs_list.append(SimpleNamespace(text_embeds=sample_text_embeds))

            return results, outputs_list

        similarity_scores = self._compute_similarity(all_images, all_texts, return_outputs=False)
        results = []
        for islice, tslice in zip(img_slices, txt_slices):
            results.append(similarity_scores[islice, tslice])

        return results

    @torch.inference_mode()
    def _compute_similarity_batched_preprocessed(
        self,
        batch: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Union[List[Tensor], tuple[List[Tensor], List[Any]]]:
        if not self._is_preprocessed_batch(batch):
            raise TypeError("Expected a preprocessed evaluation batch.")
        if not self._supports_independent_encoding():
            raise RuntimeError("Preprocessed evaluation batches require independent image/text encoding support.")

        img_slices = self._spans_to_slices(batch["img_spans"])
        txt_slices = self._spans_to_slices(batch["txt_spans"])
        if not img_slices or not txt_slices:
            if not return_outputs:
                return []
            return [], []

        image_features = self._move_features_to_device(BatchEncoding(batch["image_features"]), self.model.device)
        text_features = self._move_features_to_device(BatchEncoding(batch["text_features"]), self.model.device)

        image_embeds = self.model.get_image_features(
            pixel_values=image_features.get("pixel_values"),
            output_attentions=False,
            output_hidden_states=False,
        )
        text_embeds = self.model.get_text_features(
            **self._get_text_feature_kwargs(text_features),
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        results = self._score_sample_groups(image_embeds, text_embeds, img_slices, txt_slices)

        if not return_outputs:
            return results

        outputs_list = [
            SimpleNamespace(text_embeds=text_embeds[txt_slice])
            for txt_slice in txt_slices
        ]
        return results, outputs_list

    @staticmethod
    def _finalize_image_to_text_scores(
        batched_sim: List[Tensor],
        batched_outputs: Optional[List[Any]] = None,
        return_tot: bool = False,
        return_raw_scores: bool = False,
    ) -> List[Any]:
        final_results = []

        if not return_tot:
            for similarity_scores in batched_sim:
                prediction = int(similarity_scores.argmax() == 0)
                if return_raw_scores:
                    final_results.append((prediction, similarity_scores))
                else:
                    final_results.append(prediction)
            return final_results

        if batched_outputs is None:
            raise ValueError("`batched_outputs` is required when `return_tot=True`.")

        for similarity_scores, outputs in zip(batched_sim, batched_outputs):
            i0_c0 = similarity_scores[0, 0].item()
            i0_c1 = similarity_scores[0, 1].item()
            i0_c2 = similarity_scores[0, 2].item()

            image_correct = i0_c0 > i0_c2 and i0_c1 > i0_c2

            text_similarity_scores = torch.matmul(outputs.text_embeds, outputs.text_embeds.t())
            c0_c1 = text_similarity_scores[0, 1].item()
            c0_c2 = text_similarity_scores[0, 2].item()
            c1_c2 = text_similarity_scores[1, 2].item()

            text_correct = c0_c1 > c0_c2 and c0_c1 > c1_c2

            if return_raw_scores:
                final_results.append(
                    (int(image_correct), similarity_scores, int(text_correct), [c0_c1, c0_c2, c1_c2])
                )
            else:
                final_results.append((int(image_correct), int(text_correct)))

        return final_results

    @staticmethod
    def _finalize_text_to_image_scores(
        batched_results: List[Tensor],
        return_raw_scores: bool = False,
    ) -> List[Any]:
        final_results = []
        for similarity_scores in batched_results:
            prediction = int(similarity_scores[:, 0].argmax().item() == 0)
            if return_raw_scores:
                final_results.append((prediction, similarity_scores))
            else:
                final_results.append(prediction)
        return final_results

    @staticmethod
    def _finalize_group_scores(
        batched_results: List[Tensor],
    ) -> List[Dict[str, bool]]:
        final_results = []
        for similarity_scores in batched_results:
            c0_i0 = similarity_scores[0, 0].item()
            c0_i1 = similarity_scores[0, 1].item()
            c1_i0 = similarity_scores[1, 0].item()
            c1_i1 = similarity_scores[1, 1].item()

            text_correct = c0_i0 > c0_i1 and c1_i1 > c1_i0
            image_correct = c0_i0 > c1_i0 and c1_i1 > c0_i1
            group_correct = text_correct and image_correct

            final_results.append({
                "text_correct": text_correct,
                "image_correct": image_correct,
                "group_correct": group_correct,
            })
        return final_results

    @torch.inference_mode()
    def get_image_to_text_score_batched(
            self,
            samples_images: List[List[Image]],
            samples_texts: List[List[str]],
            return_tot: bool = False,
            return_raw_scores: bool = False
    ):
        if not return_tot:
            batched_sim = self._compute_similarity_batched(
                samples_images,
                samples_texts,
                return_outputs=False,
            )
            return self._finalize_image_to_text_scores(
                batched_sim,
                return_tot=False,
                return_raw_scores=return_raw_scores,
            )

        batched_sim, batched_outputs = self._compute_similarity_batched(
            samples_images,
            samples_texts,
            return_outputs=True,
        )
        return self._finalize_image_to_text_scores(
            batched_sim,
            batched_outputs=batched_outputs,
            return_tot=True,
            return_raw_scores=return_raw_scores,
        )

    @torch.inference_mode()
    def get_image_to_text_score_batched_fast(
            self,
            samples_images: List[List[Image]],
            samples_texts: List[List[str]],
            return_tot: bool = False,
            return_raw_scores: bool = False
    ):
        if not return_tot:
            batched_sim = self._compute_similarity_batched_fast(
                samples_images,
                samples_texts,
                return_outputs=False,
            )
            return self._finalize_image_to_text_scores(
                batched_sim,
                return_tot=False,
                return_raw_scores=return_raw_scores,
            )

        batched_sim, batched_outputs = self._compute_similarity_batched_fast(
            samples_images,
            samples_texts,
            return_outputs=True,
        )
        return self._finalize_image_to_text_scores(
            batched_sim,
            batched_outputs=batched_outputs,
            return_tot=True,
            return_raw_scores=return_raw_scores,
        )

    @torch.inference_mode()
    def get_image_to_text_score_batched_preprocessed(
        self,
        batch: Dict[str, Any],
        return_tot: bool = False,
        return_raw_scores: bool = False,
    ):
        if not return_tot:
            batched_sim = self._compute_similarity_batched_preprocessed(batch, return_outputs=False)
            return self._finalize_image_to_text_scores(
                batched_sim,
                return_tot=False,
                return_raw_scores=return_raw_scores,
            )

        batched_sim, batched_outputs = self._compute_similarity_batched_preprocessed(batch, return_outputs=True)
        return self._finalize_image_to_text_scores(
            batched_sim,
            batched_outputs=batched_outputs,
            return_tot=True,
            return_raw_scores=return_raw_scores,
        )

    @torch.inference_mode()
    def get_text_to_image_score_batched(
        self,
        samples_images: List[List[Image]],
        samples_text: List[List[str]],
        return_raw_scores: bool = False,
    ):
        batched_results = self._compute_similarity_batched(
            samples_images,
            samples_text,
            return_outputs=False,
        )
        return self._finalize_text_to_image_scores(
            batched_results,
            return_raw_scores=return_raw_scores,
        )

    @torch.inference_mode()
    def get_text_to_image_score_batched_fast(
        self,
        samples_images: List[List[Image]],
        samples_text: List[List[str]],
        return_raw_scores: bool = False,
    ):
        batched_results = self._compute_similarity_batched_fast(
            samples_images,
            samples_text,
            return_outputs=False,
        )
        return self._finalize_text_to_image_scores(
            batched_results,
            return_raw_scores=return_raw_scores,
        )

    @torch.inference_mode()
    def get_text_to_image_score_batched_preprocessed(
        self,
        batch: Dict[str, Any],
        return_raw_scores: bool = False,
    ):
        batched_results = self._compute_similarity_batched_preprocessed(batch, return_outputs=False)
        return self._finalize_text_to_image_scores(
            batched_results,
            return_raw_scores=return_raw_scores,
        )

    @torch.inference_mode()
    def get_group_score_batched(
        self,
        samples_images: List[List[Image]],
        samples_text: List[List[str]],
    ):
        batched_results = self._compute_similarity_batched(
            samples_images,
            samples_text,
            return_outputs=False,
        )
        return self._finalize_group_scores(batched_results)

    @torch.inference_mode()
    def get_group_score_batched_preprocessed(
        self,
        batch: Dict[str, Any],
    ):
        batched_results = self._compute_similarity_batched_preprocessed(batch, return_outputs=False)
        return self._finalize_group_scores(batched_results)

    def _save_result(self, result: Dict, filename: Optional[str] = None) -> None:
        """
        Saves evaluation results to a JSON file.

        Args:
            result (Dict): The evaluation results.
            filename (Optional[str]): The name of the result file.
        """
        if not self.output_dir:
            raise ValueError("Output directory is not set.")

        os.makedirs(self.output_dir, exist_ok=True)
        file_path = self._get_result_path(filename)

        try:
            with open(file_path, "w") as f:
                json.dump(result, f, indent=2)
        except IOError as e:
            raise IOError(f"Failed to save results to {file_path}: {e}")
