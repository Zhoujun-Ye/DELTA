from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL.Image import Image
from transformers import BatchEncoding
from transformers.utils import add_end_docstrings
from transformers.utils import logging

from src.common.mixin import (
    NegativeTextMining,
    VisionBindingExample,
    VISION_BINDING_NEGATIVE_TYPE_TO_ID,
)
from src.common.registry import registry
from .base import BASE_COLLATOR_DOCSTRING, BaseCollator

logger = logging.get_logger(__name__)

__all__ = [
    "ImageCollator",
    "COCOImageCollator",
]


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
@registry.register_collator("ImageCollator")
class ImageCollator(BaseCollator):
    """
    Collator for processing input data containing 'images' and 'text' keys.
    """

    def _validate_and_process_images(self, images: Any) -> List[Image]:
        if not isinstance(images, list):
            raise TypeError(f"Expected `list` for key 'images', but got {type(images)}.")
        processed_images = []
        for img in images:
            if not isinstance(img, Image):
                raise TypeError(f"Expected elements of 'images' to be `PIL.Image.Image`, but got {type(img)}.")
            processed_images.append(img.convert("RGB"))
        return processed_images

    @staticmethod
    def _validate_and_process_texts(texts: Any) -> List[str]:
        if not isinstance(texts, list):
            raise TypeError(f"Expected `list` for key 'text', but got {type(texts)}.")
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"Expected elements of 'text' to be `str`, but got {type(text)}.")
        return texts

    def prepare_images(
        self,
        images: List[Image],
    ) -> BatchEncoding:
        processed_images = self._validate_and_process_images(images)
        image_features = self.processor.image_processor(
            images=processed_images,
            return_tensors=self.return_tensors,
        )
        return BatchEncoding(dict(image_features))

    def prepare_texts(self, texts: List[str]) -> BatchEncoding:
        processed_texts = self._validate_and_process_texts(texts)
        pad_to_multiple_of = self.pad_to_multiple_of
        if pad_to_multiple_of is not None and self.truncation:
            effective_max_length = self.max_length
            if effective_max_length is None:
                tokenizer_max_length = getattr(self.processor.tokenizer, "model_max_length", None)
                if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < 1_000_000:
                    effective_max_length = tokenizer_max_length
            if effective_max_length is not None and effective_max_length % pad_to_multiple_of != 0:
                pad_to_multiple_of = None

        tokenizer_kwargs = {
            "padding": self.padding,
            "truncation": self.truncation,
            "max_length": self.max_length,
        }
        if pad_to_multiple_of is not None:
            tokenizer_kwargs["pad_to_multiple_of"] = pad_to_multiple_of
        text_features = self.processor.tokenizer(
            processed_texts,
            return_tensors=self.return_tensors,
            **tokenizer_kwargs,
        )
        return BatchEncoding(
            {
                "input_ids": text_features["input_ids"],
                "attention_mask": text_features["attention_mask"],
            }
        )

    def prepare_texts_with_offsets(self, texts: List[str]) -> BatchEncoding:
        processed_texts = self._validate_and_process_texts(texts)
        if not getattr(self.processor.tokenizer, "is_fast", False):
            raise ValueError("Vision binding requires a fast tokenizer with offset mappings.")

        pad_to_multiple_of = self.pad_to_multiple_of
        if pad_to_multiple_of is not None and self.truncation:
            effective_max_length = self.max_length
            if effective_max_length is None:
                tokenizer_max_length = getattr(self.processor.tokenizer, "model_max_length", None)
                if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < 1_000_000:
                    effective_max_length = tokenizer_max_length
            if effective_max_length is not None and effective_max_length % pad_to_multiple_of != 0:
                pad_to_multiple_of = None

        tokenizer_kwargs = {
            "padding": self.padding,
            "truncation": self.truncation,
            "max_length": self.max_length,
            "return_offsets_mapping": True,
        }
        if pad_to_multiple_of is not None:
            tokenizer_kwargs["pad_to_multiple_of"] = pad_to_multiple_of
        text_features = self.processor.tokenizer(
            processed_texts,
            return_tensors=self.return_tensors,
            **tokenizer_kwargs,
        )
        return BatchEncoding(
            {
                "input_ids": text_features["input_ids"],
                "attention_mask": text_features["attention_mask"],
                "offset_mapping": text_features["offset_mapping"],
            }
        )

    def prepare_inputs(
        self,
        images: List[Image],
        text: List[str],
    ) -> BatchEncoding:
        image_features = self.prepare_images(images=images)
        text_features = self.prepare_texts(texts=text)

        return BatchEncoding(
            {
                **dict(image_features),
                **dict(text_features),
            }
        )

    def __call__(self, inputs: Dict[str, Any]) -> BatchEncoding:
        allowed_keys = {"images", "text"}
        if not set(inputs.keys()).issubset(allowed_keys) or "images" not in inputs or "text" not in inputs:
            raise ValueError(
                "Input dictionary must contain `images` and `text`. "
                f"Found: {inputs.keys()}"
            )

        return self.prepare_inputs(
            images=inputs["images"],
            text=inputs["text"],
        )


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
@registry.register_collator("COCOImageCollator")
class COCOImageCollator(ImageCollator, NegativeTextMining):
    num_hard_negs: Optional[int] = 3
    use_negative_caption: Optional[bool] = False
    enable_text_swap: Optional[bool] = True
    enable_vision_binding: Optional[bool] = False

    def __post_init__(self):
        NegativeTextMining.__init__(self)

    @staticmethod
    def _build_char_span_token_mask(
        offset_mapping: torch.Tensor,
        char_span: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        seq_len = offset_mapping.size(0)
        mask = torch.zeros(seq_len, dtype=torch.float)
        if char_span is None:
            return mask

        char_start, char_end = char_span
        if char_end <= char_start:
            return mask

        token_starts = offset_mapping[:, 0]
        token_ends = offset_mapping[:, 1]
        valid_tokens = token_ends > token_starts
        overlaps = valid_tokens & (token_ends > char_start) & (token_starts < char_end)
        mask[overlaps] = 1.0
        return mask

    def __call__(self, inputs: List[Dict[str, Any]]) -> BatchEncoding:
        if self.enable_vision_binding and not self.use_negative_caption:
            raise ValueError("Vision binding requires `use_negative_caption=True`.")
        if self.enable_vision_binding and self.num_hard_negs != 3:
            raise ValueError("Vision binding requires typed negatives with `num_hard_negs=3`.")

        valid_images = []
        valid_texts = []
        attribute_neg_texts = []
        object_neg_texts = []
        relation_neg_texts = []
        vision_binding_neg_texts = []
        text_swap_texts = []
        text_swap_valid_mask = []
        vision_binding_examples: List[Optional[VisionBindingExample]] = []

        for input_dict in inputs:
            if "images" not in input_dict or "sentences" not in input_dict:
                raise ValueError("Invalid input dictionary. Please check the input dictionary.")

            image = input_dict["images"].convert("RGB")
            captions = input_dict["sentences"][:]
            text = str(self.rng.choice(captions, size=1, replace=False)[0])

            valid_images.append(image)
            valid_texts.append(text)

        relation_examples = [None] * len(valid_texts)
        if self.use_negative_caption or self.enable_text_swap:
            self._populate_relation_example_cache(valid_texts)
            relation_examples = [self.extract_relation_example(text) for text in valid_texts]

        noun_chunk_pool = []
        if self.use_negative_caption:
            noun_chunk_pool = self.collect_noun_chunk_candidates(valid_texts)

        def _resolve_negative(text: str, candidate_negative: str) -> str:
            if candidate_negative and candidate_negative != text:
                return candidate_negative
            return self.fallback_negative_caption(text, candidate_texts=valid_texts)

        for text, relation_example in zip(valid_texts, relation_examples):
            if self.use_negative_caption:
                if self.num_hard_negs != 3:
                    raise ValueError("Typed negative ITC requires `num_hard_negs=3`.")
                vision_binding_example = self.build_vision_binding_example(text) if self.enable_vision_binding else None
                attribute_negative = self.build_attribute_replace_negative(text)
                attribute_negative = _resolve_negative(text, attribute_negative)
                attribute_neg_texts.append(
                    attribute_negative
                )
                if self.enable_vision_binding:
                    vision_binding_examples.append(vision_binding_example)
                    vision_binding_neg_texts.append(
                        vision_binding_example.negative_text
                        if vision_binding_example is not None
                        else _resolve_negative(text, "")
                    )
                object_neg_texts.append(
                    _resolve_negative(
                        text,
                        self.build_object_replace_negative(text, noun_chunk_pool=noun_chunk_pool, example=relation_example),
                    )
                )
                relation_neg_texts.append(
                    _resolve_negative(text, self.build_relation_replace_negative(text, example=relation_example))
                )

            swap_negative = None
            if relation_example is not None:
                candidate_negative = self.build_relation_negative(relation_example)
                if candidate_negative and candidate_negative != text:
                    swap_negative = candidate_negative

            if swap_negative is None:
                text_swap_valid_mask.append(0.0)
                text_swap_texts.append(text)
            else:
                text_swap_valid_mask.append(1.0)
                text_swap_texts.append(swap_negative)

        processor_texts = list(valid_texts)
        if self.use_negative_caption:
            processor_texts.extend(attribute_neg_texts)
            processor_texts.extend(object_neg_texts)
            processor_texts.extend(relation_neg_texts)
            if self.enable_vision_binding:
                processor_texts.extend(vision_binding_neg_texts)
        logger.debug(f"images: {len(valid_images)}")
        logger.debug(f"texts: {len(processor_texts)}")

        image_features = self.prepare_images(images=valid_images)
        main_text_features = (
            self.prepare_texts_with_offsets(processor_texts)
            if self.enable_vision_binding
            else self.prepare_texts(processor_texts)
        )
        main_text_feature_dict = dict(main_text_features)
        main_text_offset_mapping = main_text_feature_dict.pop("offset_mapping", None)

        batch_output = BatchEncoding(
            {
                **dict(image_features),
                **main_text_feature_dict,
            }
        )

        if self.enable_vision_binding:
            positive_count = len(valid_texts)
            seq_len = main_text_feature_dict["input_ids"].size(1)
            object_token_masks = torch.zeros((positive_count, seq_len), dtype=torch.float)
            pos_attribute_token_masks = torch.zeros((positive_count, seq_len), dtype=torch.float)
            neg_attribute_token_masks = torch.zeros((positive_count, seq_len), dtype=torch.float)
            raw_valid_mask = torch.zeros(positive_count, dtype=torch.float)
            valid_mask = torch.zeros(positive_count, dtype=torch.float)
            negative_type_ids = torch.zeros(positive_count, dtype=torch.long)
            neg_text_block_start = positive_count * 4
            neg_text_indices = torch.arange(neg_text_block_start, neg_text_block_start + positive_count, dtype=torch.long)

            if main_text_offset_mapping is None:
                raise ValueError("Vision binding requires offset mappings for the main text batch.")

            for sample_idx, example in enumerate(vision_binding_examples):
                if example is None:
                    continue

                raw_valid_mask[sample_idx] = 1.0
                negative_type_ids[sample_idx] = VISION_BINDING_NEGATIVE_TYPE_TO_ID.get(example.negative_type, 0)

                neg_row_idx = int(neg_text_indices[sample_idx].item())
                pos_offset_mapping = main_text_offset_mapping[sample_idx]
                neg_offset_mapping = main_text_offset_mapping[neg_row_idx]

                object_mask = self._build_char_span_token_mask(pos_offset_mapping, example.object_char_span)
                pos_attribute_mask = self._build_char_span_token_mask(
                    pos_offset_mapping,
                    example.pos_attribute_char_span,
                )
                neg_attribute_mask = self._build_char_span_token_mask(
                    neg_offset_mapping,
                    example.neg_attribute_char_span,
                )

                if not (torch.any(object_mask > 0) and torch.any(pos_attribute_mask > 0) and torch.any(neg_attribute_mask > 0)):
                    continue

                object_token_masks[sample_idx] = object_mask
                pos_attribute_token_masks[sample_idx] = pos_attribute_mask
                neg_attribute_token_masks[sample_idx] = neg_attribute_mask
                valid_mask[sample_idx] = 1.0

            batch_output["vision_binding_raw_valid_mask"] = raw_valid_mask
            batch_output["vision_binding_valid_mask"] = valid_mask
            batch_output["vision_binding_negative_type"] = negative_type_ids
            batch_output["vision_binding_neg_text_indices"] = neg_text_indices
            batch_output["vision_binding_object_token_mask"] = object_token_masks
            batch_output["vision_binding_pos_attribute_token_mask"] = pos_attribute_token_masks
            batch_output["vision_binding_neg_attribute_token_mask"] = neg_attribute_token_masks

        if self.enable_text_swap:
            text_swap_features = self.prepare_texts(text_swap_texts)
            merged_text_swap_valid_mask = torch.tensor(text_swap_valid_mask, dtype=torch.float)
            for feature_name, feature_value in text_swap_features.items():
                batch_output[f"text_swap_{feature_name}"] = feature_value
            batch_output["text_swap_valid_mask"] = merged_text_swap_valid_mask

        return batch_output
