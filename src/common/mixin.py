from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span, Token

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj"}
OBJECT_DEPS = {"dobj", "obj", "iobj", "attr", "oprd", "pobj"}
ATTRIBUTE_CONFUSION_MAP = {
    "red": ["blue", "green", "yellow", "black", "white"],
    "blue": ["red", "green", "yellow", "black", "white"],
    "green": ["red", "blue", "yellow", "black", "white"],
    "yellow": ["red", "blue", "green", "black", "white"],
    "black": ["white", "red", "blue", "yellow"],
    "white": ["black", "red", "blue", "green"],
    "small": ["large", "big", "tall"],
    "large": ["small", "little", "short"],
    "big": ["small", "little", "short"],
    "little": ["big", "large", "tall"],
    "tall": ["short", "small"],
    "short": ["tall", "large"],
    "wooden": ["metal", "plastic"],
    "metal": ["wooden", "plastic"],
    "plastic": ["wooden", "metal"],
    "open": ["closed", "shut"],
    "closed": ["open"],
    "shut": ["open"],
    "old": ["young", "new"],
    "young": ["old"],
    "new": ["old"],
    "clean": ["dirty"],
    "dirty": ["clean"],
    "empty": ["full"],
    "full": ["empty"],
    "round": ["square", "rectangular"],
    "square": ["round"],
    "rectangular": ["round"],
}
RELATION_TOKEN_CONFUSION_MAP = {
    "on": ["under", "above", "below"],
    "under": ["on", "above"],
    "above": ["below", "under"],
    "below": ["above", "on"],
    "over": ["under", "below"],
    "behind": ["in front of", "beside"],
    "beside": ["behind", "in front of"],
    "inside": ["outside"],
    "outside": ["inside"],
    "left": ["right"],
    "right": ["left"],
    "holding": ["wearing", "carrying"],
    "wearing": ["holding", "carrying"],
    "carrying": ["holding", "wearing"],
    "riding": ["holding", "standing"],
    "standing": ["sitting", "riding"],
    "sitting": ["standing", "lying"],
    "looking": ["touching", "facing"],
    "touching": ["looking", "holding"],
    "covering": ["under"],
    "next": ["behind"],
    "with": ["without"],
    "without": ["with"],
}
RELATION_PHRASE_CONFUSION_MAP = {
    "in front of": ["behind", "next to"],
    "next to": ["behind", "in front of"],
    "left of": ["right of"],
    "right of": ["left of"],
    "standing on": ["standing under", "sitting on"],
    "sitting on": ["standing on", "under"],
    "looking at": ["looking away from", "touching"],
    "looking away from": ["looking at"],
}


@dataclass
class RelationExample:
    doc: "spacy.tokens.Doc"
    relation_span: Span
    subject_span: Span
    object_span: Span


@dataclass
class VisionBindingExample:
    negative_text: str
    negative_type: str
    object_char_span: Tuple[int, int]
    pos_attribute_char_span: Tuple[int, int]
    neg_attribute_char_span: Tuple[int, int]


@dataclass
class AttributeObjectPair:
    attribute_span: Span
    object_core_span: Span
    object_head_index: int


VISION_BINDING_NEGATIVE_TYPE_TO_ID = {
    "none": 0,
    "swap": 1,
}


def swap_spans(tokens: List[Token], span1: Span, span2: Span) -> List[Token]:
    if span2.start < span1.start:
        span1, span2 = span2, span1

    first_start = span1.start
    first_end = span1.end
    second_start = span2.start
    second_end = span2.end

    return (
        tokens[:first_start]
        + tokens[second_start:second_end]
        + tokens[first_end:second_start]
        + tokens[first_start:first_end]
        + tokens[second_end:]
    )


class RelationTextMixin:
    def __init__(self, rng=None):
        self.rng = rng if rng else np.random.default_rng(2024)
        self.nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])
        self._doc_cache_max_size = 4096
        self._doc_cache: "OrderedDict[str, spacy.tokens.Doc]" = OrderedDict()
        self._noun_chunk_cache_max_size = 8192
        self._noun_chunk_cache: "OrderedDict[str, str]" = OrderedDict()
        self._relation_example_cache: Dict[str, Optional[RelationExample]] = {}
        self.vp_matcher = Matcher(self.nlp.vocab)
        self.vp_matcher.add(
            "Verb phrase",
            [[
                {"POS": "VERB", "OP": "?"},
                {"POS": "ADV", "OP": "*"},
                {"POS": "AUX", "OP": "*"},
                {"POS": "VERB", "OP": "+"},
            ]],
        )

    def _ensure_relation_text_state(self) -> None:
        if not hasattr(self, "rng"):
            self.rng = np.random.default_rng(2024)
        if not hasattr(self, "nlp"):
            self.nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])
        if not hasattr(self, "_doc_cache_max_size"):
            self._doc_cache_max_size = 4096
        if not hasattr(self, "_doc_cache"):
            self._doc_cache = OrderedDict()
        if not hasattr(self, "_noun_chunk_cache_max_size"):
            self._noun_chunk_cache_max_size = 8192
        if not hasattr(self, "_noun_chunk_cache"):
            self._noun_chunk_cache = OrderedDict()
        if not hasattr(self, "_relation_example_cache"):
            self._relation_example_cache = {}
        if not hasattr(self, "vp_matcher"):
            self.vp_matcher = Matcher(self.nlp.vocab)
            self.vp_matcher.add(
                "Verb phrase",
                [[
                    {"POS": "VERB", "OP": "?"},
                    {"POS": "ADV", "OP": "*"},
                    {"POS": "AUX", "OP": "*"},
                    {"POS": "VERB", "OP": "+"},
                ]],
            )

    def _cache_doc(self, text: str, doc) -> None:
        self._ensure_relation_text_state()
        text = str(text)
        self._doc_cache[text] = doc
        self._doc_cache.move_to_end(text)
        if len(self._doc_cache) > self._doc_cache_max_size:
            self._doc_cache.popitem(last=False)

    def _cache_noun_chunk(self, chunk_text: str) -> None:
        self._ensure_relation_text_state()
        normalized = str(chunk_text).strip().lower()
        if not normalized:
            return
        self._noun_chunk_cache[normalized] = str(chunk_text).strip()
        self._noun_chunk_cache.move_to_end(normalized)
        if len(self._noun_chunk_cache) > self._noun_chunk_cache_max_size:
            self._noun_chunk_cache.popitem(last=False)

    def _get_doc(self, text: str):
        self._ensure_relation_text_state()
        text = str(text)
        doc = self._doc_cache.get(text)
        if doc is None:
            doc = self.nlp(text)
            self._cache_doc(text, doc)
        else:
            self._doc_cache.move_to_end(text)
        return doc

    def _populate_relation_example_cache(self, texts: Sequence[str]) -> None:
        self._ensure_relation_text_state()
        missing_texts = []
        seen_texts = set()
        for text in texts:
            text = str(text)
            if text not in self._relation_example_cache and text not in seen_texts:
                missing_texts.append(text)
                seen_texts.add(text)

        if not missing_texts:
            return

        batch_size = min(256, len(missing_texts))
        for text, doc in zip(missing_texts, self.nlp.pipe(missing_texts, batch_size=batch_size)):
            self._cache_doc(text, doc)
            self._relation_example_cache[text] = self._extract_relation_example_from_doc(doc)

    def _get_head_noun_chunk(self, doc, head_token: Token) -> Span:
        for chunk in doc.noun_chunks:
            if chunk.start <= head_token.i < chunk.end:
                return chunk
        return doc[head_token.i:head_token.i + 1]

    def _resolve_subject_head(self, token: Token) -> Optional[Token]:
        for child in token.children:
            if child.dep_ in SUBJECT_DEPS and child.pos_ in {"NOUN", "PROPN", "PRON"}:
                return child
        return None

    def _resolve_object_head(self, token: Token) -> Optional[Token]:
        for child in token.children:
            if child.dep_ in OBJECT_DEPS and child.pos_ in {"NOUN", "PROPN", "PRON"}:
                return child

        for child in token.children:
            if child.dep_ == "prep":
                for grand_child in child.children:
                    if grand_child.dep_ == "pobj" and grand_child.pos_ in {"NOUN", "PROPN", "PRON"}:
                        return grand_child
        return None

    def _get_relation_tokens(self, token: Token) -> List[Token]:
        relation_tokens = [token]
        for child in token.children:
            if child.dep_ in {"aux", "auxpass", "neg", "prt", "prep"}:
                relation_tokens.append(child)
        return sorted(relation_tokens, key=lambda item: item.i)

    def _build_relation_span(self, doc, relation_tokens: Sequence[Token]) -> Span:
        start = min(token.i for token in relation_tokens)
        end = max(token.i for token in relation_tokens) + 1
        return doc[start:end]

    def _find_nominal_ancestor(self, token: Token) -> Optional[Token]:
        cursor = token
        visited = set()
        while cursor is not None and cursor.i not in visited:
            if cursor.pos_ in {"NOUN", "PROPN", "PRON"}:
                return cursor
            visited.add(cursor.i)
            if cursor.head == cursor:
                break
            cursor = cursor.head
        return None

    def _extract_relation_example_from_doc(self, doc) -> Optional[RelationExample]:
        for token in doc:
            if token.pos_ not in {"VERB", "AUX"}:
                continue

            subject_head = self._resolve_subject_head(token)
            object_head = self._resolve_object_head(token)
            if subject_head is None or object_head is None:
                continue

            subject_span = self._get_head_noun_chunk(doc, subject_head)
            object_span = self._get_head_noun_chunk(doc, object_head)
            if subject_span.start == object_span.start and subject_span.end == object_span.end:
                continue
            relation_tokens = self._get_relation_tokens(token)

            return RelationExample(
                doc=doc,
                relation_span=self._build_relation_span(doc, relation_tokens),
                subject_span=subject_span,
                object_span=object_span,
            )

        for token in doc:
            if token.pos_ != "ADP":
                continue

            object_head = None
            for child in token.children:
                if child.dep_ == "pobj" and child.pos_ in {"NOUN", "PROPN", "PRON"}:
                    object_head = child
                    break

            subject_head = self._find_nominal_ancestor(token.head)
            if subject_head is None or object_head is None or subject_head.i == object_head.i:
                continue

            subject_span = self._get_head_noun_chunk(doc, subject_head)
            object_span = self._get_head_noun_chunk(doc, object_head)
            if subject_span.start == object_span.start and subject_span.end == object_span.end:
                continue

            return RelationExample(
                doc=doc,
                relation_span=doc[token.i:token.i + 1],
                subject_span=subject_span,
                object_span=object_span,
            )
        return None

    def extract_relation_example(self, text: str) -> Optional[RelationExample]:
        self._ensure_relation_text_state()
        text = str(text)
        if text not in self._relation_example_cache:
            doc = self._get_doc(text)
            self._relation_example_cache[text] = self._extract_relation_example_from_doc(doc)
        return self._relation_example_cache[text]

    def _replace_span(self, doc, span: Span, replacement: str) -> str:
        replacement = str(replacement).strip()
        if not replacement:
            return ""
        prefix = "".join(token.text_with_ws for token in doc[:span.start])
        suffix = "".join(token.text_with_ws for token in doc[span.end:])
        trailing_ws = doc[span.end - 1].whitespace_ if span.end > span.start else ""
        return f"{prefix}{replacement}{trailing_ws}{suffix}".strip()

    def _pick_replacement(self, source_text: str, candidates: Sequence[str]) -> str:
        normalized_source = str(source_text).strip().lower()
        filtered = [candidate for candidate in candidates if candidate.strip().lower() != normalized_source]
        if not filtered:
            return ""
        return str(self.rng.choice(filtered))

    def _build_token_swap_fallback(self, text: str) -> str:
        doc = self._get_doc(text)
        candidate_spans = [
            doc[token.i:token.i + 1]
            for token in doc
            if not token.is_space and not token.is_punct
        ]
        if len(candidate_spans) >= 2:
            swapped_tokens = swap_spans(list(doc), candidate_spans[0], candidate_spans[-1])
            swapped_text = "".join(token.text_with_ws for token in swapped_tokens).strip()
            if swapped_text and swapped_text != text:
                return swapped_text
        if len(candidate_spans) == 1:
            return f"not {text}".strip()
        return ""

    def _find_attribute_spans(self, doc) -> List[Span]:
        priority_spans = []
        fallback_spans = []
        seen = set()
        for token in doc:
            if token.pos_ != "ADJ":
                continue
            start = token.i
            while start > 0:
                prev_token = doc[start - 1]
                if prev_token.is_punct or prev_token.head.i != token.i:
                    break
                if prev_token.dep_ not in {"advmod", "amod", "compound"}:
                    break
                start -= 1

            span = doc[start:token.i + 1]
            span_key = (span.start, span.end)
            if span_key in seen:
                continue
            seen.add(span_key)
            if token.dep_ in {"amod", "acomp"}:
                priority_spans.append(span)
            else:
                fallback_spans.append(span)
        return priority_spans + fallback_spans

    def _find_object_like_span(self, doc) -> Optional[Span]:
        for token in doc:
            if token.dep_ in OBJECT_DEPS and token.pos_ in {"NOUN", "PROPN"}:
                return self._get_head_noun_chunk(doc, token)

        for chunk in doc.noun_chunks:
            if any(token.pos_ in {"NOUN", "PROPN"} for token in chunk):
                return chunk

        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:
                return doc[token.i:token.i + 1]
        return None

    def _get_object_core_span(self, doc, head_token: Token) -> Span:
        start = head_token.i
        while start > 0:
            prev_token = doc[start - 1]
            if prev_token.dep_ != "compound" or prev_token.head.i != head_token.i:
                break
            start -= 1
        return doc[start:head_token.i + 1]

    @staticmethod
    def _span_char_range(span: Span) -> Tuple[int, int]:
        return span.start_char, span.end_char

    def _rewrite_text(
        self,
        doc,
        replacements: Sequence[Tuple[int, int, str, str]],
    ) -> Tuple[str, Dict[str, Tuple[int, int]]]:
        text = doc.text
        cursor = 0
        current_length = 0
        pieces: List[str] = []
        span_map: Dict[str, Tuple[int, int]] = {}

        for start_char, end_char, replacement, label in sorted(replacements, key=lambda item: item[0]):
            replacement = str(replacement).strip()
            if not replacement or start_char < cursor or end_char <= start_char:
                return "", {}

            prefix = text[cursor:start_char]
            pieces.append(prefix)
            current_length += len(prefix)

            span_start = current_length
            pieces.append(replacement)
            current_length += len(replacement)
            span_map[label] = (span_start, current_length)
            cursor = end_char

        suffix = text[cursor:]
        pieces.append(suffix)
        rewritten = "".join(pieces)
        return rewritten, span_map

    def _get_attribute_replacement_candidates(self, span: Span) -> List[str]:
        key = span.text.lower().strip()
        candidates = ATTRIBUTE_CONFUSION_MAP.get(key)
        if candidates is None:
            candidates = ATTRIBUTE_CONFUSION_MAP.get(span.root.lemma_.lower())
        return list(candidates) if candidates else []

    def _collect_amod_attribute_object_pairs(self, doc) -> List[AttributeObjectPair]:
        pairs: List[AttributeObjectPair] = []
        seen = set()
        for span in self._find_attribute_spans(doc):
            if span.root.dep_ != "amod" or span.root.pos_ != "ADJ":
                continue

            object_head = span.root.head
            if object_head.pos_ not in {"NOUN", "PROPN"}:
                continue

            object_core_span = self._get_object_core_span(doc, object_head)
            pair_key = (
                span.start,
                span.end,
                object_core_span.start,
                object_core_span.end,
            )
            if pair_key in seen:
                continue
            seen.add(pair_key)
            pairs.append(
                AttributeObjectPair(
                    attribute_span=span,
                    object_core_span=object_core_span,
                    object_head_index=object_head.i,
                )
            )
        pairs.sort(key=lambda pair: (pair.attribute_span.start_char, pair.object_core_span.start_char))
        return pairs

    def _build_vision_binding_swap_example(self, doc, pairs: Sequence[AttributeObjectPair]) -> Optional[VisionBindingExample]:
        for pair_idx, pair in enumerate(pairs):
            for donor_pair in pairs[pair_idx + 1:]:
                if pair.object_head_index == donor_pair.object_head_index:
                    continue
                if pair.attribute_span.text.lower().strip() == donor_pair.attribute_span.text.lower().strip():
                    continue

                negative_text, span_map = self._rewrite_text(
                    doc,
                    [
                        (
                            pair.attribute_span.start_char,
                            pair.attribute_span.end_char,
                            donor_pair.attribute_span.text,
                            "target_attribute",
                        ),
                        (
                            donor_pair.attribute_span.start_char,
                            donor_pair.attribute_span.end_char,
                            pair.attribute_span.text,
                            "donor_attribute",
                        ),
                    ],
                )
                if not negative_text or negative_text == doc.text:
                    continue

                neg_attribute_char_span = span_map.get("target_attribute")
                if neg_attribute_char_span is None:
                    continue

                return VisionBindingExample(
                    negative_text=negative_text,
                    negative_type="swap",
                    object_char_span=self._span_char_range(pair.object_core_span),
                    pos_attribute_char_span=self._span_char_range(pair.attribute_span),
                    neg_attribute_char_span=neg_attribute_char_span,
                )
        return None

    def collect_noun_chunk_candidates(self, texts: Sequence[str]) -> List[str]:
        self._ensure_relation_text_state()
        batch_candidates = []
        seen = set()
        for text in texts:
            doc = self._get_doc(text)
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.strip()
                if not chunk_text or chunk_text.lower() in seen:
                    continue
                if not any(token.pos_ in {"NOUN", "PROPN"} for token in chunk):
                    continue
                batch_candidates.append(chunk_text)
                seen.add(chunk_text.lower())

        cached_candidates = [
            chunk_text for normalized, chunk_text in self._noun_chunk_cache.items()
            if normalized not in seen
        ]

        for chunk_text in batch_candidates:
            self._cache_noun_chunk(chunk_text)

        return batch_candidates + cached_candidates

    def build_relation_negative(self, example: RelationExample) -> str:
        swapped_tokens = swap_spans(list(example.doc), example.subject_span, example.object_span)
        return "".join(token.text_with_ws for token in swapped_tokens).strip()

    def build_vision_binding_example(self, text: str) -> Optional[VisionBindingExample]:
        doc = self._get_doc(text)
        pairs = self._collect_amod_attribute_object_pairs(doc)
        if not pairs:
            return None

        return self._build_vision_binding_swap_example(doc, pairs)

    def build_attribute_replace_negative(self, text: str) -> str:
        doc = self._get_doc(text)
        for span in self._find_attribute_spans(doc):
            candidates = self._get_attribute_replacement_candidates(span)
            if not candidates:
                continue
            replacement = self._pick_replacement(span.text, candidates)
            if not replacement:
                continue
            replaced = self._replace_span(doc, span, replacement)
            if replaced and replaced != text:
                return replaced
        return ""

    def build_object_replace_negative(
        self,
        text: str,
        noun_chunk_pool: Sequence[str],
        example: Optional[RelationExample] = None,
    ) -> str:
        relation_example = example if example is not None else self.extract_relation_example(text)
        doc = relation_example.doc if relation_example is not None else self._get_doc(text)
        object_span = relation_example.object_span if relation_example is not None else self._find_object_like_span(doc)
        if object_span is None:
            return ""

        replacement = self._pick_replacement(object_span.text, noun_chunk_pool)
        if not replacement:
            return ""

        replaced = self._replace_span(doc, object_span, replacement)
        if replaced and replaced != text:
            return replaced
        return ""

    def build_relation_replace_negative(self, text: str, example: Optional[RelationExample] = None) -> str:
        relation_example = example if example is not None else self.extract_relation_example(text)
        if relation_example is None:
            return ""

        doc = relation_example.doc
        relation_span = relation_example.relation_span
        phrase_candidates = RELATION_PHRASE_CONFUSION_MAP.get(relation_span.text.lower().strip())
        if phrase_candidates:
            replacement = self._pick_replacement(relation_span.text, phrase_candidates)
            if replacement:
                replaced = self._replace_span(doc, relation_span, replacement)
                if replaced and replaced != text:
                    return replaced

        prep_tokens = [token for token in relation_span if token.pos_ == "ADP"]
        for token in reversed(prep_tokens):
            candidates = RELATION_TOKEN_CONFUSION_MAP.get(token.text.lower())
            if not candidates:
                continue
            replacement = self._pick_replacement(token.text, candidates)
            if not replacement:
                continue
            replaced = self._replace_span(doc, doc[token.i:token.i + 1], replacement)
            if replaced and replaced != text:
                return replaced

        root = relation_span.root
        relation_key = root.lemma_.lower()
        candidates = RELATION_TOKEN_CONFUSION_MAP.get(relation_key)
        if not candidates:
            candidates = RELATION_TOKEN_CONFUSION_MAP.get(root.text.lower())
        if not candidates:
            return ""

        replacement = self._pick_replacement(root.text, candidates)
        if not replacement:
            return ""

        replaced = self._replace_span(doc, doc[root.i:root.i + 1], replacement)
        if replaced and replaced != text:
            return replaced
        return ""


class NegativeTextMining(RelationTextMixin):
    def __init__(self, max_num_texts: int = 5, rng=None):
        self.max_num_texts = max_num_texts
        super().__init__(rng=rng)

    def generate_negative_captions(self, original_text: str) -> List[str]:
        original_text = str(original_text)
        doc = self._get_doc(original_text)

        noun_phrases = [np for np in doc.noun_chunks if len(np) >= 3]
        verb_phrases = self.vp_matcher(doc, as_spans=True)
        adjectives = [doc[token.i:token.i + 1] for token in doc if token.pos_ == "ADJ"]
        adverbs = [doc[token.i:token.i + 1] for token in doc if token.pos_ == "ADV"]
        nouns = [doc[token.i:token.i + 1] for token in doc if token.pos_ == "NOUN"]

        swap_groups = [
            ("NOUN", nouns),
            ("ADJ", adjectives),
            ("ADV", adverbs),
            ("NP", noun_phrases),
            ("VP", verb_phrases),
        ]

        negative_captions = []

        for _ in range(self.max_num_texts):
            eligible_groups = [(name, group) for name, group in swap_groups if len(group) >= 2]
            if not eligible_groups:
                break

            group_idx = self.rng.choice(len(eligible_groups))
            _, group = eligible_groups[group_idx]

            span_indices = self.rng.choice(len(group), size=2, replace=False)
            span1 = group[span_indices[0]]
            span2 = group[span_indices[1]]

            new_tokens = swap_spans(list(doc), span1, span2)
            new_caption = " ".join(token.text for token in new_tokens)
            if new_caption not in negative_captions and new_caption != original_text:
                negative_captions.append(new_caption)

        return negative_captions

    def negative_caption(self, original_text: str) -> str:
        negative_captions = self.generate_negative_captions(original_text)
        if negative_captions:
            return self.rng.choice(negative_captions)
        return ""

    def fallback_negative_caption(self, original_text: str, candidate_texts: Optional[Sequence[str]] = None) -> str:
        fallback = self.negative_caption(original_text)
        if fallback and fallback != original_text:
            return fallback

        if candidate_texts is not None:
            candidates = [str(text) for text in candidate_texts if str(text) != original_text]
            if candidates:
                return str(self.rng.choice(candidates))

        fallback = self._build_token_swap_fallback(original_text)
        if fallback and fallback != original_text:
            return fallback

        return f"not {original_text}".strip()
