import logging
import os
from typing import Optional, Dict, Union, Callable, List, Tuple, Any

import safetensors
import torch
from torch import nn
from torch.utils.data import Dataset, RandomSampler, IterableDataset
from transformers import (
    is_torch_xla_available, PreTrainedModel, TrainingArguments,
    DataCollator, PreTrainedTokenizerBase, BaseImageProcessor,
    FeatureExtractionMixin, ProcessorMixin, TrainerCallback, is_torch_xpu_available, is_torch_mlu_available,
    is_torch_musa_available, is_torch_npu_available, is_apex_available
)
from transformers.trainer import TRAINING_ARGS_NAME, TRAINER_STATE_NAME
from transformers.trainer_callback import ExportableState
from transformers.trainer_utils import has_length, EvalPrediction
from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME, is_torch_mps_available, \
    is_accelerate_available

from src.common.registry import registry
from src.runners.base import BaseTrainer
from src.utils import negative_contrastive_loss

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
if is_accelerate_available():
    from accelerate.utils import (
        DistributedType,
    )

if is_apex_available():
    from apex import amp

__all__ = [
    "RandomSamplerTrainer",
    "DELTATrainer"
]

logger = logging.getLogger(__name__)


@registry.register_trainer('RandomSamplerTrainer')
class RandomSamplerTrainer(BaseTrainer):
    """
    Trainer that uses a RandomSampler for the training dataset.
    """

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_length:
            raise ValueError("Argument `group_by_length` must be `False`.")
        else:
            generator = torch.Generator()
            generator.manual_seed(2024)

            return RandomSampler(self.train_dataset, generator=generator)


@registry.register_trainer('DELTATrainer')
class DELTATrainer(RandomSamplerTrainer):
    """
    Trainer that integrates the project contrastive loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(
            inputs, **{
                'return_dict': True,
            }
        )

        outputs = model(**inputs)
        fusion_gate_mean = getattr(outputs, "fusion_gate_mean", None)
        loss = outputs.loss if getattr(outputs, "loss", None) is not None else negative_contrastive_loss(outputs.logits_per_image)

        self._last_loss_logs = {
            "cont_loss": float(outputs.cont_loss.detach().item()) if getattr(outputs, "cont_loss", None) is not None else None,
            "text_order_loss": (
                float(outputs.text_order_loss.detach().item())
                if getattr(outputs, "text_order_loss", None) is not None
                else None
            ),
            "vision_binding_loss": (
                float(outputs.vision_binding_loss.detach().item())
                if getattr(outputs, "vision_binding_loss", None) is not None
                else None
            ),
            "fusion_gate_mean": (
                float(fusion_gate_mean.detach().item())
                if fusion_gate_mean is not None
                else None
            ),
        }

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            steps = self.state.global_step - self._globalstep_last_logged
            
            logs["loss"] = round(tr_loss_scalar / steps, 4)
            if hasattr(self, "_last_loss_logs"):
                for log_name, log_value in self._last_loss_logs.items():
                    if log_value is not None:
                        logs[log_name] = round(log_value, 4)

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs, start_time)

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving DELTA model checkpoint to {output_dir}")

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if isinstance(unwrapped_model, PreTrainedModel):
            unwrapped_model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            state_dict = state_dict if state_dict is not None else unwrapped_model.state_dict()
            if self.args.save_safetensors:
                safetensors.torch.save_file(
                    state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                )
            else:
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        for cb in [
            cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
        ]:
            cb_name = cb.__class__.__name__
            cb_state = cb.state()
            if isinstance(self.state.stateful_callbacks[cb_name], list):
                self.state.stateful_callbacks[cb_name].append(cb_state)
            else:
                self.state.stateful_callbacks[cb_name] = cb_state
        # Good practice: save your training arguments together with the trained model

        self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

        return loss.detach()
