import inspect
from typing import Optional, Dict, Type, TypeVar

from datasets import Dataset, IterableDataset, interleave_datasets, concatenate_datasets
from omegaconf import OmegaConf
from transformers import (
    PreTrainedModel,
    ProcessorMixin,
    TrainingArguments,
    add_end_docstrings
)

from src.common import registry, TrainConfig
from src.tasks.base import BaseTrainTask, TaskWithCustomModel, TaskWithPretrainedModel, TRAIN_TASK_DOCSTRING

ModelType = Type[PreTrainedModel]
ProcessorType = Type[ProcessorMixin]
DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)

__all__ = [
    "SingleTrainTask",
    "SingleTrainTaskWithPretrainedModel",
    "SingleTrainTaskWithCustomModel",
    "DatasetSingleTrainTask",
    "IterableDatasetSingleTrainTask",
    "DatasetTrainTaskWithPretrainedModel",
    "IterableDatasetTrainTaskWithPretrainedModel",
    "DatasetTrainTaskWithCustomModel",
    "IterableDatasetTrainTaskWithCustomModel",
]

@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class SingleTrainTask(BaseTrainTask):
    config: TrainConfig

    def _build_collator_kwargs(self, collator_cls) -> Dict:
        collator_kwargs = dict(self.config.collator_config.config)
        collator_parameters = inspect.signature(collator_cls).parameters

        collator_kwargs.setdefault("padding", "longest")

        if "enable_text_swap" in collator_parameters:
            text_loss_enabled = bool(
                OmegaConf.select(
                    self.config.model_config,
                    "config.text_binding.loss_enabled",
                    default=True,
                )
            )
            collator_kwargs.setdefault("enable_text_swap", text_loss_enabled)
        if "enable_vision_binding" in collator_parameters:
            vision_binding_enabled = bool(
                OmegaConf.select(
                    self.config.model_config,
                    "config.vision_binding.enabled",
                    default=False,
                )
            )
            collator_kwargs.setdefault("enable_vision_binding", vision_binding_enabled)
        return collator_kwargs

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        assert "runner" in self.config.run_config, "Trainer name must be provided."

        trainer_name = self.config.run_config.runner
        trainer_cls = registry.get_trainer_class(trainer_name)
        assert trainer_cls is not None, "Trainer {} not properly registered.".format(trainer_name)

        trainer_config = trainer_config if trainer_config is not None else self.config.trainer_config
        trainer_config = dict(trainer_config)
        trainer_config.setdefault("report_to", "none")
        training_argument_parameters = inspect.signature(TrainingArguments).parameters
        dataloader_num_workers = int(trainer_config.get("dataloader_num_workers", 0) or 0)
        if dataloader_num_workers > 0:
            if "dataloader_prefetch_factor" in training_argument_parameters:
                trainer_config.setdefault("dataloader_prefetch_factor", 2)

        collator_cls = registry.get_collator_class(self.config.collator_config.collator_cls)

        assert collator_cls is not None, "Collator {} not properly registered.".format(collator_cls)

        train_dataset = self.build_datasets()

        collator = collator_cls(
            processor=self.build_processor(),
            **self._build_collator_kwargs(collator_cls),
        )

        return trainer_cls(
            model=self.build_model(),
            args=TrainingArguments(**trainer_config),
            train_dataset=train_dataset,
            data_collator=collator,
        )


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class DatasetSingleTrainTask(SingleTrainTask):

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> Dataset:
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        datasets = list()

        assert len(dataset_config) > 0, "At least one dataset has to be specified."

        for builder_cls_name, config in dataset_config.items():
            builder = registry.get_builder_class(builder_cls_name)(**config)
            dataset = builder.build_dataset()
            if not isinstance(dataset, Dataset):
                raise TypeError("DatasetTrainTask must build dataset with `Dataset` type.")
            if shuffle:
                dataset = dataset.shuffle(seed=self.config.run_config.seed, buffer_size=buffer_size)

            datasets.append(dataset)

        return concatenate_datasets(datasets)


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class IterableDatasetSingleTrainTask(SingleTrainTask):

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> IterableDataset:
        dataset_config = dataset_config if dataset_config is not None else self.config.dataset_config

        datasets = list()

        assert len(dataset_config) > 0, "At least one dataset has to be specified."

        for builder_cls_name, config in dataset_config.items():
            builder = registry.get_builder_class(builder_cls_name)(**config)
            dataset = builder.build_dataset()
            if not isinstance(dataset, IterableDataset):
                raise TypeError("DatasetTrainTask must build dataset with `IterableDataset` type.")
            if shuffle:
                dataset = dataset.shuffle(seed=self.config.run_config.seed, buffer_size=buffer_size)

            datasets.append(dataset)

        return interleave_datasets(datasets).with_format("torch")


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class SingleTrainTaskWithPretrainedModel(SingleTrainTask, TaskWithPretrainedModel):

    def build_model(self, model_config: Optional[Dict] = None):
        """Build a pretrained model from the provided configuration."""
        model_config = model_config if model_config is not None else self.config.model_config.copy()

        model_cls = registry.get_model_class(model_config.model_cls)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)
        # Initialize the model

        model = model_cls.from_pretrained(**model_config.config)

        return model.cuda().train()


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
class SingleTrainTaskWithCustomModel(SingleTrainTask, TaskWithCustomModel):

    def build_model(self, model_config: Optional[Dict] = None):
        """Build a custom model using the registered config and model classes."""

        model_config = model_config if model_config is not None else self.config.model_config.copy()

        # Get the model configuration and model class from the registry
        model_cfg_cls = registry.get_model_config_class(model_config.config_cls)
        model_cls = registry.get_model_class(model_config.model_cls)

        assert model_cls is not None, "Model {} not properly registered.".format(model_cls)
        assert model_cfg_cls is not None, "Model config {} not properly registered.".format(model_cfg_cls)

        # Initialize the model configuration and model
        model_cfg = model_cfg_cls(**model_config.config)
        model = model_cls(model_cfg)

        return model.cuda().train()


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
@registry.register_task("DatasetTrainTaskWithPretrainedModel")
class DatasetTrainTaskWithPretrainedModel(DatasetSingleTrainTask, SingleTrainTaskWithPretrainedModel):
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        return SingleTrainTaskWithPretrainedModel.build_model(
            self,
            model_config=model_config,
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> Dataset:
        return DatasetSingleTrainTask.build_datasets(
            self,
            dataset_config=dataset_config,
            shuffle=shuffle,
            buffer_size=buffer_size,
        )

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return SingleTrainTaskWithPretrainedModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        return DatasetSingleTrainTask.build_trainer(
            self,
            trainer_config=trainer_config,
        )


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
@registry.register_task("IterableDatasetTrainTaskWithPretrainedModel")
class IterableDatasetTrainTaskWithPretrainedModel(IterableDatasetSingleTrainTask, SingleTrainTaskWithPretrainedModel):
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        return SingleTrainTaskWithPretrainedModel.build_model(
            self,
            model_config=model_config,
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> IterableDataset:
        return IterableDatasetSingleTrainTask.build_datasets(
            self,
            dataset_config=dataset_config,
            shuffle=shuffle,
            buffer_size=buffer_size,
        )

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return SingleTrainTaskWithPretrainedModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        return IterableDatasetSingleTrainTask.build_trainer(
            self,
            trainer_config=trainer_config,
        )


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
@registry.register_task("DatasetTrainTaskWithCustomModel")
class DatasetTrainTaskWithCustomModel(DatasetSingleTrainTask, SingleTrainTaskWithCustomModel):
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        return SingleTrainTaskWithCustomModel.build_model(
            self,
            model_config=model_config,
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> Dataset:
        return DatasetSingleTrainTask.build_datasets(
            self,
            dataset_config=dataset_config,
            shuffle=shuffle,
            buffer_size=buffer_size,
        )

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return SingleTrainTaskWithCustomModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        return DatasetSingleTrainTask.build_trainer(
            self,
            trainer_config=trainer_config,
        )


@add_end_docstrings(TRAIN_TASK_DOCSTRING)
@registry.register_task("IterableDatasetTrainTaskWithCustomModel")
class IterableDatasetTrainTaskWithCustomModel(IterableDatasetSingleTrainTask, SingleTrainTaskWithCustomModel):
    def build_model(
            self,
            model_config: Optional[Dict] = None
    ) -> ModelType:
        return SingleTrainTaskWithCustomModel.build_model(
            self,
            model_config=model_config,
        )

    def build_datasets(
            self,
            dataset_config: Optional[Dict] = None,
            shuffle: Optional[bool] = False,
            buffer_size: Optional[int] = 10000
    ) -> IterableDataset:
        return IterableDatasetSingleTrainTask.build_datasets(
            self,
            dataset_config=dataset_config,
            shuffle=shuffle,
            buffer_size=buffer_size,
        )

    def build_processor(
            self,
            processor_config: Optional[Dict] = None
    ) -> ProcessorType:
        return SingleTrainTaskWithCustomModel.build_processor(
            self,
            processor_config=processor_config,
        )

    def build_trainer(
            self,
            trainer_config: Optional[Dict] = None
    ):
        return IterableDatasetSingleTrainTask.build_trainer(
            self,
            trainer_config=trainer_config,
        )
