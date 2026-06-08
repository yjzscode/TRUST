# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

from msgspec import field
from packaging import version as vs
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

try:
    from vllm.lora.models import LoRAModel
except ModuleNotFoundError:
    from vllm.lora.lora_model import LoRAModel

from verl.third_party.vllm import get_version


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class VLLMHijack:
    @staticmethod
    def hijack():
        def _call_with_supported_kwargs(func, /, *args, **kwargs):
            sig = inspect.signature(func)
            supported = {name for name in sig.parameters if name != "self"}
            filtered = {k: v for k, v in kwargs.items() if k in supported}
            return func(*args, **filtered)

        def _get_lora_extra_vocab_size(manager) -> int:
            value = getattr(manager.lora_config, "lora_extra_vocab_size", None)
            if value is None:
                return 0
            try:
                return int(value)
            except Exception:
                return 0

        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            """
            based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors

            Reason:
            VLLM does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
            To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to
            load memory-based LoRA tensors.
            """
            try:
                supported_lora_modules = self._adapter_manager.supported_lora_modules
                packed_modules_mapping = self._adapter_manager.packed_modules_mapping
                expected_lora_lst: list[str] = []
                for module in supported_lora_modules:
                    if module in packed_modules_mapping:
                        expected_lora_lst.extend(packed_modules_mapping[module])
                    else:
                        expected_lora_lst.append(module)
                    if module == "experts":
                        expected_lora_lst.append(module)
                expected_lora_modules = set(expected_lora_lst)

                lora_tensors = None
                from vllm.lora.peft_helper import PEFTHelper

                if isinstance(lora_request, TensorLoRARequest):
                    peft_config = lora_request.peft_config
                    lora_tensors = lora_request.lora_tensors
                    peft_helper = PEFTHelper.from_dict(peft_config)
                else:
                    lora_path = get_adapter_absolute_path(lora_request.lora_path)
                    peft_helper = _call_with_supported_kwargs(
                        PEFTHelper.from_local_dir,
                        lora_path,
                        self.max_position_embeddings,
                        tensorizer_config_dict=getattr(lora_request, "tensorizer_config_dict", None),
                    )

                # Validates the LoRA configuration against requirements before
                # loading weights, throwing an exception if validation fails.
                peft_helper.validate_legal(self.lora_config)

                # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
                # to ensure correct loading of lora weights.
                model = self._adapter_manager.model
                hf_to_vllm_mapper = None
                if hasattr(model, "hf_to_vllm_mapper") and model.hf_to_vllm_mapper is not None:
                    hf_to_vllm_mapper = model.hf_to_vllm_mapper

                if isinstance(lora_request, TensorLoRARequest):
                    lora_extra_vocab_size = _get_lora_extra_vocab_size(self)
                    lora = _call_with_supported_kwargs(
                        self._lora_model_cls.from_lora_tensors,
                        lora_model_id=lora_request.lora_int_id,
                        tensors=lora_tensors,
                        peft_helper=peft_helper,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        model_vocab_size=self.vocab_size,
                        target_embedding_padding=self.vocab_size + lora_extra_vocab_size,
                        embedding_modules=getattr(self, "embedding_modules", None),
                        embedding_padding_modules=getattr(self, "embedding_padding_modules", None),
                        weights_mapper=hf_to_vllm_mapper,
                    )
                else:
                    lora = _call_with_supported_kwargs(
                        self._lora_model_cls.from_local_checkpoint,
                        lora_path,
                        expected_lora_modules,
                        peft_helper=peft_helper,
                        lora_model_id=lora_request.lora_int_id,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        model_vocab_size=self.vocab_size,
                        tensorizer_config_dict=getattr(lora_request, "tensorizer_config_dict", None),
                        embedding_modules=getattr(self, "embedding_modules", None),
                        embedding_padding_modules=getattr(self, "embedding_padding_modules", None),
                        weights_mapper=hf_to_vllm_mapper,
                    )
            except Exception as e:
                raise e

            extra_vocab_size = getattr(lora, "extra_vocab_size", 0)
            lora_extra_vocab_size = _get_lora_extra_vocab_size(self)
            if extra_vocab_size > lora_extra_vocab_size:
                raise ValueError(
                    f"LoRA added vocab size {extra_vocab_size} is greater than lora_extra_vocab_size "
                    f"{lora_extra_vocab_size}."
                )
            return lora

        def do_hijack(target_cls, target_method_name, hooking_method):
            setattr(target_cls, target_method_name, hooking_method)

        do_hijack(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)


def is_version_ge(pkg: str = "vllm", minver: str = "0.7.3"):
    """check if the package version is greater than or equal to the minimum version"""
    return vs.parse(get_version(pkg)) >= vs.parse(minver)
