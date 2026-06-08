# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import dataclasses
import inspect
import logging
import os
from typing import Any, Optional

import ray
import sglang.srt.entrypoints.engine
import torch
from ray.actor import ActorHandle
from sglang.srt.entrypoints.http_server import (
    ServerArgs,
    _GlobalState,
    _launch_subprocesses,
    app,
    set_global_state,
)
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter, _set_envs_and_config
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address, run_unvicorn

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
_ASYNC_SGLANG_BUILD_TAG = os.getenv("VERL_ASYNC_SGLANG_BUILD_TAG", "2026-04-16-outputids-v1")


def _normalize_gpu_ids(gpu_ids: list[Any] | None) -> list[str]:
    normalized: list[str] = []
    for gpu_id in gpu_ids or []:
        if gpu_id is None:
            continue
        gpu_id_str = str(gpu_id).strip()
        if not gpu_id_str:
            continue
        normalized.append(gpu_id_str)
    return normalized


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _get_worker_node_gpu_info(_worker) -> tuple[str, list[str], str | None]:
    runtime_context = ray.get_runtime_context()
    accelerator_ids = runtime_context.get_accelerator_ids()
    gpu_ids = accelerator_ids.get("GPU")
    if gpu_ids is None:
        try:
            gpu_ids = ray.get_gpu_ids()
        except Exception:
            gpu_ids = []
    return (
        runtime_context.get_node_id(),
        _normalize_gpu_ids(gpu_ids),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def _patch_sglang_rwlock_py310_compat() -> None:
    """Patch sglang RWLock for Python 3.10 asyncio.Condition(lock) compatibility.

    In this environment, ``asyncio.Condition(lock)`` still validates that the passed
    lock is already bound to the current loop. sglang's RWLock constructs
    ``asyncio.Lock()`` and immediately passes it into ``asyncio.Condition(...)``,
    which raises ``ValueError: loop argument must agree with lock`` inside Ray async
    actors running under uvloop.
    """

    try:
        import sglang.srt.lora.lora_registry as lora_registry_mod
        import sglang.srt.managers.tokenizer_manager as tokenizer_manager_mod
        import sglang.srt.utils.aio_rwlock as aio_rwlock_mod
    except Exception:
        return

    if "loop" not in inspect.signature(asyncio.Condition.__init__).parameters:
        return

    if getattr(aio_rwlock_mod.RWLock, "_verl_py310_compat", False):
        return

    class CompatRWLock:
        _verl_py310_compat = True

        def __init__(self):
            self._lock = asyncio.Lock()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and getattr(self._lock, "_loop", None) is None:
                self._lock._loop = loop

            self._cond = asyncio.Condition(self._lock)
            self._readers = 0
            self._writer_active = False
            self._waiting_writers = 0

        @property
        def reader_lock(self):
            return aio_rwlock_mod._ReaderLock(self)

        @property
        def writer_lock(self):
            return aio_rwlock_mod._WriterLock(self)

        async def acquire_reader(self):
            async with self._lock:
                while self._writer_active or self._waiting_writers > 0:
                    await self._cond.wait()
                self._readers += 1

        async def release_reader(self):
            async with self._lock:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

        async def acquire_writer(self):
            async with self._lock:
                self._waiting_writers += 1
                try:
                    while self._writer_active or self._readers > 0:
                        await self._cond.wait()
                    self._writer_active = True
                finally:
                    self._waiting_writers -= 1

        async def release_writer(self):
            async with self._lock:
                self._writer_active = False
                self._cond.notify_all()

    aio_rwlock_mod.RWLock = CompatRWLock
    tokenizer_manager_mod.RWLock = CompatRWLock
    lora_registry_mod.RWLock = CompatRWLock


def _patch_sglang_resume_memory_occupation() -> None:
    """Make SGLang memory resume tolerant of already-resident memory tags."""
    try:
        import sglang.srt.managers.scheduler_update_weights_mixin as mixin_mod
        from sglang.srt.constants import (
            GPU_MEMORY_ALL_TYPES,
            GPU_MEMORY_TYPE_CUDA_GRAPH,
            GPU_MEMORY_TYPE_KV_CACHE,
            GPU_MEMORY_TYPE_WEIGHTS,
        )
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqOutput
    except Exception:
        return

    cls = getattr(mixin_mod, "SchedulerUpdateWeightsMixin", None)
    if cls is None or getattr(cls, "_verl_resume_memory_compat", False):
        return

    def resume_memory_occupation(self, recv_req):
        tags = recv_req.tags
        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        active_tags = []
        for tag in tags:
            if tag in self.offload_tags:
                self.offload_tags.discard(tag)
                active_tags.append(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in active_tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in active_tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            if hasattr(self, "stashed_model_static_state"):
                mixin_mod._import_static_state(
                    self.tp_worker.model_runner.model,
                    self.stashed_model_static_state,
                )
                del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in active_tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

        return ResumeMemoryOccupationReqOutput()

    cls.resume_memory_occupation = resume_memory_occupation
    cls._verl_resume_memory_compat = True


@ray.remote(num_cpus=1)
class SGLangHttpServer:
    """SGLang http server in single node, this is equivalent to launch server with command line:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 1 ...
    ```

    Args:
        config (DictConfig): full config.
        rollout_mode (RolloutMode): rollout mode.
        replica_rank (int): replica rank, a replica may contain multiple nodes.
        node_rank (int): node rank.
        nnodes (int): number of nodes.
        cuda_visible_devices (str): cuda visible devices.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        print(f"SGLang http server: {rollout_mode=}, {replica_rank=}, {node_rank=}, {nnodes=}, {cuda_visible_devices=}")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        assert torch.cuda.is_available(), "SGLang http server should run on GPU node"

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        print(
            "[async-sglang-build] "
            f"tag={_ASYNC_SGLANG_BUILD_TAG} model_path={self.model_config.local_path} "
            f"load_format={self.config.load_format} skip_tokenizer_init={self.config.skip_tokenizer_init}"
        )
        self.config.max_model_len = self.config.prompt_length + self.config.response_length
        self.rollout_mode = rollout_mode
        self.workers = workers
        self._debug_generate_count = 0
        self._debug_tokenizer = None
        self._memory_released = False
        self.stop_token_ids = self._resolve_stop_token_ids()

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for NCCL process group
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address)
            logger.info(
                f"SGLangHttpServer, replica_rank: {self.replica_rank}, "
                f"master address: {self._master_address}, port: {self._master_port}"
            )
        else:
            self._master_address = None
            self._master_port = None

    def _resolve_stop_token_ids(self) -> list[int]:
        stop_token_ids: list[int] = []
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.local_path,
                trust_remote_code=self.model_config.trust_remote_code,
            )
            self._debug_tokenizer = tokenizer
            for token_id in (
                getattr(tokenizer, "eos_token_id", None),
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ):
                if isinstance(token_id, int) and token_id >= 0 and token_id not in stop_token_ids:
                    stop_token_ids.append(token_id)
        except Exception:
            logger.warning("Failed to resolve stop token ids for async SGLang server", exc_info=True)
        if not stop_token_ids:
            stop_token_ids = [151645]
        return stop_token_ids

    def _decode_debug(self, token_ids: list[int], limit: int = 256) -> str:
        if not token_ids or self._debug_tokenizer is None:
            return ""
        try:
            preview = self._debug_tokenizer.decode(token_ids[:limit], skip_special_tokens=False)
        except Exception:
            return ""
        return preview.replace("\n", "\\n")[:800]

    def get_master_address(self):
        """Get master address and port for init NCCL process group."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        attention_backend = engine_kwargs.pop("attention_backend", None)
        dist_init_addr = (
            f"[{self._master_address}]:{self._master_port}"
            if is_valid_ipv6_address(self._master_address)
            else f"{self._master_address}:{self._master_port}"
        )

        load_format = "dummy" if str(self.config.load_format).startswith("dummy") else self.config.load_format
        args = {
            "model_path": self.model_config.local_path,
            "dtype": self.config.dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "disable_cuda_graph": self.config.enforce_eager,
            "enable_memory_saver": True,
            "base_gpu_id": 0,
            "gpu_id_step": 1,
            "tp_size": self.config.tensor_model_parallel_size,
            "dp_size": self.config.data_parallel_size,
            "ep_size": self.config.expert_parallel_size,
            "node_rank": self.node_rank,
            "load_format": load_format,
            "dist_init_addr": dist_init_addr,
            "nnodes": self.nnodes,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_running_requests": self.config.get("max_num_seqs", None),
            "log_level": "error",
            "mm_attention_backend": "fa3",
            "attention_backend": attention_backend if attention_backend is not None else "fa3",
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "skip_server_warmup": True,
            **engine_kwargs,
        }

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

            # start sglang metrics
            args["enable_metrics"] = True

        # enable_weights_cpu_backup is supported in sglang>=0.5.3
        if "enable_weights_cpu_backup" in [f.name for f in dataclasses.fields(ServerArgs)]:
            enable_weights_cpu_backup = True if self.rollout_mode == RolloutMode.COLOCATED else False
            args["enable_weights_cpu_backup"] = enable_weights_cpu_backup

        # NOTE: We can't directly call SGLang's launch_server since it's not an async function.
        # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py
        sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        _patch_sglang_rwlock_py310_compat()
        _patch_sglang_resume_memory_occupation()
        server_args = ServerArgs(**args)
        self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = _launch_subprocesses(
            server_args=server_args
        )

        # In multi-node cases, non-zero rank nodes should not launch http server.
        if self.node_rank > 0:
            return

        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                template_manager=self.template_manager,
                scheduler_info=self.scheduler_info,
            )
        )
        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_args = (server_args, None, None)
        self._server_port, self._server_task = await run_unvicorn(app, server_args, self._server_address)
        self.tokenizer_manager.server_status = ServerStatus.Up

    async def wake_up(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            # Call all workers to switch between trainer mode and rollout mode.
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Keep weights resident in this SGLang version. Its weights tag state can
            # desynchronize and crash the scheduler on resume; kv_cache offload is enough
            # for the colocated rollout/training handoff used here.
            if not self._memory_released:
                logger.info("skip SGLang resume_memory_occupation because memory was not released")
                return
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache"])
            await self.tokenizer_manager.resume_memory_occupation(obj, None)
            await self.tokenizer_manager.flush_cache()
            self._memory_released = False
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
            await self.tokenizer_manager.release_memory_occupation(obj, None)
            self._memory_released = True
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.detach().cpu().tolist()
        else:
            prompt_ids = list(prompt_ids)
        sampling_params = dict(sampling_params)
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(prompt_ids) - 1)
        sampling_params["max_new_tokens"] = max_new_tokens
        existing_stop_token_ids = sampling_params.get("stop_token_ids") or []
        merged_stop_token_ids = []
        for token_id in list(existing_stop_token_ids) + list(self.stop_token_ids):
            if isinstance(token_id, int) and token_id not in merged_stop_token_ids:
                merged_stop_token_ids.append(token_id)
        sampling_params["ignore_eos"] = False
        sampling_params["stop_token_ids"] = merged_stop_token_ids
        sampling_params.setdefault("skip_special_tokens", True)
        return_logprob = sampling_params.pop("logprobs", False)

        request = GenerateReqInput(
            rid=request_id,
            input_ids=prompt_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            image_data=image_data,
        )
        output = await self.tokenizer_manager.generate_request(request, None).__anext__()
        output_ids = list(output.get("output_ids") or [])
        output_token_logprobs = output.get("meta_info", {}).get("output_token_logprobs") or []
        if return_logprob:
            logprob_token_ids = [token_id for _, token_id, _ in output_token_logprobs]
            log_probs = [log_prob for log_prob, _, _ in output_token_logprobs]
        else:
            logprob_token_ids = []
            log_probs = None
        token_ids = output_ids if output_ids else logprob_token_ids
        if log_probs is not None and len(log_probs) != len(token_ids):
            logger.warning(
                "Async SGLang output/logprob length mismatch: request_id=%s output_ids=%s logprob_ids=%s",
                request_id,
                len(output_ids),
                len(logprob_token_ids),
            )
            if len(log_probs) >= len(token_ids):
                log_probs = log_probs[: len(token_ids)]
            else:
                log_probs = None
        if self._debug_generate_count < 12:
            self._debug_generate_count += 1
            finish_reason = output.get("meta_info", {}).get("finish_reason")
            print(
                "[async sglang generate] "
                f"request_id={request_id} prompt_len={len(prompt_ids)} output_ids_len={len(output_ids)} "
                f"logprob_ids_len={len(logprob_token_ids)} finish_reason={finish_reason} "
                f"output_ids_head={self._decode_debug(output_ids)} "
                f"logprob_ids_head={self._decode_debug(logprob_token_ids)}"
            )
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)


_rollout_worker_actor_cls = ray.remote(ServerAdapter)


class SGLangReplica(RolloutReplica):
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # Ray training workers may expose a logical/local CUDA_VISIBLE_DEVICES like "0" even when
        # they are actually placed on different physical GPUs. Use Ray's accelerator ids as the
        # source of truth, and only fall back to worker env vars if Ray does not report ids.
        worker_infos = await asyncio.gather(
            *[worker.__ray_call__.remote(_get_worker_node_gpu_info) for worker in self.workers]
        )
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        for node_rank in range(self.nnodes):
            workers = self.workers[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            node_worker_infos = worker_infos[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            node_gpu_ids: list[str] = []
            for _, gpu_ids, worker_cuda_visible_devices in node_worker_infos:
                if gpu_ids:
                    node_gpu_ids.extend(gpu_ids)
                    continue
                if worker_cuda_visible_devices:
                    node_gpu_ids.extend(
                        [gpu_id.strip() for gpu_id in worker_cuda_visible_devices.split(",") if gpu_id.strip()]
                    )
            node_gpu_ids = _dedupe_preserve_order(node_gpu_ids)
            node_cuda_visible_devices = ",".join(node_gpu_ids)
            node_id = worker_node_ids[node_rank * self.gpus_per_node]
            logger.info(
                "SGLangReplica launch topology: replica_rank=%s node_rank=%s node_id=%s worker_infos=%s resolved_cuda_visible_devices=%s",
                self.replica_rank,
                node_rank,
                node_id,
                node_worker_infos,
                node_cuda_visible_devices,
            )
            print(
                "SGLangReplica launch topology: "
                f"replica_rank={self.replica_rank}, node_rank={node_rank}, node_id={node_id}, "
                f"worker_infos={node_worker_infos}, resolved_cuda_visible_devices={node_cuda_visible_devices}"
            )
            if not node_cuda_visible_devices:
                raise RuntimeError(
                    "Failed to resolve CUDA_VISIBLE_DEVICES for async SGLang server. "
                    f"worker_infos={node_worker_infos}"
                )
            name = (
                f"sglang_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"sglang_server_reward_{self.replica_rank}_{node_rank}"
            )
            server = SGLangHttpServer.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
