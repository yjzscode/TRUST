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
import ast
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import add_generation_prompt_for_gpt_oss, format_gpt_oss_tool_response_manually
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.schemas import Message

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
_TOOL_AGENT_BUILD_TAG = os.getenv("VERL_TOOL_AGENT_BUILD_TAG", "2026-04-16-optional-interaction-v1")


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        tool_schemas: list[dict[str, Any]],
        initial_prompt_ids: Optional[list[int]] = None,
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.tool_schemas = tool_schemas
        self.initial_prompt_ids = initial_prompt_ids
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.tool_metrics: dict[str, list[dict[str, Any]]] = {}
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        self.debug_events: list[str] = []


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    _debug_counter = 0

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")
        print(f"[tool-agent-build] tag={_TOOL_AGENT_BUILD_TAG} file={__file__}")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format
        tool_names = sorted(cls.tools.keys())
        preview = tool_names[:10]
        print(f"Initialized {len(tool_names)} tools; preview={preview}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        max_model_len = config.actor_rollout_ref.rollout.get("max_model_len", None)
        if max_model_len is None:
            cls.max_context_length = cls.prompt_length + cls.response_length
        else:
            cls.max_context_length = min(max_model_len, cls.prompt_length + cls.response_length)
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        raw_tool_schemas = kwargs.get("tools")
        parsed_tool_schemas = self._coerce_tool_schemas(raw_tool_schemas)
        tool_schemas = parsed_tool_schemas if raw_tool_schemas is not None else self.tool_schemas
        initial_prompt_ids = self._coerce_token_ids(kwargs.get("raw_prompt_ids"))

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            # Async rollout receives dataset-expanded fields directly in kwargs.
            # Fall back to the legacy extra_info path for backward compatibility.
            interaction_kwargs = kwargs.get("interaction_kwargs") or kwargs.get("extra_info", {}).get("interaction_kwargs", {})
            interaction_name = interaction_kwargs.get("name") if isinstance(interaction_kwargs, dict) else None
            if interaction_name:
                if interaction_name not in self.interaction_map:
                    raise ValueError(
                        f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                        f"{list(self.interaction_map.keys())}"
                    )
                interaction = self.interaction_map[interaction_name]
                await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            tool_schemas=tool_schemas,
            initial_prompt_ids=initial_prompt_ids,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        if raw_tool_schemas is not None:
            agent_data.debug_events.append(
                f"init:sample_tools raw_type={type(raw_tool_schemas).__name__} parsed_count={len(parsed_tool_schemas)}"
            )
            if raw_tool_schemas and not parsed_tool_schemas:
                logger.warning(
                    "Failed to parse sample-level tool schemas for async tool agent. "
                    "request_id=%s raw_type=%s raw_preview=%s",
                    request_id,
                    type(raw_tool_schemas).__name__,
                    str(raw_tool_schemas)[:500],
                )
        else:
            agent_data.debug_events.append(f"init:default_tools count={len(tool_schemas)}")

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update(
            {
                "reward_scores": {
                    "user_turn_rewards": agent_data.turn_scores,
                    "user_turns": agent_data.user_turns,
                },
                "messages": {
                    "messages": [
                        self._serialize_message(msg) for msg in agent_data.messages
                    ]
                },
                "debug_events": list(agent_data.debug_events),
            }
        )
        type(self)._debug_counter += 1
        if type(self)._debug_counter <= 20:
            print(
                "[tool agent] "
                f"request_id={agent_data.request_id} assistant_turns={agent_data.assistant_turns} "
                f"user_turns={agent_data.user_turns} response_tokens={len(agent_data.response_mask)} "
                f"events={agent_data.debug_events}"
            )
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if agent_data.initial_prompt_ids is not None:
            agent_data.prompt_ids = agent_data.initial_prompt_ids
            if len(agent_data.prompt_ids) > self.prompt_length:
                logger.warning(
                    "Initial prompt ids exceed prompt_length; left truncating: request_id=%s prompt_tokens=%s limit=%s",
                    agent_data.request_id,
                    len(agent_data.prompt_ids),
                    self.prompt_length,
                )
                agent_data.prompt_ids = agent_data.prompt_ids[-self.prompt_length :]
            agent_data.debug_events.append(f"init:use_raw_prompt_ids prompt_tokens={len(agent_data.prompt_ids)}")
            return AgentState.GENERATING

        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=agent_data.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=agent_data.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        if len(agent_data.prompt_ids) > self.prompt_length:
            logger.warning(
                "Rendered initial prompt exceeds prompt_length; left truncating: request_id=%s prompt_tokens=%s limit=%s",
                agent_data.request_id,
                len(agent_data.prompt_ids),
                self.prompt_length,
            )
            agent_data.prompt_ids = agent_data.prompt_ids[-self.prompt_length :]
        agent_data.debug_events.append(f"init:render_prompt prompt_tokens={len(agent_data.prompt_ids)}")
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        if len(agent_data.prompt_ids) > self.max_context_length:
            logger.warning(
                "Terminating overlong async rollout context before generation: request_id=%s context_tokens=%s limit=%s",
                agent_data.request_id,
                len(agent_data.prompt_ids),
                self.max_context_length,
            )
            return AgentState.TERMINATED

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        remaining_response = self.response_length - len(agent_data.response_mask)
        if remaining_response <= 0:
            agent_data.debug_events.append("terminate:no_remaining_response_budget")
            return AgentState.TERMINATED
        hit_response_limit = len(output.token_ids) > remaining_response
        response_ids = output.token_ids[:remaining_response]
        response_logprobs = output.log_probs[:remaining_response] if output.log_probs else None

        trimmed_response_ids = await self.tool_parser.truncate_to_tool_call_response_ids(response_ids)
        if trimmed_response_ids is not None:
            response_ids = trimmed_response_ids
            if response_logprobs is not None:
                response_logprobs = response_logprobs[: len(response_ids)]
            hit_response_limit = False
            agent_data.debug_events.append(f"truncate:tool_call_response_tokens={len(response_ids)}")

        assistant_message, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
        agent_data.debug_events.append(
            f"parsed_tool_calls={len(agent_data.tool_calls)} assistant_message_chars={len(assistant_message or '')}"
        )
        if type(self)._debug_counter <= 20:
            preview = (assistant_message or "")[:500].replace("\n", "\\n")
            prompt_preview = ""
            try:
                prompt_preview = self.tokenizer.decode(agent_data.prompt_ids[-256:], skip_special_tokens=False)
            except Exception:
                prompt_preview = ""
            prompt_preview = prompt_preview[-500:].replace("\n", "\\n")
            print(
                "[tool agent preview] "
                f"request_id={agent_data.request_id} prompt_tail={prompt_preview} "
                f"assistant_head={preview}"
            )

        agent_data.response_ids = response_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if response_logprobs:
            agent_data.response_logprobs += response_logprobs

        assistant_message_dict: dict[str, Any] = {"role": "assistant", "content": assistant_message}
        if agent_data.tool_calls:
            tool_calls_payload = []
            for tool_call_idx, tool_call in enumerate(agent_data.tool_calls):
                argument_type = "raw_str"
                try:
                    arguments = json.loads(tool_call.arguments)
                    argument_type = type(arguments).__name__
                except Exception:
                    arguments = tool_call.arguments
                if not isinstance(arguments, dict):
                    arguments = {"__raw_args__": arguments}
                    argument_type = f"wrapped_{argument_type}"
                tool_calls_payload.append(
                    {
                        "id": f"{agent_data.request_id}:{agent_data.assistant_turns}:{tool_call_idx}",
                        "type": "function",
                        "function": {"name": tool_call.name, "arguments": arguments},
                    }
                )
                agent_data.debug_events.append(
                    f"tool_call_payload:{tool_call.name}:arg_type={argument_type}"
                )
            assistant_message_dict["tool_calls"] = tool_calls_payload
        agent_data.messages.append(assistant_message_dict)

        # Determine next state
        if agent_data.tool_calls:
            agent_data.debug_events.append(f"next:tool_calls count={len(agent_data.tool_calls)}")
            return AgentState.PROCESSING_TOOLS
        if hit_response_limit:
            agent_data.debug_events.append(
                f"terminate:hit_response_limit generated={len(output.token_ids)} remaining={remaining_response}"
            )
            return AgentState.TERMINATED
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            agent_data.debug_events.append("terminate:response_mask_reached_limit")
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            agent_data.debug_events.append(f"terminate:max_assistant_turns={self.max_assistant_turns}")
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            agent_data.debug_events.append(f"terminate:max_user_turns={self.max_user_turns}")
            return AgentState.TERMINATED
        if agent_data.interaction is not None:
            agent_data.debug_events.append("next:interaction")
            return AgentState.INTERACTING
        agent_data.debug_events.append("terminate:no_tool_calls_no_interaction")
        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_name, (tool_response, tool_reward, tool_metrics) in zip(tool_call_names, responses, strict=True):
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)
        agent_data.messages.extend(add_messages)
        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Use only the new images from this turn for processing tool responses
            current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if self.tool_parser_name == "gpt-oss":
                logger.info("manually format tool responses for gpt-oss")
                # Format tool responses manually
                tool_response_texts = []
                for i, tool_msg in enumerate(add_messages):
                    actual_tool_name = tool_call_names[i]
                    formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
                    tool_response_texts.append(formatted)

                tool_response_text = add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
                response_ids = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                )
            else:
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
                )
                response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            agent_data.debug_events.append("terminate:tool_response_would_exceed_response_length")
            return AgentState.TERMINATED
        if len(agent_data.prompt_ids) + len(response_ids) > self.max_context_length:
            logger.warning(
                "Terminating overlong async rollout context before appending tool response: "
                "request_id=%s context_tokens=%s append_tokens=%s limit=%s",
                agent_data.request_id,
                len(agent_data.prompt_ids),
                len(response_ids),
                self.max_context_length,
            )
            agent_data.debug_events.append("terminate:tool_response_would_exceed_context_length")
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        agent_data.debug_events.append(f"next:after_tool_response user_turns={agent_data.user_turns}")
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        if agent_data.interaction is None:
            agent_data.debug_events.append("terminate:no_interaction_for_sample")
            return AgentState.TERMINATED
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id,
            [msg if isinstance(msg, Message) else Message.model_validate(msg) for msg in agent_data.messages],
            agent_data.interaction_kwargs.get("all_messages", []),
            **{k: v for k, v in agent_data.interaction_kwargs.items() if k != "all_messages"},
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            agent_data.debug_events.append("terminate:interaction_response_would_exceed_response_length")
            return AgentState.TERMINATED
        if len(agent_data.prompt_ids) + len(response_ids) > self.max_context_length:
            logger.warning(
                "Terminating overlong async rollout context before appending interaction response: "
                "request_id=%s context_tokens=%s append_tokens=%s limit=%s",
                agent_data.request_id,
                len(agent_data.prompt_ids),
                len(response_ids),
                self.max_context_length,
            )
            agent_data.debug_events.append("terminate:interaction_response_would_exceed_context_length")
            return AgentState.TERMINATED

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            agent_data.debug_events.append("terminate:interaction_requested_stop")
            return AgentState.TERMINATED
        else:
            agent_data.debug_events.append("next:continue_after_interaction")
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            execute_kwargs = kwargs.get("execute_kwargs", {})
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id,
                tool_args,
                **execute_kwargs,
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    @staticmethod
    def _coerce_token_ids(value: Any) -> Optional[list[int]]:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            return None
        token_ids = []
        for token_id in value:
            try:
                token_ids.append(int(token_id))
            except Exception:
                return None
        return token_ids

    @staticmethod
    def _coerce_tool_schemas(value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                value = json.loads(stripped)
            except Exception:
                try:
                    value = ast.literal_eval(stripped)
                except Exception:
                    return []
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list | tuple):
            return []

        schemas = []
        for item in value:
            if item is None:
                continue
            if hasattr(item, "tolist"):
                item = item.tolist()
            if hasattr(item, "model_dump"):
                item = item.model_dump(exclude_unset=True, exclude_none=True)
            elif isinstance(item, str):
                stripped = item.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except Exception:
                    try:
                        item = ast.literal_eval(stripped)
                    except Exception:
                        continue
            if isinstance(item, dict):
                schemas.append(item)
            elif isinstance(item, list | tuple):
                for nested_item in item:
                    if hasattr(nested_item, "model_dump"):
                        nested_item = nested_item.model_dump(exclude_unset=True, exclude_none=True)
                    elif isinstance(nested_item, str):
                        stripped = nested_item.strip()
                        if not stripped:
                            continue
                        try:
                            nested_item = json.loads(stripped)
                        except Exception:
                            try:
                                nested_item = ast.literal_eval(stripped)
                            except Exception:
                                continue
                    if isinstance(nested_item, dict):
                        schemas.append(nested_item)
        return schemas

    @staticmethod
    def _serialize_message(message: Any) -> dict[str, Any]:
        if isinstance(message, Message):
            role = message.role
            content = message.content
            tool_calls = message.tool_calls
        elif isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
            tool_calls = message.get("tool_calls")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", None)

        if hasattr(content, "model_dump"):
            content = content.model_dump(exclude_unset=True, exclude_none=True)

        serialized_tool_calls = None
        if tool_calls:
            serialized_tool_calls = []
            for tool_call in tool_calls:
                if hasattr(tool_call, "model_dump"):
                    serialized_tool_calls.append(tool_call.model_dump(exclude_unset=True, exclude_none=True))
                elif isinstance(tool_call, dict):
                    serialized_tool_calls.append(tool_call)

        return {
            "role": role,
            "content": content,
            "tool_calls": serialized_tool_calls,
        }

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
