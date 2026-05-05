# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Union

import numpy as np

try:
    import transformers

    HAVE_TRANSFORMERS = True
except ModuleNotFoundError:
    HAVE_TRANSFORMERS = False


# fmt: off
nemotron_h_aligned_custom_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + message['content'].strip() + '\n' + '<SPECIAL_11>Assistant\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() + '\n' }}{% endif %}{% endfor %}""" # pylint: disable=line-too-long
nemotron_nano_v2_custom_template = """{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<SPECIAL_11>Assistant\n' + content.strip() + '\n<SPECIAL_12>\n' }}{% endif %}{% endfor %}""" # pylint: disable=line-too-long
identity_template = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""
# fmt: on

DEEPSEEK_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DEEPSEEK_EOS_TOKEN = "<｜end▁of▁sentence｜>"
DEEPSEEK_USER_TOKEN = "<｜User｜>"
DEEPSEEK_ASSISTANT_TOKEN = "<｜Assistant｜>"
DEEPSEEK_THINKING_END_TOKEN = "</think>"
DEEPSEEK_DSML_TOKEN = "｜DSML｜"

DEEPSEEK_TOOLS_SYSTEM_TEMPLATE = """## Tools

You have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a "<{dsml_token}function_calls>" block like the following as part of your reply to the user:
<{dsml_token}function_calls>
<{dsml_token}invoke name="$FUNCTION_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$FUNCTION_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}function_calls>

String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The "string" attribute should be set to "true" for string type parameters and "false" for other types (numbers, booleans, arrays, objects).

If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:

<{dsml_token}function_calls>
...
</{dsml_token}function_calls>

<function_results>
...
</function_results>

<think>...thinking about results</think>

Here are the functions available in JSONSchema format:
<functions>
{tool_schemas}
</functions>
"""


IGNORE_INDEX = -100


@dataclass
class PromptConfig:
    """Config options for different prompt formats."""

    # How many tokens are used for the assistant prefix, e.g. "<|im_start|>assistant\n".
    # Used for masking the assistant prefix.
    assistant_prefix_len: int
    # Padding token ID.
    pad_token_id: int
    # For overriding the default chat format template.
    custom_chat_template: str
    # If the tokenizer inserts BOS token by default.
    has_bos: bool
    # If the tokenizer supports a separate role for system messages.
    has_system_role: bool
    # Wether to force a specific system message.
    force_system_message: bool = False
    system_default: dict = None


class SFTTokenizer:
    """SFT Tokenizer."""

    def __init__(self, tokenizer_path: str, prompt_format: str):
        """
        Note: Currently, only HuggingFaceTokenizer is supported as the underlying text tokenizer.

        Args:
            tokenizer_path (str): Underlying tokenizer path.
            prompt_format (str): Prompt format for the tokenizer.
        """
        if HAVE_TRANSFORMERS:
            # Currently, only HuggingFace tokenizers are supported.
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_path
            )
        else:
            raise ImportError(
                "SFTTokenizer currently requires transformers library to be installed"
            )

        self._vocab_size = len(tokenizer)
        self._tokenizer = tokenizer

        if prompt_format == "nemotron-nano-v2":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=nemotron_nano_v2_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "nemotron-h-aligned":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<SPECIAL_233>"),
                custom_chat_template=nemotron_h_aligned_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "identity":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=identity_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "default":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                custom_chat_template=tokenizer.chat_template,
                has_bos=tokenizer.bos_token_id is not None,
                has_system_role=True,
            )
        elif prompt_format in ("deepseek-v3.2", "deepseek-v32"):
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                custom_chat_template=None,
                has_bos=True,
                has_system_role=True,
            )
        else:
            raise NotImplementedError("unknown SFT prompt format", prompt_format)

        self._prompt_format = prompt_format

    def tokenize_conversation(
        self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool
    ):
        """Convert a conversation to tokens.

        Args:
            conversation (List[Dict]): Sequence of system/user/assistant messages.
                Must be in the following format:
                [
                    {"role": "system", "content": "something"},
                    {"role": "user", "content": "something1"},
                    {"role": "assistant", "content": "something2"},
                ]
            return_target (bool): Return target tokens with system and assistant masked.
            add_generation_prompt (bool): Add assistant prefix to the end.
        """
        if self._prompt_format in ("deepseek-v3.2", "deepseek-v32"):
            return self._tokenize_deepseek_v32_conversation(
                conversation=conversation,
                return_target=return_target,
                add_generation_prompt=add_generation_prompt,
            )

        # Skip system message if the tokenizer doesn't have a system role.
        if not self._prompt_config.has_system_role and conversation[0]["role"] == "system":
            conversation = conversation[1:]

        tokens = self._tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_assistant_token_mask=False,
            return_tensors="np",
            chat_template=self._prompt_config.custom_chat_template,
        )[0]

        if not return_target:
            return tokens

        target = tokens.copy()

        # When using the default prompt format, we do not replace any tokens with IGNORE_INDEX.
        # Instead, all token losses will be used for simplicity.
        if self._prompt_format == "default":
            return tokens, target

        # Mask system and user tokens in the target.
        idx = 0
        for turn_idx, turn in enumerate(conversation):

            if turn["role"].lower() == "assistant" and len(turn["content"]) == 0:
                raise ValueError(f"empty assistant turn in conversation: {conversation}.")
            if turn["role"].lower() == "assistant":
                assert conversation[turn_idx - 1]["role"].lower() in ("user", "tool")

            turn_tokens = self._tokenizer.apply_chat_template(
                [turn], tokenize=True, chat_template=self._prompt_config.custom_chat_template
            )

            # There should be only one BOS at the very beginning.
            # After the first turn, skip BOS token.
            if self._prompt_config.has_bos and turn_idx > 0:
                turn_tokens = turn_tokens[1:]
            turn_len = len(turn_tokens)

            role = turn["role"].lower()
            if role in ("system", "user", "tool"):
                target[idx : idx + turn_len] = IGNORE_INDEX
            elif role == "assistant":
                if self._prompt_config.assistant_prefix_len > 0:
                    target[idx : idx + self._prompt_config.assistant_prefix_len] = IGNORE_INDEX
            else:
                raise ValueError("Wrong role value.")

            assert np.allclose(
                tokens[idx : idx + turn_len], turn_tokens
            ), f"expected turn tokens to match tokens in conversation {conversation}"

            idx += turn_len

        assert idx == len(tokens), f"mismatch in target masking the conversation {conversation}"

        return tokens, target

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @classmethod
    def _maybe_json_loads(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @classmethod
    def _deepseek_tools_from_openai_format(cls, tools: Any) -> List[Dict[str, Any]]:
        tools = cls._maybe_json_loads(tools)
        if not tools:
            return []
        normalized = []
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                normalized.append(tool["function"])
            else:
                normalized.append(tool)
        return normalized

    @classmethod
    def _deepseek_tool_calls_from_openai_format(cls, tool_calls: Any) -> List[Dict[str, str]]:
        tool_calls = cls._maybe_json_loads(tool_calls)
        if not tool_calls:
            return []

        normalized = []
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                arguments = function.get("arguments", "{}")
                name = function.get("name", "")
            else:
                arguments = tool_call.get("arguments", "{}")
                name = tool_call.get("name", "")
            if not isinstance(arguments, str):
                arguments = cls._json_dumps(arguments)
            normalized.append({"name": name, "arguments": arguments})
        return normalized

    @classmethod
    def _deepseek_render_tools(cls, tools: Any) -> str:
        tool_schemas = [cls._json_dumps(tool) for tool in cls._deepseek_tools_from_openai_format(tools)]
        return DEEPSEEK_TOOLS_SYSTEM_TEMPLATE.format(
            tool_schemas="\n".join(tool_schemas),
            dsml_token=DEEPSEEK_DSML_TOKEN,
        )

    @classmethod
    def _deepseek_render_tool_call_arguments(cls, tool_call: Dict[str, str]) -> str:
        parameter_template = (
            '<{dsml_token}parameter name="{key}" string="{is_str}">{value}'
            '</{dsml_token}parameter>'
        )
        arguments = cls._maybe_json_loads(tool_call.get("arguments", "{}"))
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}

        rendered = []
        for key, value in arguments.items():
            rendered.append(
                parameter_template.format(
                    dsml_token=DEEPSEEK_DSML_TOKEN,
                    key=key,
                    is_str="true" if isinstance(value, str) else "false",
                    value=value if isinstance(value, str) else cls._json_dumps(value),
                )
            )
        return "\n".join(rendered)

    @classmethod
    def _deepseek_render_tool_calls(cls, tool_calls: Any) -> str:
        rendered_calls = []
        for tool_call in cls._deepseek_tool_calls_from_openai_format(tool_calls):
            rendered_calls.append(
                '<{dsml_token}invoke name="{name}">\n{arguments}\n'
                '</{dsml_token}invoke>'.format(
                    dsml_token=DEEPSEEK_DSML_TOKEN,
                    name=tool_call.get("name", ""),
                    arguments=cls._deepseek_render_tool_call_arguments(tool_call),
                )
            )
        if not rendered_calls:
            return ""
        return (
            "\n\n"
            f"<{DEEPSEEK_DSML_TOKEN}function_calls>\n"
            + "\n".join(rendered_calls)
            + f"\n</{DEEPSEEK_DSML_TOKEN}function_calls>"
        )

    @classmethod
    def _deepseek_render_message(
        cls, index: int, conversation: List[Dict[str, Any]]
    ) -> str:
        message = conversation[index]
        role = message.get("role")
        content = message.get("content") or ""

        if role == "system":
            prompt = content
            if message.get("tools"):
                prompt += "\n\n" + cls._deepseek_render_tools(message["tools"])
            if message.get("response_format"):
                prompt += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
                prompt += cls._json_dumps(message["response_format"])
            return prompt

        if role == "developer":
            developer_content = ""
            if message.get("tools"):
                developer_content += "\n\n" + cls._deepseek_render_tools(message["tools"])
            if message.get("response_format"):
                developer_content += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
                developer_content += cls._json_dumps(message["response_format"])
            developer_content += f"\n\n# The user's message is: {content}"
            return f"{DEEPSEEK_USER_TOKEN}{developer_content}{DEEPSEEK_ASSISTANT_TOKEN}{DEEPSEEK_THINKING_END_TOKEN}"

        if role == "user":
            return f"{DEEPSEEK_USER_TOKEN}{content}{DEEPSEEK_ASSISTANT_TOKEN}{DEEPSEEK_THINKING_END_TOKEN}"

        if role == "assistant":
            tool_calls = cls._deepseek_render_tool_calls(message.get("tool_calls"))
            return f"{content}{tool_calls}{DEEPSEEK_EOS_TOKEN}"

        if role == "tool":
            previous_assistant_idx = index - 1
            while (
                previous_assistant_idx >= 0
                and conversation[previous_assistant_idx].get("role") == "tool"
            ):
                previous_assistant_idx -= 1
            if (
                previous_assistant_idx < 0
                or conversation[previous_assistant_idx].get("role") != "assistant"
            ):
                raise ValueError(f"tool message must follow an assistant tool call: {conversation}")

            assistant_tool_calls = cls._deepseek_tool_calls_from_openai_format(
                conversation[previous_assistant_idx].get("tool_calls")
            )
            if not assistant_tool_calls:
                raise ValueError(f"tool message missing preceding assistant tool_calls: {conversation}")

            tool_call_order = index - previous_assistant_idx
            prompt = ""
            if tool_call_order == 1:
                prompt += "\n\n<function_results>"
            prompt += f"\n<result>{content}</result>"
            if tool_call_order == len(assistant_tool_calls):
                prompt += f"\n</function_results>\n\n{DEEPSEEK_THINKING_END_TOKEN}"
            return prompt

        raise ValueError(f"Wrong role value {role}.")

    def _encode_deepseek_segment(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _tokenize_deepseek_v32_conversation(
        self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool
    ):
        """Tokenize DeepSeek-V3.2 chat/tool data and mask non-assistant spans."""
        all_tokens: List[int] = []
        all_targets: List[int] = []

        bos_tokens = self._encode_deepseek_segment(DEEPSEEK_BOS_TOKEN)
        all_tokens.extend(bos_tokens)
        all_targets.extend(bos_tokens)
        if return_target:
            all_targets[: len(bos_tokens)] = [IGNORE_INDEX] * len(bos_tokens)

        for index, message in enumerate(conversation):
            rendered = self._deepseek_render_message(index, conversation)
            segment_tokens = self._encode_deepseek_segment(rendered)
            all_tokens.extend(segment_tokens)

            role = message.get("role")
            if return_target and role in ("system", "user", "developer", "tool"):
                all_targets.extend([IGNORE_INDEX] * len(segment_tokens))
            else:
                all_targets.extend(segment_tokens)

        tokens = np.asarray(all_tokens, dtype=np.int64)
        if not return_target:
            return tokens

        return tokens, np.asarray(all_targets, dtype=np.int64)

    def text_to_ids(self, text: Union[str, List[Dict]]):
        """Tokenize conversation or string input."""
        if isinstance(text, list):
            # This code path is used by the inference code currently.
            return self.tokenize_conversation(
                text, return_target=False, add_generation_prompt=True
            ).tolist()

        return self._tokenizer.encode(text)

    def tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def ids_to_text(self, tokens: List[int]):
        """Detokenize tokens."""
        return self._tokenizer.decode(tokens)

    def ids_to_tokens(self):
        """Converts ids to tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def text_to_tokens(self):
        """Converts text to tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def tokens_to_text(self):
        """Converts tokens to text."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def get_special_tokens(self):
        """Get special tokens."""
        return self._tokenizer.get_added_vocab()

    def add_special_tokens(self):
        """Add special tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    @property
    def pad_id(self):
        """Pad token ID."""
        return self._prompt_config.pad_token_id

    @property
    def bos_id(self):
        """Beginning of sequence token ID."""
        return self._tokenizer.bos_token_id

    @property
    def eod(self):
        """End of sentence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def vocab(self):
        """Vocab."""
        return NotImplementedError("not used")

    @property
    def inv_vocab(self):
        """Inverse vocab."""
        return NotImplementedError("not used")

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size
