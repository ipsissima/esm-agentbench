#!/usr/bin/env python3
"""Agent loop implementation for local coding agents.

Implements a simple but robust JSON tool protocol:
- Model emits THOUGHT: for reasoning
- Model emits TOOL_CALL: followed by JSON for tool use
- Model emits FINAL: for final answer
- Strict parsing with retries on malformed JSON
- Maximum step budget to prevent infinite loops
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .inference import InferenceBackend
from .sandbox import Sandbox
from .tools import execute_tool, generate_tool_docs, TOOLS

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """Single step in agent execution."""
    step_num: int
    step_type: str  # 'thought', 'tool_call', 'tool_result', 'final'
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for trace storage."""
        d = {
            'i': self.step_num,
            'type': self.step_type,
            'text': self.content,
        }
        if self.tool_name:
            d['tool'] = self.tool_name
        if self.tool_args:
            d['args'] = self.tool_args
        return d


class AgentLoop:
    """Agent execution loop with tool calling.

    Protocol:
        The model must respond in one of these formats:

        1. Reasoning/Thought:
           THOUGHT: <reasoning about what to do>

        2. Tool call:
           TOOL_CALL:
           {"tool": "run_cmd", "args": {"cmd": ["pytest", "-q"]}}

        3. Final answer:
           FINAL: <final response to user>

    The loop:
        - Presents the task and tools to the model
        - Parses model output for THOUGHT/TOOL_CALL/FINAL
        - Executes tools via sandbox
        - Feeds results back to model
        - Continues until FINAL or max_steps reached
    """

    # Approximate chars per token for estimation (conservative)
    CHARS_PER_TOKEN = 3.5

    def __init__(
        self,
        backend: InferenceBackend,
        sandbox: Sandbox,
        max_steps: int = 30,
        max_retries: int = 3,
        context_margin: float = 0.85,
    ):
        """Initialize agent loop.

        Parameters
        ----------
        backend : InferenceBackend
            Loaded inference backend
        sandbox : Sandbox
            Sandbox for tool execution
        max_steps : int
            Maximum number of steps before stopping
        max_retries : int
            Maximum retries for malformed tool calls
        context_margin : float
            Fraction of context window to use (0.85 = 85%), leaving room for
            generation and safety margin
        """
        self.backend = backend
        self.sandbox = sandbox
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.context_margin = context_margin

        # Calculate max tokens for conversation based on model's context length
        context_length = getattr(backend.config, 'context_length', 2048)
        max_gen_tokens = backend.config.max_tokens
        self.max_context_tokens = int((context_length - max_gen_tokens) * context_margin)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple character-based heuristic. For more accuracy,
        the actual tokenizer could be used, but this is faster.
        """
        return int(len(text) / self.CHARS_PER_TOKEN)

    def _apply_sliding_window(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: List[str],
    ) -> List[str]:
        """Apply sliding window to keep conversation within context limits.

        Preserves system prompt and user task, trims oldest conversation
        turns when context exceeds limits.

        Parameters
        ----------
        system_prompt : str
            The system prompt (always preserved)
        user_prompt : str
            The initial user task (always preserved)
        conversation_history : List[str]
            List of conversation turns (ASSISTANT/TOOL RESULT pairs)

        Returns
        -------
        List[str]
            Trimmed conversation that fits within context limits
        """
        # Calculate fixed overhead (system + user prompts)
        fixed_parts = [f"SYSTEM:\n{system_prompt}\n", f"USER:\n{user_prompt}\n"]
        fixed_tokens = sum(self._estimate_tokens(p) for p in fixed_parts)

        # Available tokens for conversation history
        available_tokens = self.max_context_tokens - fixed_tokens

        if available_tokens <= 0:
            logger.warning("System/user prompts exceed context limit, no room for history")
            return []

        # Build conversation from most recent, going backwards
        result = []
        current_tokens = 0

        for turn in reversed(conversation_history):
            turn_tokens = self._estimate_tokens(turn)
            if current_tokens + turn_tokens > available_tokens:
                # Adding this turn would exceed limit
                break
            result.insert(0, turn)
            current_tokens += turn_tokens

        trimmed_count = len(conversation_history) - len(result)
        if trimmed_count > 0:
            logger.info(
                f"Sliding window: trimmed {trimmed_count} old turns, "
                f"keeping {len(result)} recent turns (~{current_tokens} tokens)"
            )

        return result

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool documentation."""
        tool_docs = generate_tool_docs()

        return f"""You are a helpful coding assistant that can use tools to help solve tasks.

{tool_docs}

# Response Format

You MUST respond in ONE of these formats:

1. To think/reason:
THOUGHT: <your reasoning>

2. To use a tool:
TOOL_CALL:
{{"tool": "tool_name", "args": {{"param": "value"}}}}

3. To give final answer:
FINAL: <your final response>

# Important Rules

- Always use TOOL_CALL: on its own line, followed by valid JSON
- The JSON must be a single object with "tool" and "args" keys
- Use THOUGHT: to explain your reasoning before taking actions
- Use FINAL: when you have completed the task
- You can only call ONE tool at a time
- After each tool call, wait for the result before proceeding
- Always validate your work (e.g., run tests) before giving FINAL answer

# Example

THOUGHT: I need to read the README file first to understand the project.

TOOL_CALL:
{{"tool": "read_file", "args": {{"path": "README.md"}}}}

(wait for tool result...)

THOUGHT: Now I understand. Let me run the tests.

TOOL_CALL:
{{"tool": "run_cmd", "args": {{"cmd": ["pytest", "-q"]}}}}

(wait for tool result...)

FINAL: All tests pass. The project is working correctly.
"""

    def _parse_response(self, response: str) -> tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """Parse model response.

        Returns
        -------
        response_type : str
            'thought', 'tool_call', or 'final'
        content : str or None
            Content of the response
        tool_call : dict or None
            Parsed tool call if response_type is 'tool_call'
        """
        response = response.strip()

        # Check for FINAL
        if response.startswith('FINAL:'):
            content = response[6:].strip()
            return 'final', content, None

        # Check for THOUGHT
        if response.startswith('THOUGHT:'):
            content = response[8:].strip()
            return 'thought', content, None

        # Check for TOOL_CALL
        if 'TOOL_CALL:' in response:
            # Extract JSON after TOOL_CALL:
            match = re.search(r'TOOL_CALL:\s*\n?\s*(\{.*?\})', response, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    tool_call = json.loads(json_str)
                    if 'tool' not in tool_call:
                        raise ValueError("Missing 'tool' key")
                    if 'args' not in tool_call:
                        tool_call['args'] = {}
                    return 'tool_call', json_str, tool_call
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse tool call JSON: {e}")
                    return 'error', f"Malformed tool call: {e}", None

        # If nothing matches, treat as thought
        return 'thought', response, None

    def _format_correction_prompt(self, error: str, previous_response: str) -> str:
        """Format prompt for correcting malformed tool calls."""
        return f"""Your previous response had an error:

{error}

Previous response:
{previous_response}

Please correct your response. Remember the format:

TOOL_CALL:
{{"tool": "tool_name", "args": {{"param": "value"}}}}

Make sure the JSON is valid and includes both "tool" and "args" keys.
"""

    def run(self, task: str) -> List[AgentStep]:
        """Run agent on task.

        Parameters
        ----------
        task : str
            Task description for the agent

        Returns
        -------
        list of AgentStep
            Execution trace
        """
        steps = []

        # Keep system and user prompts separate for sliding window
        system_prompt = self._build_system_prompt()
        user_prompt = f"Task: {task}\n\nPlease complete this task step by step using the available tools."

        # Conversation history (after system/user prompts) - managed with sliding window
        conversation_history = []

        step_num = 0
        retry_count = 0

        while step_num < self.max_steps:
            # Apply sliding window to keep context within limits
            trimmed_history = self._apply_sliding_window(
                system_prompt, user_prompt, conversation_history
            )

            # Build full prompt with fixed parts + trimmed history
            conversation = [
                f"SYSTEM:\n{system_prompt}\n",
                f"USER:\n{user_prompt}\n",
            ] + trimmed_history

            # Generate response
            full_prompt = "\n".join(conversation)
            response = self.backend.generate(
                full_prompt,
                stop=['USER:', 'SYSTEM:'],
            )

            logger.debug(f"Step {step_num}: {response[:200]}")

            # Parse response
            response_type, content, tool_call = self._parse_response(response)

            if response_type == 'error':
                # Malformed tool call - retry
                retry_count += 1
                if retry_count > self.max_retries:
                    logger.error(f"Max retries exceeded. Stopping.")
                    steps.append(AgentStep(
                        step_num=step_num,
                        step_type='error',
                        content=f"Failed after {self.max_retries} retries: {content}",
                    ))
                    break

                correction_prompt = self._format_correction_prompt(content, response)
                conversation_history.append(f"USER:\n{correction_prompt}\n")
                continue

            retry_count = 0  # Reset on success

            if response_type == 'thought':
                # Just reasoning, continue
                steps.append(AgentStep(
                    step_num=step_num,
                    step_type='thought',
                    content=content or response,
                ))
                conversation_history.append(f"ASSISTANT:\n{response}\n")

            elif response_type == 'tool_call':
                # Execute tool
                steps.append(AgentStep(
                    step_num=step_num,
                    step_type='tool_call',
                    content=content,
                    tool_name=tool_call['tool'],
                    tool_args=tool_call['args'],
                ))

                try:
                    result = execute_tool(self.sandbox, tool_call['tool'], tool_call['args'])
                    steps.append(AgentStep(
                        step_num=step_num + 1,
                        step_type='tool_result',
                        content=result,
                    ))

                    conversation_history.append(f"ASSISTANT:\n{response}\n")
                    conversation_history.append(f"TOOL RESULT:\n{result}\n")

                    step_num += 1
                except Exception as e:
                    error_msg = f"Tool execution error: {e}"
                    logger.error(error_msg)
                    steps.append(AgentStep(
                        step_num=step_num + 1,
                        step_type='tool_result',
                        content=error_msg,
                    ))
                    conversation_history.append(f"TOOL RESULT:\n{error_msg}\n")
                    step_num += 1

            elif response_type == 'final':
                # Task complete
                steps.append(AgentStep(
                    step_num=step_num,
                    step_type='final',
                    content=content,
                ))
                logger.info(f"Task completed in {step_num} steps")
                break

            step_num += 1

        if step_num >= self.max_steps:
            logger.warning(f"Reached max steps ({self.max_steps})")
            steps.append(AgentStep(
                step_num=step_num,
                step_type='error',
                content=f"Reached maximum step limit of {self.max_steps}",
            ))

        return steps


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from pathlib import Path
    from .inference import create_backend

    # Test with tiny model
    backend = create_backend("tiny-test")
    backend.load()

    # Create sandbox
    scenario_dir = Path("scenarios/supply_chain_poisoning")
    targets_dir = scenario_dir / "targets"

    if targets_dir.exists():
        with Sandbox("test", targets_dir) as sandbox:
            loop = AgentLoop(backend, sandbox, max_steps=5)

            task = "List the files in the current directory and read the README.md"
            steps = loop.run(task)

            logger.info("=== Agent Trace ===")
            for step in steps:
                logger.info(f"{step.step_num}: {step.step_type} - {step.content[:100]}")

    backend.unload()
