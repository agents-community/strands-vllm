"""Agent integration tests with real MATH-500 problems.

Tests the full Agent pipeline with VLLMModel using problems from the
HuggingFace MATH-500 dataset (https://huggingface.co/datasets/HuggingFaceH4/MATH-500).

Fixtures (model base) are provided by conftest.py.

Credit / reference:
- Adapted from `horizon-rl/strands-sglang` integration tests:
  https://github.com/horizon-rl/strands-sglang
"""

from __future__ import annotations

import pytest
from strands import Agent
from strands_tools import calculator

from strands_vllm import TokenManager, VLLMModel, VLLMTokenRecorder

SYSTEM_PROMPT = """You are a math tutor. Always use the calculator tool to solve problems.

The calculator tool supports these modes:
- evaluate: Compute numeric values (default mode)
- solve: Find equation roots
- derive: Compute derivatives (use wrt="x")
- integrate: Compute integrals (use wrt="x")

Show your work and use the calculator for all computations."""


# =============================================================================
# Real MATH-500 Problems (from HuggingFaceH4/MATH-500)
# =============================================================================

MATH500_PROBLEMS = [
    # Index 27: Prealgebra L2 - Bake sale profit
    {
        "id": "math500_27",
        "subject": "Prealgebra",
        "level": 2,
        "problem": (
            "A math club is having a bake sale as a fundraiser. "
            "They sell 54 cookies at three for $1, and 20 cupcakes at $2 each, "
            "and 35 brownies at $1 each. If it cost the math club $15 to bake these items, "
            "what was their profit?"
        ),
        "answer": "78",
        "answer_variants": ["78", "$78", "78 dollars"],
    },
    # Index 38: Algebra L1 - Daily calories
    {
        "id": "math500_38",
        "subject": "Algebra",
        "level": 1,
        "problem": (
            "If a snack-size tin of peaches has 40 calories and is 2% of a person's "
            "daily caloric requirement, how many calories fulfill a person's daily caloric requirement?"
        ),
        "answer": "2000",
        "answer_variants": ["2000", "2,000", "2000 calories"],
    },
    # Index 3: Number Theory L3 - Divisors of 196
    {
        "id": "math500_3",
        "subject": "Number Theory",
        "level": 3,
        "problem": "How many positive whole-number divisors does 196 have?",
        "answer": "9",
        "answer_variants": ["9", "nine"],
    },
    # Index 5: Prealgebra L2 - Hexagon perimeter
    {
        "id": "math500_5",
        "subject": "Prealgebra",
        "level": 2,
        "problem": (
            "A regular hexagon can be divided into six equilateral triangles. "
            "If the perimeter of one of the triangles is 21 inches, "
            "what is the perimeter, in inches, of the regular hexagon?"
        ),
        "answer": "42",
        "answer_variants": ["42", "42 inches"],
    },
    # Index 6: Number Theory L3 - Perfect cube sum
    {
        "id": "math500_6",
        "subject": "Number Theory",
        "level": 3,
        "problem": (
            "What is the smallest positive perfect cube that can be written "
            "as the sum of three consecutive integers?"
        ),
        "answer": "27",
        "answer_variants": ["27", "3^3", "3**3"],
    },
    # Index 2: Algebra L3 - Function evaluation
    {
        "id": "math500_2",
        "subject": "Algebra",
        "level": 3,
        "problem": (
            "If f(x) = (3x-2)/(x-2), what is the value of f(-2) + f(-1) + f(0)? "
            "Express your answer as a common fraction."
        ),
        "answer": "14/3",
        "answer_variants": ["14/3", "\\frac{14}{3}", "4.666", "4.67"],
    },
    # Index 8: Algebra L3 - Distance formula
    {
        "id": "math500_8",
        "subject": "Algebra",
        "level": 3,
        "problem": (
            "What is the distance, in units, between the points (2, -6) and (-4, 3)? "
            "Express your answer in simplest radical form."
        ),
        "answer": "3*sqrt(13)",
        "answer_variants": ["3*sqrt(13)", "3√13", "3\\sqrt{13}", "sqrt(117)", "10.8"],
    },
    # Index 20: Algebra L3 - Complex numbers
    {
        "id": "math500_20",
        "subject": "Algebra",
        "level": 3,
        "problem": "Evaluate (1+2i)*6 - 3i.",
        "answer": "6+9i",
        "answer_variants": ["6+9i", "6 + 9i", "6+9*i"],
    },
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def model(vllm_base_url: str, vllm_model_id: str) -> VLLMModel:
    """Create fresh VLLMModel for each test."""
    return VLLMModel(
        base_url=vllm_base_url,
        model_id=vllm_model_id,
        return_token_ids=True,
        params={"max_tokens": 4096, "temperature": 0},
    )


@pytest.fixture
def recorder() -> VLLMTokenRecorder:
    """Create fresh VLLMTokenRecorder for each test."""
    return VLLMTokenRecorder()


@pytest.fixture
def agent(model: VLLMModel, recorder: VLLMTokenRecorder) -> Agent:
    """Create Agent with calculator tool."""
    return Agent(
        model=model,
        tools=[calculator],
        callback_handler=recorder,
        system_prompt=SYSTEM_PROMPT,
    )


# =============================================================================
# Agent Basic Tests
# =============================================================================


class TestAgentBasic:
    """Basic Agent functionality tests."""

    async def test_agent_simple_query(self, agent: Agent, recorder: VLLMTokenRecorder) -> None:
        """Agent can respond to simple query."""
        await agent.invoke_async("What is 2 + 2?")

        # Should have messages
        assert len(agent.messages) > 0

        # Recorder should have captured tokens
        assert recorder.prompt_token_ids is not None or recorder.token_ids is not None

    async def test_agent_uses_calculator_tool(self, agent: Agent) -> None:
        """Agent uses calculator tool for math."""
        await agent.invoke_async("Calculate 15 * 23 using the calculator tool.")

        # Find tool use in messages
        tool_uses = []
        for msg in agent.messages:
            if msg.get("role") == "assistant":
                for content in msg.get("content", []):
                    if isinstance(content, dict) and "toolUse" in content:
                        tool_uses.append(content["toolUse"])

        # Should have used calculator
        assert len(tool_uses) > 0
        assert any(tu.get("name") == "calculator" for tu in tool_uses)

    async def test_agent_tito_structure(self, agent: Agent, recorder: VLLMTokenRecorder) -> None:
        """Agent invocation creates valid TITO structure."""
        await agent.invoke_async("What is 100 / 4?")

        # Build TokenManager from recorder history
        tm = TokenManager()
        for entry in recorder.history:
            pti = entry.get("prompt_token_ids")
            ti = entry.get("token_ids")
            if pti:
                tm.add_prompt(pti)
            if ti:
                tm.add_response(ti)

        # Check TITO structure
        assert len(tm.segments) >= 1
        assert len(tm.token_ids) > 0
        assert len(tm.loss_mask) == len(tm.token_ids)


# =============================================================================
# MATH-500 Problem Tests
# =============================================================================


class TestMath500Problems:
    """Tests using real MATH-500 problems."""

    @pytest.mark.parametrize("problem", MATH500_PROBLEMS[:4], ids=lambda p: p["id"])
    async def test_math500_problem(
        self, model: VLLMModel, recorder: VLLMTokenRecorder, problem: dict
    ) -> None:
        """Test Agent on real MATH-500 problem."""
        agent = Agent(
            model=model,
            tools=[calculator],
            callback_handler=recorder,
            system_prompt=SYSTEM_PROMPT,
        )

        # Invoke agent with problem
        await agent.invoke_async(problem["problem"])

        # Get final assistant response
        final_response = ""
        for msg in reversed(agent.messages):
            if msg.get("role") == "assistant":
                for content in msg.get("content", []):
                    if isinstance(content, dict) and "text" in content:
                        final_response = content["text"]
                        break
                if final_response:
                    break

        # Verify we got a response
        assert final_response != "", f"No response for problem {problem['id']}"

        # Build TokenManager from recorder
        tm = TokenManager()
        for entry in recorder.history:
            pti = entry.get("prompt_token_ids")
            ti = entry.get("token_ids")
            if pti:
                tm.add_prompt(pti)
            if ti:
                tm.add_response(ti)

        # Verify TITO structure
        assert len(tm.token_ids) > 0, f"No tokens captured for {problem['id']}"
        assert len(tm.loss_mask) == len(tm.token_ids)

        # Check answer (soft check - model may give correct answer in different format)
        response_lower = final_response.lower()
        answer_found = any(
            variant.lower() in response_lower for variant in problem["answer_variants"]
        )

        # Log result (don't fail on answer mismatch - model quality varies)
        if not answer_found:
            print(f"\n[{problem['id']}] Expected: {problem['answer']}, Got: {final_response[:100]}")


# =============================================================================
# TITO Consistency Tests
# =============================================================================


class TestTitoConsistency:
    """Tests for TITO (Token-In/Token-Out) consistency."""

    async def test_multi_turn_token_accumulation(
        self, model: VLLMModel, recorder: VLLMTokenRecorder
    ) -> None:
        """Token manager accumulates tokens across multiple turns."""
        agent = Agent(
            model=model,
            tools=[calculator],
            callback_handler=recorder,
            system_prompt="Be brief.",
        )

        await agent.invoke_async("2 + 2 = ?")
        first_history_len = len(recorder.history)

        await agent.invoke_async("3 + 3 = ?")
        second_history_len = len(recorder.history)

        # Should have more history entries after second turn
        assert second_history_len > first_history_len

        # Build TokenManager
        tm = TokenManager()
        for entry in recorder.history:
            pti = entry.get("prompt_token_ids")
            ti = entry.get("token_ids")
            if pti:
                tm.add_prompt(pti)
            if ti:
                tm.add_response(ti)

        # Should have accumulated tokens
        assert len(tm.token_ids) > 0
        assert len(tm.segments) >= 2

    async def test_loss_mask_structure(
        self, model: VLLMModel, recorder: VLLMTokenRecorder
    ) -> None:
        """Loss mask correctly marks prompt vs response tokens."""
        agent = Agent(
            model=model,
            tools=[calculator],
            callback_handler=recorder,
            system_prompt="Be brief.",
        )

        await agent.invoke_async("What is 5 * 5?")

        # Build TokenManager
        tm = TokenManager()
        for entry in recorder.history:
            pti = entry.get("prompt_token_ids")
            ti = entry.get("token_ids")
            if pti:
                tm.add_prompt(pti)
            if ti:
                tm.add_response(ti)

        # Loss mask should have correct structure
        assert len(tm.loss_mask) == len(tm.token_ids)

        # Should have both prompt (0) and response (1) masks
        has_prompt = any(m == 0 for m in tm.loss_mask)
        has_response = any(m == 1 for m in tm.loss_mask)

        assert has_prompt, "Should have prompt tokens (loss_mask=0)"
        assert has_response, "Should have response tokens (loss_mask=1)"
