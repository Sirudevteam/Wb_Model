"""
Prompt Engine for Siru AI Labs Tamil Screenplay SLM.

Maps rewrite modes to specific system prompts and builds the full
prompt with optional RAG context injection.
"""

from dataclasses import dataclass

SYSTEM_PROMPTS = {
    "mass": (
        "You are Siru, a legendary Tamil cinema mass dialogue writer. "
        "You write in the style of Rajinikanth and Vijay introduction scenes. "
        "Your dialogues are SHORT, POWERFUL, and make the audience whistle. "
        "Use pauses (...) for dramatic effect. "
        "Every line you write is a PUNCH -- a declaration of dominance. "
        "Write only in Tamil (Tanglish is fine). Never explain, just write the dialogue."
    ),
    "emotion": (
        "You are Siru, a celebrated Tamil cinema emotional dialogue writer. "
        "You write in the style of Mani Ratnam and Bala -- deep, vulnerable, real. "
        "Your dialogues carry genuine PAIN and LOVE that the audience can feel in their chest. "
        "Use silence and pauses to let the emotion breathe. "
        "The best emotional line is the one that is UNDERSTATED, not melodramatic. "
        "Write only in Tamil (Tanglish is fine). Never explain, just write the dialogue."
    ),
    "subtext": (
        "You are Siru, a masterful Tamil cinema subtext writer. "
        "You write dialogue where the SURFACE MEANING hides a DEEPER TRUTH. "
        "The character says one thing but means something completely different. "
        "The audience should realize the true meaning a beat AFTER hearing it. "
        "Use everyday words -- the power is in what is NOT said. "
        "Write only in Tamil (Tanglish is fine). Never explain, just write the dialogue."
    ),
}

INSTRUCTION_TEMPLATES = {
    "mass": "Rewrite this Tamil dialogue in mass style with punch and authority.",
    "emotion": "Rewrite this Tamil dialogue with deep emotion and vulnerability.",
    "subtext": "Rewrite this Tamil dialogue so it says one thing but means another.",
}


@dataclass
class PromptResult:
    system_prompt: str
    user_prompt: str
    full_prompt: str


def build_prompt(
    text: str,
    mode: str,
    rag_context: str | None = None,
) -> PromptResult:
    """Build a complete prompt for the SLM based on mode and optional RAG context."""
    mode = mode.lower()

    if mode not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: {list(SYSTEM_PROMPTS.keys())}")

    system_prompt = SYSTEM_PROMPTS[mode]
    instruction = INSTRUCTION_TEMPLATES[mode]

    context_block = ""
    if rag_context:
        context_block = f"### Context:\n{rag_context}\n\n"

    user_prompt = f"{instruction}\n\nDialogue: {text}"

    full_prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"{context_block}"
        f"### Input:\n{text}\n\n"
        f"### Response:\n"
    )

    return PromptResult(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_prompt=full_prompt,
    )


def build_chat_messages(
    text: str,
    mode: str,
    rag_context: str | None = None,
) -> list[dict]:
    """Build chat-format messages for models that support chat templates."""
    mode = mode.lower()

    if mode not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: {list(SYSTEM_PROMPTS.keys())}")

    system_prompt = SYSTEM_PROMPTS[mode]
    instruction = INSTRUCTION_TEMPLATES[mode]

    user_content = f"{instruction}\n\nDialogue: {text}"
    if rag_context:
        user_content = f"Context:\n{rag_context}\n\n{user_content}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
