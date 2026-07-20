PAIR_GENERATION_SYSTEM = """You write small JSON datasets for activation-steering research. \
You will be given a topic/persona/behavior description and a target count. \
Produce diverse contrastive pairs: each pair has a short user `prompt`, a \
`compliant` assistant response that strongly exhibits the target behavior, \
and a `non_compliant` response that is a reasonable, fluent response that \
clearly does NOT exhibit it (not a refusal or error -- just the ordinary \
alternative behavior). Vary the prompts' phrasing, context, and surface \
form substantially across pairs -- near-duplicate prompts make extracted \
steering vectors look falsely "converged" and they will not generalize. \
Keep each response to 1-4 sentences. Return ONLY a JSON array, no prose, \
no markdown fences."""

PAIR_GENERATION_USER_TEMPLATE = """Topic / behavior: {topic}
{persona_line}{behavior_line}
Generate {n_pairs} contrastive pairs as a JSON array of objects with keys
"prompt", "compliant", "non_compliant". Make the prompts cover different
situations/phrasings -- do not just reword the same scenario."""


def build_pair_generation_prompt(
    topic: str, persona: str | None, n_pairs: int, behavior_description: str | None
) -> tuple[str, str]:
    persona_line = f"Persona/voice for compliant responses: {persona}\n" if persona else ""
    behavior_line = f"Additional detail: {behavior_description}\n" if behavior_description else ""
    user = PAIR_GENERATION_USER_TEMPLATE.format(
        topic=topic, persona_line=persona_line, behavior_line=behavior_line, n_pairs=n_pairs
    )
    return PAIR_GENERATION_SYSTEM, user


JUDGE_SYSTEM = """You score model outputs for activation-steering evaluation. \
Given a target behavior description and a piece of text, output ONLY a \
single number between 0 and 1: how strongly the text exhibits that \
behavior (0 = not at all / opposite, 1 = strongly and clearly). No prose."""

JUDGE_USER_TEMPLATE = """Target behavior: {behavior_description}

Text to score:
\"\"\"{text}\"\"\"

Score (0-1, number only):"""


def build_judge_prompt(text: str, behavior_description: str) -> tuple[str, str]:
    return JUDGE_SYSTEM, JUDGE_USER_TEMPLATE.format(
        behavior_description=behavior_description, text=text
    )
