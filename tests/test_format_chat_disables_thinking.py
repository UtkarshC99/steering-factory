"""format_chat must pass enable_thinking=False whenever the tokenizer's
own chat template accepts that kwarg (Qwen3-style), and must NOT pass it
to a template that doesn't accept it (Gemma-style) -- passing an
unexpected kwarg to a template that ignores **kwargs is harmless, but a
template that doesn't accept **kwargs at all would raise, so this is
worth pinning down explicitly rather than assuming.

Real-run motivation: 81.7% of one run's Qwen3 harmful_instruction_
compliance outputs were still inside an unclosed <think> block at the
96-token cutoff -- the model never produced a delivered answer, only
interrupted reasoning, which a keyword/judge scorer then scores as if it
were the real response. See steering-factory memory
max-new-tokens-must-fit-the-scorer.md.
"""
from steering_factory.model_utils import format_chat


class _FakeQwenTokenizer:
    """Mimics Qwen3's template: accepts **kwargs including enable_thinking,
    and its own chat_template string literally contains that name (the
    real jinja source does -- format_chat's chokepoint detects it by
    substring on the template text, not by introspecting the template
    compiler)."""

    chat_template = "{% if enable_thinking is defined and not enable_thinking %}<think>\\n\\n</think>\\n\\n{% endif %}...{{ messages }}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        assert kwargs.get("enable_thinking") is False
        return f"RENDERED:{messages}:thinking={kwargs.get('enable_thinking')}"


class _FakeGemmaTokenizer:
    """Mimics Gemma's template: no enable_thinking support at all, and
    raises TypeError on an unexpected kwarg (as a real Jinja template
    compiled without **kwargs support would)."""

    chat_template = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"RENDERED:{messages}"


def test_format_chat_disables_thinking_when_template_supports_it():
    tokenizer = _FakeQwenTokenizer()
    result = format_chat(tokenizer, "hello")
    assert "thinking=False" in result


def test_format_chat_omits_enable_thinking_when_template_lacks_it():
    tokenizer = _FakeGemmaTokenizer()
    # Must not raise (a real strict template would TypeError on an
    # unexpected kwarg) and must still render normally.
    result = format_chat(tokenizer, "hello")
    assert "RENDERED:" in result
