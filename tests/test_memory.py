from backend.services.memory import condense_history, naive_summarize


def test_short_history_returned_unchanged():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    result = condense_history(history, keep_recent=8, summarize_fn=lambda m: "summary")
    assert result == history


def test_long_history_is_condensed_to_summary_plus_recent():
    history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
    result = condense_history(history, keep_recent=4, summarize_fn=lambda m: "summary text")

    assert len(result) == 5  # 1 summary turn + 4 recent
    assert result[0]["role"] == "system"
    assert "summary text" in result[0]["content"]
    assert result[1:] == history[-4:]


def test_naive_summarize_extracts_user_topics():
    messages = [
        {"role": "user", "content": "I need help with my order"},
        {"role": "assistant", "content": "Sure, what's the issue?"},
    ]
    summary = naive_summarize(messages)
    assert "order" in summary


def test_naive_summarize_handles_empty_input():
    assert naive_summarize([]) == "No prior topics."
