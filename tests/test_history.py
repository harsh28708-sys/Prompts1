"""Tests for the local SQLite history store (promteval/history.py). Every test
points at a throwaway db_path inside tmp_path -- never the real
~/.promteval/history.db -- so tests can't pollute (or be polluted by) a real
user's history."""

from promteval.history import clear_history, delete_run, get_run, list_runs, save_run


def test_save_run_returns_a_new_id_each_time(tmp_path):
    db = tmp_path / "history.db"

    first_id = save_run("improve", "Summarize: {input}", "groq/llama-3.1-8b-instant", {"prompt": "x"}, {"score": 4}, db_path=db)
    second_id = save_run("improve", "Another prompt", "groq/llama-3.1-8b-instant", {"prompt": "y"}, {"score": 3}, db_path=db)

    assert first_id != second_id


def test_list_runs_returns_newest_first(tmp_path):
    db = tmp_path / "history.db"
    save_run("improve", "First", "m", {}, {}, db_path=db)
    save_run("compare", "Second", "m", {}, {}, db_path=db)

    entries = list_runs(db_path=db)

    assert [e.summary for e in entries] == ["Second", "First"]


def test_list_runs_respects_the_limit(tmp_path):
    db = tmp_path / "history.db"
    for i in range(5):
        save_run("improve", f"Run {i}", "m", {}, {}, db_path=db)

    entries = list_runs(limit=2, db_path=db)

    assert len(entries) == 2
    assert entries[0].summary == "Run 4"  # newest


def test_get_run_returns_the_full_record_including_request_and_response(tmp_path):
    db = tmp_path / "history.db"
    request = {"prompt": "Summarize: {input}", "model": "groq/llama-3.1-8b-instant"}
    response = {"score": 4, "reasoning": "Solid.", "improved_prompt": "Summarize concisely: {input}"}
    run_id = save_run("improve", "Summarize: {input}", "groq/llama-3.1-8b-instant", request, response, db_path=db)

    entry = get_run(run_id, db_path=db)

    assert entry is not None
    assert entry.kind == "improve"
    assert entry.request == request
    assert entry.response == response


def test_get_run_returns_none_for_a_missing_id(tmp_path):
    db = tmp_path / "history.db"

    assert get_run(999, db_path=db) is None


def test_delete_run_removes_just_that_entry(tmp_path):
    db = tmp_path / "history.db"
    keep_id = save_run("improve", "Keep me", "m", {}, {}, db_path=db)
    remove_id = save_run("improve", "Remove me", "m", {}, {}, db_path=db)

    deleted = delete_run(remove_id, db_path=db)

    assert deleted is True
    assert get_run(remove_id, db_path=db) is None
    assert get_run(keep_id, db_path=db) is not None


def test_delete_run_returns_false_for_a_missing_id(tmp_path):
    db = tmp_path / "history.db"

    assert delete_run(999, db_path=db) is False


def test_clear_history_removes_everything_and_returns_the_count(tmp_path):
    db = tmp_path / "history.db"
    save_run("improve", "One", "m", {}, {}, db_path=db)
    save_run("compare", "Two", "m", {}, {}, db_path=db)

    deleted_count = clear_history(db_path=db)

    assert deleted_count == 2
    assert list_runs(db_path=db) == []


def test_a_summary_longer_than_200_chars_is_truncated(tmp_path):
    db = tmp_path / "history.db"
    long_summary = "x" * 500

    run_id = save_run("improve", long_summary, "m", {}, {}, db_path=db)

    entry = get_run(run_id, db_path=db)
    assert len(entry.summary) == 200
