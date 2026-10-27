"""בדיקות אוטומטיות לאג'נט. רצות בלי מפתח Gemini (משתמשות במוח הכללים או ב"מודל מזויף")."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import memory as memory_mod  # noqa: E402
from agent import Agent, GeminiBrain, RuleBrain  # noqa: E402
from tools import AgentState, ToolError, validate_args  # noqa: E402


@pytest.fixture
def mem_path(tmp_path):
    p = tmp_path / "memory.json"
    shutil.copy(ROOT / "memory.json", p)
    return p


def run(mem_path, topic, pids, goal="prepare_meeting", summary="", answers=(), brain=None):
    answers = list(answers)
    state = AgentState(goal=goal, topic=topic, participant_ids=pids, meeting_date="2026-10-27",
                       memory_path=str(mem_path), summary_text=summary,
                       answer_provider=lambda q, opts: opts[answers.pop(0) - 1] if answers else None)
    result = Agent(brain or RuleBrain(), verbose=False).run(state)
    return state, result


def tools_used(result):
    return [s["tool_used"] for s in result["steps"] if s["status"] == "ok"]


def test_continuation_with_overdue_task(mem_path):
    state, result = run(mem_path, "המשך שיפור תהליך ההזמנה – בדיקת השדה החדש", ["P001", "P002", "P003"])
    assert result["status"] == "completed"
    assert state.context["kind"] == "continuation" and state.context["topic_id"] == "T001"
    overdue = [t for t in state.open_tasks if t["overdue"]]
    assert [t["task_id"] for t in overdue] == ["TASK003"]
    assert "אין להניח שהמשימה לא הושלמה" in result["report"]


def test_new_topic_same_participants_does_not_mix_topics(mem_path):
    state, result = run(mem_path, "קליטת עובד חדש לצוות", ["P001", "P002", "P003"])
    assert state.context["kind"] == "new"
    assert "get_topic_history" not in tools_used(result)     # לא שולף החלטות לא קשורות
    assert "D001" not in result["report"] and "D002" not in result["report"]


def test_indirect_relation_is_flagged(mem_path):
    state, result = run(mem_path, "שיפור מהירות הדוחות למנהלים", ["P001", "P003"])
    assert state.context["kind"] == "related"
    assert "טעון אימות" in result["report"]


def test_unclear_context_asks_user_and_follows_answer(mem_path):
    state, result = run(mem_path, "מעקב אחרי השינויים בעיצוב ובביצועים", ["P002", "P004"], answers=[2])
    assert "ask_user" in tools_used(result)
    assert state.context["topic_id"] == "T003"


def test_unclear_context_without_answer_stops(mem_path):
    state, result = run(mem_path, "מעקב אחרי השינויים בעיצוב ובביצועים", ["P002", "P004"])
    assert result["status"] == "needs_clarification"
    assert state.history == {}       # לא ניחש ולא שלף כלום


def test_participants_change_detected(mem_path):
    state, _ = run(mem_path, "המשך שדרוג מסד הנתונים", ["P002", "P005"])
    assert state.participants_info["new_to_topic"] == ["P005"]
    assert set(state.participants_info["absent_from_last"]) == {"P001", "P003"}


SUMMARY = (ROOT / "samples/summary_order_meeting.txt").read_text(encoding="utf-8")


def test_conflict_confirmed_supersedes_old_decision(mem_path):
    state, result = run(mem_path, "המשך שיפור תהליך ההזמנה", ["P001", "P002", "P004"],
                        goal="summarize_meeting", summary=SUMMARY, answers=[1])
    assert result["status"] == "completed" and state.verified
    saved = json.loads(mem_path.read_text(encoding="utf-8"))
    decisions = {d["decision_id"]: d for m in saved["meetings"] for d in m["decisions"]}
    assert decisions["D002"]["status"] == "superseded"
    new = next(m for m in saved["meetings"] if m["meeting_id"] == "M004")
    assert len(new["tasks"]) == 2
    task003 = next(t for m in saved["meetings"] for t in m["tasks"] if t["task_id"] == "TASK003")
    assert task003["status"] == "done"
    assert "נשמר ואומת" in result["report"]


def test_conflict_not_confirmed_keeps_old_decision(mem_path):
    state, _ = run(mem_path, "המשך שיפור תהליך ההזמנה", ["P001", "P002", "P004"],
                   goal="summarize_meeting", summary=SUMMARY, answers=[2])
    saved = json.loads(mem_path.read_text(encoding="utf-8"))
    decisions = {d["decision_id"]: d for m in saved["meetings"] for d in m["decisions"]}
    assert decisions["D002"]["status"] == "active"
    assert any(d["status"] == "needs_clarification" for d in decisions.values())


def test_corrupt_memory_stops_without_writing(tmp_path):
    bad = tmp_path / "memory.json"
    shutil.copy(ROOT / "samples/memory_corrupt.json", bad)
    before = bad.read_bytes()
    state, result = run(bad, "המשך שיפור תהליך ההזמנה", ["P001"], goal="summarize_meeting", summary=SUMMARY)
    assert result["status"] == "failed"
    assert bad.read_bytes() == before
    assert "לא בוצעה שום כתיבה" in result["report"]


def test_failed_save_keeps_original_and_reports_failure(mem_path, monkeypatch):
    before = mem_path.read_bytes()

    def boom(*a, **k):
        raise OSError("הדיסק מלא")
    monkeypatch.setattr(memory_mod.os, "replace", boom)
    state, result = run(mem_path, "המשך שיפור תהליך ההזמנה", ["P001", "P002", "P004"],
                        goal="summarize_meeting", summary=SUMMARY, answers=[1])
    assert result["status"] == "failed"
    assert mem_path.read_bytes() == before
    assert "השמירה נכשלה" in result["report"] and "נשמר ואומת" not in result["report"]


def test_tool_argument_validation():
    with pytest.raises(ToolError):
        validate_args("delete_everything", {})
    with pytest.raises(ToolError):
        validate_args("search_topics", {})
    with pytest.raises(ToolError):
        validate_args("search_topics", {"query": 5})
    with pytest.raises(ToolError):
        validate_args("search_topics", {"query": "x", "hack": True})


class FakeLLM:
    """מחליף את Gemini בבדיקות: מחזיר תשובות מוכנות מראש."""

    def __init__(self, replies):
        self.replies = list(replies)

    def generate_json(self, system, prompt):
        r = self.replies.pop(0) if self.replies else {"action": "finish"}
        if isinstance(r, Exception):
            raise r
        return r


def test_model_cannot_skip_steps_or_use_unknown_tools(mem_path):
    llm = FakeLLM([
        {"decision": "x", "reason": "", "action": "finish", "args": {}},             # סיום מוקדם → נדחה
        {"decision": "x", "reason": "", "action": "rm_rf", "args": {}},              # כלי לא קיים → נדחה
        {"decision": "x", "reason": "", "action": "load_memory", "args": {}},
        {"decision": "x", "reason": "", "action": "search_topics", "args": {"query": "קליטת עובד חדש"}},
        {"decision": "x", "reason": "", "action": "set_context",                     # קישור לנושא שלא נמצא → נדחה
         "args": {"kind": "continuation", "topic_id": "T001", "reason": "ניחוש"}},
        {"decision": "x", "reason": "", "action": "set_context", "args": {"kind": "new", "reason": "אין התאמה"}},
        {"decision": "x", "reason": "", "action": "finish", "args": {}},
    ])
    state, result = run(mem_path, "קליטת עובד חדש", ["P001"], brain=GeminiBrain(llm))
    statuses = [s["status"] for s in result["steps"]]
    assert statuses[:2] == ["rejected", "rejected"] and statuses[4] == "rejected"
    assert state.context["kind"] == "new" and result["status"] == "completed"


def test_model_failure_falls_back_to_rules(mem_path):
    llm = FakeLLM([RuntimeError("network")] * 20)
    state, result = run(mem_path, "המשך שדרוג מסד הנתונים", ["P002", "P005"], brain=GeminiBrain(llm))
    assert result["status"] == "completed"
    assert all("גיבוי" in s["brain"] for s in result["steps"])


def test_llm_extraction_rejects_invented_items(mem_path):
    state = AgentState(goal="summarize_meeting", topic="x", participant_ids=["P001"], meeting_date="2026-10-27",
                       memory_path=str(mem_path), summary_text=SUMMARY,
                       llm=FakeLLM([{"decisions": [{"subject": "תקציב", "text": "אושר תקציב של מיליון שקל לשיווק"}],
                                     "tasks": [], "task_updates": [{"task_id": "TASK999", "status": "done"}]}]))
    from tools import extract_summary, load_memory_tool
    load_memory_tool(state)
    out = extract_summary(state)
    assert out["decisions"] == [] and out["task_updates"] == []
    assert len(out["issues"]) == 2


def test_step_limit_stops_agent(mem_path):
    llm = FakeLLM([{"decision": "loop", "reason": "", "action": "finish", "args": {}}] * 50)
    _, result = run(mem_path, "המשך שדרוג מסד הנתונים", ["P002"], brain=GeminiBrain(llm))
    assert result["stop_reason"] == "max_steps" and result["status"] == "stopped_limit"


def test_gemini_client_waits_and_retries_on_rate_limit():
    import agent

    class RateLimit(Exception):
        code = 429

    calls = []

    class Models:
        def generate_content(self, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RateLimit("Please retry in 0.1s")
            return type("R", (), {"text": '{"action": "load_memory"}'})()

    client = agent.GeminiClient.__new__(agent.GeminiClient)
    client.client = type("C", (), {"models": Models()})()
    client.model, client.min_interval, client.max_retries, client._last_call = "m", 0, 2, 0
    messages = []
    client.notify = messages.append
    assert client.generate_json("s", "p") == {"action": "load_memory"}
    assert len(calls) == 2 and messages
