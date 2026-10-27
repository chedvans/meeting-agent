"""קריאה, אימות ושמירה בטוחה של קובץ הזיכרון (memory.json)."""
import json
import os
import shutil
import tempfile
from pathlib import Path


class MemoryError_(Exception):
    """שגיאה בקובץ הזיכרון (פגום, חסר או שהשמירה נכשלה)."""


REQUIRED = {
    "participants": ["participant_id", "name", "role"],
    "topics": ["topic_id", "title", "keywords"],
    "meetings": ["meeting_id", "topic_id", "date", "participants", "decisions", "tasks"],
}
DECISION_FIELDS = ["decision_id", "subject", "text", "status"]
TASK_FIELDS = ["task_id", "owner_id", "description", "status"]


def validate(memory: dict) -> list[str]:
    """מחזיר רשימת בעיות במבנה הזיכרון. רשימה ריקה = תקין."""
    problems = []
    if not isinstance(memory, dict):
        return ["שורש הקובץ אינו אובייקט JSON"]
    for section, fields in REQUIRED.items():
        items = memory.get(section)
        if not isinstance(items, list):
            problems.append(f"חסרה הרשימה '{section}'")
            continue
        for i, item in enumerate(items):
            for f in fields:
                if not isinstance(item, dict) or f not in item:
                    problems.append(f"{section}[{i}]: חסר השדה '{f}'")
    if problems:
        return problems

    ids = set()
    for m in memory["meetings"]:
        for d in m["decisions"]:
            problems += [f"{m['meeting_id']}: בהחלטה חסר '{f}'" for f in DECISION_FIELDS if f not in d]
        for t in m["tasks"]:
            problems += [f"{m['meeting_id']}: במשימה חסר '{f}'" for f in TASK_FIELDS if f not in t]
    # מזהים ייחודיים – מונע כפילויות
    all_ids = ([p["participant_id"] for p in memory["participants"]]
               + [t["topic_id"] for t in memory["topics"]]
               + [m["meeting_id"] for m in memory["meetings"]]
               + [d.get("decision_id") for m in memory["meetings"] for d in m["decisions"]]
               + [t.get("task_id") for m in memory["meetings"] for t in m["tasks"]])
    for x in all_ids:
        if x in ids:
            problems.append(f"מזהה כפול: {x}")
        ids.add(x)
    return problems


def load_memory(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise MemoryError_(f"קובץ הזיכרון לא נמצא: {path}")
    try:
        memory = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise MemoryError_(f"קובץ הזיכרון פגום (JSON לא תקין, שורה {e.lineno}): {path.name}") from e
    problems = validate(memory)
    if problems:
        raise MemoryError_("מבנה הזיכרון אינו תקין: " + "; ".join(problems[:5]))
    return memory


def save_memory(path: str | Path, memory: dict) -> Path:
    """שמירה אטומית: מאמת, יוצר גיבוי, כותב לקובץ זמני ורק אז מחליף.
    אם משהו נכשל – הקובץ המקורי נשאר כמו שהיה."""
    path = Path(path)
    problems = validate(memory)
    if problems:
        raise MemoryError_("השמירה בוטלה – הנתונים החדשים אינם תקינים: " + "; ".join(problems[:5]))
    backup = path.with_suffix(path.suffix + ".bak")
    tmp_name = None
    try:
        if path.exists():
            shutil.copy2(path, backup)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".memory_", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
        tmp_name = None
    except OSError as e:
        raise MemoryError_(f"השמירה נכשלה ({e}); הקובץ המקורי לא שונה") from e
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.remove(tmp_name)
    return backup
