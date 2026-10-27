"""נקודת הכניסה לאג'נט הכנת הפגישות.

דוגמאות:
  python main.py prepare --topic "המשך שיפור תהליך ההזמנה" --participants P001,P002,P003
  python main.py summarize --topic "המשך שיפור תהליך ההזמנה" --participants P001,P002,P004 \
      --summary-file samples/summary_order_meeting.txt
  python main.py demo --out demo_output.txt
"""
import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

from agent import Agent, make_brain
from tools import AgentState

ROOT = Path(__file__).parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


class Output:
    """מדפיס למסך ובמקביל (אם צריך) לקובץ."""

    def __init__(self, path=None):
        self.file = open(path, "w", encoding="utf-8") if path else None

    def __call__(self, text=""):
        print(text)
        if self.file:
            self.file.write(text + "\n")

    def close(self):
        if self.file:
            self.file.close()


def interactive_answer(out):
    def provider(question, options):
        out(f"\n❓ {question}")
        for i, o in enumerate(options, 1):
            out(f"   {i}. {o}")
        if not sys.stdin.isatty():
            out("   (אין קלט אינטראקטיבי – השאלה נשארת פתוחה)")
            return None
        raw = input("   בחירה (מספר): ").strip()
        return options[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(options) else None
    return provider


def scripted_answer(out, answers):
    """לתרחישי ההדגמה: תשובות מוכנות מראש במקום הקלדה."""
    answers = list(answers)

    def provider(question, options):
        out(f"\n❓ {question}")
        for i, o in enumerate(options, 1):
            out(f"   {i}. {o}")
        if not answers:
            out("   (לא סופקה תשובה)")
            return None
        choice = answers.pop(0)
        out(f"   ← המשתמש בחר: {choice}. {options[choice - 1]}")
        return options[choice - 1]
    return provider


def run_once(args_ns, out, provider, memory_path, summary_text="", goal="prepare_meeting"):
    brain, llm, mode = make_brain(args_ns.offline, notify=out)
    out(f"מצב הפעלה: {mode}")
    state = AgentState(goal=goal, topic=args_ns.topic, participant_ids=args_ns.participant_ids,
                       meeting_date=args_ns.date, memory_path=str(memory_path), summary_text=summary_text,
                       answer_provider=provider, llm=llm)
    out(f"מטרה: {goal} | נושא: {state.topic} | משתתפים: {', '.join(state.participant_ids)}")
    out("\n--- יומן החלטות (שלב אחר שלב) ---")
    result = Agent(brain, printer=out).run(state)
    out("\n--- פלט ---")
    out(result["report"])
    out(f"\nסטטוס ריצה: {result['status']} | צעדים: {len(result['steps'])} | סיבת עצירה: {result['stop_reason']}")
    return result


DEMO = [
    ("תרחיש 1+5: המשך ישיר של נושא קודם + משימה באיחור", "prepare_meeting",
     "המשך שיפור תהליך ההזמנה – בדיקת השדה החדש", ["P001", "P002", "P003"], None, []),
    ("תרחיש 2: נושא חדש עם אותם משתתפים", "prepare_meeting",
     "קליטת עובד חדש לצוות", ["P001", "P002", "P003"], None, []),
    ("תרחיש 3: נושא חדש שקשור בעקיפין לנושא קודם", "prepare_meeting",
     "שיפור מהירות הדוחות למנהלים", ["P001", "P003"], None, []),
    ("תרחיש 4: הקשר לא ברור – האג'נט שואל במקום לנחש", "prepare_meeting",
     "מעקב אחרי השינויים בעיצוב ובביצועים", ["P002", "P004"], None, [1]),
    ("תרחיש 7: שינוי במשתתפים", "prepare_meeting",
     "המשך שדרוג מסד הנתונים", ["P002", "P005"], None, []),
    ("תרחיש 6: סיכום אחרי פגישה – החלטה שסותרת החלטה קודמת + עדכון זיכרון", "summarize_meeting",
     "המשך שיפור תהליך ההזמנה", ["P001", "P002", "P004"], "samples/summary_order_meeting.txt", [1]),
    ("תרחיש 8: קובץ זיכרון פגום", "prepare_meeting",
     "המשך שיפור תהליך ההזמנה", ["P001", "P002"], None, []),
]


def run_demo(args_ns):
    out = Output(args_ns.out)
    work = ROOT / "runs"
    work.mkdir(exist_ok=True)
    demo_memory = work / "demo_memory.json"
    shutil.copy(ROOT / "memory.json", demo_memory)   # הקובץ המקורי לא משתנה בהדגמה
    out("הדגמת אג'נט הכנת פגישות – כל התרחישים רצים דרך אותה לולאת אג'נט")
    out(f"תאריך הפגישות בהדגמה: {args_ns.date}. הזיכרון בהדגמה: עותק בתיקייה runs/")
    for title, goal, topic, pids, summary_file, answers in DEMO:
        out("\n" + "#" * 70)
        out(f"# {title}")
        out("#" * 70)
        args_ns.topic, args_ns.participant_ids = topic, pids
        mem = ROOT / "samples/memory_corrupt.json" if "פגום" in title else demo_memory
        summary = (ROOT / summary_file).read_text(encoding="utf-8") if summary_file else ""
        run_once(args_ns, out, scripted_answer(out, answers), mem, summary, goal)
    out.close()


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="אג'נט להכנת פגישות ועדכון זיכרון פגישות")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "summarize"):
        s = sub.add_parser(name)
        s.add_argument("--topic", required=True, help="נושא הפגישה")
        s.add_argument("--participants", required=True, help="מזהי משתתפים מופרדים בפסיק, למשל P001,P002")
        s.add_argument("--memory", default=str(ROOT / "memory.json"))
        if name == "summarize":
            s.add_argument("--summary-file", required=True, help="קובץ טקסט עם סיכום/תמלול הפגישה")
    d = sub.add_parser("demo", help="מריץ את כל תרחישי ההדגמה")
    d.add_argument("--out", help="שמירת הפלט גם לקובץ")
    for s in sub.choices.values():
        s.add_argument("--date", default=date.today().isoformat(), help="תאריך הפגישה (YYYY-MM-DD)")
        s.add_argument("--offline", action="store_true", help="להריץ בלי Gemini (מוח מבוסס כללים)")
    a = p.parse_args(argv)
    try:
        date.fromisoformat(a.date)
    except ValueError:
        p.error("תאריך לא תקין, הפורמט הנדרש: YYYY-MM-DD")

    if a.command == "demo":
        if a.date == date.today().isoformat() and "--date" not in (argv or sys.argv):
            a.date = "2026-10-27"   # תאריך קבוע כדי שההדגמה תהיה עקבית
        run_demo(a)
        return 0
    a.participant_ids = [x.strip() for x in a.participants.split(",") if x.strip()]
    out = Output()
    summary = ""
    if a.command == "summarize":
        path = Path(a.summary_file)
        if not path.exists():
            out(f"שגיאה: קובץ הסיכום לא נמצא: {path}")
            return 1
        summary = path.read_text(encoding="utf-8")
    goal = "summarize_meeting" if a.command == "summarize" else "prepare_meeting"
    result = run_once(a, out, interactive_answer(out), a.memory, summary, goal)
    return 0 if result["status"] in ("completed", "needs_clarification") else 1


if __name__ == "__main__":
    sys.exit(main())
