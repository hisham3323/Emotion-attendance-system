import argparse
import sqlite3
import sys
from pathlib import Path
from textwrap import indent
from datetime import datetime

DB_PATH = Path(__file__).with_name("face_recognition.db")   # same folder as this script


# ──────────────────────────────────────────────────────────────
def connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        sys.exit(f"[!] Database not found at {DB_PATH.resolve()}")
    return sqlite3.connect(DB_PATH)


# USERS ─────────────────────────────────────────────────────────
def list_users() -> None:
    with connect() as conn:
        rows = conn.execute("SELECT name, age, email FROM users ORDER BY name").fetchall()

    if not rows:
        print("No users in the database yet.")
        return

    print(f"{'Name':<20} {'Age':<5}  Email")
    print("-" * 60)
    for name, age, email in rows:
        print(f"{name:<20} {age:<5}  {email}")


# ATTENDANCE ───────────────────────────────────────────────────
def list_attendance() -> None:
    with connect() as conn:
        rows = conn.execute(
            "SELECT name, timestamp FROM attendance ORDER BY timestamp DESC"
        ).fetchall()

    if not rows:
        print("No attendance records yet.")
        return

    print(f"{'Timestamp':<20}  Name")
    print("-" * 40)
    for name, ts in rows:
        # make the timestamp a little friendlier
        try:
            ts_h = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            ts_h = ts
        print(f"{ts_h:<20}  {name}")


# DELETE ───────────────────────────────────────────────────────
def delete_users(names: list[str]) -> None:
    # confirm first
    print("You are about to delete the following user(s):")
    print(indent("\n".join(names), "  • "))
    ans = input("Proceed? [y/N] ").strip().lower()
    if ans != "y":
        print("Aborted.")
        return

    with connect() as conn:
        cur = conn.cursor()
        for n in names:
            cur.execute("DELETE FROM users      WHERE name=?", (n,))
            cur.execute("DELETE FROM attendance WHERE name=?", (n,))
        conn.commit()

    print("User(s) deleted.")


# ──────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Inspect or modify the face_recognition.db SQLite database."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("users", help="List all registered users")
    sub.add_parser("attendance", help="Show attendance log")
    del_p = sub.add_parser("delete", help="Delete one or more users (also clears their attendance)")
    del_p.add_argument("names", nargs="+", help="Name(s) of the user(s) to remove")

    args = p.parse_args()

    if args.cmd == "users":
        list_users()
    elif args.cmd == "attendance":
        list_attendance()
    elif args.cmd == "delete":
        delete_users(args.names)


if __name__ == "__main__":
    main()
