"""SQLite 台账：把 agent 的每一步、每个工具调用、每道门禁、每次提交都落盘。

为什么不用内存：agent 必须能断点续跑（跑到一半被 kill、超预算、host 重启），
且「提交门禁」需要能读到「这个 run 之前的门禁是否真的过了」——
如果门禁状态只活在对话上下文里，模型一句 "I passed the gate" 就能骗过系统。

所有写操作串行化（单连接 + 互斥锁）；WAL 模式支持并发读。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  comp TEXT NOT NULL,
  phase TEXT NOT NULL DEFAULT 'SELECT',
  status TEXT NOT NULL DEFAULT 'running',
  goal TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  steps INTEGER NOT NULL DEFAULT 0,
  tokens_in INTEGER NOT NULL DEFAULT 0,
  tokens_out INTEGER NOT NULL DEFAULT 0,
  submissions INTEGER NOT NULL DEFAULT 0,
  shell_calls INTEGER NOT NULL DEFAULT 0,
  config TEXT,
  note TEXT
);
CREATE TABLE IF NOT EXISTS steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL,
  idx INTEGER NOT NULL,
  role TEXT NOT NULL,
  payload TEXT NOT NULL,
  tokens_in INTEGER DEFAULT 0,
  tokens_out INTEGER DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, idx);
CREATE TABLE IF NOT EXISTS tool_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL,
  step_idx INTEGER NOT NULL,
  name TEXT NOT NULL,
  args TEXT NOT NULL,
  ok INTEGER NOT NULL,
  result TEXT NOT NULL,
  ms INTEGER NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tc_run ON tool_calls(run_id);
CREATE TABLE IF NOT EXISTS gates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  passed INTEGER NOT NULL,
  evidence TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gates_run ON gates(run_id, name);
CREATE TABLE IF NOT EXISTS submissions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL,
  comp TEXT NOT NULL,
  file TEXT,
  message TEXT,
  status TEXT,
  score REAL,
  ref TEXT,
  raw TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
  run_id INTEGER NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """老库补列：submissions.ref（提交号用来回查榜分）。"""
        try:
            cols = {r[1] for r in self.conn.execute("PRAGMA table_info(submissions)")}
        except sqlite3.Error:
            return
        if cols and "ref" not in cols:
            with self._lock:
                self.conn.execute("ALTER TABLE submissions ADD COLUMN ref TEXT")
                self.conn.execute(
                    "UPDATE submissions SET ref=json_extract(raw,'$.ref') "
                    "WHERE ref IS NULL AND raw LIKE '%\"ref\"%'")
                self.conn.commit()

    # ---------- runs ----------
    def new_run(self, comp: str, goal: str, config: dict | None = None) -> int:
        now = time.time()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO runs(comp, goal, created_at, updated_at, config) VALUES(?,?,?,?,?)",
                (comp, goal, now, now, json.dumps(config or {}, ensure_ascii=False)),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def run(self, run_id: int) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def runs(self, limit: int = 20, comp: str | None = None) -> list[dict]:
        q = "SELECT * FROM runs"
        args: list[Any] = []
        if comp:
            q += " WHERE comp=?"
            args.append(comp)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def update_run(self, run_id: int, **kw) -> None:
        if not kw:
            return
        kw["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in kw)
        with self._lock:
            self.conn.execute(f"UPDATE runs SET {cols} WHERE id=?", (*kw.values(), run_id))
            self.conn.commit()

    def bump(self, run_id: int, **deltas) -> None:
        cols = ", ".join(f"{k}={k}+?" for k in deltas)
        with self._lock:
            self.conn.execute(
                f"UPDATE runs SET {cols}, updated_at=? WHERE id=?",
                (*deltas.values(), time.time(), run_id),
            )
            self.conn.commit()

    # ---------- messages ----------
    def append_message(self, run_id: int, idx: int, message: dict, tin: int = 0, tout: int = 0) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO steps(run_id, idx, role, payload, tokens_in, tokens_out, created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (run_id, idx, message.get("role", "?"), json.dumps(message, ensure_ascii=False),
                 tin, tout, time.time()),
            )
            self.conn.commit()

    def messages(self, run_id: int) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT payload FROM steps WHERE run_id=? ORDER BY idx, id", (run_id,)
            ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def step_count(self, run_id: int) -> int:
        with self._lock:
            r = self.conn.execute("SELECT COUNT(*) c FROM steps WHERE run_id=?", (run_id,)).fetchone()
        return int(r["c"])

    # ---------- tool calls ----------
    def log_tool(self, run_id: int, step_idx: int, name: str, args: dict,
                 ok: bool, result: Any, ms: int) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO tool_calls(run_id, step_idx, name, args, ok, result, ms, created_at)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (run_id, step_idx, name, json.dumps(args, ensure_ascii=False), int(ok),
                 json.dumps(result, ensure_ascii=False, default=str)[:20000], ms, time.time()),
            )
            self.conn.commit()

    def tool_calls(self, run_id: int, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM tool_calls WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- gates ----------
    def record_gate(self, run_id: int, name: str, passed: bool, evidence: Any) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO gates(run_id, name, passed, evidence, created_at) VALUES(?,?,?,?,?)",
                (run_id, name, int(bool(passed)),
                 json.dumps(evidence, ensure_ascii=False, default=str)[:50000], time.time()),
            )
            self.conn.commit()

    def gate_state(self, run_id: int) -> dict[str, dict]:
        """每道门取最新一次记录。"""
        with self._lock:
            rows = self.conn.execute(
                "SELECT g.* FROM gates g JOIN (SELECT name, MAX(id) mid FROM gates WHERE run_id=?"
                " GROUP BY name) x ON g.id=x.mid", (run_id,)
            ).fetchall()
        out = {}
        for r in rows:
            out[r["name"]] = {
                "passed": bool(r["passed"]),
                "evidence": json.loads(r["evidence"]) if r["evidence"] else {},
                "at": r["created_at"],
            }
        return out

    def gate_pass(self, run_id: int, name: str) -> bool:
        st = self.gate_state(run_id)
        return bool(st.get(name, {}).get("passed"))

    # ---------- submissions ----------
    def add_submission(self, run_id: int, comp: str, file: str, message: str,
                       status: str, score: float | None, raw: Any = None,
                       ref: str | None = None) -> None:
        if ref is None and isinstance(raw, dict):
            ref = str(raw.get("ref") or "") or None
        with self._lock:
            self.conn.execute(
                "INSERT INTO submissions(run_id, comp, file, message, status, score, ref, raw,"
                " created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, comp, file, message, status, score, ref,
                 json.dumps(raw, ensure_ascii=False, default=str)[:20000], time.time()),
            )
            self.conn.commit()

    def submissions(self, run_id: int | None = None, comp: str | None = None) -> list[dict]:
        q, a = "SELECT * FROM submissions", []
        where = []
        if run_id:
            where.append("run_id=?"); a.append(run_id)
        if comp:
            where.append("comp=?"); a.append(comp)
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY id DESC"
        with self._lock:
            return [dict(r) for r in self.conn.execute(q, a).fetchall()]

    # ---------- kv / events ----------
    def kv_set(self, run_id: int, key: str, value: Any) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO kv(run_id,key,value,updated_at) VALUES(?,?,?,?)"
                " ON CONFLICT(run_id,key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (run_id, key, json.dumps(value, ensure_ascii=False, default=str), time.time()),
            )
            self.conn.commit()

    def kv_get(self, run_id: int, key: str, default: Any = None) -> Any:
        with self._lock:
            r = self.conn.execute("SELECT value FROM kv WHERE run_id=? AND key=?", (run_id, key)).fetchone()
        if not r:
            return default
        try:
            return json.loads(r["value"])
        except Exception:
            return r["value"]

    def log_event(self, run_id: int, kind: str, payload: Any) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO events(run_id, kind, payload, created_at) VALUES(?,?,?,?)",
                (run_id, kind, json.dumps(payload, ensure_ascii=False, default=str)[:20000], time.time()),
            )
            self.conn.commit()

    def events(self, run_id: int, limit: int = 100) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit)
            ).fetchall()]

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass
