"""
GovernanceDataEngine.py

GovernanceDataEngin
-------------------
A dedicated data shaping/query layer for UI.

Design principles:
- Strictly separate LIVE vs QUERY:
  - LIVE: memory-only, no DB access, no time range filters.
  - QUERY: DB queries by time range, no merging with LIVE (by design).
- Provide stable DTOs for frontend rendering.
- Provide a lightweight watermark for UI refresh optimization.
- Use a single, centralized "results" rate logic for success_rate.

Notes:
- This engine expects a `governor` object compatible with your GovernanceManager:
  - governor.event_buffer: deque of event dicts
  - governor.stats_lock: threading.RLock
  - governor.round_contexts: dict[str, GroupRoundContext]
  - governor.runtime_groups: dict[str, dict] (registered groups in current session)
  - governor.known_anchors: set[list_url]
  - governor.db: database handler with execute/fetch_one/fetch_all_dict (for QUERY only)
  - governor._hash(url): optional; else we compute md5 locally
- Anchor LIVE data is expected to be maintained in memory by core (Option B).
  This engine does NOT query DB for anchor status in LIVE mode.
"""

from __future__ import annotations

import csv
import hashlib
import io
import time
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union


# ---- Status constants (mirror your Status IntEnum values) ----
STATUS_PENDING = 0
STATUS_RUNNING = 1
STATUS_SUCCESS = 2
STATUS_TEMP_FAIL = 3
STATUS_PERM_FAIL = 4
STATUS_SKIPPED = 5
STATUS_STOPPED = 6
STATUS_CACHED = 7
STATUS_IGNORED = 8


TerminalFailSet = {STATUS_TEMP_FAIL, STATUS_PERM_FAIL, STATUS_STOPPED}
TerminalSuccessSet = {STATUS_SUCCESS}
IgnoredForTotalsSet = {STATUS_PENDING, STATUS_SKIPPED, STATUS_IGNORED, STATUS_CACHED}


class GovernanceDataEngine:
    """
    Mode selection:
    - GovernanceDataEngine does NOT auto-switch between LIVE and QUERY.
    - The caller (usually GovernanceManager or an API layer) must explicitly choose:
        * LIVE = no time range
        * QUERY = time range provided

    API Mode Contract
    -----------------
    LIVE mode:
    - Used when no time window is provided.
    - Data comes exclusively from in-memory structures.
    - Very fast; optimized for real-time dashboard updates.

    QUERY mode:
    - Used when the API caller provides start_ts / end_ts.
    - Data comes exclusively from database tables (crawl_log, crawl_status).
    - Used for history, trends, exports, and time-window analytics.

    The backend will NOT mix memory data with DB data.
    If a time window is provided → QUERY.
    If no time window is provided → LIVE.
    """

    def __init__(self, governor: Any):
        self.gov = governor

    # ---------------------------------------------------------------------
    # Common helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _to_ms(ts: Any) -> int:
        """
        Convert timestamps to epoch milliseconds.

        Accepted:
        - None -> 0
        - int/float seconds -> ms (if <= 1e12)
        - int ms -> ms (if > 1e12)
        - datetime -> ms
        - ISO string -> ms (best-effort; if fails -> 0)
        """
        if ts is None:
            return 0
        if isinstance(ts, (int, float)):
            # Heuristic: treat > 1e12 as ms, else seconds
            return int(ts) if ts > 1_000_000_000_000 else int(ts * 1000)
        if isinstance(ts, datetime.datetime):
            return int(ts.timestamp() * 1000)
        if isinstance(ts, str):
            # Best-effort parse: allow "YYYY-MM-DD HH:MM:SS" or ISO
            try:
                s = ts.strip()
                if "T" not in s and " " in s:
                    s = s.replace(" ", "T")
                # If no timezone info, treat as local time.
                d = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
                return int(d.timestamp() * 1000)
            except Exception:
                return 0
        return 0

    @staticmethod
    def _md5(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _url_hash(self, url: str) -> str:
        """
        Use governor._hash if available, otherwise md5(url).
        """
        fn = getattr(self.gov, "_hash", None)
        if callable(fn):
            try:
                return str(fn(url))
            except Exception:
                pass
        return self._md5(url)

    @staticmethod
    def _match_filters(
        item: Dict[str, Any],
        group_path: Optional[str] = None,
        spider: Optional[str] = None,
        status: Optional[int] = None,
    ) -> bool:
        """
        Memory-only filtering for LIVE endpoints.
        """
        if group_path and item.get("group_path") != group_path:
            return False
        if spider and item.get("spider") != spider and item.get("spider_name") != spider:
            return False
        if status is not None and item.get("status") != status:
            return False
        return True

    @staticmethod
    def _make_watermark(max_updated_ms: int, buffer_len: int) -> str:
        """
        A cheap watermark. Frontend can compare string equality to skip redraw.
        """
        return f"{max_updated_ms}:{buffer_len}"

    @staticmethod
    def _init_counts() -> Dict[str, int]:
        return {"total": 0, "success": 0, "failed": 0, "running": 0}

    @staticmethod
    def _init_perf() -> Dict[str, Any]:
        return {"min": 0.0, "max": 0.0, "sum": 0.0, "count": 0, "avg": 0.0}

    @staticmethod
    def _accumulate_counts(counts: Dict[str, int], st: int, n: int = 1) -> None:
        """
        Unified counting rule (LIVE aggregation).

        Counting philosophy:
        - running: count RUNNING
        - totals: count only terminal success/fail (exclude PENDING/SKIPPED/IGNORED/CACHED)
        - success: SUCCESS
        - failed: TEMP_FAIL / PERM_FAIL / STOPPED
        """
        if st == STATUS_RUNNING:
            counts["running"] += n
            return
        if st in IgnoredForTotalsSet:
            return
        # terminal -> total
        counts["total"] += n
        if st in TerminalSuccessSet:
            counts["success"] += n
        elif st in TerminalFailSet:
            counts["failed"] += n

    @staticmethod
    def calc_success_rate(results_counts: Dict[str, int]) -> float:
        """
        Compute success rate based on RESULTS counts (unique outcome).
        Return percentage in [0, 100], rounded to 1 decimal.
        """
        total = int(results_counts.get("total", 0) or 0)
        success = int(results_counts.get("success", 0) or 0)
        if total <= 0:
            return 0.0
        return round((success / total) * 100.0, 1)

    # ---------------------------------------------------------------------
    # LIVE (memory-only)
    # ---------------------------------------------------------------------

    def live_logs(
        self,
        group_path: Optional[str] = None,
        spider: Optional[str] = None,
        status: Optional[int] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """
        LIVE: Return recent event logs from memory ring buffer (no DB).

        Output item fields (stable contract):
        - id
        - url
        - group_path
        - spider
        - status
        - http_code
        - duration_s
        - start_ts_ms
        - updated_ts_ms   (UI primary time)
        - state_msg
        """
        items: List[Dict[str, Any]] = []
        max_updated_ms = 0

        with self.gov.stats_lock:
            # newest first
            for e in reversed(self.gov.event_buffer):
                if len(items) >= limit:
                    break

                # Normalize timestamps
                start_ms = self._to_ms(e.get("ts"))
                updated_ms = self._to_ms(e.get("updated_ts", e.get("ts")))
                if updated_ms > max_updated_ms:
                    max_updated_ms = updated_ms

                view = {
                    "id": int(e.get("id") or 0),
                    "url": e.get("url"),
                    "group_path": e.get("group_path"),
                    "spider": e.get("spider"),
                    "status": int(e.get("status", 0)),
                    "http_code": e.get("http_code"),  # may be None if core doesn't set
                    "duration_s": float(e.get("duration") or 0.0),
                    "start_ts_ms": start_ms,
                    "updated_ts_ms": updated_ms,
                    "state_msg": e.get("state_msg"),
                }

                if not self._match_filters(view, group_path=group_path, spider=spider, status=status):
                    continue

                items.append(view)

            watermark = self._make_watermark(max_updated_ms, len(self.gov.event_buffer))

        return {
            "meta": {
                "mode": "LIVE",
                "source": "MEMORY",
                "schema": 1,
                "server_ts_ms": self._now_ms(),
                "watermark": watermark,
            },
            "items": items,
        }

    def live_statuses(
        self,
        group_path: Optional[str] = None,
        spider: Optional[str] = None,
        status: Optional[int] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """
        LIVE: Return de-duplicated latest URL statuses from memory.

        Dedup strategy:
        - Iterate memory events from newest to oldest.
        - Keep first occurrence for each URL.

        Output item fields:
        - url
        - url_hash
        - group_path
        - spider
        - status
        - http_code
        - duration_s
        - start_ts_ms
        - updated_ts_ms
        - state_msg
        """
        items: List[Dict[str, Any]] = []
        seen: set = set()
        max_updated_ms = 0

        with self.gov.stats_lock:
            for e in reversed(self.gov.event_buffer):
                if len(items) >= limit:
                    break
                url = e.get("url")
                if not url or url in seen:
                    continue

                start_ms = self._to_ms(e.get("ts"))
                updated_ms = self._to_ms(e.get("updated_ts", e.get("ts")))
                if updated_ms > max_updated_ms:
                    max_updated_ms = updated_ms

                view = {
                    "url": url,
                    "url_hash": self._url_hash(url),  # hash-based SNAP support
                    "group_path": e.get("group_path"),
                    "spider": e.get("spider"),
                    "status": int(e.get("status", 0)),
                    "http_code": e.get("http_code"),
                    "duration_s": float(e.get("duration") or 0.0),
                    "start_ts_ms": start_ms,
                    "updated_ts_ms": updated_ms,
                    "state_msg": e.get("state_msg"),
                }

                if not self._match_filters(view, group_path=group_path, spider=spider, status=status):
                    continue

                seen.add(url)
                items.append(view)

            watermark = self._make_watermark(max_updated_ms, len(self.gov.event_buffer))

        return {
            "meta": {
                "mode": "LIVE",
                "source": "MEMORY",
                "schema": 1,
                "server_ts_ms": self._now_ms(),
                "watermark": watermark,
            },
            "items": items,
        }

    def live_round(self, group_path: str) -> Dict[str, Any]:
        """
        LIVE: Return round snapshot for a given group (memory-only).
        """
        with self.gov.stats_lock:
            ctx = self.gov.round_contexts.get(group_path)
            snap = ctx.get_snapshot() if ctx else {}
        return {
            "meta": {
                "mode": "LIVE",
                "source": "MEMORY",
                "schema": 1,
                "server_ts_ms": self._now_ms(),
                "watermark": f"round:{group_path}:{snap.get('round_id', 0)}:{snap.get('last_update_ts_ms', 0)}",
            },
            "round": snap,
        }

    def live_summary(
        self,
        spider: Optional[str] = None,
        group_path: Optional[str] = None,
        include_round: bool = True,
        limit_groups: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        LIVE: Return aggregated UI data for Explorer and group header (memory-only).

        Output:
        - global: session + results-based success_rate + running/pending
        - groups: list of group summaries (registered in current session)
        - meta: watermark (based on event buffer)

        Note:
        - Anchor status must be provided from core memory cache.
          This engine will read it if governor exposes `anchor_status_cache`.
        """
        # 1) Prepare group registry (runtime only)
        with self.gov.stats_lock:
            paths = sorted(self.gov.runtime_groups.keys())
            if limit_groups is not None:
                paths = paths[: max(0, int(limit_groups))]

            active_groups = []
            for p in paths:
                g = self.gov.runtime_groups[p]
                if spider and not p.startswith(spider):
                    continue
                if group_path and p != group_path:
                    continue
                active_groups.append(
                    {
                        "group_path": p,
                        "name": g.get("name"),
                        "spider": g.get("spider"),
                        "list_url": g.get("list_url"),
                    }
                )

        # 2) Compute LIVE-only group stats from memory events
        stats_map, watermark = self._aggregate_live_group_stats()

        # 3) Build group DTOs (with anchor + round)
        groups_out: List[Dict[str, Any]] = []
        with self.gov.stats_lock:
            for g in active_groups:
                gp = g["group_path"]
                st = stats_map.get(gp, self._default_group_stats())

                # Anchor status: memory-only cache (expected to be maintained by core)
                anchor = None
                list_url = g.get("list_url")
                cache = getattr(self.gov, "anchor_status_cache", None)
                if list_url and isinstance(cache, dict):
                    anchor = cache.get(list_url)

                round_obj = None
                if include_round:
                    ctx = self.gov.round_contexts.get(gp)
                    if ctx:
                        round_obj = ctx.get_snapshot()

                groups_out.append(
                    {
                        "group_path": gp,
                        "name": g.get("name"),
                        "spider": g.get("spider"),
                        "list_url": list_url,
                        "anchor": anchor,
                        "stats": st,
                        "round": round_obj,
                    }
                )

        # 4) Global results-based success rate (sum over groups)
        global_results = self._init_counts()
        global_running = 0
        for g in groups_out:
            r = (g.get("stats") or {}).get("results") or {}
            global_results["total"] += int(r.get("total", 0) or 0)
            global_results["success"] += int(r.get("success", 0) or 0)
            global_results["failed"] += int(r.get("failed", 0) or 0)
            global_running += int(r.get("running", 0) or 0)

        pending_count = 0
        # LIVE principle: pending_count is persistent state, but you may accept it as "summary" info.
        # If you want absolute purity (memory-only), remove this and let UI query it separately.
        try:
            pending_count = int(self.gov.get_pending_count())
        except Exception:
            pending_count = 0

        session_start_ms = 0
        try:
            # Prefer a numeric timestamp from core if available
            t = getattr(self.gov, "session_start_time", None)
            session_start_ms = self._to_ms(t)
        except Exception:
            session_start_ms = 0

        return {
            "meta": {
                "mode": "LIVE",
                "source": "MEMORY",
                "schema": 1,
                "server_ts_ms": self._now_ms(),
                "watermark": watermark,
            },
            "global": {
                "session_start_ts_ms": session_start_ms,
                "results": global_results,
                "success_rate": self.calc_success_rate(global_results),
                "running_count": global_running,
                "pending_count": pending_count,
                "updated_ts_ms": self._now_ms(),
            },
            "groups": groups_out,
        }

    # ---------------------------------------------------------------------
    # QUERY (DB-only by design, no merging with LIVE)
    # ---------------------------------------------------------------------

    def query_logs(
        self,
        start_ts: float,
        end_ts: float,
        group_path: Optional[str] = None,
        spider: Optional[str] = None,
        status: Optional[int] = None,
        limit: int = 5000,
    ) -> Dict[str, Any]:
        """
        QUERY: Read logs from DB by time range (seconds), no merge with memory.

        Output item fields match LIVE logs as much as possible.
        """
        if not getattr(self.gov, "db", None):
            return {
                "meta": {
                    "mode": "QUERY",
                    "source": "DB",
                    "schema": 1,
                    "server_ts_ms": self._now_ms(),
                    "watermark": "db:missing",
                },
                "items": [],
            }

        start_s = int(start_ts)
        end_s = int(end_ts)
        if end_s <= start_s:
            end_s = start_s + 1

        sql = """
            SELECT l.*, s.url_hash
            FROM crawl_log l
            LEFT JOIN crawl_status s ON l.url = s.url
            WHERE l.created_at BETWEEN ? AND ?
        """
        params: List[Any] = [start_s, end_s]

        if spider:
            sql += " AND l.group_path LIKE ?"
            params.append(f"{spider}%")
        if group_path:
            sql += " AND l.group_path = ?"
            params.append(group_path)
        if status is not None:
            sql += " AND l.status = ?"
            params.append(int(status))

        sql += " ORDER BY l.id DESC LIMIT ?"
        params.append(int(limit))

        rows = self.gov.db.fetch_all_dict(sql, tuple(params))
        items: List[Dict[str, Any]] = []

        max_id = 0
        for r in rows or []:
            # created_at in DB is epoch seconds
            created_ms = self._to_ms(r.get("created_at"))
            dur = float(r.get("duration") or 0.0)
            approx_updated = created_ms + int(dur * 1000) if dur > 0 else created_ms

            if int(r.get("id") or 0) > max_id:
                max_id = int(r.get("id") or 0)

            items.append(
                {
                    "id": int(r.get("id") or 0),
                    "url": r.get("url"),
                    "url_hash": r.get("url_hash") or (self._url_hash(r.get("url") or "") if r.get("url") else None),
                    "group_path": r.get("group_path"),
                    "spider": r.get("spider_name"),
                    "status": int(r.get("status", 0)),
                    "http_code": r.get("http_code"),
                    "duration_s": dur,
                    "start_ts_ms": created_ms,
                    "updated_ts_ms": approx_updated,
                    "state_msg": r.get("state_msg"),
                }
            )

        watermark = f"db:logs:{max_id}:{start_s}:{end_s}"
        return {
            "meta": {
                "mode": "QUERY",
                "source": "DB",
                "schema": 1,
                "server_ts_ms": self._now_ms(),
                "watermark": watermark,
                "window": {"start_ts": start_s, "end_ts": end_s},
            },
            "items": items,
        }

    def query_trend(
        self,
        start_ts: float,
        end_ts: float,
        bucket_minutes: int = 60,
        group_filter: Optional[str] = None,
        use_updated_at: bool = True,
        include_cached_as_success: bool = False,
    ) -> Dict[str, Any]:
        """
        QUERY: Trend buckets for bar chart (DB-only).
        This intentionally does not run on a timer; frontend triggers it manually.
        """
        if not getattr(self.gov, "db", None):
            return {
                "meta": {
                    "mode": "QUERY",
                    "source": "DB",
                    "schema": 1,
                    "server_ts_ms": self._now_ms(),
                    "watermark": "db:missing",
                },
                "items": [],
            }

        bucket_seconds = max(1, int(bucket_minutes)) * 60
        time_col = "updated_at" if use_updated_at else "last_run_at"

        sql = f"""
            SELECT
                (s.{time_col} / ?) * ? as bucket_ts,
                s.status as status,
                COUNT(*) as cnt
            FROM crawl_status s
            LEFT JOIN task_groups g ON s.url = g.list_url
            WHERE s.{time_col} IS NOT NULL
              AND s.{time_col} BETWEEN ? AND ?
              AND g.list_url IS NULL
        """
        params: List[Any] = [bucket_seconds, bucket_seconds, int(start_ts), int(end_ts)]

        if group_filter:
            sql += " AND s.group_path = ?"
            params.append(group_filter)

        sql += " GROUP BY bucket_ts, status ORDER BY bucket_ts ASC"

        rows = self.gov.db.fetch_all_dict(sql, tuple(params)) or []

        # Fill empty buckets for stable chart
        def floor_bucket(ts: float) -> int:
            return int(ts // bucket_seconds) * bucket_seconds

        start_bucket = floor_bucket(start_ts)
        end_bucket = floor_bucket(end_ts)
        timeline: Dict[int, Dict[str, Any]] = {}

        cur = start_bucket
        while cur <= end_bucket:
            timeline[cur] = {
                "bucket_ts_ms": int(cur * 1000),
                "success": 0,
                "fail": 0,
                "total": 0,
            }
            cur += bucket_seconds

        success_set = set(TerminalSuccessSet)
        if include_cached_as_success:
            success_set.add(STATUS_CACHED)

        fail_set = set(TerminalFailSet)

        for r in rows:
            b = int(r.get("bucket_ts") or 0)
            st = int(r.get("status") or 0)
            cnt = int(r.get("cnt") or 0)
            if b not in timeline:
                timeline[b] = {"bucket_ts_ms": int(b * 1000), "success": 0, "fail": 0, "total": 0}
            if st in success_set:
                timeline[b]["success"] += cnt
                timeline[b]["total"] += cnt
            elif st in fail_set:
                timeline[b]["fail"] += cnt
                timeline[b]["total"] += cnt
            else:
                # ignore pending/running/skipped/ignored by default
                pass

        items = [timeline[k] for k in sorted(timeline.keys())]
        watermark = f"db:trend:{int(start_ts)}:{int(end_ts)}:{bucket_minutes}:{group_filter or 'all'}:{int(use_updated_at)}:{int(include_cached_as_success)}"

        return {
            "meta": {
                "mode": "QUERY",
                "source": "DB",
                "schema": 1,
                "server_ts_ms": self._now_ms(),
                "watermark": watermark,
                "window": {"start_ts": int(start_ts), "end_ts": int(end_ts), "bucket_minutes": int(bucket_minutes)},
            },
            "items": items,
        }

    def export_csv(
        self,
        export_type: str,
        group_path: Optional[str] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> str:
        """
        QUERY: Generate CSV content.

        Supported export_type:
        - "global"        : current live summary snapshot (results-based)
        - "group_logs"    : DB logs by time window (requires start_ts/end_ts) OR latest N if window omitted
        - "group_status"  : reserved (status query not required now)
        """
        out = io.StringIO()
        writer = csv.writer(out)

        if export_type == "global":
            # Snapshot export: uses LIVE summary output.
            payload = self.live_summary(include_round=True)
            writer.writerow(
                [
                    "GroupPath",
                    "Spider",
                    "Name",
                    "RoundID",
                    "Phase",
                    "ResultsTotal",
                    "ResultsSuccess",
                    "ResultsFailed",
                    "Running",
                    "AvgDuration(s)",
                    "AnchorStatus",
                    "AnchorHttpCode",
                ]
            )
            for g in payload.get("groups", []):
                stats = (g.get("stats") or {}).get("results") or {}
                perf = (g.get("stats") or {}).get("perf") or {}
                round_obj = g.get("round") or {}
                anchor = g.get("anchor") or {}
                writer.writerow(
                    [
                        g.get("group_path"),
                        g.get("spider"),
                        g.get("name"),
                        round_obj.get("round_id", 0),
                        round_obj.get("phase", "IDLE"),
                        stats.get("total", 0),
                        stats.get("success", 0),
                        stats.get("failed", 0),
                        stats.get("running", 0),
                        perf.get("avg", 0),
                        anchor.get("status", ""),
                        anchor.get("http_code", ""),
                    ]
                )
            return out.getvalue()

        if export_type == "group_logs":
            if not group_path:
                writer.writerow(["ERROR", "Missing group_path"])
                return out.getvalue()

            # If time window provided -> query DB, else export latest N from memory logs
            if start_ts is not None and end_ts is not None:
                payload = self.query_logs(start_ts, end_ts, group_path=group_path, limit=10000)
                rows = payload.get("items", [])
                writer.writerow(["ID", "UpdatedTime(ms)", "StartTime(ms)", "URL", "Status", "HTTP", "Duration(s)"])
                for r in rows:
                    writer.writerow(
                        [
                            r.get("id"),
                            r.get("updated_ts_ms"),
                            r.get("start_ts_ms"),
                            r.get("url"),
                            r.get("status"),
                            r.get("http_code"),
                            r.get("duration_s"),
                        ]
                    )
            else:
                payload = self.live_logs(group_path=group_path, limit=10000)
                rows = payload.get("items", [])
                writer.writerow(["ID", "UpdatedTime(ms)", "StartTime(ms)", "URL", "Status", "HTTP", "Duration(s)"])
                for r in rows:
                    writer.writerow(
                        [
                            r.get("id"),
                            r.get("updated_ts_ms"),
                            r.get("start_ts_ms"),
                            r.get("url"),
                            r.get("status"),
                            r.get("http_code"),
                            r.get("duration_s"),
                        ]
                    )
            return out.getvalue()

        writer.writerow(["ERROR", f"Unsupported export_type: {export_type}"])
        return out.getvalue()

    # ---------------------------------------------------------------------
    # Schema/Migration helpers (hash-based SNAP support)
    # ---------------------------------------------------------------------

    def ensure_crawl_status_url_hash_column(self) -> Dict[str, Any]:
        """
        Ensure crawl_status.url_hash exists (non-destructive migration).

        This method:
        - inspects table schema via PRAGMA table_info
        - adds url_hash column if missing

        Note:
        - SQLite allows ALTER TABLE ADD COLUMN without dropping data.
        - Backfill of url_hash values can be performed separately.
        """
        if not getattr(self.gov, "db", None):
            return {"ok": False, "error": "db not available"}

        try:
            cols = self.gov.db.fetch_all_dict("PRAGMA table_info(crawl_status)") or []
            names = {str(c.get("name")) for c in cols}
            if "url_hash" in names:
                return {"ok": True, "changed": False, "note": "url_hash already exists"}

            # Add column (TEXT)
            self.gov.db.execute("ALTER TABLE crawl_status ADD COLUMN url_hash TEXT")
            return {"ok": True, "changed": True, "note": "url_hash column added"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def backfill_crawl_status_url_hash(self, batch_size: int = 2000) -> Dict[str, Any]:
        """
        Backfill crawl_status.url_hash for rows where url_hash is NULL/empty.

        Note:
        - Uses md5(url) by default (or governor._hash(url) if available).
        - Batch update to reduce transaction cost.
        """
        if not getattr(self.gov, "db", None):
            return {"ok": False, "error": "db not available"}

        try:
            rows = self.gov.db.fetch_all_dict(
                "SELECT url FROM crawl_status WHERE url_hash IS NULL OR url_hash = '' LIMIT ?",
                (int(batch_size),),
            ) or []
            if not rows:
                return {"ok": True, "changed": False, "updated": 0}

            updated = 0
            for r in rows:
                url = r.get("url")
                if not url:
                    continue
                h = self._url_hash(url)
                self.gov.db.execute("UPDATE crawl_status SET url_hash = ? WHERE url = ?", (h, url))
                updated += 1

            return {"ok": True, "changed": True, "updated": updated}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---------------------------------------------------------------------
    # Internal aggregation (LIVE only)
    # ---------------------------------------------------------------------

    def _default_group_stats(self) -> Dict[str, Any]:
        return {
            "results": self._init_counts(),
            "traffic": self._init_counts(),
            "perf": {"min": 0.0, "max": 0.0, "avg": 0.0, "count": 0},
        }

    def _aggregate_live_group_stats(self) -> Tuple[Dict[str, Dict[str, Any]], str]:
        """
        Aggregate memory events into group stats.

        Returns:
        - stats_map: {group_path: {results, traffic, perf}}
        - watermark: derived from max updated_ts_ms and buffer length
        """
        stats_map: Dict[str, Dict[str, Any]] = {}

        # For unique results: track latest status per URL per group
        latest_by_group_url: Dict[str, Dict[str, int]] = {}

        max_updated_ms = 0
        buf_len = 0

        with self.gov.stats_lock:
            buf = list(self.gov.event_buffer)
            buf_len = len(buf)

        for e in buf:
            gp = e.get("group_path") or ""
            url = e.get("url") or ""
            st = int(e.get("status", 0))
            dur = float(e.get("duration") or 0.0)

            updated_ms = self._to_ms(e.get("updated_ts", e.get("ts")))
            if updated_ms > max_updated_ms:
                max_updated_ms = updated_ms

            if gp not in stats_map:
                stats_map[gp] = {
                    "results": self._init_counts(),
                    "traffic": self._init_counts(),
                    "perf": self._init_perf(),
                }
                latest_by_group_url[gp] = {}

            # 1) Traffic: count each event
            self._accumulate_counts(stats_map[gp]["traffic"], st, 1)

            # 2) Perf: only terminal states with duration > 0
            if st not in {STATUS_PENDING, STATUS_RUNNING, STATUS_SKIPPED, STATUS_IGNORED} and dur > 0:
                p = stats_map[gp]["perf"]
                if p["count"] == 0:
                    p["min"] = dur
                    p["max"] = dur
                else:
                    p["min"] = min(p["min"], dur)
                    p["max"] = max(p["max"], dur)
                p["sum"] += dur
                p["count"] += 1

            # 3) Unique results: overwrite latest status for URL
            if url:
                latest_by_group_url[gp][url] = st

        # Finalize results + perf avg
        for gp, data in stats_map.items():
            # Results from dedup map
            url_map = latest_by_group_url.get(gp, {})
            for st in url_map.values():
                self._accumulate_counts(data["results"], st, 1)

            # Perf avg
            p = data["perf"]
            if p["count"] > 0:
                p["avg"] = round(p["sum"] / p["count"], 3)
            else:
                p["min"] = 0.0
                p["max"] = 0.0
                p["avg"] = 0.0

            # Strip internal fields
            p.pop("sum", None)

        watermark = self._make_watermark(max_updated_ms, buf_len)
        return stats_map, watermark

    def resolve_snapshot_path(self, url_hash: str) -> Optional[str]:
        """
        Map url_hash -> file_path via DB.
        Returns absolute or stored file_path, or None if not found.
        """
        if not getattr(self.gov, "db", None):
            return None
        try:
            row = self.gov.db.fetch_one_dict(
                "SELECT file_path FROM crawl_status WHERE url_hash = ? AND file_path IS NOT NULL AND file_path != '' LIMIT 1",
                (url_hash,)
            )
            if not row:
                return None
            return row.get("file_path")
        except Exception:
            return None
