import os
import time
import threading
import datetime

from typing import Optional
from urllib.parse import urljoin

from flask_cors import CORS
from flask import Flask, jsonify, request, send_file, render_template

try:
    from CrawlerGovernanceCore import GovernanceManager
    from GovernanceDataEngine import GovernanceDataEngine
except Exception as e:
    print(str(e))
    from IntelligenceCrawler.CrawlerGovernanceCore import GovernanceManager
    from IntelligenceCrawler.GovernanceDataEngine import GovernanceDataEngine

self_path = os.path.dirname(os.path.abspath(__file__))


class CrawlerGovernanceBackend:
    def __init__(self,
                 governor: GovernanceManager,
                 host: Optional[str] = "0.0.0.0",
                 port: Optional[int] = 8002,
                 app: Optional[Flask] = None,
                 base_url: Optional[str] = ''
                 ):
        """
        Initialize the Governance Backend Service.
        Decoupled from DB. Relies entirely on GovernanceManager interfaces.
        """
        self.governor = governor
        self.host = host
        self.port = port
        self.app = app
        self.base_url = base_url

        self.own_app = not app
        self.flask_thread = None

        self.data_engine = GovernanceDataEngine(governor)

    def start_service(self, blocking: bool = False):
        """Start the Flask web service."""
        if not self.app:
            self.app = Flask(__name__)
            self.app.secret_key = os.urandom(24)
            CORS(self.app)

        self._register_routes(wrapper=None)
        self.app.template_folder = self_path

        print(f"Starting Governance API Server on http://{self.host}:{self.port}")

        if self.own_app:
            if blocking:
                self.app.run(debug=False, host=self.host, port=self.port, use_reloader=False, threaded=True)
            else:
                def run_flask():
                    self.app.run(debug=False, host=self.host, port=self.port, use_reloader=False, threaded=True)

                self.flask_thread = threading.Thread(target=run_flask, daemon=True, name='CrawlerGovernanceBackend')
                self.flask_thread.start()
                time.sleep(1)
                print(f"Flask server running in background.")

    def _register_routes(self, wrapper):

        def maybe_wrap(fn): return wrapper(fn) if wrapper else fn

        def build_url(endpoint: str) -> str: return urljoin(self.base_url, endpoint)

        # UI
        self.app.add_url_rule(build_url('/'), 'read_root', maybe_wrap(self.read_root))

        # Data APIs (GET)
        self.app.add_url_rule(build_url('/api/dashboard/stats'), 'get_dashboard_stats',
                              maybe_wrap(self.get_dashboard_stats), methods=['GET'])
        self.app.add_url_rule(build_url('/api/flow/snapshot'), 'get_flow_snapshot',
                              maybe_wrap(self.get_flow_snapshot), methods=['GET'])
        self.app.add_url_rule(build_url('/api/groups'), 'get_groups',
                              maybe_wrap(self.get_groups), methods=['GET'])
        self.app.add_url_rule(build_url('/api/logs'), 'get_logs',
                              maybe_wrap(self.get_logs), methods=['GET'])
        self.app.add_url_rule(build_url('/api/status/recent'), 'get_recent_statuses',
                              maybe_wrap(self.get_recent_statuses), methods=['GET'])
        self.app.add_url_rule(build_url('/api/snapshot/<url_hash>'), 'get_snapshot',
                              maybe_wrap(self.get_snapshot), methods=['GET'])
        self.app.add_url_rule(build_url('/api/dashboard/chart'), 'get_trend_chart',
                              maybe_wrap(self.get_trend_chart), methods=['GET'])

        self.app.add_url_rule(build_url('/api/entry/status'), 'get_entry_status',
                              maybe_wrap(self.get_entry_status), methods=['GET'])
        self.app.add_url_rule(build_url('/api/entry/history'), 'get_entry_history',
                              maybe_wrap(self.get_entry_history), methods=['GET'])
        self.app.add_url_rule(build_url('/api/entry/articles'), 'get_entry_articles',
                              maybe_wrap(self.get_entry_articles), methods=['GET'])

        self.app.add_url_rule(build_url('/api/history/stats'), 'get_history_stats',
                              maybe_wrap(self.get_history_stats), methods=['GET'])
        self.app.add_url_rule(build_url('/api/group/round_status'), 'get_group_round_status',
                              maybe_wrap(self.get_group_round_status), methods=['GET'])
        self.app.add_url_rule(build_url('/api/export/<target>'), 'export_data',
                              maybe_wrap(self.export_data), methods=['GET'])

        # Control APIs (POST)
        self.app.add_url_rule(build_url('/api/control/<action>'), 'system_control',
                              maybe_wrap(self.system_control), methods=['POST'])
        self.app.add_url_rule(build_url('/api/control/reset_stats'), 'reset_stats',
                              maybe_wrap(self.reset_stats), methods=['POST'])

        # RPC APIs (POST)
        self.app.add_url_rule(build_url('/rpc/register_group'), 'rpc_register_group',
                              maybe_wrap(self.rpc_register_group), methods=['POST'])
        self.app.add_url_rule(build_url('/rpc/should_crawl'), 'rpc_should_crawl',
                              maybe_wrap(self.rpc_should_crawl), methods=['POST'])
        self.app.add_url_rule(build_url('/rpc/report_result'), 'rpc_report_result',
                              maybe_wrap(self.rpc_report_result), methods=['POST'])
        self.app.add_url_rule(build_url('/rpc/round/lifecycle'), 'rpc_round_lifecycle',
                              maybe_wrap(self.rpc_round_lifecycle), methods=['POST'])

        # 给所有 API 响应添加禁止缓存头
        @self.app.after_request
        def add_header(response):
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            return response

    # --- Helper: Time Parsing ---

    def _get_since_time(self) -> Optional[datetime.datetime]:
        """Parses 'since' query parameter (timestamp float)."""
        ts = request.args.get('since', type=float)
        if ts:
            try:
                return datetime.datetime.fromtimestamp(ts)
            except (ValueError, OSError):
                return None
        return None

    # --- Web Service Methods ---

    def read_root(self):
        return render_template("crawler_governance_frontend.html")

    def get_dashboard_stats(self):
        """
        Aggregated Global + Groups summary (LIVE only).
        Query:
          - spider: optional
          - group_path: optional
          - include_round: optional int (1/0), default 1
          - limit_groups: optional int
        """
        if not self.governor:
            return jsonify({"error": "Init failed"}), 500

        spider = request.args.get('spider')
        group_path = request.args.get('group_path')
        include_round = request.args.get('include_round', default=1, type=int)
        limit_groups = request.args.get('limit_groups', type=int)

        payload = self.data_engine.live_summary(
            spider=spider,
            group_path=group_path,
            include_round=bool(include_round),
            limit_groups=limit_groups
        )
        return jsonify(payload)

    def get_flow_snapshot(self):
        if not self.governor or not self.governor.scheduler:
            return jsonify({}), 500
        limit = int(request.args.get("limit", "50"))
        snap = self.governor.get_scheduler_snapshot(max_items_per_state=limit)
        return jsonify(snap)

    def get_groups(self):
        """
        Group registry + per-group stats (LIVE only).
        Returns only the 'groups' array from live_summary to keep endpoint semantics.
        """
        if not self.governor:
            return jsonify({"error": "Init failed"}), 500

        spider = request.args.get('spider')
        group_path = request.args.get('group_path')
        include_round = request.args.get('include_round', default=1, type=int)
        limit_groups = request.args.get('limit_groups', type=int)

        summary = self.data_engine.live_summary(
            spider=spider,
            group_path=group_path,
            include_round=bool(include_round),
            limit_groups=limit_groups
        )
        return jsonify(summary.get("groups", []))

    def get_logs(self):
        """
        Logs:
          - LIVE: no time window -> latest memory events with optional filters
          - QUERY: with start_ts & end_ts -> DB logs
        Query:
          - limit: int (LIVE only, default 100)
          - status: int (optional)
          - spider: str (optional)
          - group_path: str (optional)
          - start_ts: float seconds (QUERY only)
          - end_ts: float seconds (QUERY only)
        """
        if not self.governor:
            return jsonify({"error": "Init failed"}), 500

        # Filters
        limit = request.args.get('limit', 100, type=int)
        status = request.args.get('status', type=int)
        spider = request.args.get('spider')
        group_path = request.args.get('group_path')

        # Time window -> QUERY
        start_ts = request.args.get('start_ts', type=float)
        end_ts = request.args.get('end_ts', type=float)

        if start_ts is not None and end_ts is not None:
            if end_ts <= start_ts:
                end_ts = start_ts + 1.0
            payload = self.data_engine.query_logs(
                start_ts=float(start_ts),
                end_ts=float(end_ts),
                group_path=group_path,
                spider=spider,
                status=status,
                limit=max(1, int(limit))
            )
            return jsonify(payload)

        # No window -> LIVE
        payload = self.data_engine.live_logs(
            group_path=group_path,
            spider=spider,
            status=status,
            limit=max(1, int(limit))
        )
        return jsonify(payload)

    def get_recent_statuses(self):
        """
        Latest URL statuses (LIVE-only, dedup by URL).
        Query:
          - limit: int (default 100)
          - spider: str (optional)
          - group_path: str (optional)
          - status: int (optional)
        """
        if not self.governor:
            return jsonify({"error": "Init failed"}), 500

        limit = request.args.get('limit', 100, type=int)
        spider = request.args.get('spider')
        group_path = request.args.get('group_path')
        status = request.args.get('status', type=int)

        payload = self.data_engine.live_statuses(
            group_path=group_path,
            spider=spider,
            status=status,
            limit=max(1, int(limit))
        )
        return jsonify(payload)

    def get_snapshot(self, url_hash: str):
        """
        Serve snapshot file by url_hash.
        Requires Engine to resolve hash -> file path via DB crawl_status.url_hash.
        """
        if not self.governor:
            return jsonify({"error": "Init failed"}), 500

        if not url_hash.isalnum():
            return jsonify({"error": "Invalid hash"}), 400

        try:
            file_path = self.data_engine.resolve_snapshot_path(url_hash)
        except AttributeError:
            # Engine 尚未实现该方法
            return jsonify({"error": "Engine missing resolve_snapshot_path(url_hash)"}), 501

        if not file_path:
            return jsonify({"error": "Snapshot not found"}), 404

        if not os.path.exists(file_path):
            return jsonify({"error": "File on disk missing"}), 404

        return send_file(file_path)

    def get_trend_chart(self):
        """
        Trend chart (QUERY-only).
        Query:
          - start_ts, end_ts: float seconds
          - bucket_minutes: int (default 60)
          - group_filter: str (optional)
          - use_updated_at: int 1/0 (default 1)
          - include_cached_as_success: int 1/0 (default 0)
        """
        if not self.governor:
            return jsonify({"error": "Init failed"}), 500

        now = time.time()

        start_ts = request.args.get('start_ts', type=float)
        end_ts = request.args.get('end_ts', type=float)

        if start_ts is None:
            start_ts = now - 3600
        if end_ts is None:
            end_ts = now
        if end_ts <= start_ts:
            end_ts = start_ts + 1.0

        bucket_minutes = request.args.get('bucket_minutes', type=int)
        if bucket_minutes is None or bucket_minutes <= 0:
            bucket_minutes = 60

        group_filter = request.args.get('group_filter', type=str)

        use_updated_at = request.args.get('use_updated_at', default=1, type=int)
        include_cached = request.args.get('include_cached_as_success', default=0, type=int)

        payload = self.data_engine.query_trend(
            start_ts=float(start_ts),
            end_ts=float(end_ts),
            bucket_minutes=int(bucket_minutes),
            group_filter=group_filter,
            use_updated_at=bool(int(use_updated_at)),
            include_cached_as_success=bool(int(include_cached))
        )
        return jsonify(payload)

    def get_history_stats(self):
        """
        Deprecated: no Engine counterpart for 'history stats' yet.
        Please use:
          - /api/dashboard/chart   for trend (QUERY)
          - /api/logs              for logs (LIVE / QUERY)
        """
        return jsonify({"error": "deprecated endpoint; use /api/dashboard/chart or /api/logs"}), 410

    def get_group_round_status(self):
        """
        [API] 获取指定 Group 的实时轮次状态 (进度、统计、倒计时)。
        前端轮询此接口以更新进度条。
        Query Params:
            - group: string (必填, e.g., "news/tech")
        """
        if not self.governor: return jsonify({"error": "Init failed"}), 500

        group_path = request.args.get('group')
        if not group_path:
            return jsonify({"error": "Missing group parameter"}), 400

        # 调用 Governor 新增的接口
        status_data = self.governor.get_group_round_status(group_path)
        return jsonify(status_data)

    def get_entry_status(self):
        """
        [API] Get the current Entry Round live snapshot (memory) for a group.
        Query Params:
            - group: string (required)
        """
        if not self.governor:
            return jsonify({"error": "Init failed"}), 500

        group_path = request.args.get('group')
        if not group_path:
            return jsonify({"error": "Missing group parameter"}), 400

        snapshot = self.governor.get_entry_round_status(group_path)
        return jsonify({
            "meta": {
                "mode": "LIVE",
                "source": "MEMORY",
                "schema": 1,
                "server_ts_ms": int(time.time() * 1000),
            },
            "entry_round": snapshot or {},
        })

    def get_entry_history(self):
        """
        [API] Get Entry Round history for a group (DB).
        Query Params:
            - group: string (required)
            - limit: int (default 50)
            - offset: int (default 0)
        """
        if not self.governor:
            return jsonify({"error": "Init failed"}), 500

        group_path = request.args.get('group')
        if not group_path:
            return jsonify({"error": "Missing group parameter"}), 400

        limit = request.args.get('limit', default=50, type=int)
        offset = request.args.get('offset', default=0, type=int)

        items = self.governor.get_entry_round_history(group_path, limit=limit, offset=offset)
        return jsonify({
            "meta": {
                "mode": "QUERY",
                "source": "DB",
                "schema": 1,
                "server_ts_ms": int(time.time() * 1000),
                "limit": limit,
                "offset": offset,
            },
            "items": items,
        })

    def get_entry_articles(self):
        """
        [API] Get article crawl records under an Entry Round (DB).
        Query Params:
            - entry_round_id: int (required)
            - limit: int (default 100)
            - offset: int (default 0)
        """
        if not self.governor:
            return jsonify({"error": "Init failed"}), 500

        entry_round_id = request.args.get('entry_round_id', type=int)
        if entry_round_id is None:
            return jsonify({"error": "Missing entry_round_id parameter"}), 400

        limit = request.args.get('limit', default=100, type=int)
        offset = request.args.get('offset', default=0, type=int)

        items = self.governor.get_entry_round_articles(entry_round_id, limit=limit, offset=offset)
        return jsonify({
            "meta": {
                "mode": "QUERY",
                "source": "DB",
                "schema": 1,
                "server_ts_ms": int(time.time() * 1000),
                "limit": limit,
                "offset": offset,
            },
            "items": items,
        })

    def export_data(self, target):
        """
        CSV export.
        target: 'global' | 'group_logs' | 'group_status' (reserved)
        Query:
          - group: str (for group_* targets)
          - start_ts, end_ts: float seconds (optional; if provided and target == group_logs -> QUERY)
        """
        from flask import Response

        if not self.governor:
            return jsonify({"error": "Init failed"}), 500

        group = request.args.get('group')
        start_ts = request.args.get('start_ts', type=float)
        end_ts = request.args.get('end_ts', type=float)

        if target == 'global':
            csv_content = self.data_engine.export_csv('global')
            filename = f"global_{int(time.time())}.csv"

        elif target == 'group_logs':
            if not group:
                return jsonify({"error": "Missing group"}), 400
            # 如果提供了时间窗 -> 传给 Engine 走 QUERY；否则 Engine 会走 LIVE
            csv_content = self.data_engine.export_csv(
                'group_logs', group_path=group, start_ts=start_ts, end_ts=end_ts
            )
            filename = f"logs_{group.replace('/', '_')}.csv"

        elif target == 'group_status':
            # Engine 里目前是 reserved/未实现；直接返回错误更明确
            return jsonify({"error": "export type 'group_status' not supported yet"}), 400

        else:
            return jsonify({"error": "Invalid export target"}), 400

        return Response(
            csv_content or "",
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={filename}"}
        )

    def reset_stats(self):
        if self.governor:
            self.governor.reset_statistics()
        return jsonify({"status": "reset"})

    def system_control(self, action: str):
        if not self.governor: return jsonify({"error": "Init failed"}), 500
        action = action.upper()
        if action == "PAUSE":
            self.governor.pause()
        elif action == "RESUME":
            self.governor.resume()
        elif action == "IMMEDIATE":
            self.governor.trigger_immediate()
        else:
            return jsonify({"error": "Unknown action"}), 400
        return jsonify({"status": "ok", "action": action})

    # --- RPC Methods (Unchanged logic, just clean) ---

    def rpc_register_group(self):
        if not self.governor: return jsonify({"error": "Init failed"}), 500
        data = request.get_json() or {}

        group = data.get('group_path') or data.get('group')
        if not group: return jsonify({"error": "Missing group"}), 400

        self.governor.register_group_metadata(
            group_path=group,
            list_url=data.get('list_url') or data.get('url'),
            friendly_name=data.get('name')
        )
        return jsonify({"status": "registered"})

    def rpc_should_crawl(self):
        if not self.governor: return jsonify({"error": "Init failed"}), 500
        data = request.get_json() or {}
        if 'url' not in data: return jsonify({"error": "Missing url"}), 400

        should = self.governor.should_crawl(data['url'], data.get('max_retries', 3))
        return jsonify({"should_crawl": should})

    def rpc_report_result(self):
        if not self.governor: return jsonify({"error": "Init failed"}), 500
        data = request.get_json() or {}
        if not all(k in data for k in ['url', 'group_path', 'status']):
            return jsonify({"error": "Missing fields"}), 400

        gp = data.get('group_path') or ""
        spider = data.get('spider') or (gp.split('/')[0] if gp else "")

        self.governor._handle_task_finish(
            log_id=None,
            url=data['url'],
            spider=spider,
            group_path=data['group_path'],
            status=int(data['status']),
            duration=data.get('duration', 0.0),
            http_code=data.get('http_code', 0),
            state_msg=data.get('error_msg') or data.get('state_msg'),
            file_path=data.get('file_path')
        )
        return jsonify({"status": "acked"})

    def rpc_round_lifecycle(self):
        """
        [RPC] 外部爬虫控制轮次生命周期。
        POST JSON Payload:
            - action: "start" | "finish"
            - group: "news/tech"
            - expected_count: int (仅 start 需要)
            - next_run_delay: float (仅 finish 需要, 单位秒)
        """
        if not self.governor: return jsonify({"error": "Init failed"}), 500
        data = request.get_json() or {}

        action = data.get('action')
        group = data.get('group')

        if not action or not group:
            return jsonify({"error": "Missing action or group"}), 400

        if action == "start":
            expected = int(data.get('expected_count', 0))
            self.governor.start_round(group, expected_count=expected)
            return jsonify({"status": "started", "group": group})

        elif action == "finish":
            delay = float(data.get('next_run_delay', 0))
            self.governor.finish_round(group, next_run_delay=int(delay))
            return jsonify({"status": "finished", "group": group})

        else:
            return jsonify({"error": "Invalid action"}), 400
