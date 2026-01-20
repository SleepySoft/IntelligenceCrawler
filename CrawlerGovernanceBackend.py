import os
import time
import threading
import datetime

from typing import Optional
from urllib.parse import urljoin

from flask_cors import CORS
from flask import Flask, jsonify, request, send_file, render_template

# Import the core logic (Assuming relative import or package structure)
# from IntelligenceCrawler.CrawlerGovernanceCore import GovernanceManager, Status

self_path = os.path.dirname(os.path.abspath(__file__))


class CrawlerGovernanceBackend:
    def __init__(self,
                 governor,  # Type: GovernanceManager
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
                self.app.run(debug=True, host=self.host, port=self.port, use_reloader=False, threaded=True)
            else:
                def run_flask():
                    self.app.run(debug=True, host=self.host, port=self.port, use_reloader=False, threaded=True)

                self.flask_thread = threading.Thread(target=run_flask, daemon=True)
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
        self.app.add_url_rule(build_url('/api/groups'), 'get_groups', maybe_wrap(self.get_groups), methods=['GET'])
        self.app.add_url_rule(build_url('/api/logs'), 'get_logs', maybe_wrap(self.get_logs), methods=['GET'])
        self.app.add_url_rule(build_url('/api/status/recent'), 'get_recent_statuses',
                              maybe_wrap(self.get_recent_statuses), methods=['GET'])
        self.app.add_url_rule(build_url('/api/snapshot/<url_hash>'), 'get_snapshot', maybe_wrap(self.get_snapshot),
                              methods=['GET'])

        # Control APIs (POST)
        self.app.add_url_rule(build_url('/api/control/<action>'), 'system_control', maybe_wrap(self.system_control),
                              methods=['POST'])
        self.app.add_url_rule(build_url('/api/control/reset_stats'), 'reset_stats', maybe_wrap(self.reset_stats),
                              methods=['POST'])

        # RPC APIs (POST)
        self.app.add_url_rule(build_url('/rpc/register_group'), 'rpc_register_group',
                              maybe_wrap(self.rpc_register_group), methods=['POST'])
        self.app.add_url_rule(build_url('/rpc/should_crawl'), 'rpc_should_crawl', maybe_wrap(self.rpc_should_crawl),
                              methods=['POST'])
        self.app.add_url_rule(build_url('/rpc/report_result'), 'rpc_report_result', maybe_wrap(self.rpc_report_result),
                              methods=['POST'])

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
        Aggregated Global Statistics.
        Supports 'since' parameter for incremental updates.
        """
        if not self.governor: return jsonify({}), 500

        since_time = self._get_since_time()

        # 1. Session Stats (Memory or DB Aggregation via Governor)
        session_stats = self.governor.get_session_stats(since_time=since_time)

        # 2. Pending Count (Persistent State via Governor Interface)
        pending_count = self.governor.get_pending_count()

        return jsonify({
            "active_spiders": 0,  # Placeholder, or use len(governor.group_stats)
            "success_rate": session_stats['success_rate'],
            "total_requests": session_stats['total'],
            "network_errors": session_stats['failed'],
            "running_count": session_stats.get('running', 0),  # Added running count
            "pending_count": pending_count,
            "session_start": session_stats['session_start']
        })

    def get_groups(self):
        """
        Group Hierarchy & Statistics.
        Supports 'since' parameter.
        Only returns groups registered in the current runtime session.
        """
        if not self.governor: return jsonify({"error": "Init failed"}), 500

        spider_filter = request.args.get('spider')
        since_time = self._get_since_time()

        # Delegate to Governor's smart aggregation
        summary = self.governor.get_dashboard_summary(spider_filter=spider_filter, since_time=since_time)
        return jsonify(summary)

    def get_logs(self):
        """
        Streaming Logs.
        Supports 'since' parameter to fetch new logs only.
        """
        if not self.governor: return jsonify({"error": "Init failed"}), 500

        limit = request.args.get('limit', 100, type=int)
        status = request.args.get('status', type=int)
        spider = request.args.get('spider')
        since_time = self._get_since_time()

        # Delegate to Governor
        logs = self.governor.get_logs(limit=limit, status=status, spider=spider, since_time=since_time)
        return jsonify(logs)

    def get_recent_statuses(self):
        """
        Latest URL Statuses (Live Memory View).
        Fetch the latest activity directly from memory.
        NO 'since' parameter used here as it returns a snapshot of the current state.
        """
        if not self.governor: return jsonify({"error": "Init failed"}), 500

        limit = request.args.get('limit', 100, type=int)
        spider = request.args.get('spider')
        status = request.args.get('status', type=int)

        # Removed 'since_time' logic as this is a memory snapshot
        # Delegate to Governor (Memory Only)
        statuses = self.governor.get_recent_statuses(limit=limit, spider=spider, status=status)
        return jsonify(statuses)

    def get_snapshot(self, url_hash: str):
        """
        Serve file content.
        """
        if not url_hash.isalnum(): return jsonify({"error": "Invalid hash"}), 400

        # Delegate path lookup to Governor
        file_path = self.governor.get_snapshot_path(url_hash)

        if not file_path:
            return jsonify({"error": "Snapshot not found"}), 404

        if not os.path.exists(file_path):
            return jsonify({"error": "File on disk missing"}), 404

        return send_file(file_path)

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

        spider = data.get('spider') or self.governor._extract_spider_name(data['group_path'])

        # Stateless call (log_id=None)
        self.governor._handle_task_finish(
            log_id=None,
            url=data['url'],
            spider=spider,
            group_path=data['group_path'],
            status=int(data['status']),  # Ensure int
            duration=data.get('duration', 0.0),
            http_code=data.get('http_code', 0),
            state_msg=data.get('error_msg') or data.get('state_msg'),
            file_path=data.get('file_path')
        )
        return jsonify({"status": "acked"})
