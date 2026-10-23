"""静态扫描项目内 crawler config/task，不执行配置文件。"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class CrawlerCatalogEntry:
    name: str
    group: str
    config_path: Path
    task_path: Optional[Path]
    discoverer: str = ""
    extractor: str = ""
    entry_count: int = 0
    has_declarative_filter: bool = False
    has_custom_filter: bool = False
    regression_sample_count: int = 0
    disabled: bool = False


def _literal(node: Optional[ast.AST], default: Any = None) -> Any:
    if node is None:
        return default
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return default


def _config_dict_node(tree: ast.Module) -> Optional[ast.Dict]:
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(isinstance(target, ast.Name) and target.id == "CRAWLER_CONFIG" for target in targets):
            return statement.value if isinstance(statement.value, ast.Dict) else None
    return None


def _dict_nodes(node: Optional[ast.Dict]) -> Dict[str, ast.AST]:
    if node is None:
        return {}
    result = {}
    for key, value in zip(node.keys, node.values):
        literal_key = _literal(key)
        if isinstance(literal_key, str):
            result[literal_key] = value
    return result


def _task_import_map(task_root: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for task_path in task_root.rglob("task_crawl_*.py"):
        try:
            tree = ast.parse(task_path.read_text(encoding="utf-8-sig"), filename=str(task_path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("crawler_config_"):
                result[node.module] = task_path
    return result


def _disabled_tasks(config_path: Optional[Path]) -> set[str]:
    if not config_path or not config_path.exists():
        return set()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        values = payload.get("collector", {}).get("disabled_tasks", [])
        return {str(value) for value in values}
    except (OSError, ValueError, TypeError):
        return set()


def scan_crawler_catalog(
    task_root: Path | str,
    collector_config: Path | str | None = None,
) -> List[CrawlerCatalogEntry]:
    root = Path(task_root).resolve()
    task_map = _task_import_map(root)
    disabled = _disabled_tasks(Path(collector_config) if collector_config else None)
    entries: List[CrawlerCatalogEntry] = []

    for config_path in sorted(root.rglob("crawler_config_*.py")):
        try:
            tree = ast.parse(config_path.read_text(encoding="utf-8-sig"), filename=str(config_path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        values = _dict_nodes(_config_dict_node(tree))
        module_name = config_path.stem
        task_path = task_map.get(module_name)
        entry_points = _literal(values.get("entry_points"), {})
        filter_node = values.get("article_filter")
        declarative_filter = _literal(values.get("url_filter"))
        regression_samples = _literal(values.get("url_filter_samples"), [])
        has_custom_filter = not (
            filter_node is None
            or isinstance(filter_node, ast.Constant) and filter_node.value is None
        )
        entries.append(CrawlerCatalogEntry(
            name=module_name.removeprefix("crawler_config_"),
            group=str(config_path.parent.relative_to(root)) or ".",
            config_path=config_path,
            task_path=task_path,
            discoverer=str(_literal(values.get("discoverer_name"), "")),
            extractor=str(_literal(values.get("extractor_name"), "")),
            entry_count=len(entry_points) if isinstance(entry_points, (dict, list, tuple)) else 0,
            has_declarative_filter=bool(declarative_filter),
            has_custom_filter=has_custom_filter,
            regression_sample_count=(
                len(regression_samples) if isinstance(regression_samples, list) else 0
            ),
            disabled=bool(task_path and task_path.stem in disabled),
        ))
    return entries
