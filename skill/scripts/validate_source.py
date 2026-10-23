#!/usr/bin/env python3
"""Offline validation for one IIS crawler configuration."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_KEYS = {
    "d_fetcher_name", "d_fetcher_init_param", "e_fetcher_name",
    "e_fetcher_init_param", "discoverer_name", "discoverer_init_param",
    "extractor_name", "extractor_init_param", "entry_points", "period_filter",
    "channel_filter", "d_fetcher_kwargs", "e_fetcher_kwargs", "extractor_kwargs",
    "article_filter", "content_handler", "exception_handler",
}


def _find_project_root(config_path: Path) -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents, *config_path.parents]
    for candidate in candidates:
        if (candidate / "IntelligenceCrawler").is_dir() and (candidate / "CrawlTasks").is_dir():
            return candidate
    raise RuntimeError("Could not locate IIS project root containing IntelligenceCrawler and CrawlTasks")


def _load_config(path: Path) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location(f"_validated_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "CRAWLER_CONFIG", None)
    if not isinstance(config, dict):
        raise ValueError("CRAWLER_CONFIG must be a dictionary")
    return config


def _validate_task_wrapper(path: Path) -> List[str]:
    """静态验证包装器契约，不导入会启动基础设施的任务模块。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        return [f"task wrapper is not parseable: {exc}"]

    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    errors = []
    if "module_init" not in functions:
        errors.append("task wrapper must define module_init(service_context)")
    start_task = functions.get("start_task")
    if start_task is None:
        errors.append("task wrapper must define start_task(stop_event)")
        return errors
    parameter_names = {argument.arg for argument in start_task.args.args}
    if "stop_event" not in parameter_names:
        errors.append("start_task must accept stop_event")

    run_calls = [
        node for node in ast.walk(start_task)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_common_flow"
    ]
    if not run_calls:
        errors.append("start_task must call FLOW.run_common_flow")
        return errors
    call = run_calls[0]
    has_config_copy = any(
        isinstance(argument, ast.Call)
        and isinstance(argument.func, ast.Attribute)
        and isinstance(argument.func.value, ast.Name)
        and argument.func.value.id == "CRAWLER_CONFIG"
        and argument.func.attr == "copy"
        for argument in call.args
    )
    if not has_config_copy:
        errors.append("run_common_flow must receive CRAWLER_CONFIG.copy()")
    if not any(isinstance(argument, ast.Name) and argument.id == "stop_event" for argument in call.args):
        errors.append("run_common_flow must receive stop_event")
    return errors


def validate(path: Path, allow_missing_samples: bool = False) -> Dict[str, Any]:
    root = _find_project_root(path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from IntelligenceCrawler.CrawlerCodeGenerator import CrawlerCodeGenerator
    from IntelligenceCrawler.ProjectCatalog import scan_crawler_catalog
    from IntelligenceCrawler.URLFilter import normalize_url_filter, validate_url_filter_samples

    config = _load_config(path)
    errors: List[str] = []
    warnings: List[str] = []

    missing = sorted(REQUIRED_KEYS - config.keys())
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")
    if not isinstance(config.get("entry_points"), dict) or not config.get("entry_points"):
        errors.append("entry_points must be a non-empty dictionary")
    for key in ("d_fetcher_name", "e_fetcher_name", "discoverer_name", "extractor_name"):
        if not isinstance(config.get(key), str) or not config.get(key):
            errors.append(f"{key} must be a non-empty string")
    if config.get("content_handler") is not None:
        errors.append("content_handler must be None in a committed source config")
    if config.get("exception_handler") is not None:
        errors.append("exception_handler must be None in a committed source config")

    try:
        normalize_url_filter(config.get("url_filter"))
    except ValueError as exc:
        errors.append(str(exc))

    samples = config.get("url_filter_samples") or []
    if not samples:
        message = "url_filter_samples is empty; add accepted and rejected regression URLs"
        (warnings if allow_missing_samples else errors).append(message)
    else:
        if not any(bool(item.get("allowed", True)) for item in samples if isinstance(item, dict)):
            errors.append("url_filter_samples must contain at least one accepted URL")
        if not any(not bool(item.get("allowed", True)) for item in samples if isinstance(item, dict)):
            errors.append("url_filter_samples must contain at least one rejected URL")
        try:
            failures = validate_url_filter_samples(config)
            if failures:
                errors.append(f"filter regression failures: {failures}")
        except ValueError as exc:
            errors.append(str(exc))

    try:
        generated = CrawlerCodeGenerator().generate_code_from_config(config)
        namespace: Dict[str, Any] = {}
        exec(compile(generated, f"<generated:{path.name}>", "exec"), namespace)
        if set(namespace["CRAWLER_CONFIG"]) != set(config):
            errors.append("generated config keys differ from source config")
    except Exception as exc:
        errors.append(f"generated config is not executable: {type(exc).__name__}: {exc}")

    catalog = scan_crawler_catalog(root / "CrawlTasks", root / "_config" / "collector.json")
    entry = next((item for item in catalog if item.config_path == path.resolve()), None)
    if entry is None:
        errors.append("config is not discoverable under CrawlTasks")
    elif entry.task_path is None:
        errors.append("no task_crawl_*.py wrapper imports this config")
    else:
        errors.extend(_validate_task_wrapper(entry.task_path))

    return {
        "config": str(path),
        "entry_points": len(config.get("entry_points") or {}),
        "filter_samples": len(samples),
        "task": str(entry.task_path) if entry and entry.task_path else None,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--allow-missing-samples", action="store_true",
        help="Warn instead of failing for a legacy config without regression samples.",
    )
    args = parser.parse_args()
    path = args.config.resolve()
    if not path.is_file():
        parser.error(f"config does not exist: {path}")
    try:
        result = validate(path, allow_missing_samples=args.allow_missing_samples)
    except Exception as exc:
        result = {"config": str(path), "ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
