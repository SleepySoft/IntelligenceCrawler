"""声明式文章 URL 过滤。

该模块是 Playground 与 CrawlPipeline 的唯一过滤实现。自定义 ``article_filter``
仍作为复杂站点的逃生口，并与声明式规则按 AND 组合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class URLFilterDecision:
    allowed: bool
    reason: str = "accepted"


def _string_list(config: Dict[str, Any], key: str) -> List[str]:
    value = config.get(key) or []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"url_filter.{key} must be a string list")
    return [str(item).strip().lower() for item in value if str(item).strip()]


def normalize_url_filter(config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """校验并规范化规则；空规则返回 ``None``。"""
    if not config:
        return None
    if not isinstance(config, dict):
        raise ValueError("url_filter must be a dictionary")

    query_policy = str(config.get("query_policy") or "allow").strip().lower()
    if query_policy not in {"allow", "reject"}:
        raise ValueError("url_filter.query_policy must be 'allow' or 'reject'")

    normalized = {
        "allowed_hosts": _string_list(config, "allowed_hosts"),
        "path_includes": _string_list(config, "path_includes"),
        "path_excludes": _string_list(config, "path_excludes"),
        "query_policy": query_policy,
        "include_subdomains": bool(config.get("include_subdomains", False)),
    }
    if not any((
        normalized["allowed_hosts"],
        normalized["path_includes"],
        normalized["path_excludes"],
        normalized["query_policy"] == "reject",
    )):
        return None
    return normalized


def _normalize_host(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"//{value}")
    return (parsed.hostname or value).strip().lower().rstrip(".")


def evaluate_url_filter(
    url: str,
    config: Optional[Dict[str, Any]],
) -> URLFilterDecision:
    rules = normalize_url_filter(config)
    if rules is None:
        return URLFilterDecision(True)

    parsed = urlparse(str(url or ""))
    host = (parsed.hostname or "").lower().rstrip(".")
    path = (parsed.path or "/").lower()
    if not host:
        return URLFilterDecision(False, "invalid_url")

    allowed_hosts = [_normalize_host(item) for item in rules["allowed_hosts"]]
    if allowed_hosts:
        if rules["include_subdomains"]:
            host_allowed = any(host == item or host.endswith(f".{item}") for item in allowed_hosts)
        else:
            host_allowed = host in allowed_hosts
        if not host_allowed:
            return URLFilterDecision(False, "host_not_allowed")

    if rules["path_includes"] and not any(item in path for item in rules["path_includes"]):
        return URLFilterDecision(False, "path_not_included")
    if any(item in path for item in rules["path_excludes"]):
        return URLFilterDecision(False, "path_excluded")
    if rules["query_policy"] == "reject" and parsed.query:
        return URLFilterDecision(False, "query_rejected")
    return URLFilterDecision(True)


def evaluate_article_url(
    url: str,
    declarative_filter: Optional[Dict[str, Any]] = None,
    article_filter: Optional[Callable[[str], bool]] = None,
) -> URLFilterDecision:
    """按声明式规则、再按旧 callable 过滤；两者必须同时通过。"""
    decision = evaluate_url_filter(url, declarative_filter)
    if not decision.allowed:
        return decision
    if article_filter and not article_filter(url):
        return URLFilterDecision(False, "custom_filter_rejected")
    return URLFilterDecision(True)


def filter_article_urls(
    urls: Iterable[str],
    declarative_filter: Optional[Dict[str, Any]] = None,
    article_filter: Optional[Callable[[str], bool]] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """返回接受 URL，以及 ``(URL, reason)`` 拒绝记录。"""
    accepted: List[str] = []
    rejected: List[Tuple[str, str]] = []
    for url in urls:
        decision = evaluate_article_url(url, declarative_filter, article_filter)
        if decision.allowed:
            accepted.append(url)
        else:
            rejected.append((url, decision.reason))
    return accepted, rejected


def validate_url_filter_samples(crawler_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """运行配置内的 ``url_filter_samples``，仅返回失败样本。"""
    samples = crawler_config.get("url_filter_samples") or []
    if not isinstance(samples, list):
        raise ValueError("url_filter_samples must be a list")
    failures = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not str(sample.get("url") or "").strip():
            raise ValueError(f"url_filter_samples[{index}] must contain a URL")
        url = str(sample["url"])
        expected = bool(sample.get("allowed", True))
        decision = evaluate_article_url(
            url,
            crawler_config.get("url_filter"),
            crawler_config.get("article_filter"),
        )
        if decision.allowed != expected:
            failures.append({
                "index": index,
                "url": url,
                "expected": expected,
                "actual": decision.allowed,
                "reason": decision.reason,
            })
    return failures
