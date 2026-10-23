# Crawler source examples

Use the closest pattern, then narrow it using live evidence. These examples use reserved domains and
are templates, not runnable sources.

## Example A: narrow RSS source

```python
"""Example News: only the regional politics feed."""


CRAWLER_CONFIG = {
    "d_fetcher_name": "RequestsFetcher",
    "d_fetcher_init_param": {"log_callback": print, "proxy": None, "timeout_s": 20},
    "e_fetcher_name": "RequestsFetcher",
    "e_fetcher_init_param": {"log_callback": print, "proxy": None, "timeout_s": 25},
    "discoverer_name": "RSSDiscoverer",
    "discoverer_init_param": {"verbose": False},
    "extractor_name": "Trafilatura",
    "extractor_init_param": {"verbose": False},
    "entry_points": {
        "regional_politics": "https://news.example.com/rss/regional-politics.xml",
    },
    "period_filter": (None, None),
    "channel_filter": None,
    "d_fetcher_kwargs": {
        "wait_until": "networkidle", "wait_for_selector": None,
        "wait_for_timeout_s": 20, "scroll_pages": 0,
    },
    "e_fetcher_kwargs": {
        "wait_until": "networkidle", "wait_for_selector": None,
        "wait_for_timeout_s": 25, "scroll_pages": 0,
    },
    "extractor_kwargs": {},
    "url_filter": {
        "allowed_hosts": ["news.example.com"],
        "path_includes": ["/region/"],
        "path_excludes": ["/opinion/", "/video/"],
        "query_policy": "allow",
    },
    "url_filter_samples": [
        {"url": "https://news.example.com/region/article-1", "allowed": True},
        {"url": "https://news.example.com/opinion/article-2", "allowed": False},
        {"url": "https://outside.test/region/article-3", "allowed": False},
    ],
    "article_filter": None,
    "content_handler": None,
    "exception_handler": None,
}
```

## Example B: static list page

```python
ARTICLE_SELECTOR = (
    'a[href^="/country/"]:not([href*="?"]), '
    'a[href^="/business/"]:not([href*="?"])'
)

CRAWLER_CONFIG = {
    # Fetcher/extractor/common fields are identical to Example A.
    "discoverer_name": "ListPageDiscoverer",
    "discoverer_init_param": {
        "verbose": False,
        "scope_selector": ARTICLE_SELECTOR,
        "manual_specified_signature": None,
        "extraction_mode": "card",
        "same_host_only": True,
    },
    "entry_points": {"country": "https://www.example.com/country"},
    "url_filter": {
        "allowed_hosts": ["www.example.com"],
        "path_includes": ["/country/", "/business/"],
        "path_excludes": ["/sport/", "/culture/"],
        "query_policy": "reject",
    },
    "url_filter_samples": [
        {"url": "https://www.example.com/country/2026/article", "allowed": True},
        {"url": "https://www.example.com/sport/2026/article", "allowed": False},
        {"url": "https://www.example.com/country/2026/article?ref=home", "allowed": False},
    ],
    # Include every common key from the contract; omitted here only for readability.
}
```

Do not omit the common keys in a real config.

## Example C: exceptional custom condition

Use a callable only when declarative OR and deny rules cannot express the condition:

```python
def _is_relevant_article(url: str) -> bool:
    value = str(url or "").lower()
    return (
        "/en/" in value
        and "/article/" in value
        and any(section in value for section in ("/politics/", "/economy/"))
    )


CRAWLER_CONFIG = {
    # ...common fields...
    "url_filter": {"allowed_hosts": ["agency.example"], "query_policy": "allow"},
    "article_filter": _is_relevant_article,
    "url_filter_samples": [
        {"url": "https://agency.example/en/politics/article/a", "allowed": True},
        {"url": "https://agency.example/en/sport/article/b", "allowed": False},
        {"url": "https://outside.test/en/politics/article/c", "allowed": False},
    ],
}
```

## Example D: task wrapper

```python
from typing import Optional

from CrawlerServiceEngine import ServiceContext
from crawler_config_example import CRAWLER_CONFIG
from Workflow.IntelligenceCrawlFlow import CommonIntelligenceCrawlFlow

NAME = "example"
FLOW: Optional[CommonIntelligenceCrawlFlow] = None


def module_init(service_context: ServiceContext):
    global FLOW
    FLOW = CommonIntelligenceCrawlFlow(NAME, service_context)


def start_task(stop_event):
    if FLOW:
        FLOW.run_common_flow(CRAWLER_CONFIG.copy(), stop_event)
```

For a domestic site requiring the domestic proxy policy, pass `global_site=False` to
`run_common_flow`. Do not hard-code deployment proxy addresses in committed configs.
