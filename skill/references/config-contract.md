# CRAWLER_CONFIG contract

Read this reference whenever adding or changing a crawler config.

## Required shape

```python
CRAWLER_CONFIG = {
    "d_fetcher_name": "RequestsFetcher",
    "d_fetcher_init_param": {"log_callback": print, "proxy": None, "timeout_s": 20},
    "e_fetcher_name": "RequestsFetcher",
    "e_fetcher_init_param": {"log_callback": print, "proxy": None, "timeout_s": 25},
    "discoverer_name": "RSSDiscoverer",
    "discoverer_init_param": {"verbose": False},
    "extractor_name": "Trafilatura",
    "extractor_init_param": {"verbose": False},
    "entry_points": {"world": "https://example.com/world.xml"},
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
    "url_filter": None,
    "url_filter_samples": [],
    "article_filter": None,
    "content_handler": None,
    "exception_handler": None,
}
```

Do not remove unknown fields while editing an existing config. Playground deliberately preserves
fields it cannot display.

## Component choice

| Evidence | Discoverer/fetcher |
|---|---|
| Current official RSS/Atom | `RSSDiscoverer` + `RequestsFetcher` |
| Static list contains canonical article links | `ListPageDiscoverer` + `RequestsFetcher` |
| Links only appear after JavaScript rendering | `ListPageDiscoverer` + `PlaywrightFetcher` |
| Stable sitemap with useful dates | `SitemapDiscoverer` |

Default extraction is `Trafilatura`. Use `Generic CSS` only when a stable site-specific selector is
demonstrably better. Do not select a heavier component speculatively.

For `ListPageDiscoverer`, use:

```python
"discoverer_init_param": {
    "verbose": False,
    "scope_selector": 'a[href^="/world/"]',
    "manual_specified_signature": None,
    "extraction_mode": "card",
    "same_host_only": True,
}
```

`same_host_only=True` is the default security and quality boundary. Add
`allowed_article_hosts` only for a verified first-party or syndication host.

## Declarative URL filter

```python
"url_filter": {
    "allowed_hosts": ["www.example.com"],
    "include_subdomains": False,
    "path_includes": ["/world/", "/politics/"],
    "path_excludes": ["/sports/", "/opinion/", "/video/"],
    "query_policy": "reject",  # allow | reject
}
```

Semantics:

- `allowed_hosts` is exact by default; `notexample.com` never matches `example.com`.
- `include_subdomains=True` permits `news.example.com` for `example.com`.
- `path_includes` is OR: at least one fragment must occur.
- `path_excludes` is OR: any matching fragment rejects the URL.
- `query_policy=reject` rejects every URL with a query string.
- The declarative filter runs first. A non-`None` `article_filter` then runs as an additional AND
  condition.

Stable rejection reasons are `invalid_url`, `host_not_allowed`, `path_not_included`,
`path_excluded`, `query_rejected`, and `custom_filter_rejected`.

## Executable filter samples

Every new source should include at least:

```python
"url_filter_samples": [
    {"url": "https://www.example.com/world/article-1", "allowed": True},
    {"url": "https://www.example.com/sports/article-2", "allowed": False},
    {"url": "https://outside.test/world/article-3", "allowed": False},
]
```

Add one rejected case for every important exclusion and query rule. Playground validates these on
load; `validate_source.py` validates them without opening the UI.

## Runtime hooks

- `article_filter`: module-level `def` for exceptional URL logic. Prefer a self-contained function.
- `content_handler`: always `None` in committed source configs.
- `exception_handler`: always `None` in committed source configs.

Avoid lambdas for new configs even though the generator supports simple legacy lambdas: named
functions provide better tracebacks and tests.
