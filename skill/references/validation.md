# Source investigation and validation

Read this reference for new sources, sample shortages, or filter changes.

## Source acceptance

Verify all of the following before implementation:

- The feed or list page is current, not merely HTTP 200.
- At least two discovered article pages contain meaningful article bodies and metadata.
- The source offers a useful regional or editorial perspective not already overrepresented.
- A narrow category or selector can exclude sport, lifestyle, video, navigation, and unrelated links.
- Access does not require bypassing a paywall, login, CAPTCHA, robots restriction, or deliberate block.

Prefer authoritative publisher pages and feeds. Search-engine caches and unofficial mirrors are not
source contracts.

## Offline checks

```bash
python IntelligenceCrawler/skill/scripts/validate_source.py \
  CrawlTasks/news/crawler_config_<name>.py

python -m py_compile \
  CrawlTasks/news/crawler_config_<name>.py \
  CrawlTasks/news/task_crawl_<name>.py

python -m pytest -q \
  IntelligenceCrawler/tests \
  Test/test_<name>_crawler_config.py

python Scripts/deployment_preflight.py --component crawler
```

The validator checks the flat contract, handler safety, declarative rules, regression samples,
generated-code execution, and task-wrapper discovery.

## Bounded live trial

Use CrawlerPlayground “Full Trial Run” or the formal Pipeline with a small maximum. Keep handlers
disabled. A reasonable initial trial is 1–5 articles per source.

Report the funnel rather than pasting raw pages or tracebacks:

```text
entry_points: 1
channels_unique: 1
articles_discovered: 20
articles_duplicate: 2
articles_accepted: 4
article_filter.path_excluded: 8
article_filter.host_not_allowed: 6
extraction_attempted: 4
extraction_succeeded: 3
extraction.no_content: 1
```

Inspect at most a few rejected samples per reason. If a page fails, summarize its stage, category,
URL, and actionable reason; do not dump full HTML or a long backtrace into the working context.

## Tests expected for a new source

- Entry-point set is intentionally narrow.
- At least one canonical accepted URL.
- Off-site URL rejected.
- Each excluded section rejected.
- Query URL rejected when `query_policy=reject`.
- `validate_url_filter_samples(config) == []`.
- Config generates and executes through `CrawlerCodeGenerator`.
- Task wrapper is found by `ProjectCatalog` and service discovery.

Do not make automated tests depend on a changing live website. Keep live smoke checks separate from
deterministic regression tests.
