---
name: iis-crawler-source
description: Add or revise IIS media crawler configurations and task wrappers using the shared IntelligenceCrawler framework. Use for integrating a news source, changing URL filters, diagnosing low or incorrect samples, or validating Playground and production Pipeline parity.
---

# IIS Crawler Source

Produce a framework-native crawler source that behaves identically in CrawlerPlayground and the
production `CrawlPipeline`.

## Start here

1. Read the repository `AGENTS.md`, then `IntelligenceCrawler/AGENTS.md`.
2. For a new or changed config, read [references/config-contract.md](references/config-contract.md).
3. For concrete RSS, list-page, custom-filter, and task-wrapper patterns, read
   [references/examples.md](references/examples.md).
4. For investigation and acceptance thresholds, read
   [references/validation.md](references/validation.md).

## Non-negotiable invariants

- Use the flat `CRAWLER_CONFIG` consumed by Playground, generated files, and Pipeline. Do not add a
  task-local alternative schema or duplicate filtering logic.
- Prefer `RSSDiscoverer`, then a static `ListPageDiscoverer`; use browser rendering only when live
  evidence shows it is necessary. Do not bypass paywalls, authentication, CAPTCHAs, or access controls.
- Bound list-page discovery to the intended site and sections. Prefer declarative `url_filter` rules;
  retain `article_filter` only for logic the declarative contract cannot express.
- Add `url_filter_samples` containing both accepted and rejected URLs. Samples are executable
  regression cases, not comments.
- Keep `content_handler` and `exception_handler` as `None`; the IIS task flow injects them. A local
  full trial must disable both to avoid submission or disk side effects.
- The task wrapper must call `FLOW.run_common_flow(CRAWLER_CONFIG.copy(), stop_event)`.
- Do not restart a running crawler, enable a source, or submit collected intelligence unless the user
  requested that operational change. New or changed task modules require a crawler restart because
  the filesystem observer is currently disabled.

## Expected files

For a news source named `<name>`:

```text
CrawlTasks/news/crawler_config_<name>.py
CrawlTasks/news/task_crawl_<name>.py
Test/test_<name>_crawler_config.py       # or extend a focused regional test file
```

Keep source-specific selection and filter facts in the config. Keep the task wrapper mechanical.

## Workflow

1. Verify the official feed or list page is current and fetchable. Sample at least two article pages;
   reject frozen, inaccessible, empty-body, or predominantly paywalled sources.
2. Select the narrowest useful categories. The objective is regional signal, not indiscriminate
   ingestion of Latest, sport, lifestyle, video, or opinion pages.
3. Implement the config and task wrapper from the examples. Use exact host boundaries by default.
4. Add accepted and rejected filter samples, including an off-site URL and each important exclusion.
5. Run the offline validator:

   ```bash
   python IntelligenceCrawler/skill/scripts/validate_source.py \
     CrawlTasks/news/crawler_config_<name>.py
   ```

6. Run focused tests and crawler preflight. If network validation is appropriate, use a bounded full
   Pipeline trial with handlers disabled; report discovered, filtered, extracted, and failed counts
   from `CrawlTrace`.
7. Commit the `IntelligenceCrawler` submodule first when it changed, then commit the parent repository
   pointer and source files. Do not include unrelated dirty submodules.

## Completion evidence

Report the source scope, accepted categories, deliberate exclusions, test results, bounded live-trial
result if performed, commits, and whether the crawler service was restarted. Never claim a source is
live merely because its files were committed.
