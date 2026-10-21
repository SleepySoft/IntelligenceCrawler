from IntelligenceCrawler.Discoverer import ListPageDiscoverer, discoverer_factory


def test_list_page_defaults_to_exact_channel_hostname():
    discoverer = ListPageDiscoverer(fetcher=None, verbose=False)

    result = discoverer._filter_article_urls(
        "https://news.example.com/world",
        [
            "https://news.example.com/article/1",
            "https://special.example.com/article/2",
            "https://example.com/article/3",
            "https://outside.test/article/4",
        ],
    )

    assert result == ["https://news.example.com/article/1"]
    assert "special.example.com=1" in "\n".join(discoverer.log_messages)


def test_list_page_accepts_explicit_additional_hosts():
    discoverer = ListPageDiscoverer(
        fetcher=None,
        verbose=False,
        allowed_article_hosts=["special.example.com", "https://partner.test/path"],
    )

    result = discoverer._filter_article_urls(
        "https://news.example.com/world",
        [
            "https://news.example.com/article/1",
            "https://special.example.com/article/2",
            "https://partner.test/article/3",
            "https://outside.test/article/4",
        ],
    )

    assert result == [
        "https://news.example.com/article/1",
        "https://special.example.com/article/2",
        "https://partner.test/article/3",
    ]


def test_list_page_can_explicitly_disable_host_boundary():
    discoverer = discoverer_factory("ListPageDiscoverer", {
        "fetcher": None,
        "same_host_only": False,
    })

    links = ["https://outside.test/article/1"]
    assert discoverer._filter_article_urls(
        "https://news.example.com/world", links) == links
