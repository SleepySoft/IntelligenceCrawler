import time
import statistics
import re
from html.parser import HTMLParser

from IntelligenceCrawler.Fetcher import PlaywrightFetcher


# 假设你的类在 fetcher.py 或类似路径
# from fetcher import PlaywrightFetcher


# ----------------------------
# 小工具：HTML -> 文本摘要（仅用于 demo 输出）
# ----------------------------
class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data and data.strip():
            self.parts.append(data.strip())

def html_to_text_snippet(html_bytes: bytes, max_len=200) -> str:
    if not html_bytes:
        return ""
    try:
        s = html_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    p = _TextExtractor()
    p.feed(s)
    text = " ".join(p.parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


# ----------------------------
# 统一的 demo runner（不使用测试框架）
# ----------------------------
def run_case(name, func):
    print(f"\n=== CASE: {name} ===")
    t0 = time.perf_counter()
    try:
        out = func()
        dt = time.perf_counter() - t0
        print(f"[OK] {name}  cost={dt:.3f}s")
        return True, dt, out
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"[FAIL] {name} cost={dt:.3f}s  err={repr(e)}")
        return False, dt, None


def summarize(latencies):
    if not latencies:
        return "n/a"
    latencies = sorted(latencies)
    avg = sum(latencies) / len(latencies)
    p50 = latencies[int(0.50 * (len(latencies) - 1))]
    p90 = latencies[int(0.90 * (len(latencies) - 1))]
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    return f"n={len(latencies)} avg={avg:.3f}s p50={p50:.3f}s p90={p90:.3f}s p95={p95:.3f}s"


def main():
    # 你可以在这里开关 stealth / proxy / render_page 默认值
    fetcher = PlaywrightFetcher(
        timeout_s=20,
        stealth=False,
        pause_browser=False,
        render_page=True
    )

    # 统一：避免 demo 卡死，主线程等待给一个上限
    # 如果你已经把 get_content 改成“默认无限等待”，这里显式加 overall_timeout_s 更安全
    COMMON_WAIT = dict(overall_timeout_s=60, poll_interval_s=0.5)

    # --- 测试 URL ---
    BOOKS_PAGE = "https://books.toscrape.com/catalogue/page-1.html"   # 有图片，适合测资源阻断 [1](http://toscrape.com/)
    QUOTES_DEFAULT = "https://quotes.toscrape.com/"                   # 基础页 [2](https://github.com/zytedata/spidyquotes)
    QUOTES_JS = "https://quotes.toscrape.com/js/"                     # JS 生成内容 [2](https://github.com/zytedata/spidyquotes)
    QUOTES_JS_DELAYED = "https://quotes.toscrape.com/js-delayed/?delay=3000"  # 延迟 JS [2](https://github.com/zytedata/spidyquotes)
    QUOTES_SCROLL = "https://quotes.toscrape.com/scroll/"             # 无限滚动，AJAX 加载 [2](https://github.com/zytedata/spidyquotes)[3](https://docs.zyte.com/web-scraping/tutorials/main/js.html)
    QUOTES_TABLEFUL = "https://quotes.toscrape.com/tableful/"         # 混乱布局 [2](https://github.com/zytedata/spidyquotes)
    QUOTES_LOGIN = "https://quotes.toscrape.com/login/"               # CSRF 登录 [2](https://github.com/zytedata/spidyquotes)[1](http://toscrape.com/)
    QUOTES_VIEWSTATE = "https://quotes.toscrape.com/viewState/"       # ViewState 模拟 [2](https://github.com/zytedata/spidyquotes)[1](http://toscrape.com/)
    QUOTES_RANDOM = "https://quotes.toscrape.com/random/"             # 随机内容 [2](https://github.com/zytedata/spidyquotes)[1](http://toscrape.com/)

    results = []

    # =========================================================
    # CASE 1: render_page=False vs True（原始 body vs 渲染 DOM）
    # =========================================================
    def case_render_raw_vs_rendered():
        raw = fetcher.get_content(
            BOOKS_PAGE,
            render_page=False,
            wait_mode="domcontentloaded",
            **COMMON_WAIT
        )
        rendered = fetcher.get_content(
            BOOKS_PAGE,
            render_page=True,
            wait_mode="domcontentloaded",
            wait_for_selector="article.product_pod",
            **COMMON_WAIT
        )
        print("raw_len=", len(raw) if raw else 0, "rendered_len=", len(rendered) if rendered else 0)
        print("rendered_text_snip:", html_to_text_snippet(rendered))
        return raw, rendered

    results.append(run_case("render_page raw vs rendered (books)", case_render_raw_vs_rendered))

    # =========================================================
    # CASE 2: 资源阻断对性能影响（禁图片/字体/媒体）
    # =========================================================
    def case_block_images_speedup():
        # 不阻断
        t = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = fetcher.get_content(
                BOOKS_PAGE,
                render_page=True,
                wait_mode="domcontentloaded",
                wait_for_selector="article.product_pod",
                block_resources={"image": False, "media": True, "font": True, "stylesheet": False, "script": False},
                **COMMON_WAIT
            )
            t.append(time.perf_counter() - t0)
        print("no-block:", summarize(t))

        # 阻断图片
        t2 = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = fetcher.get_content(
                BOOKS_PAGE,
                render_page=True,
                wait_mode="domcontentloaded",
                wait_for_selector="article.product_pod",
                block_resources={"image": True, "media": True, "font": True, "stylesheet": False, "script": False},
                **COMMON_WAIT
            )
            t2.append(time.perf_counter() - t0)
        print("block-images:", summarize(t2))
        return t, t2

    results.append(run_case("performance: block images/fonts/media (books)", case_block_images_speedup))

    # =========================================================
    # CASE 3: wait_mode 对比（domcontentloaded / load / networkidle）
    # =========================================================
    def case_wait_modes():
        modes = ["domcontentloaded", "load", "networkidle"]
        times = {}
        for m in modes:
            t = []
            for _ in range(3):
                t0 = time.perf_counter()
                _ = fetcher.get_content(
                    QUOTES_DEFAULT,
                    render_page=True,
                    wait_mode=m,
                    wait_for_selector=".quote",
                    **COMMON_WAIT
                )
                t.append(time.perf_counter() - t0)
            times[m] = t
            print(f"{m}: {summarize(t)}")
        return times

    results.append(run_case("wait_mode compare (quotes default)", case_wait_modes))

    # =========================================================
    # CASE 4: wait_for_selector（最佳努力等待）
    # =========================================================
    def case_wait_for_selector():
        content = fetcher.get_content(
            QUOTES_DEFAULT,
            render_page=True,
            wait_mode="domcontentloaded",
            wait_for_selector=".quote",
            wait_for_timeout_s=10,
            **COMMON_WAIT
        )
        print("text_snip:", html_to_text_snippet(content))
        return content

    results.append(run_case("wait_for_selector (.quote)", case_wait_for_selector))

    # =========================================================
    # CASE 5: JS 生成内容（必须 render_page=True） + wait_for_function
    # =========================================================
    def case_js_generated():
        content = fetcher.get_content(
            QUOTES_JS,
            render_page=True,
            wait_mode="domcontentloaded",
            wait_for_function="() => document.querySelectorAll('.quote').length > 0",
            **COMMON_WAIT
        )
        print("quotes_count_hint:", html_to_text_snippet(content))
        return content

    results.append(run_case("JS-generated content (/js) + wait_for_function", case_js_generated))

    # =========================================================
    # CASE 6: JS 延迟内容（/js-delayed/?delay=3000） + wait_for_text
    # =========================================================
    def case_js_delayed():
        content = fetcher.get_content(
            QUOTES_JS_DELAYED,
            render_page=True,
            wait_mode="domcontentloaded",
            wait_for_text="Quotes to Scrape",
            **COMMON_WAIT
        )
        print("text_snip:", html_to_text_snippet(content))
        return content

    results.append(run_case("JS-delayed content (/js-delayed) + wait_for_text", case_js_delayed))

    # =========================================================
    # CASE 7: 无限滚动（/scroll） + scroll_pages + wait_for_response_url
    #   /scroll 的内容通过站点自己的 API 动态加载（/api/quotes?page=...）[3](https://docs.zyte.com/web-scraping/tutorials/main/js.html)
    # =========================================================
    def case_infinite_scroll():
        content = fetcher.get_content(
            QUOTES_SCROLL,
            render_page=True,
            wait_mode="domcontentloaded",
            scroll_pages=4,  # 滚动触发更多内容加载
            wait_for_response_url="/api/quotes?page=",
            wait_for_function="() => document.querySelectorAll('.quote').length >= 30",
            **COMMON_WAIT
        )
        print("text_snip:", html_to_text_snippet(content))
        return content

    results.append(run_case("infinite scroll (/scroll) + scroll_pages + wait_for_response_url", case_infinite_scroll))

    # =========================================================
    # CASE 8: post_extra_action callable —— 登录（/login, CSRF, any user/pass works）[2](https://github.com/zytedata/spidyquotes)[1](http://toscrape.com/)
    # =========================================================
    def case_login_callable():
        def do_login(page):
            # 填写用户名密码（站点说明任意可用）
            page.fill("input[name='username']", "demo_user")
            page.fill("input[name='password']", "demo_pass")
            page.click("input[type='submit']")
            # 登录后通常会跳转到首页，等待出现 Logout
            page.wait_for_selector("a[href='/logout']", timeout=10000)
            # 再访问首页，确保已登录状态（示例：登录后页面会显示额外链接/状态）
            page.goto("https://quotes.toscrape.com/", wait_until="domcontentloaded")
            page.wait_for_selector("a[href='/logout']", timeout=10000)

        content = fetcher.get_content(
            QUOTES_LOGIN,
            render_page=True,
            wait_mode="domcontentloaded",
            post_extra_action=do_login,
            **COMMON_WAIT
        )
        print("after-login snippet:", html_to_text_snippet(content))
        return content

    results.append(run_case("post_extra_action callable: login (/login)", case_login_callable))

    # =========================================================
    # CASE 9: post_extra_action list —— 触发 ActionEngine 路径（需要你项目的 action schema）
    # =========================================================

    def case_action_engine_basic_nav():
        actions = [
            # 1) 故意失败：页面上一般不存在这个文本，用很短 timeout 模拟“可选弹窗关闭”
            {"text": "I do not exist", "action": "click", "timeout": 200},

            # 2) 点击一个 Top Ten Tag（主页右侧通常有 tag，如 inspirational/life 等）
            {"text": "inspirational", "action": "click", "timeout": 3000, "wait_after": 300},

            # 3) 点击分页 Next →（主页模板里就是 “Next →” 文本）
            {"text": "Next →", "action": "click", "timeout": 3000, "wait_after": 300},

            # 4) 再点 ←Previous（可能存在也可能不存在；用短 timeout 测“可选元素”）
            {"text": "←Previous", "action": "click", "timeout": 500},
        ]
        content = fetcher.get_content(
            "https://quotes.toscrape.com/",
            render_page=True,
            wait_mode="domcontentloaded",
            post_extra_action=actions,
            **COMMON_WAIT
        )
        print("snippet:", html_to_text_snippet(content))
        return content

    def case_action_engine_login_list():
        actions = [
            {"selector": "input[name='username']", "action": "fill", "value": "demo_user", "timeout": 3000},
            # password 在 hosted 版可能存在（即使不校验也无妨）；不存在则会失败但继续
            {"selector": "input[name='password']", "action": "fill", "value": "demo_pass", "timeout": 500},
            # 在密码框按 Enter（如果 password 不存在，这步会失败但继续）
            {"selector": "input[name='password']", "action": "press", "key": "Enter", "timeout": 500},
            # 兜底：点击提交按钮（更稳）
            {"selector": "input[type='submit'], button[type='submit']", "action": "click", "timeout": 5000,
             "wait_after": 500},
            # 登录后一般会有 /logout 链接，点一下验证会话保持（不存在也不会中止）
            {"selector": "a[href='/logout']", "action": "click", "timeout": 1000},
        ]

        content = fetcher.get_content(
            "https://quotes.toscrape.com/login",
            render_page=True,
            wait_mode="domcontentloaded",
            post_extra_action=actions,
            **COMMON_WAIT
        )
        print("after-login snippet:", html_to_text_snippet(content))
        return content

    def case_action_engine_viewstate_flow():
        actions = [
            # 1) 选作者：点击 author 下拉框，再点某个作者 option（用 text 定位）
            {"selector": "select[name='author']", "action": "click", "timeout": 3000},
            {"text": "Albert Einstein", "action": "click", "timeout": 3000, "wait_after": 300},

            # 2) 第一次提交（让服务端返回该作者对应的 tag 列表）
            {"selector": "input[type='submit'], button[type='submit']", "action": "click", "timeout": 5000,
             "wait_after": 800},

            # 3) 选 tag（此时页面通常出现 tag 下拉）
            {"selector": "select[name='tag']", "action": "click", "timeout": 3000},
            {"text": "life", "action": "click", "timeout": 1000, "wait_after": 300},

            # 4) 第二次提交（返回过滤后的 quotes）
            {"selector": "input[type='submit'], button[type='submit']", "action": "click", "timeout": 5000,
             "wait_after": 800},

            # 5) 故意做个“快速失败”的可选点击：验证失败不影响后续（可删）
            {"selector": ".non-existent", "action": "click", "timeout": 200},
        ]

        content = fetcher.get_content(
            "https://quotes.toscrape.com/search.aspx",
            render_page=True,
            wait_mode="domcontentloaded",
            post_extra_action=actions,
            **COMMON_WAIT
        )
        print("viewstate snippet:", html_to_text_snippet(content))
        return content

    def case_action_engine_all_actions_data_url():
        # 一个完全可控的小页面：checkbox + input + button + status text
        html = """
        <html><body>
          <label><input id="agree" type="checkbox"> Agree</label>
          <input id="q" type="text" value="">
          <button id="btn" onclick="document.getElementById('out').innerText='clicked:'+document.getElementById('q').value;">Go</button>
          <div id="out">empty</div>
          <script>
            // 支持 Enter：按下 Enter 就触发按钮点击
            document.getElementById('q').addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ document.getElementById('btn').click(); }});
          </script>
        </body></html>
        """
        # 注意 data URL 需要简单 url-encode（这里只处理空格/换行够用；更严谨可用 urllib.parse.quote）
        data_url = "data:text/html," + (
            html.replace("\n", "").replace("  ", " ").replace("#", "%23").replace("%", "%25").replace(" ", "%20")
        )

        actions = [
            {"selector": "#agree", "action": "check", "timeout": 3000},
            {"selector": "#agree", "action": "uncheck", "timeout": 3000},

            {"selector": "#q", "action": "fill", "value": "hello", "timeout": 3000},

            # press Enter 触发按钮 click（测试 press）
            {"selector": "#q", "action": "press", "key": "Enter", "timeout": 3000, "wait_after": 200},

            # 再手动 click 一次按钮（测试 click）
            {"selector": "#btn", "action": "click", "timeout": 3000, "wait_after": 200},

            # 故意失败一步：验证失败继续
            {"selector": "#not-exist", "action": "click", "timeout": 200},
        ]

        content = fetcher.get_content(
            data_url,
            render_page=True,
            wait_mode="domcontentloaded",
            post_extra_action=actions,
            # data URL 不需要等太久
            overall_timeout_s=20,
            poll_interval_s=0.2,
        )
        print("data-url snippet:", html_to_text_snippet(content))
        return content

    results.append(run_case("AE basic nav + optional fail", case_action_engine_basic_nav))
    results.append(run_case("AE login flow (list)", case_action_engine_login_list))
    results.append(run_case("AE viewstate flow (2-step)", case_action_engine_viewstate_flow))
    results.append(run_case("AE all actions on data URL", case_action_engine_all_actions_data_url))

    # =========================================================
    # CASE 10: allowed_domains / block_third_party
    #   - allowlist 错误配置应导致失败（可靠性：能否明确失败而不是假成功）
    #   - 正确 allowlist 应正常返回
    # =========================================================
    def case_domain_policies():
        # 正确 allowlist（允许 quotes.toscrape.com）
        ok = fetcher.get_content(
            QUOTES_DEFAULT,
            render_page=True,
            wait_mode="domcontentloaded",
            allowed_domains=["quotes.toscrape.com"],
            **COMMON_WAIT
        )
        print("allowlist ok snippet:", html_to_text_snippet(ok))

        # 错误 allowlist（不允许目标域），应触发 page.goto 失败
        try:
            _ = fetcher.get_content(
                QUOTES_DEFAULT,
                render_page=True,
                wait_mode="domcontentloaded",
                allowed_domains=["example.com"],
                **COMMON_WAIT
            )
            print("unexpected: allowlist wrong still succeeded")
        except Exception as e:
            print("expected fail with wrong allowlist:", repr(e))

        # block_third_party：此站大多资源同域，功能开关应不影响主文档
        ok2 = fetcher.get_content(
            QUOTES_DEFAULT,
            render_page=True,
            wait_mode="domcontentloaded",
            block_third_party=True,
            **COMMON_WAIT
        )
        print("block_third_party snippet:", html_to_text_snippet(ok2))
        return ok, ok2

    results.append(run_case("domain policies: allowed_domains + block_third_party", case_domain_policies))

    # =========================================================
    # CASE 11: tableful / viewState / random —— 不同页面结构与随机性
    # =========================================================
    def case_misc_endpoints():
        a = fetcher.get_content(QUOTES_TABLEFUL, render_page=True, wait_mode="domcontentloaded", **COMMON_WAIT)
        b = fetcher.get_content(QUOTES_VIEWSTATE, render_page=True, wait_mode="domcontentloaded", **COMMON_WAIT)
        c = fetcher.get_content(QUOTES_RANDOM, render_page=True, wait_mode="domcontentloaded", **COMMON_WAIT)

        print("tableful:", html_to_text_snippet(a))
        print("viewState:", html_to_text_snippet(b))
        print("random:", html_to_text_snippet(c))
        return a, b, c

    results.append(run_case("misc endpoints: /tableful /viewState /random", case_misc_endpoints))

    # =========================================================
    # CASE 12: Context 复用的性能收益（同站重复请求耗时下降）
    # =========================================================
    def case_context_reuse_benchmark():
        lat = []
        for i in range(10):
            t0 = time.perf_counter()
            _ = fetcher.get_content(
                QUOTES_DEFAULT,
                render_page=True,
                wait_mode="domcontentloaded",
                wait_for_selector=".quote",
                **COMMON_WAIT
            )
            lat.append(time.perf_counter() - t0)
        print("reuse benchmark:", summarize(lat))
        print("first vs later:", lat[0], "->", statistics.mean(lat[1:]))
        return lat

    results.append(run_case("context reuse benchmark (10x same site)", case_context_reuse_benchmark))

    # --- 汇总 ---
    ok = [dt for success, dt, _ in results if success]
    fail = [dt for success, dt, _ in results if not success]
    print("\n==================== SUMMARY ====================")
    print(f"success={len(ok)} fail={len(fail)}")
    print("success latency:", summarize(ok))
    if fail:
        print("fail latency:", summarize(fail))

    fetcher.close()


if __name__ == "__main__":
    main()
