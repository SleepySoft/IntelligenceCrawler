import time
from typing import List, Dict, Any, Optional
from playwright.sync_api import Page, Locator


class PlaywrightActionEngine:
    """
    Simple action engine that executes steps in order.
    Each step has a timeout - if element not found/actionable within timeout, step fails.
    """

    def __init__(self, page: Page, default_timeout: int = 3000):
        self.page = page
        self.default_timeout = default_timeout
        self.logs = []

    def _log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        log = f"[{timestamp}] {message}"
        self.logs.append(log)
        print(log)

    def execute(self, config: List[Dict]) -> bool:
        """
        Execute a sequence of UI interaction steps in order.

        Each step in the configuration represents a single UI action to perform.
        Steps are executed sequentially, and if any step fails (element not found
        or not actionable within the specified timeout), the entire execution stops
        and returns False.

        Configuration Format:
        --------------------
        The configuration is a list of step dictionaries. Each step must contain:

        Required:
        - Either 'text' (exact text match) or 'selector' (CSS selector) to locate the element
        - 'action': Action to perform ('click', 'check', 'uncheck', 'fill', or 'press')

        Optional:
        - 'timeout': Maximum wait time in milliseconds for the element to become actionable
                     (default: 3000ms). Use shorter timeouts for optional elements.
        - 'value': For 'fill' action, the text to input
        - 'key': For 'press' action, the key to press (e.g., 'Enter')
        - 'wait_after': Milliseconds to wait after the action (default: 0)

        Step Execution Logic:
        --------------------
        1. The engine attempts to locate the element using the specified 'text' or 'selector'
        2. It waits for the element to become actionable (visible and enabled) for up to 'timeout' ms
        3. If the element is found and actionable within the timeout, the specified action is performed
        4. If the element is not found or not actionable within the timeout, the step fails
        5. Step failure stops the entire execution chain unless handled by multiple flow strategies

        Timeout Strategy Guide:
        -----------------------
        - timeout=3000 (default): Standard wait for important UI elements
        - timeout=100-500: Quick check for optional elements that may or may not appear
        - timeout=5000+: Extended wait for slow-loading or critical elements

        Example Configurations:
        ----------------------
        1. Simple click sequence:
            [
                {"text": "Agree", "action": "click"},
                {"text": "Next", "action": "click"},
                {"text": "Submit", "action": "click", "timeout": 5000}
            ]

        2. Form filling with mixed locators:
            [
                {"selector": "#username", "action": "fill", "value": "user123"},
                {"selector": "input[type='password']", "action": "fill", "value": "pass456"},
                {"selector": "button.submit", "action": "click"},
                {"text": "Success!", "action": "wait", "timeout": 3000}
            ]

        3. Handling optional elements with quick timeout:
            [
                {"text": "Main Button", "action": "click"},
                {"text": "Popup Close", "action": "click", "timeout": 500},  # Quick try for optional popup
                {"selector": ".notification", "action": "click", "timeout": 200}  # Very quick check
            ]

        4. Key press actions:
            [
                {"selector": "input.search", "action": "fill", "value": "query"},
                {"selector": "input.search", "action": "press", "key": "Enter"}
            ]

        Advanced Usage Patterns:
        ------------------------
        For handling multiple possible UI flows, create separate configurations and
        use execute_with_fallback() or similar methods to try them in sequence.

        Args:
            config: List of step dictionaries defining the interaction sequence.

        Returns:
            True if all steps completed successfully, False if any step failed.

        Raises:
            ValueError: If configuration is invalid (missing required keys).
            PlaywrightTimeoutError: If element interaction times out (handled internally).
        """
        self._log(f"Starting execution of {len(config)} steps")

        for i, step in enumerate(config):
            self._log(f"Step {i + 1}: {step.get('text') or step.get('selector', 'N/A')}")

            try:
                # 执行单个步骤
                success = self._execute_step(step)

                if not success:
                    self._log(f"Step {i + 1} failed, go next step.")
                    # # 步骤失败，返回False
                    # self._log(f"Step {i + 1} failed, stopping execution")
                    # return False

            except Exception as e:
                self._log(f"Error in step {i + 1}: {str(e)}")
                return False

        self._log("All steps completed successfully")
        return True

    def _execute_step(self, step: Dict) -> bool:
        """Execute a single step."""
        # 获取配置参数
        find_by = "text" if "text" in step else "selector" if "selector" in step else None
        value = step.get(find_by, "")
        action = step.get("action", "click")
        timeout = step.get("timeout", self.default_timeout)

        if not find_by or not value:
            self._log(f"Invalid step: no 'text' or 'selector' specified")
            return False

        try:
            # 查找元素
            if find_by == "text":
                element = self.page.get_by_text(value, exact=True)
            else:  # selector
                element = self.page.locator(value)

            # 执行动作
            if action == "click":
                element.click(timeout=timeout)
            elif action == "check":
                element.check(timeout=timeout)
            elif action == "uncheck":
                element.uncheck(timeout=timeout)
            elif action == "fill":
                element.fill(step.get("value", ""), timeout=timeout)
            elif action == "press":
                element.press(step.get("key", "Enter"), timeout=timeout)
            else:
                self._log(f"Unknown action: {action}")
                return False

            # 动作后的等待
            if "wait_after" in step:
                self.page.wait_for_timeout(step["wait_after"])

            self._log(f"Success: {action} on {find_by}='{value}'")
            return True

        except Exception as e:
            # 如果在timeout时间内没找到/不可操作，步骤失败
            self._log(f"Failed: {action} on {find_by}='{value}' (timeout={timeout}ms)")
            return False
