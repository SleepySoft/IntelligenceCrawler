import time
from typing import Any, Dict, List, Optional, Tuple, Union

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError


JumpDirective = Union[int, str]  # relative jump offset or STOP_OK/STOP_FAIL


class PlaywrightActionEngine:
    # Supported actions (v1)
    _ELEMENT_ACTIONS = {"wait", "click", "fill", "press", "check", "uncheck"}
    _PAGE_ACTIONS = {"sleep", "scroll"}  # may be decisive if scroll has locator

    _STOP_OK = "STOP_OK"
    _STOP_FAIL = "STOP_FAIL"

    def __init__(
        self,
        page: Page,
        default_timeout: int = 3000,
        validate_config: bool = True,
        # Safety guards (help avoid infinite loops with jumps)
        max_total_step_executions: int = 2000,
        max_visits_per_step: int = 50,
        # Whether to ignore success/fail on non-decisive steps (recommended)
        ignore_control_flow_on_non_decisive: bool = True,
    ):
        self.page = page
        self.default_timeout = default_timeout
        self.validate_config = validate_config

        self.max_total_step_executions = max_total_step_executions
        self.max_visits_per_step = max_visits_per_step
        self.ignore_control_flow_on_non_decisive = ignore_control_flow_on_non_decisive

        self.logs: List[str] = []

    # -------------------------
    # Logging helpers
    # -------------------------
    def _log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        self.logs.append(line)
        print(line)

    # -------------------------
    # Public API
    # -------------------------
    def execute(self, config: List[Dict[str, Any]]) -> bool:
        """
        Execute steps sequentially with optional relative jumps and stop directives.
        Returns True on STOP_OK or when reaching the end without STOP_FAIL.
        Returns False on STOP_FAIL or fatal guard violation.
        """
        if not isinstance(config, list):
            raise ValueError("Config must be a list of step dictionaries.")

        if self.validate_config:
            errors = self.validate(config)
            if errors:
                # Validation is optional but when enabled, fail fast.
                msg = "\n".join(f"- {e}" for e in errors)
                raise ValueError(f"Invalid action config:\n{msg}")

        self._log(f"Starting execution: {len(config)} steps")

        step_count = len(config)
        idx = 0

        total_exec = 0
        visits = [0] * step_count

        while 0 <= idx < step_count:
            total_exec += 1
            if total_exec > self.max_total_step_executions:
                self._log(f"[Guard] Exceeded max_total_step_executions={self.max_total_step_executions}. STOP_FAIL.")
                return False

            visits[idx] += 1
            if visits[idx] > self.max_visits_per_step:
                self._log(f"[Guard] Step {idx + 1} visited too many times ({visits[idx]}). STOP_FAIL.")
                return False

            step = config[idx]
            step_desc = self._step_desc(step)
            self._log(f"Step {idx + 1}/{step_count}: {step_desc}")

            outcome, decisive = self._run_step(step)

            # Default "continue"
            next_idx = idx + 1

            # Control flow only applies meaningfully to decisive steps.
            if not decisive and self.ignore_control_flow_on_non_decisive:
                if "success" in step or "fail" in step:
                    self._log("[Warning] success/fail specified on non-decisive step; ignored.")
                idx = next_idx
                continue

            # Apply control flow if directives exist; otherwise ignore and continue.
            if outcome is True:
                if "success" in step:
                    directive = step.get("success")
                    jump = self._apply_directive(directive, on="success")
                    if jump is None:
                        return True  # STOP_OK
                    next_idx = idx + jump
            else:
                if "fail" in step:
                    directive = step.get("fail")
                    jump = self._apply_directive(directive, on="fail")
                    if jump is None:
                        return False  # STOP_FAIL
                    next_idx = idx + jump

            idx = next_idx

        # Natural completion (no STOP_FAIL triggered)
        self._log("Reached end of steps.")
        return True

    # -------------------------
    # Validation (optional)
    # -------------------------
    def validate(self, config: List[Dict[str, Any]]) -> List[str]:
        errors: List[str] = []
        for i, step in enumerate(config):
            prefix = f"Step {i + 1}:"

            if not isinstance(step, dict):
                errors.append(f"{prefix} step must be a dict.")
                continue

            action = step.get("action")
            if not isinstance(action, str) or not action.strip():
                errors.append(f"{prefix} missing or invalid 'action'.")
                continue

            action = action.strip()

            # Validate locator exclusivity
            has_selector = "selector" in step
            has_text = "text" in step
            if has_selector and has_text:
                errors.append(f"{prefix} 'selector' and 'text' are mutually exclusive.")

            # Validate timeout
            if "timeout" in step and not self._is_nonneg_int(step.get("timeout")):
                errors.append(f"{prefix} 'timeout' must be a non-negative integer (ms).")

            # Validate directives
            for key, allowed_stop in (("success", self._STOP_OK), ("fail", self._STOP_FAIL)):
                if key in step:
                    d = step.get(key)
                    if not self._valid_directive(d, allowed_stop):
                        errors.append(
                            f"{prefix} '{key}' must be an int (relative jump) or '{allowed_stop}'."
                        )

            # Validate action support
            if action not in (self._ELEMENT_ACTIONS | self._PAGE_ACTIONS):
                errors.append(f"{prefix} unsupported action '{action}'.")
                continue

            # Validate locator requirements:
            # Element actions generally require a locator.
            # Exceptions:
            # - wait: if no locator, treat as fixed wait (sleep-like) and use timeout duration
            # - scroll: locator optional
            # - sleep: locator should not be used (but we won't hard-fail; just warn)
            locator_present = has_selector or has_text

            if action in self._ELEMENT_ACTIONS:
                if action == "wait":
                    # wait may be locator-less (fixed wait), but then timeout is required
                    if not locator_present and "timeout" not in step:
                        errors.append(f"{prefix} 'wait' without locator requires 'timeout' (fixed wait).")
                else:
                    if not locator_present:
                        errors.append(f"{prefix} action '{action}' requires 'selector' or 'text'.")
            elif action == "sleep":
                if locator_present:
                    errors.append(f"{prefix} action 'sleep' should not have 'selector' or 'text'.")
                if "timeout" not in step:
                    errors.append(f"{prefix} action 'sleep' requires 'timeout' duration (ms).")
            elif action == "scroll":
                # scroll may be page-level or element-level; if page-level, value is recommended
                pass

        return errors

    # -------------------------
    # Core execution
    # -------------------------
    def _run_step(self, step: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Returns:
          (outcome, decisive)
            outcome: True if step succeeded, False otherwise
            decisive: True if step has an element target (or element-scoped scroll), else False
        """
        action = str(step.get("action", "")).strip()
        timeout = step.get("timeout", self.default_timeout)
        value = step.get("value", None)

        locator, decisive = self._resolve_locator(step, action)

        # Non-decisive page-level actions: we still execute them but do not recommend driving control flow.
        try:
            if action == "wait":
                return self._do_wait(locator, timeout, decisive), decisive

            if action == "click":
                return self._do_click(locator, timeout), True

            if action == "fill":
                text_to_fill = "" if value is None else str(value)
                return self._do_fill(locator, text_to_fill, timeout), True

            if action == "press":
                key = "Enter" if value is None else str(value)
                return self._do_press(locator, key, timeout), True

            if action == "check":
                return self._do_check(locator, timeout), True

            if action == "uncheck":
                return self._do_uncheck(locator, timeout), True

            if action == "sleep":
                return self._do_sleep(timeout), False

            if action == "scroll":
                return self._do_scroll(locator, value, timeout, decisive), decisive

            # Should never reach here due to validation/support set.
            self._log(f"[Error] Unsupported action at runtime: {action}")
            return False, decisive

        except Exception as e:
            # Treat exceptions as step failure.
            self._log(f"[Error] Step raised exception: {e!r}")
            return False, decisive

    # -------------------------
    # Locator resolution
    # -------------------------
    def _resolve_locator(self, step: Dict[str, Any], action: str):
        """
        Return (locator, decisive).
        locator may be None for page-level actions and for locator-less wait.
        decisive means the step outcome is meaningfully tied to a target object existence/action success.
        """
        if "selector" in step:
            sel = step.get("selector")
            if isinstance(sel, str) and sel.strip():
                return self.page.locator(sel.strip()), True

        if "text" in step:
            t = step.get("text")
            if isinstance(t, str) and t.strip():
                # Exact match is consistent and deterministic for automation scripts.
                return self.page.get_by_text(t.strip(), exact=True), True

        # No locator:
        # - wait: allowed as fixed wait (non-decisive)
        # - scroll: allowed as page-level (non-decisive)
        # - sleep: page-level (non-decisive)
        # - other element actions are invalid (caught by validation or will fail)
        return None, False

    # -------------------------
    # Action implementations
    # -------------------------
    def _do_wait(self, locator, timeout: int, decisive: bool) -> bool:
        # If no locator, treat wait as fixed delay (sleep-like) using timeout duration.
        if locator is None:
            if timeout is None:
                timeout = self.default_timeout
            self.page.wait_for_timeout(int(timeout))
            self._log(f"wait (fixed): {timeout}ms")
            return True

        try:
            # Use visible as the default "ready" state for determinism.
            locator.wait_for(state="visible", timeout=int(timeout))
            self._log("wait: element visible")
            return True
        except PlaywrightTimeoutError:
            self._log(f"wait: timeout after {timeout}ms")
            return False
        except Exception:
            self._log(f"wait: failed within {timeout}ms")
            return False

    def _do_click(self, locator, timeout: int) -> bool:
        if locator is None:
            self._log("click: missing locator")
            return False
        try:
            locator.click(timeout=int(timeout))
            self._log("click: success")
            return True
        except PlaywrightTimeoutError:
            self._log(f"click: timeout after {timeout}ms")
            return False
        except Exception:
            self._log(f"click: failed within {timeout}ms")
            return False

    def _do_fill(self, locator, value: str, timeout: int) -> bool:
        if locator is None:
            self._log("fill: missing locator")
            return False
        try:
            locator.fill(value, timeout=int(timeout))
            self._log("fill: success")
            return True
        except PlaywrightTimeoutError:
            self._log(f"fill: timeout after {timeout}ms")
            return False
        except Exception:
            self._log(f"fill: failed within {timeout}ms")
            return False

    def _do_press(self, locator, key: str, timeout: int) -> bool:
        if locator is None:
            self._log("press: missing locator")
            return False
        try:
            locator.press(key, timeout=int(timeout))
            self._log(f"press: success ({key})")
            return True
        except PlaywrightTimeoutError:
            self._log(f"press: timeout after {timeout}ms")
            return False
        except Exception:
            self._log(f"press: failed within {timeout}ms")
            return False

    def _do_check(self, locator, timeout: int) -> bool:
        if locator is None:
            self._log("check: missing locator")
            return False
        try:
            locator.check(timeout=int(timeout))
            self._log("check: success")
            return True
        except PlaywrightTimeoutError:
            self._log(f"check: timeout after {timeout}ms")
            return False
        except Exception:
            self._log(f"check: failed within {timeout}ms")
            return False

    def _do_uncheck(self, locator, timeout: int) -> bool:
        if locator is None:
            self._log("uncheck: missing locator")
            return False
        try:
            locator.uncheck(timeout=int(timeout))
            self._log("uncheck: success")
            return True
        except PlaywrightTimeoutError:
            self._log(f"uncheck: timeout after {timeout}ms")
            return False
        except Exception:
            self._log(f"uncheck: failed within {timeout}ms")
            return False

    def _do_sleep(self, timeout: int) -> bool:
        # Unconditional delay; not decisive by definition.
        try:
            self.page.wait_for_timeout(int(timeout))
            self._log(f"sleep: {timeout}ms")
            return True
        except Exception:
            self._log(f"sleep: failed ({timeout}ms)")
            return False

    def _do_scroll(self, locator, value: Any, timeout: int, decisive: bool) -> bool:
        """
        - If locator is provided: scroll target into view (decisive).
        - If locator is None: page scroll (non-decisive). Validate result with a subsequent wait step.
        """
        try:
            if locator is not None:
                # Element-scoped scroll is "decisive": target must exist.
                locator.scroll_into_view_if_needed(timeout=int(timeout))
                self._log("scroll: element into view")
                return True

            # Page-level scroll: non-decisive. `value` determines behavior.
            mode = "page_down" if value is None else str(value).strip().lower()
            if mode in ("page_down", "pagedown", "down"):
                self.page.evaluate("window.scrollBy(0, window.innerHeight)")
                self._log("scroll: page_down")
                return True
            if mode in ("page_up", "pageup", "up"):
                self.page.evaluate("window.scrollBy(0, -window.innerHeight)")
                self._log("scroll: page_up")
                return True
            if mode in ("bottom", "end"):
                self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                self._log("scroll: bottom")
                return True
            if mode in ("top", "start"):
                self.page.evaluate("window.scrollTo(0, 0)")
                self._log("scroll: top")
                return True

            # Optional numeric pixels support: value like "1200" or "-800"
            if mode.lstrip("-").isdigit():
                pixels = int(mode)
                self.page.evaluate("window.scrollBy(0, arguments[0])", pixels)
                self._log(f"scroll: pixels {pixels}")
                return True

            # Unknown scroll mode: treat as failure (but still non-decisive)
            self._log(f"scroll: unknown mode '{value}', no-op")
            return False

        except PlaywrightTimeoutError:
            self._log(f"scroll: timeout after {timeout}ms")
            return False
        except Exception:
            self._log(f"scroll: failed")
            return False

    # -------------------------
    # Directive helpers
    # -------------------------
    def _apply_directive(self, directive: Any, on: str) -> Optional[int]:
        """
        Convert directive into a relative jump offset.
        Returns None for STOP_OK/STOP_FAIL which signal termination.
        """
        if isinstance(directive, str):
            d = directive.strip()
            if on == "success" and d == self._STOP_OK:
                self._log("success: STOP_OK")
                return None
            if on == "fail" and d == self._STOP_FAIL:
                self._log("fail: STOP_FAIL")
                return None
            # Unknown string directive
            self._log(f"[Warning] Unknown {on} directive string: {directive!r}. Ignored (continue).")
            return 1

        if isinstance(directive, int):
            # Relative jump offset
            return directive

        self._log(f"[Warning] Invalid {on} directive type: {type(directive).__name__}. Ignored (continue).")
        return 1

    def _valid_directive(self, directive: Any, allowed_stop: str) -> bool:
        if isinstance(directive, int):
            return True
        if isinstance(directive, str) and directive.strip() == allowed_stop:
            return True
        return False

    # -------------------------
    # Misc helpers
    # -------------------------
    def _is_nonneg_int(self, v: Any) -> bool:
        return isinstance(v, int) and v >= 0

    def _step_desc(self, step: Dict[str, Any]) -> str:
        action = step.get("action", "N/A")
        if "selector" in step:
            return f"{action} selector={step.get('selector')!r}"
        if "text" in step:
            return f"{action} text={step.get('text')!r}"
        if "value" in step:
            return f"{action} value={step.get('value')!r}"
        return f"{action}"
