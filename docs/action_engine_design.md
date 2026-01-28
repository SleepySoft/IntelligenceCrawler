# Playwright Action Engine – Flat Step Schema & Design Spec (v1)

## 1. Overview

### 1.1 Purpose

Provide a **flat, readable, line-by-line** action script format to execute browser UI interactions sequentially, with:

*   Built-in waiting semantics per step (bounded by `timeout`)
*   Optional control-flow directives (`success` / `fail`) using **relative jumps** or **immediate termination**
*   Minimal redundancy (avoid “wait for same element” + “do action on same element” unless explicitly desired)

### 1.2 Design Goals

*   **Non-nested configuration**: each step is a single dictionary, “one line, one intent”.
*   **Low overlap, high composability**:
    *   Element-bound actions already wait for their target; “result waits” are explicit steps.
*   **Deterministic semantics**:
    *   Clear definition of what `timeout`, `success`, `fail`, and “success/failure” mean.

### 1.3 Non-Goals

*   Full scripting language (variables, loops, complex conditions)
*   Deep branching syntax beyond relative jumps and stop directives
*   Assertions framework (can be layered later)

***

## 2. Step Object Schema

Each step is a JSON-like dictionary.

### 2.1 Required Fields

```text
action: string
```

### 2.2 Optional Fields

```text
selector: string          # CSS selector locator (mutually exclusive with `text`)
text: string              # exact text locator (mutually exclusive with `selector`)

timeout: integer          # milliseconds
value: string             # action parameter (meaning depends on action)

success: integer|string   # relative jump or STOP_OK
fail: integer|string      # relative jump or STOP_FAIL
```

### 2.3 Locator Rules (Mutual Exclusivity)

A step may specify **at most one** of:

*   `selector`
*   `text`

If both are present, the step is **invalid**.

Some actions **require** a locator (element-bound). Some **do not** (page-bound).

***

## 3. Actions (Valid Options & Behavior)

Actions are divided into **Decisive (element-bound)** and **Non-decisive (page-bound)**.

### 3.1 Decisive Actions (Element-Bound)

These actions require a locator (`selector` or `text`).  
They are considered **decisive** because step outcome depends on:

*   whether the element is found within `timeout`, and
*   whether the action succeeds (no exception) within `timeout`.

Supported (v1 baseline):

*   `wait`
*   `click`
*   `fill`
*   `press`
*   `check`
*   `uncheck`

#### 3.1.1 `wait`

*   **Meaning**: wait until the target element is found and reaches the “ready state” required by the engine (default: visible/attached), but **perform no interaction**.
*   **Outcome**:
    *   Success if element becomes ready within timeout
    *   Fail otherwise (timeout / locator not found)

#### 3.1.2 `click`

*   Waits for the element to appear and become clickable (visible and actionable).
*   Performs a click.
*   **Important**: This does **not** wait for navigation or any downstream result. If you need to wait for result, add a separate `wait` step.

#### 3.1.3 `fill`

*   Waits for input element to be ready.
*   Sets its value to `value` (default: empty string if omitted).
*   Note: `fill` is a direct set (fast). If you later want human-like typing, you can add `type` as an additional action, still flat.

#### 3.1.4 `press`

*   Waits for the element to be ready and focused-capable.
*   Sends a key specified by `value` (default: `"Enter"` if omitted).

#### 3.1.5 `check` / `uncheck`

*   Waits for checkbox/radio-like element.
*   Checks/unchecks.

***

### 3.2 Non-Decisive Actions (Page-Bound)

These actions **do not require** a locator. Their execution may still fail due to runtime exceptions, but they are **not intended** to define “business success/failure” (e.g., “scroll to bottom” is not a meaningful completion criterion).

Supported (v1 baseline):

*   `sleep`
*   `scroll`

#### 3.2.1 `sleep`

*   If no locator is provided, `timeout` is treated as a fixed sleep duration in ms.
*   `timeout` is required for meaningful behavior.
*   This is the explicit “unconditional delay” option.

#### 3.2.2 `scroll`

Two modes:

**A) Page scroll (no locator)**

*   Uses `value` to indicate scroll behavior.
*   Examples of `value` (recommended minimal set):
    *   `"page_down"` / `"page_up"`
    *   `"bottom"` / `"top"`
    *   numeric string like `"1200"` meaning pixel delta (optional; can be added later)

**B) Element scroll (locator present)**

*   Scrolls an element into view (or scrolls a container).
*   In this mode, `scroll` becomes **decisive** because it has a target object.

> Policy: **Scroll-to-bottom without a locator is non-decisive**: it should not be relied on for `success`/`fail`. Use a subsequent `wait` step to validate the actual result.

***

## 4. Timeout Semantics

### 4.1 For Element-Bound (Decisive) Steps

*   `timeout` is the maximum time to:
    1.  locate the element, and
    2.  wait it to be ready (visible/actionable), and
    3.  complete the action successfully.

If the element cannot be found or the action cannot complete in time ⇒ step **fails**.

### 4.2 For Page-Bound Steps (No Locator)

*   `sleep`: `timeout` is the sleep duration.
*   `scroll` page-mode: `timeout` may be used as a safety cap for the underlying operation (optional), but generally scroll is immediate; any post-scroll “result waiting” must be explicit.

***

## 5. Control Flow: `success` and `fail`

### 5.1 Defaults (Important)

If `success` and/or `fail` is **omitted**, the engine **ignores control flow overrides** and proceeds to the next step (i.e., *continue*).

### 5.2 Allowed Values

*   `success`:
    *   `"STOP_OK"`: terminate immediately with overall success
    *   `int`: relative jump offset (e.g., `+2`, `-1`)
*   `fail`:
    *   `"STOP_FAIL"`: terminate immediately with overall failure
    *   `int`: relative jump offset

### 5.3 Relative Jump Semantics

Let current step index be `i` (0-based). Next step index is:

*   on success: `i + (success_offset if provided else +1)`
*   on fail: `i + (fail_offset if provided else +1)`

Examples:

*   `fail: +1` → fail but continue
*   `fail: +2` → fail and skip one step
*   `fail: -1` → fail and retry previous step (use safeguards!)

### 5.4 Control Flow Applicability (Decisive vs Non-Decisive)

*   For **decisive steps** (element-bound, including `wait/click/fill/press/check/uncheck`):
    *   `success`/`fail` are honored based on step outcome.
*   For **non-decisive steps** (page scroll without locator, sleep):
    *   Recommendation: treat these steps as “neutral utilities”.
    *   **Spec rule (recommended)**: if `success`/`fail` is set on a non-decisive step, engine should **log a warning and ignore** those directives (or treat as config invalid—choose one).
        *   I recommend **warn+ignore** for ergonomics.

***

## 6. Step Outcome Definition (Your Point #3)

### 6.1 Decisive Steps

A decisive step is **SUCCESS** iff:

1.  The target object exists/appears within `timeout`, and
2.  The action completes without exception (within timeout).

A decisive step is **FAIL** otherwise.

### 6.2 Non-Decisive Steps

*   `scroll` (page-mode) and `sleep` do not constitute meaningful business success by themselves.
*   They may fail only on runtime errors (rare), but do not drive control flow by default.

**Result validation should be done by a subsequent decisive `wait` step**, e.g.:

*   scroll down → wait for “new items” selector

***

## 7. Validation Rules

A configuration is valid if:

1.  Each step has `action`.
2.  At most one locator key is present (`selector` XOR `text`).
3.  If action is element-bound, a locator is required.
4.  `timeout` is an integer ≥ 0 when provided.
5.  `success`/`fail` are either:
    *   `"STOP_OK"`/`"STOP_FAIL"`, or
    *   integer offsets
6.  (Recommended runtime safety) Engine enforces:
    *   `max_total_step_executions`
    *   `max_visits_per_step`
        to prevent infinite loops with negative/zero jumps.

***

## 8. Examples (Flat, Sequential)

### 8.1 Click then Wait for Result (No Redundant Wait)

```python
[
  {"selector": "#btn", "action": "click", "timeout": 3000, "fail": "STOP_FAIL"},
  {"selector": ".result", "action": "wait", "timeout": 8000, "fail": "STOP_FAIL"},
]
```

### 8.2 Optional Popup Close (Fail Continues)

```python
[
  {"text": "Close", "action": "click", "timeout": 300, "fail": +1},
  {"selector": ".main", "action": "wait", "timeout": 3000, "fail": "STOP_FAIL"},
]
```

### 8.3 Page Scroll then Validate by Waiting for New Content

```python
[
  {"action": "scroll", "value": "page_down", "timeout": 0},
  {"selector": ".quote", "action": "wait", "timeout": 5000, "fail": "STOP_FAIL"},
]
```

### 8.4 Fail-Back Jump (Retry Pattern) + Stop on Success

```python
[
  {"selector": "#submit", "action": "click", "timeout": 2000, "fail": -1},
  {"selector": ".done", "action": "wait", "timeout": 5000, "success": "STOP_OK", "fail": -1},
]
```

***

## 9. Open Choices (Minor) Before Coding

To keep the implementation consistent, you should decide (I can implement either way):

1.  **What “wait” means exactly**:
    *   default state: `visible`? or `attached`?
    *   (Simplest v1): `visible` for all element-bound actions.

2.  **Non-decisive steps with `success/fail`**:
    *   warn+ignore (recommended)
    *   treat as invalid configuration (strict)

3.  **Scroll `value` vocabulary**:
    *   minimal set: `page_down`, `page_up`, `bottom`, `top`
    *   optionally allow numeric pixels as string

***

# Summary (Your Design, Polished)

Your proposed design is solid. With the clarifications above, it becomes a clean “flat DSL”:

*   **Action waits for its own target** (not downstream results).
*   **Result waiting is explicit** via separate `wait` steps.
*   `success/fail` are **optional**; absent means “just proceed”.
*   **Decisive success/failure** is tied to target existence and action completion; **scroll-to-bottom is not a decisive criterion**—validate via subsequent waits.
*   `success/fail` use **relative jumps** and STOP directives.

***
