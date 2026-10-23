"""Playground 配置的无损往返状态。

UI 只能表达 ``CRAWLER_CONFIG`` 的一部分。加载现有配置后，不能用 UI 的默认值
覆盖未暴露字段；本模块以原配置为基线，只把用户实际改变的投影字段合并回去。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


_MISSING = object()


@dataclass(frozen=True)
class ConfigChange:
    path: str
    before: Any
    after: Any


def _collect_changes(before: Any, after: Any, path: str = "") -> List[ConfigChange]:
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        changes: List[ConfigChange] = []
        for key in sorted(before.keys() | after.keys(), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            changes.extend(_collect_changes(
                before.get(key, _MISSING),
                after.get(key, _MISSING),
                child_path,
            ))
        return changes
    return [ConfigChange(
        path=path or "<root>",
        before="<missing>" if before is _MISSING else before,
        after="<missing>" if after is _MISSING else after,
    )]


def _merge_changed_value(base: Any, before: Any, after: Any) -> Any:
    """把 ``before -> after`` 的变化递归应用到 ``base``。"""
    if before == after:
        return deepcopy(base)

    if isinstance(before, dict) and isinstance(after, dict):
        result = deepcopy(base) if isinstance(base, dict) else {}
        for key in before.keys() | after.keys():
            old_value = before.get(key, _MISSING)
            new_value = after.get(key, _MISSING)
            if old_value == new_value:
                continue
            if new_value is _MISSING:
                result.pop(key, None)
                continue
            base_value = result.get(key, _MISSING)
            if base_value is _MISSING:
                result[key] = deepcopy(new_value)
            else:
                result[key] = _merge_changed_value(
                    base_value,
                    None if old_value is _MISSING else old_value,
                    new_value,
                )
        return result

    return deepcopy(after)


class PlaygroundConfigState:
    """保存已加载配置，并合并 UI 中真正发生的修改。"""

    _COMPONENT_GROUPS = (
        ("discoverer_name", "discoverer_init_param"),
        ("d_fetcher_name", "d_fetcher_init_param", "d_fetcher_kwargs"),
        ("e_fetcher_name", "e_fetcher_init_param", "e_fetcher_kwargs"),
        ("extractor_name", "extractor_init_param", "extractor_kwargs"),
    )

    def __init__(self) -> None:
        self._baseline: Optional[Dict[str, Any]] = None
        self._ui_projection: Optional[Dict[str, Any]] = None

    @property
    def has_loaded_config(self) -> bool:
        return self._baseline is not None

    def capture(self, config: Dict[str, Any], ui_projection: Dict[str, Any]) -> None:
        self._baseline = deepcopy(config)
        self._ui_projection = deepcopy(ui_projection)

    def clear(self) -> None:
        self._baseline = None
        self._ui_projection = None

    def loaded_value(self, key: str, default: Any = None) -> Any:
        if self._baseline is None:
            return default
        return self._baseline.get(key, default)

    def projection_changed(self, current_projection: Dict[str, Any], *keys: str) -> bool:
        """判断指定 UI 字段是否相对加载时发生了实际变化。"""
        if self._ui_projection is None:
            return True
        return any(
            self._ui_projection.get(key, _MISSING)
            != current_projection.get(key, _MISSING)
            for key in keys
        )

    def merge(self, current_projection: Dict[str, Any]) -> Dict[str, Any]:
        """返回运行/保存配置；未被 UI 修改的值保持加载时的原样。"""
        if self._baseline is None or self._ui_projection is None:
            return deepcopy(current_projection)

        result = deepcopy(self._baseline)
        before = self._ui_projection

        replaced_keys = set()
        for group in self._COMPONENT_GROUPS:
            identity_key = group[0]
            if before.get(identity_key, _MISSING) == current_projection.get(identity_key, _MISSING):
                continue
            for key in group:
                replaced_keys.add(key)
                if key in current_projection:
                    result[key] = deepcopy(current_projection[key])
                else:
                    result.pop(key, None)

        for key in before.keys() | current_projection.keys():
            if key in replaced_keys:
                continue
            old_value = before.get(key, _MISSING)
            new_value = current_projection.get(key, _MISSING)
            if old_value == new_value:
                continue
            if new_value is _MISSING:
                result.pop(key, None)
                continue
            base_value = result.get(key, _MISSING)
            if base_value is _MISSING:
                result[key] = deepcopy(new_value)
            else:
                result[key] = _merge_changed_value(
                    base_value,
                    None if old_value is _MISSING else old_value,
                    new_value,
                )

        return result

    def changes(self, current_projection: Dict[str, Any]) -> List[ConfigChange]:
        if self._baseline is None:
            return []
        return _collect_changes(self._baseline, self.merge(current_projection))
