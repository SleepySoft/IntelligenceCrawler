# -*- coding: utf-8 -*-
"""
ActionEditorDialog — 易用的 Playwright 网页操作序列编辑器

设计目标：
- 独立可复用，不依赖 CrawlerPlayground 主体
- 面向“配置者”而非“工程师”，用自然语言描述每个步骤
- 标准数据接口：输入/输出为 list[dict]，与 PlaywrightActionEngine_v2 兼容
"""
import json
import copy
from typing import List, Dict, Any, Optional

from PyQt5.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QListWidget, QListWidgetItem, QPushButton, QComboBox, QLineEdit,
    QSpinBox, QLabel, QFrame, QSplitter, QTextEdit, QMessageBox,
    QSizePolicy, QDialogButtonBox, QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont


# ---------------------------------------------------------------------------
# Friendly labels & icons
# ---------------------------------------------------------------------------

ACTION_META = {
    "click":   {"label": "🖱️ 点击元素",   "needs_target": True,  "needs_value": False, "value_label": None},
    "fill":    {"label": "⌨️ 输入文字",   "needs_target": True,  "needs_value": True,  "value_label": "输入内容"},
    "press":   {"label": "🔘 按下按键",   "needs_target": True,  "needs_value": True,  "value_label": "按键（如 Enter）"},
    "check":   {"label": "☑️ 勾选框",     "needs_target": True,  "needs_value": False, "value_label": None},
    "uncheck": {"label": "⬜ 取消勾选",   "needs_target": True,  "needs_value": False, "value_label": None},
    "wait":    {"label": "⏳ 等待元素",   "needs_target": "opt", "needs_value": False, "value_label": None},
    "sleep":   {"label": "😴 固定等待",   "needs_target": False, "needs_value": False, "value_label": None},
    "scroll":  {"label": "📜 滚动页面",   "needs_target": "opt", "needs_value": "opt", "value_label": "滚动模式"},
}

TARGET_LABELS = {
    "selector": "CSS 选择器",
    "text": "文本内容",
    "none": "（无需目标）",
}

FLOW_LABELS = {
    "continue": "继续下一步",
    "STOP_OK": "结束（成功）",
    "STOP_FAIL": "结束（失败）",
}


def _fmt_action(step: dict, idx: int) -> str:
    """把一个步骤 dict 渲染成人类可读的短描述。"""
    action = step.get("action", "?")
    meta = ACTION_META.get(action, {})
    label = meta.get("label", action)

    target = step.get("selector") or step.get("text") or ""
    value = step.get("value", "")
    timeout = step.get("timeout", 3000)

    # 自然语言拼接
    if action in ("click", "check", "uncheck"):
        return f"{idx}. {label} 「{target}」"

    if action == "fill":
        return f'{idx}. {label} 「{target}」→ "{value}"'

    if action == "press":
        key = value or "Enter"
        return f'{idx}. {label} 「{target}」按键 {key}'

    if action == "wait":
        if target:
            return f"{idx}. {label} 「{target}」({timeout}ms)"
        return f"{idx}. {label} {timeout}ms"

    if action == "sleep":
        return f"{idx}. {label} {timeout}ms"

    if action == "scroll":
        if target:
            return f"{idx}. {label} 元素「{target}」"
        mode = value or "向下翻页"
        return f"{idx}. {label} ({mode})"

    return f"{idx}. {label}"


def _make_default_step(action: str = "click") -> dict:
    """新建一个默认步骤。"""
    return {
        "action": action,
        "selector": "",
        "text": "",
        "value": "",
        "timeout": 3000,
        "success": "continue",
        "fail": "continue",
    }


def _clean_step(step: dict) -> dict:
    """清理步骤：去掉空值，保留有效字段，输出与 PlaywrightActionEngine 兼容。"""
    out = {"action": step.get("action", "click")}

    target_type = step.get("_target_type", "selector")
    if target_type == "selector" and step.get("selector"):
        out["selector"] = step["selector"]
    elif target_type == "text" and step.get("text"):
        out["text"] = step["text"]

    if step.get("value"):
        out["value"] = step["value"]

    if step.get("timeout") is not None and step["timeout"] != 3000:
        out["timeout"] = int(step["timeout"])

    for key in ("success", "fail"):
        v = step.get(key, "continue")
        if v and v != "continue":
            out[key] = v

    return out


def _expand_step(step: dict) -> dict:
    """把外部导入的干净 dict 扩展为内部编辑用的富 dict（带 _target_type 等辅助字段）。"""
    out = copy.deepcopy(step)
    out.setdefault("action", "click")
    out.setdefault("selector", "")
    out.setdefault("text", "")
    out.setdefault("value", "")
    out.setdefault("timeout", 3000)
    out.setdefault("success", "continue")
    out.setdefault("fail", "continue")

    # 推断 target_type
    if out.get("selector"):
        out["_target_type"] = "selector"
    elif out.get("text"):
        out["_target_type"] = "text"
    else:
        out["_target_type"] = "none"
    return out


# ---------------------------------------------------------------------------
# ActionEditorDialog
# ---------------------------------------------------------------------------

class ActionEditorDialog(QDialog):
    """
    网页操作序列编辑器。

    Usage:
        dialog = ActionEditorDialog(parent)
        dialog.set_actions([{"action":"click","selector":"#btn"}])
        if dialog.exec_() == QDialog.Accepted:
            actions = dialog.get_actions()   # -> list[dict]
    """

    def __init__(self, parent: Optional[QWidget] = None, title: str = "网页操作步骤"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(720, 520)
        self._steps: List[dict] = []   # 内部富格式 dict 列表
        self._current_index = -1       # 当前正在编辑的步骤索引

        self._build_ui()
        self._connect_signals()
        self._refresh_list()

    # ---------- UI Construction ----------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # --- Header ---
        header = QLabel("<b>配置页面加载后的自动化操作</b>（如点击按钮、输入文字、滚动等）")
        header.setWordWrap(True)
        root.addWidget(header)

        # --- Splitter: List | Editor ---
        splitter = QSplitter(Qt.Horizontal)

        # ---- Left: Step List ----
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.step_list = QListWidget()
        self.step_list.setAlternatingRowColors(True)
        self.step_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.step_list.setMinimumWidth(260)
        left_layout.addWidget(self.step_list)

        # List toolbar
        list_toolbar = QHBoxLayout()
        self.btn_add = QPushButton("+ 添加")
        self.btn_remove = QPushButton("- 删除")
        self.btn_up = QPushButton("↑")
        self.btn_down = QPushButton("↓")
        self.btn_up.setMaximumWidth(36)
        self.btn_down.setMaximumWidth(36)
        list_toolbar.addWidget(self.btn_add)
        list_toolbar.addWidget(self.btn_remove)
        list_toolbar.addStretch(1)
        list_toolbar.addWidget(self.btn_up)
        list_toolbar.addWidget(self.btn_down)
        left_layout.addLayout(list_toolbar)

        splitter.addWidget(left_frame)

        # ---- Right: Editor Form ----
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(8, 0, 0, 0)

        # Title for editor
        self.editor_title = QLabel("<b>步骤详情</b>（先点击左侧步骤）")
        right_layout.addWidget(self.editor_title)

        # Form
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setSpacing(10)

        self.f_action = QComboBox()
        for key, meta in ACTION_META.items():
            self.f_action.addItem(meta["label"], key)
        form.addRow("动作类型:", self.f_action)

        self.f_target_type = QComboBox()
        for key, label in TARGET_LABELS.items():
            self.f_target_type.addItem(label, key)
        form.addRow("查找方式:", self.f_target_type)

        self.f_target_value = QLineEdit()
        self.f_target_value.setPlaceholderText('例如：#cookie-accept 或 "同意并继续"')
        form.addRow("目标值:", self.f_target_value)

        self.f_value = QLineEdit()
        self.f_value.setPlaceholderText("根据动作类型填写，如输入的文字或按键名称")
        form.addRow("输入值:", self.f_value)

        self.f_timeout = QSpinBox()
        self.f_timeout.setRange(0, 60000)
        self.f_timeout.setSingleStep(500)
        self.f_timeout.setSuffix(" ms")
        self.f_timeout.setValue(3000)
        form.addRow("超时时间:", self.f_timeout)

        # Control flow
        flow_frame = QFrame()
        flow_frame.setFrameShape(QFrame.StyledPanel)
        flow_layout = QFormLayout(flow_frame)
        flow_layout.setSpacing(8)

        self.f_success = QComboBox()
        self.f_success.addItem("继续下一步", "continue")
        self.f_success.addItem("结束（成功）", "STOP_OK")
        self.f_success.addItem("结束（失败）", "STOP_FAIL")
        flow_layout.addRow("✅ 成功时:", self.f_success)

        self.f_fail = QComboBox()
        self.f_fail.addItem("继续下一步", "continue")
        self.f_fail.addItem("结束（成功）", "STOP_OK")
        self.f_fail.addItem("结束（失败）", "STOP_FAIL")
        flow_layout.addRow("❌ 失败时:", self.f_fail)

        form.addRow(flow_frame)

        right_layout.addWidget(form_widget)
        right_layout.addStretch(1)

        # Hint label
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #666; font-size: 12px;")
        right_layout.addWidget(self.hint_label)

        splitter.addWidget(right_frame)
        splitter.setSizes([280, 420])
        root.addWidget(splitter, 1)

        # --- Bottom buttons ---
        bottom = QHBoxLayout()

        self.btn_preview = QPushButton("📋 查看 JSON")
        self.btn_preview.setToolTip("查看当前步骤列表的标准 JSON 格式")
        bottom.addWidget(self.btn_preview)

        bottom.addStretch(1)

        self.btn_ok = QPushButton("确定")
        self.btn_ok.setDefault(True)
        self.btn_cancel = QPushButton("取消")
        bottom.addWidget(self.btn_ok)
        bottom.addWidget(self.btn_cancel)

        root.addLayout(bottom)

        # --- Editor state ---
        self._set_editor_enabled(False)

    def _connect_signals(self):
        self.step_list.currentRowChanged.connect(self._on_list_selection_changed)

        self.btn_add.clicked.connect(self._on_add)
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_up.clicked.connect(self._on_move_up)
        self.btn_down.clicked.connect(self._on_move_down)

        self.f_action.currentIndexChanged.connect(self._on_form_changed)
        self.f_target_type.currentIndexChanged.connect(self._on_form_changed)
        self.f_target_value.textChanged.connect(self._on_form_changed)
        self.f_value.textChanged.connect(self._on_form_changed)
        self.f_timeout.valueChanged.connect(self._on_form_changed)
        self.f_success.currentIndexChanged.connect(self._on_form_changed)
        self.f_fail.currentIndexChanged.connect(self._on_form_changed)

        self.btn_preview.clicked.connect(self._on_preview_json)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    # ---------- Public API ----------

    def set_actions(self, actions: List[dict]):
        """从外部载入动作列表（标准格式）。"""
        self._steps = [_expand_step(s) for s in (actions or [])]
        self._refresh_list()

    def get_actions(self) -> List[dict]:
        """导出为标准格式（list[dict]），可直接传给 PlaywrightActionEngine。"""
        return [_clean_step(s) for s in self._steps]

    # ---------- List Management ----------

    def _refresh_list(self):
        self.step_list.clear()
        for i, step in enumerate(self._steps, 1):
            item = QListWidgetItem(_fmt_action(step, i))
            item.setData(Qt.UserRole, i - 1)
            self.step_list.addItem(item)

        if self._steps:
            if self._current_index < 0 or self._current_index >= len(self._steps):
                self._current_index = 0
            self.step_list.setCurrentRow(self._current_index)
        else:
            self._current_index = -1
            self._set_editor_enabled(False)
            self.editor_title.setText("<b>步骤详情</b>（先点击左侧步骤）")

    def _on_list_selection_changed(self, row: int):
        if row < 0 or row >= len(self._steps):
            self._current_index = -1
            self._set_editor_enabled(False)
            return
        self._current_index = row
        self._load_step_into_editor(self._steps[row])
        self._set_editor_enabled(True)

    def _on_add(self):
        # 根据当前选中的动作类型，或默认 click
        default_action = "click"
        if 0 <= self._current_index < len(self._steps):
            default_action = self._steps[self._current_index].get("action", "click")

        new_step = _make_default_step(default_action)
        insert_idx = self._current_index + 1 if self._current_index >= 0 else len(self._steps)
        self._steps.insert(insert_idx, new_step)
        self._refresh_list()
        self.step_list.setCurrentRow(insert_idx)

    def _on_remove(self):
        row = self.step_list.currentRow()
        if 0 <= row < len(self._steps):
            del self._steps[row]
            self._current_index = min(row, len(self._steps) - 1)
            self._refresh_list()

    def _on_move_up(self):
        row = self.step_list.currentRow()
        if row > 0:
            self._steps[row - 1], self._steps[row] = self._steps[row], self._steps[row - 1]
            self._refresh_list()
            self.step_list.setCurrentRow(row - 1)

    def _on_move_down(self):
        row = self.step_list.currentRow()
        if 0 <= row < len(self._steps) - 1:
            self._steps[row + 1], self._steps[row] = self._steps[row], self._steps[row + 1]
            self._refresh_list()
            self.step_list.setCurrentRow(row + 1)

    # ---------- Editor Logic ----------

    def _set_editor_enabled(self, enabled: bool):
        widgets = [
            self.f_action, self.f_target_type, self.f_target_value,
            self.f_value, self.f_timeout, self.f_success, self.f_fail,
        ]
        for w in widgets:
            w.setEnabled(enabled)
        self.btn_remove.setEnabled(len(self._steps) > 0)
        self.btn_up.setEnabled(self.step_list.currentRow() > 0)
        self.btn_down.setEnabled(
            0 <= self.step_list.currentRow() < len(self._steps) - 1
        )

    def _load_step_into_editor(self, step: dict):
        self._ui_silence = True  # 防止信号递归
        try:
            action = step.get("action", "click")
            idx = self.f_action.findData(action)
            self.f_action.setCurrentIndex(max(0, idx))

            tt = step.get("_target_type", "none")
            idx = self.f_target_type.findData(tt)
            self.f_target_type.setCurrentIndex(max(0, idx))

            self.f_target_value.setText(step.get("selector") or step.get("text") or "")
            self.f_value.setText(step.get("value", ""))
            self.f_timeout.setValue(int(step.get("timeout", 3000)))

            succ = step.get("success", "continue")
            idx = self.f_success.findData(succ)
            self.f_success.setCurrentIndex(max(0, idx))

            fail = step.get("fail", "continue")
            idx = self.f_fail.findData(fail)
            self.f_fail.setCurrentIndex(max(0, idx))

            self._update_editor_ui_by_action(action)
            self.editor_title.setText(f"<b>步骤 {self._current_index + 1} 详情</b>")
        finally:
            self._ui_silence = False

    def _update_editor_ui_by_action(self, action: str):
        """根据动作类型，动态显示/隐藏相关字段。"""
        meta = ACTION_META.get(action, {})
        needs_target = meta.get("needs_target", False)
        needs_value = meta.get("needs_value", False)
        value_label = meta.get("value_label", "输入值")

        # target_type
        if needs_target is True:
            self.f_target_type.setEnabled(True)
            # 不允许选 "none"
            if self.f_target_type.currentData() == "none":
                self.f_target_type.setCurrentIndex(0)  # selector
        elif needs_target == "opt":
            self.f_target_type.setEnabled(True)
        else:
            self.f_target_type.setEnabled(False)
            self.f_target_type.setCurrentIndex(self.f_target_type.findData("none"))

        # target_value
        self.f_target_value.setEnabled(needs_target is not False and self.f_target_type.currentData() != "none")

        # value
        self.f_value.setEnabled(bool(needs_value))
        form = self.f_value.parentWidget().layout()
        if form:
            label = form.labelForField(self.f_value)
            if label:
                label.setText(value_label + ":" if needs_value else "输入值:（无需填写）")

        # hints
        hints = {
            "click":   "等待目标元素出现，然后点击。若超时则视为失败。",
            "fill":    "等待目标元素出现，然后填入文字。",
            "press":   "等待目标元素出现，然后按下指定按键（默认 Enter）。",
            "check":   "等待目标复选框出现，然后勾选。",
            "uncheck": "等待目标复选框出现，然后取消勾选。",
            "wait":    "等待目标元素出现；若不填目标，则变为固定等待（与 sleep 相同）。",
            "sleep":   "固定等待一段时间，不进行任何页面操作。",
            "scroll":  "有目标=滚动到元素可见；无目标=页面滚动（可指定模式如 page_down/bottom）。",
        }
        self.hint_label.setText(hints.get(action, ""))

    def _on_form_changed(self):
        if getattr(self, '_ui_silence', False):
            return
        if self._current_index < 0 or self._current_index >= len(self._steps):
            return

        step = self._steps[self._current_index]

        action = self.f_action.currentData()
        step["action"] = action

        target_type = self.f_target_type.currentData()
        step["_target_type"] = target_type
        step["selector"] = ""
        step["text"] = ""
        if target_type == "selector":
            step["selector"] = self.f_target_value.text()
        elif target_type == "text":
            step["text"] = self.f_target_value.text()

        step["value"] = self.f_value.text()
        step["timeout"] = self.f_timeout.value()
        step["success"] = self.f_success.currentData()
        step["fail"] = self.f_fail.currentData()

        # 同步更新列表中的显示文本
        item = self.step_list.item(self._current_index)
        if item:
            item.setText(_fmt_action(step, self._current_index + 1))

        self._update_editor_ui_by_action(action)
        self._set_editor_enabled(True)

    # ---------- Preview ----------

    def _on_preview_json(self):
        data = self.get_actions()
        text = json.dumps(data, indent=2, ensure_ascii=False)

        dlg = QDialog(self)
        dlg.setWindowTitle("动作序列 JSON")
        dlg.setMinimumSize(480, 360)
        lay = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setPlainText(text)
        te.setReadOnly(True)
        te.setFont(QFont("Courier", 10))
        lay.addWidget(te)
        btn = QPushButton("关闭")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.exec_()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    # 示例数据：模拟一个“同意 Cookie → 搜索 → 等待结果”的流程
    sample = [
        {"action": "click", "selector": "#cookie-accept", "timeout": 3000},
        {"action": "fill", "selector": "input[name='q']", "value": "artificial intelligence", "timeout": 3000},
        {"action": "press", "selector": "input[name='q']", "value": "Enter", "timeout": 3000},
        {"action": "wait", "selector": ".search-results", "timeout": 5000},
        {"action": "scroll", "value": "page_down"},
        {"action": "sleep", "timeout": 1000},
    ]

    dialog = ActionEditorDialog()
    dialog.set_actions(sample)

    if dialog.exec_() == QDialog.Accepted:
        print("=== Accepted ===")
        print(json.dumps(dialog.get_actions(), indent=2, ensure_ascii=False))
    else:
        print("=== Cancelled ===")

    sys.exit(app.exec_())
