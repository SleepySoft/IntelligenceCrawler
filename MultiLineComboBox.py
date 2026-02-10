# -*- coding: utf-8 -*-
from PyQt5.QtGui import QTextDocument
from PyQt5.QtCore import Qt, QEvent, QSize, QRect, QRectF, QPoint, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QComboBox, QListView, QStyledItemDelegate, QStyle, QStyleOptionComboBox,
    QLineEdit, QPlainTextEdit, QFrame, QHBoxLayout, QPushButton, QSizeGrip
)


# -------------------- 1) 下拉列表多行显示 Delegate --------------------
class WrappedItemDelegate(QStyledItemDelegate):
    """
    使用 QTextDocument 实现：
    - item 文本自动换行
    - item 高度随内容自适应
    """
    def __init__(self, parent=None, wrap_width=0, h_margin=8, v_margin=6):
        super().__init__(parent)
        self.wrap_width = wrap_width
        self.h_margin = h_margin
        self.v_margin = v_margin

    def setWrapWidth(self, w: int):
        self.wrap_width = max(0, int(w))
        view = self.parent()
        if view:
            view.doItemsLayout()
            view.updateGeometries()
            view.viewport().update()

    def _make_doc(self, option, index):
        text = index.data(Qt.DisplayRole) or ""
        doc = QTextDocument()
        doc.setDefaultFont(option.font)
        doc.setPlainText(text)

        # wrap_width 来自 viewport 宽度；减去边距用于排版
        width = self.wrap_width if self.wrap_width > 0 else max(200, option.rect.width())
        doc.setTextWidth(max(1, width - self.h_margin * 2))
        return doc

    def paint(self, painter, option, index):
        painter.save()

        # 选中背景
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
            painter.setPen(option.palette.highlightedText().color())
        else:
            painter.setPen(option.palette.text().color())

        doc = self._make_doc(option, index)

        # 留边距的文本区域
        text_rect = option.rect.adjusted(self.h_margin, self.v_margin, -self.h_margin, -self.v_margin)

        painter.setClipRect(option.rect)
        painter.translate(text_rect.topLeft())

        # 关键修复：drawContents 第二参数必须 QRectF
        doc.drawContents(painter, QRectF(0, 0, text_rect.width(), text_rect.height()))

        painter.restore()

    def sizeHint(self, option, index):
        doc = self._make_doc(option, index)
        height = int(doc.size().height()) + self.v_margin * 2
        # 宽度由 view 控制，这里返回 option.rect.width() 即可
        return QSize(option.rect.width(), height)


# -------------------- 2) 可扩展高度的多行编辑器 --------------------
class ExpandingPlainTextEdit(QPlainTextEdit):
    def __init__(self, parent=None, min_lines=1, max_lines=6):
        super().__init__(parent)
        self.min_lines = max(1, min_lines)
        self.max_lines = max(self.min_lines, max_lines)

        self.setFrameStyle(0)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.document().setDocumentMargin(2)
        self.textChanged.connect(self._update_height)

    def _line_height(self):
        return self.fontMetrics().lineSpacing()

    def _update_height(self):
        # 让文档按当前宽度重新排版
        self.document().setTextWidth(self.viewport().width())
        content_h = self.document().size().height()

        lh = self._line_height()
        min_h = int(self.min_lines * lh + 8)
        max_h = int(self.max_lines * lh + 8)

        new_h = int(min(max(content_h + 6, min_h), max_h))
        self.setFixedHeight(new_h)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_height()


# -------------------- 3) 你的原始 Combo（保留宽度特性） --------------------
class AdjustableWidthComboBox(QComboBox):
    """
    保留你现有特性：
    - 下拉列表最小宽度 >= QComboBox 宽度
    - 下拉列表最大宽度受限
    """
    def __init__(self, parent=None, max_dropdown_width=800):
        super().__init__(parent)
        self.max_dropdown_width = max_dropdown_width

        lv = QListView()
        self.setView(lv)
        lv.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        lv.setMaximumWidth(self.max_dropdown_width)

    def showPopup(self):
        lv = self.view()
        lv.setMinimumWidth(self.width())          # 最小宽度跟随 combobox
        lv.setMaximumWidth(self.max_dropdown_width)
        super().showPopup()


# -------------------- 4) 组合：多行输入 + 多行下拉 --------------------
class MultiLineComboBox(AdjustableWidthComboBox):
    """
    功能：
    - 点击输入框：展开多行输入（覆盖式 QPlainTextEdit），高度自适应
    - 下拉列表：item 多行显示（换行+高度自适应）
    - 保留 AdjustableWidthComboBox 的宽度逻辑
    """
    def __init__(self, parent=None, max_dropdown_width=800, editor_min_lines=1, editor_max_lines=6):
        super().__init__(parent=parent, max_dropdown_width=max_dropdown_width)

        # 让 combo 看起来可编辑，但真实编辑用覆盖式 editor
        super().setEditable(True)

        self._preview_line = QLineEdit(self)
        self._preview_line.setReadOnly(True)
        self._preview_line.setPlaceholderText("点击输入（支持多行）…")
        self.setLineEdit(self._preview_line)

        # 关键：点击输入框事件，必须装在 lineEdit 上，否则 mousePressEvent 很容易收不到
        self._preview_line.installEventFilter(self)

        # 覆盖式多行编辑器
        self._editor = ExpandingPlainTextEdit(self, min_lines=editor_min_lines, max_lines=editor_max_lines)
        self._editor.hide()

        # 记录“完整文本”（因为预览只显示第一行）
        self._full_text = ""

        # 下拉列表多行显示 delegate
        lv = self.view()
        lv.setUniformItemSizes(False)
        lv.setWordWrap(True)
        self._delegate = WrappedItemDelegate(lv)
        lv.setItemDelegate(self._delegate)
        lv.viewport().installEventFilter(self)

        # 选择 item 时同步完整文本
        self.activated[str].connect(self._on_item_activated)

    # ------- 对外：如果你需要取“完整多行文本”，用这个 -------
    def fullText(self) -> str:
        if self.currentIndex() >= 0:
            return self.itemText(self.currentIndex())
        return self._full_text

    # ------- 保留 showPopup：并更新 wrap_width，保证 item 高度计算准确 -------
    def showPopup(self):
        self._collapse_editor(accept=True)

        lv = self.view()
        lv.setMinimumWidth(self.width())
        lv.setMaximumWidth(self.max_dropdown_width)

        # 更新 wrap 宽度，确保 sizeHint/换行正确
        self._delegate.setWrapWidth(lv.viewport().width() or self.width())

        super().showPopup()

    # ------- editor 位置跟随 edit field -------
    def _edit_field_rect(self) -> QRect:
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        return self.style().subControlRect(QStyle.CC_ComboBox, opt, QStyle.SC_ComboBoxEditField, self)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        r = self._edit_field_rect()
        self._editor.setGeometry(r)

    # ------- eventFilter：捕捉 lineEdit 点击 / viewport resize -------
    def eventFilter(self, obj, event):
        # 1) 点击预览输入框 => 展开多行编辑器
        if obj is self._preview_line and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self._expand_editor()
                return True

        # 2) 下拉视口尺寸变化 => 更新 wrap 宽度
        if obj is self.view().viewport() and event.type() == QEvent.Resize:
            self._delegate.setWrapWidth(obj.width())

        return super().eventFilter(obj, event)

    # ------- 展开/收起 editor -------
    def _expand_editor(self):
        # 如果当前是列表项，取 itemText；否则取 _full_text
        text = self.fullText()
        self._editor.setPlainText(text)
        self._editor.selectAll()
        self._editor.show()
        self._editor.setFocus()
        self._editor._update_height()

    def _collapse_editor(self, accept: bool):
        if not self._editor.isVisible():
            return

        if accept:
            text = self._editor.toPlainText()
            self._full_text = text
            self._set_preview_text(text)

            # 如果你希望“输入的内容也成为一个新 item”，取消注释：
            # if text and text not in [self.itemText(i) for i in range(self.count())]:
            #     self.addItem(text)
            #     self.setCurrentIndex(self.count() - 1)

        self._editor.hide()
        self.setFocus()

    def _set_preview_text(self, text: str):
        # 预览仅显示第一行（你也可以改成显示前两行等）
        first = (text or "").splitlines()[0] if text else ""
        fm = self._preview_line.fontMetrics()
        w = self._edit_field_rect().width() - 6
        self._preview_line.setText(fm.elidedText(first, Qt.ElideRight, w))

    # ------- 选择 item -------
    def _on_item_activated(self, text: str):
        self._full_text = text or ""
        self._set_preview_text(self._full_text)

    # ------- 键盘：在 editor 中 Enter=确认；Esc=取消 -------
    def keyPressEvent(self, e):
        if self._editor.isVisible():
            if e.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._collapse_editor(accept=True)
                return
            if e.key() == Qt.Key_Escape:
                self._collapse_editor(accept=False)
                return
        else:
            # 如果用户直接在 combo 上输入字符，也自动展开编辑器并接收输入
            if e.text() and not (e.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
                self._expand_editor()
                self._editor.insertPlainText(e.text())
                return

        super().keyPressEvent(e)

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        # editor 失焦后自动确认收起
        if self._editor.isVisible() and not self._editor.hasFocus():
            self._collapse_editor(accept=True)


class PopupTextEditor(QFrame):
    accepted = pyqtSignal(str)
    rejected = pyqtSignal()

    def __init__(self, anchor_widget, min_lines=8, max_lines=18):
        """
        min_lines: 默认最小行数（你要求至少5行，这里默认给更高的8行，参数可调）
        max_lines: 自动扩展的“舒适上限”，再多就靠竖向滚动条
        """
        super().__init__(anchor_widget, Qt.Popup | Qt.FramelessWindowHint)
        self.anchor = anchor_widget
        self.min_lines = max(1, min_lines)
        self.max_lines = max(self.min_lines, max_lines)

        self._accepted_by_user = False
        self._prefer_below = True

        # 合并更新：避免 textChanged -> resize 风暴导致卡死
        self._pending_resize = False
        self._last_suggest_h = -1

        # 用户是否手动拖拽调整过大小：一旦拖拽过，我们就不“强制缩回去”
        self._user_resized = False

        self.setObjectName("PopupTextEditor")
        self.setStyleSheet("""
            QFrame#PopupTextEditor {
                background: white;
                border: 1px solid #BFBFBF;
                border-radius: 6px;
            }
            QPushButton { padding: 4px 14px; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # --- 编辑区 ---
        self.edit = QPlainTextEdit(self)
        self.edit.setFrameStyle(0)

        # 关键：杜绝横向滚动条出现/隐藏引发的抖动
        self.edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.edit.setLineWrapMode(QPlainTextEdit.WidgetWidth)

        # 关键：竖滚动条固定 AlwaysOn，避免“按需出现”改变视口宽度导致反复重排
        self.edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.edit.document().setDocumentMargin(2)
        self.edit.textChanged.connect(self._request_resize)
        self.edit.installEventFilter(self)
        root.addWidget(self.edit, 1)

        # --- 底部按钮栏（含 QSizeGrip） ---
        bottom = QWidget(self)
        bottom_lay = QHBoxLayout(bottom)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        bottom_lay.setSpacing(8)

        bottom_lay.addStretch(1)

        self.btn_cancel = QPushButton("取消", bottom)
        self.btn_ok = QPushButton("确认", bottom)
        self.btn_ok.setDefault(True)

        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_ok.clicked.connect(self._ok)

        bottom_lay.addWidget(self.btn_cancel)
        bottom_lay.addWidget(self.btn_ok)

        # 右下角拖拽调整大小（QSizeGrip）
        self._grip = QSizeGrip(bottom)
        bottom_lay.addWidget(self._grip, 0, Qt.AlignRight | Qt.AlignBottom)

        root.addWidget(bottom, 0)

        # 最小高度：编辑区至少 min_lines 行（按钮栏不算在这 5/8 行里）
        self._apply_min_edit_height()

        # 允许窗口被 QSizeGrip 改变大小：不能用 fixedHeight
        self.setMinimumHeight(self.layout().sizeHint().height())

    # ---------- 对外接口 ----------
    def setText(self, text: str):
        self.edit.setPlainText(text or "")
        self.edit.moveCursor(self.edit.textCursor().End)
        self._user_resized = False  # 每次打开可重新回到“自动建议高度”
        self._request_resize()

    def text(self) -> str:
        return self.edit.toPlainText()

    def open_near_anchor(self, prefer_below=True):
        """打开并定位：宽度严格等于 anchor（combobox）宽度"""
        self._accepted_by_user = False
        self._prefer_below = prefer_below

        # 严格对齐宽度
        self.setFixedWidth(self.anchor.width())

        # 先建议一个更舒适的初始高度（更像网页 textarea）
        self._apply_resize(force=True)
        self._reposition()

        self.show()
        self.edit.setFocus()
        self.edit.selectAll()

    def sync_width_to_anchor(self):
        """当 combobox 宽度变化时，popup 跟随（严格等宽）"""
        if not self.isVisible():
            return
        self.setFixedWidth(self.anchor.width())
        # 宽度变化会影响换行，从而影响建议高度
        self._request_resize()
        self._reposition()

    # ---------- 高度策略 ----------
    def _apply_min_edit_height(self):
        fm = self.edit.fontMetrics()
        line_h = fm.lineSpacing()
        min_edit_h = int(self.min_lines * line_h + 18)
        self.edit.setMinimumHeight(min_edit_h)

    def _suggest_edit_height(self) -> int:
        """根据内容/换行计算一个“建议的编辑区高度”（限制到 max_lines）"""
        fm = self.edit.fontMetrics()
        line_h = fm.lineSpacing()

        # 让 doc 按当前视口宽度排版（换行高度才准）
        self.edit.document().setTextWidth(max(10, self.edit.viewport().width()))
        content_h = self.edit.document().size().height()

        min_h = int(self.min_lines * line_h + 18)
        max_h = int(self.max_lines * line_h + 18)

        return int(min(max(content_h + 10, min_h), max_h))

    def _request_resize(self):
        if self._pending_resize:
            return
        self._pending_resize = True
        QTimer.singleShot(0, self._apply_resize)

    def _apply_resize(self, force=False):
        self._pending_resize = False
        self._apply_min_edit_height()

        suggested_edit_h = self._suggest_edit_height()

        # 计算当前窗口里 edit 可以占用的高度（扣掉按钮栏+边距）
        # 这里用 layout 的 sizeHint 来近似扣除按钮栏与边距
        chrome_h = self.layout().sizeHint().height() - self.edit.height()
        # 目标总高度：让 edit 达到建议高度
        suggested_total_h = suggested_edit_h + chrome_h

        if (not force) and (suggested_total_h == self._last_suggest_h):
            return
        self._last_suggest_h = suggested_total_h

        # 关键：不再 setFixedHeight（否则无法拖拽）
        # 规则：
        # - 如果用户没手动拖拽过：窗口高度跟随建议高度（可变大也可变小）
        # - 如果用户手动拖拽过：只在“当前高度不够容纳建议高度”时才增高，不主动缩小
        cur_h = self.height()
        if (not self._user_resized) or (cur_h < suggested_total_h):
            self.resize(self.width(), suggested_total_h)
            self.setMinimumHeight(self.layout().sizeHint().height())

        if self.isVisible():
            self._reposition()

    # ---------- 位置（屏幕边界保护） ----------
    def _reposition(self):
        screen = QApplication.screenAt(self.anchor.mapToGlobal(self.anchor.rect().center()))
        if screen is None:
            screen = QApplication.primaryScreen()
        avail = screen.availableGeometry()

        w = self.width()
        h = self.height()

        below_pos = self.anchor.mapToGlobal(self.anchor.rect().bottomLeft()) + QPoint(0, 2)
        above_pos = self.anchor.mapToGlobal(self.anchor.rect().topLeft()) - QPoint(0, h + 2)

        # X：严格左对齐 combobox，若超屏则回收
        x = below_pos.x()
        if x + w > avail.right():
            x = max(avail.left(), avail.right() - w)
        if x < avail.left():
            x = avail.left()

        # Y：优先下方，否则上方
        if self._prefer_below and (below_pos.y() + h <= avail.bottom()):
            y = below_pos.y()
        else:
            y = above_pos.y()
            if y < avail.top():
                y = avail.top()

        self.move(x, y)

    # ---------- 用户拖拽识别 ----------
    def resizeEvent(self, e):
        # 只要可见状态下发生 resize，通常就是 grip 或系统调整
        if self.isVisible():
            self._user_resized = True

        # 宽度变化会影响换行/文档高度，触发一次合并更新（但不会风暴）
        self._request_resize()
        super().resizeEvent(e)

    # ---------- 关闭策略：点击外部关闭 => 取消 ----------
    def closeEvent(self, e):
        if not self._accepted_by_user:
            self.rejected.emit()
        super().closeEvent(e)

    # ---------- 确认/取消 ----------
    def _ok(self):
        self._accepted_by_user = True
        self.accepted.emit(self.text())
        self.close()

    def _cancel(self):
        self._accepted_by_user = False
        self.rejected.emit()
        self.close()

    # ---------- 键盘：Ctrl+Enter=确认，Esc=取消 ----------
    def eventFilter(self, obj, event):
        if obj is self.edit and event.type() == QEvent.KeyPress:
            if (event.key() in (Qt.Key_Return, Qt.Key_Enter)) and (event.modifiers() & Qt.ControlModifier):
                self._ok()
                return True
            if event.key() == Qt.Key_Escape:
                self._cancel()
                return True
        return super().eventFilter(obj, event)


class MultiLinePopupComboBox(AdjustableWidthComboBox):
    def __init__(self, parent=None, max_dropdown_width=800, editor_min_lines=8, editor_max_lines=18):
        super().__init__(parent=parent, max_dropdown_width=max_dropdown_width)

        super().setEditable(True)
        self._preview = QLineEdit(self)
        self._preview.setReadOnly(True)
        self._preview.setPlaceholderText("点击输入（多行浮层，可拖拽调整大小）…")
        self.setLineEdit(self._preview)
        self._preview.installEventFilter(self)

        self._full_text = ""
        self._snapshot_before_edit = ""

        self._popup = PopupTextEditor(self, min_lines=editor_min_lines, max_lines=editor_max_lines)
        self._popup.accepted.connect(self._apply_text)
        self._popup.rejected.connect(self._cancel_input)

        self.activated[str].connect(self._apply_text)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # 严格等宽：combo 宽度变化时同步 popup
        self._popup.sync_width_to_anchor()

    def eventFilter(self, obj, event):
        if obj is self._preview and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self.hidePopup()
                self._snapshot_before_edit = self._full_text or self.currentText()
                self._popup.setText(self._snapshot_before_edit)
                self._popup.open_near_anchor(prefer_below=True)
                return True
        return super().eventFilter(obj, event)

    def _apply_text(self, text: str):
        self._full_text = text or ""
        self._update_preview()

    def _cancel_input(self):
        # 点空白/取消：回滚
        self._full_text = self._snapshot_before_edit
        self._update_preview()

    def _update_preview(self):
        first = self._full_text.splitlines()[0] if self._full_text else ""
        fm = self._preview.fontMetrics()
        self._preview.setText(fm.elidedText(first, Qt.ElideRight, self._preview.width() - 8))

    def fullText(self) -> str:
        return self._full_text or self.currentText()

    def currentText(self):
        return self._full_text

    def setPlaceholderText(self, text: str):
        self._preview.setPlaceholderText(text)

# -------------------- Demo --------------------
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    w = QWidget()
    lay = QVBoxLayout(w)

    combo = MultiLinePopupComboBox(max_dropdown_width=600, editor_min_lines=1, editor_max_lines=5)
    combo.addItem("这是一个很长很长的项目文本，用于演示：下拉列表会自动换行显示，并且高度自适应。")
    combo.addItem("第一行\n第二行\n第三行：显式换行会被保留。")
    combo.addItem("A very very very very very long English line that should wrap nicely within the dropdown.")
    combo.addItem("短文本")
    combo.setCurrentIndex(0)

    lay.addWidget(combo)
    w.resize(560, 260)
    w.show()
    sys.exit(app.exec_())
