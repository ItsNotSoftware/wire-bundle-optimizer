"""
Copyright (c) 2025 Diogo Ferreira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import sys
import yaml
import numpy as np
from time import perf_counter
from typing import List, Dict, Any

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox,
    QGroupBox,
    QFormLayout,
    QRadioButton,
    QComboBox,
    QFrame,
    QListWidgetItem,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
)
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QKeySequence, QShortcut
from PyQt6.QtCore import Qt

from optimizer import WireBundleOptimizer


def load_wire_types(filepath: str = "wire_types.yaml") -> dict:
    """Load predefined wire types from a YAML file."""
    try:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return data
    except FileNotFoundError:
        QMessageBox.warning(
            None,
            "File Not Found",
            f"Wire types file '{filepath}' not found. Please create it to use predefined sizes.",
        )
        return {}


def load_sleeve_types(filepath: str = "sleeve_types.yaml") -> dict:
    """Load predefined sleeve thicknesses from a YAML file (label -> thickness mm)."""
    try:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return data
    except FileNotFoundError:
        # No predefined sleeves available; UI will allow custom entry
        return {}


class WirePlotWidget(QWidget):
    """
    QWidget that visualizes the wire bundle layout including previously
    optimized shielded layers.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.positions = np.empty((0, 2))
        self.radii = np.array([])
        self.outer_radius = 0.0
        self.colors: List[str] = []

        # Layers history: list of dicts:
        # { "coords": Nx2, "radii": N, "colors": [..], "inner_R": float, "outer_R": float }
        self.layers: List[Dict[str, Any]] = []

        # Current frozen core radius (inner exclusion for current run)
        self.inner_exclusion_radius: float = 0.0

        self.setMinimumSize(300, 300)

    def set_layers(
        self, layers: List[Dict[str, Any]], inner_exclusion_radius: float
    ) -> None:
        self.layers = layers
        self.inner_exclusion_radius = float(inner_exclusion_radius)
        self.update()

    def update_scene(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        outer_radius: float,
        colors: List[str],
    ) -> None:
        self.positions = positions if positions is not None else np.empty((0, 2))
        self.radii = radii if radii is not None else np.array([])
        self.outer_radius = float(outer_radius) if outer_radius is not None else 0.0
        self.colors = colors or []
        self.update()

    def _global_max_radius(self) -> float:
        max_r = self.outer_radius
        for L in self.layers:
            max_r = max(max_r, float(L.get("outer_R", 0.0)))
        # also show the inner exclusion ring if larger than outer_radius
        max_r = max(max_r, self.inner_exclusion_radius)
        # add some padding to avoid touching edges
        return max_r * 1.05 if max_r > 0 else 1.0

    def paintEvent(self, a0) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        max_r = self._global_max_radius()
        if max_r <= 0:
            return

        scale = min(w, h) / (2 * (max_r))
        painter.translate(w / 2, h / 2)

        # Draw historical layers (sleeve rings + their wires)
        for L in self.layers:
            inner_R = float(L["inner_R"])
            outer_R = float(L["outer_R"])

            # --- Shield ring: draw a true annulus (no "punching" the center) ---
            from PyQt6.QtGui import QPainterPath

            ring_path = QPainterPath()
            # outer ellipse
            ring_path.addEllipse(
                int(-outer_R * scale),
                int(-outer_R * scale),
                int(2 * outer_R * scale),
                int(2 * outer_R * scale),
            )
            # inner ellipse; OddEvenFill makes it a ring
            ring_path.addEllipse(
                int(-inner_R * scale),
                int(-inner_R * scale),
                int(2 * inner_R * scale),
                int(2 * inner_R * scale),
            )
            ring_path.setFillRule(Qt.FillRule.OddEvenFill)

            painter.setPen(Qt.PenStyle.NoPen)
            ring_color = L.get("ring_color", "#888888")
            c = QColor(ring_color)
            c.setAlpha(90)
            painter.setBrush(c)
            painter.drawPath(ring_path)

            # ring outline
            ring_pen = QPen(QColor(ring_color))
            ring_pen.setWidth(1)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                int(-outer_R * scale),
                int(-outer_R * scale),
                int(2 * outer_R * scale),
                int(2 * outer_R * scale),
            )
            painter.drawEllipse(
                int(-inner_R * scale),
                int(-inner_R * scale),
                int(2 * inner_R * scale),
                int(2 * inner_R * scale),
            )

            # Wires of that layer (optional for sleeve-only layers)
            coords = L.get("coords", np.empty((0, 2)))
            radii = L.get("radii", np.array([]))
            colors = L.get("colors", [])
            for (x, y), r, col in zip(coords, radii, colors):
                painter.setPen(QPen(QColor(col)))
                painter.setBrush(QBrush(QColor(col)))
                painter.drawEllipse(
                    int((x - r) * scale),
                    int((y - r) * scale),
                    int(2 * r * scale),
                    int(2 * r * scale),
                )

        # Current inner exclusion ring
        if self.inner_exclusion_radius > 0:
            core_pen = QPen(QColor("#555555"))
            core_pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(core_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            R_in = self.inner_exclusion_radius
            painter.drawEllipse(
                int(-R_in * scale),
                int(-R_in * scale),
                int(2 * R_in * scale),
                int(2 * R_in * scale),
            )

        # Current outer boundary (dashed)
        if self.outer_radius > 0:
            outer_pen = QPen(QColor("gray"))
            outer_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(outer_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                int(-self.outer_radius * scale),
                int(-self.outer_radius * scale),
                int(2 * self.outer_radius * scale),
                int(2 * self.outer_radius * scale),
            )

        # Current wires
        for (x, y), r, color in zip(self.positions, self.radii, self.colors):
            painter.setPen(QPen(QColor(color)))
            painter.setBrush(QBrush(QColor(color)))
            painter.drawEllipse(
                int((x - r) * scale),
                int((y - r) * scale),
                int(2 * r * scale),
                int(2 * r * scale),
            )


class WireBundleApp(QWidget):
    """
    Main GUI application for defining wire types, optimizing layout, and
    layering shields.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Wire Bundle Optimizer")

        # Current working wire defs: list of tuples (count, diameter_mm, color, label)
        self.wire_defs: List[tuple[int, float, str, str]] = []

        # Record of previous layers (shielded cores)
        self.layers: List[Dict[str, Any]] = []
        self.frozen_core_radius: float = 0.0  # Inner exclusion radius for next runs

        # Last optimization result (for promoting to a shielded layer)
        self._last_coords: np.ndarray | None = None
        self._last_radii: np.ndarray | None = None
        self._last_R: float | None = None
        self._last_colors: List[str] | None = None

        self.predefined_types = load_wire_types()
        self.predefined_sleeves = load_sleeve_types()
        self._setup_ui()

    def _setup_ui(self) -> None:
        # --- outer container with a scroll area so large content can be scrolled ---
        outer_layout = QVBoxLayout(self)
        outer_layout.setSpacing(0)  # tight around the scroll area

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()  # actual content widget inside the scroll area
        layout = QVBoxLayout(content)  # original layout now belongs to 'content'
        layout.setSpacing(12)

        # Color palette (wires)
        self.color_palette = [
            "#007acc",  # blue
            "#cc0000",  # red
            "#009933",  # green
            "#ffd700",  # yellow
            "#ff8800",  # orange
            "#9933cc",  # purple
            "#a5722a",  # brown
            "#ffffff",  # white
            "#444444",  # dark grey
            "#888888",  # medium grey
        ]
        self.selected_color = self.color_palette[0]
        self.color_buttons: List[QPushButton] = []

        # ── Section 1: Define Wire Types ───────────────────────────────────────
        wire_group = QGroupBox("1. Define Wire Types")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(8)

        # Count
        self.count_input = QSpinBox()
        self.count_input.setRange(1, 999)
        self.count_input.setFixedWidth(70)
        form.addRow("Count:", self.count_input)

        # Diameter mode: custom vs standard
        size_layout = QHBoxLayout()
        size_layout.setSpacing(10)

        self.custom_radio = QRadioButton("Custom")
        self.custom_radio.toggled.connect(self._update_size_mode)
        size_layout.addWidget(self.custom_radio)

        self.diameter_input = QDoubleSpinBox()
        self.diameter_input.setRange(0.001, 1000.0)
        self.diameter_input.setDecimals(3)
        self.diameter_input.setValue(1.0)
        self.diameter_input.setFixedWidth(90)
        size_layout.addWidget(self.diameter_input)

        size_layout.addSpacing(20)

        self.predef_size = QRadioButton("Predefined Sizes")
        self.predef_size.toggled.connect(self._update_size_mode)
        size_layout.addWidget(self.predef_size)

        self.predef_selector = QComboBox()
        self.predef_selector.addItems(list(self.predefined_types.keys()))
        self.predef_selector.setEnabled(False)
        size_layout.addWidget(self.predef_selector)

        form.addRow("Wire Diameter (mm):", size_layout)
        self.custom_radio.setChecked(True)

        # Color picker
        color_layout = QHBoxLayout()
        for color in self.color_palette:
            btn = QPushButton()
            btn.setFixedSize(20, 20)
            btn.setStyleSheet(
                self._color_button_style(color, selected=(color == self.selected_color))
            )
            btn.clicked.connect(lambda _, c=color: self._set_color(c))
            self.color_buttons.append(btn)
            color_layout.addWidget(btn)
        form.addRow("Color:", color_layout)

        # Add Wire
        self.add_button = QPushButton("Add Wire")
        self.add_button.setFixedHeight(28)
        self.add_button.clicked.connect(self._add_wire)
        form.addRow("", self.add_button)

        wire_group.setLayout(form)
        layout.addWidget(wire_group)

        # ── Section 2: Optimization Parameters ─────────────────────────────────
        opt_group = QGroupBox("2. Optimization Parameters")
        opt_main_layout = QVBoxLayout()

        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(20)
        row1_layout.addWidget(
            QLabel("Number of Solver Initializations (higher = better, slower):")
        )
        self.inits_input = QSpinBox()
        self.inits_input.setRange(1, 1000)
        self.inits_input.setValue(4)
        self.inits_input.setFixedWidth(70)
        row1_layout.addWidget(self.inits_input)

        row1_layout.addWidget(QLabel("Max Solver Iterations:"))
        self.max_iter_input = QSpinBox()
        self.max_iter_input.setRange(1, 9999999)
        self.max_iter_input.setValue(10000)
        self.max_iter_input.setFixedWidth(70)
        row1_layout.addWidget(self.max_iter_input)
        opt_main_layout.addLayout(row1_layout)

        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(20)
        row2_layout.addWidget(
            QLabel("Manufacturing Tolerance Margin (extra spacing between wires):")
        )
        self.margin_input = QDoubleSpinBox()
        self.margin_input.setRange(0.0, 100.0)
        self.margin_input.setSuffix(" %")
        self.margin_input.setDecimals(1)
        self.margin_input.setValue(0.0)
        self.margin_input.setFixedWidth(80)
        row2_layout.addWidget(self.margin_input)
        opt_main_layout.addLayout(row2_layout)

        opt_group.setLayout(opt_main_layout)
        layout.addWidget(opt_group)

        # ── Section 3: Defined Wires ──────────────────────────────────────────
        section_label = QLabel("3. Defined Wires")
        section_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(section_label)
        self.wire_summary_label = QLabel("No wires added yet.")
        self.wire_summary_label.setStyleSheet("color: #555555;")
        self.wire_summary_label.setWordWrap(True)
        layout.addWidget(self.wire_summary_label)

        self.wire_list = QListWidget()
        self.wire_list.setFixedHeight(90)
        layout.addWidget(self.wire_list)

        row_remove = QHBoxLayout()
        remove_button = QPushButton("Remove Selected Wire")
        remove_button.setFixedHeight(28)
        remove_button.clicked.connect(self._remove_selected_wire)

        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setFixedHeight(28)
        clear_all_btn.setToolTip("Clear all layers, results and defined wires.")
        clear_all_btn.clicked.connect(self._clear_all)

        row_remove.addWidget(remove_button)
        row_remove.addWidget(clear_all_btn)
        layout.addLayout(row_remove)

        # ── Section 4: Sleeving ──────────────────────────────────────────────
        sleeve_group = QGroupBox("4. Sleeving")
        sleeve_form = QFormLayout()
        sleeve_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        sleeve_form.setHorizontalSpacing(20)
        sleeve_form.setVerticalSpacing(8)

        # Sleeve thickness: Custom vs Predefined
        sleeve_size_layout = QHBoxLayout()
        self.sleeve_custom_radio = QRadioButton("Custom")
        self.sleeve_custom_radio.toggled.connect(self._update_sleeve_size_mode)
        sleeve_size_layout.addWidget(self.sleeve_custom_radio)

        self.sleeve_thickness = QDoubleSpinBox()
        self.sleeve_thickness.setRange(0.01, 10000.0)
        self.sleeve_thickness.setDecimals(3)
        self.sleeve_thickness.setValue(0.1)
        self.sleeve_thickness.setFixedWidth(100)
        sleeve_size_layout.addWidget(self.sleeve_thickness)

        sleeve_size_layout.addSpacing(20)

        self.sleeve_predef_radio = QRadioButton("Predefined Sleeves")
        self.sleeve_predef_radio.toggled.connect(self._update_sleeve_size_mode)
        sleeve_size_layout.addWidget(self.sleeve_predef_radio)

        self.sleeve_predef_selector = QComboBox()
        self.sleeve_predef_selector.addItems(list(self.predefined_sleeves.keys()))
        self.sleeve_predef_selector.setEnabled(False)
        sleeve_size_layout.addWidget(self.sleeve_predef_selector)

        sleeve_form.addRow("Sleeve thickness (mm):", sleeve_size_layout)
        self.sleeve_custom_radio.setChecked(True)

        # Sleeve color picker — reuse same palette as wires
        self.sleeve_color_palette = self.color_palette
        self.selected_sleeve_color = self.sleeve_color_palette[-1]
        self.sleeve_color_buttons: List[QPushButton] = []
        sleeve_color_layout = QHBoxLayout()
        for color in self.sleeve_color_palette:
            btn = QPushButton()
            btn.setFixedSize(20, 20)
            btn.setStyleSheet(
                self._color_button_style(
                    color, selected=(color == self.selected_sleeve_color)
                )
            )
            btn.clicked.connect(lambda _, c=color: self._set_sleeve_color(c))
            self.sleeve_color_buttons.append(btn)
            sleeve_color_layout.addWidget(btn)
        sleeve_form.addRow("Sleeve color:", sleeve_color_layout)

        # Add Sleeve button
        self.add_sleeve_btn = QPushButton("Add Sleeve")
        self.add_sleeve_btn.setToolTip(
            "Add one sleeve ring around the current core. You can add multiple sleeves in sequence."
        )
        self.add_sleeve_btn.setEnabled(False)
        self.add_sleeve_btn.clicked.connect(self._add_sleeve)

        self.undo_layer_btn = QPushButton("Undo Last Layer")
        self.undo_layer_btn.setToolTip(
            "Remove the most recently added layer (wires and sleeve)."
        )
        self.undo_layer_btn.setEnabled(False)
        self.undo_layer_btn.clicked.connect(self._undo_last_layer)

        sleeve_buttons_layout = QHBoxLayout()
        sleeve_buttons_layout.setSpacing(10)
        sleeve_buttons_layout.addWidget(self.add_sleeve_btn)
        sleeve_buttons_layout.addWidget(self.undo_layer_btn)

        sleeve_buttons_container = QWidget()
        sleeve_buttons_container.setLayout(sleeve_buttons_layout)
        sleeve_form.addRow("", sleeve_buttons_container)

        sleeve_group.setLayout(sleeve_form)
        layout.addWidget(sleeve_group)

        # ── Section 5: Optimize ───────────────────────────────────────────────
        self.optimize_button = QPushButton("Optimize and Plot")
        self.optimize_button.setToolTip(
            "Run the solver to arrange the defined wires in the smallest bundle."
        )
        self.optimize_button.setFixedHeight(32)
        self.optimize_button.setEnabled(False)
        self.optimize_button.clicked.connect(self._optimize)
        layout.addWidget(self.optimize_button)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # ── Section 6: Results ────────────────────────────────────────────────
        results_group = QGroupBox("6. Results & Status")
        results_layout = QVBoxLayout()

        self.diameter_label = QLabel("")
        self.diameter_label.setStyleSheet("font-weight: bold;")
        results_layout.addWidget(self.diameter_label)

        summary_row = QHBoxLayout()
        summary_row.setSpacing(16)
        self.total_layers_label = QLabel("Layers: 0")
        self.bundle_outer_label = QLabel("Bundle outer Ø: —")
        summary_row.addWidget(self.total_layers_label)
        summary_row.addWidget(self.bundle_outer_label)
        summary_row.addStretch(1)
        results_layout.addLayout(summary_row)

        self.layer_table = QTableWidget(0, 5)
        self.layer_table.setHorizontalHeaderLabels(
            [
                "#",
                "Description",
                "Inner Ø (mm / in)",
                "Outer Ø (mm / in)",
                "Thickness (mm / in)",
            ]
        )
        self.layer_table.verticalHeader().setVisible(False)
        self.layer_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.layer_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.layer_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.layer_table.setAlternatingRowColors(True)
        self.layer_table.setShowGrid(False)
        self.layer_table.setStyleSheet(
            "QTableWidget::item { padding: 4px; } "
            "QHeaderView::section { background-color: #f0f4ff; border: none; padding: 6px; font-weight: bold; }"
        )
        self.layer_table.setFixedHeight(180)
        header = self.layer_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        results_layout.addWidget(self.layer_table)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Optimizing…")
        results_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color: #555555;")
        results_layout.addWidget(self.status_label)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.plot_widget = WirePlotWidget()
        self.plot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.plot_widget)

        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self._undo_last_layer)

        # Mount content into the scroll area and finish
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        self.setMinimumSize(640, 680)
        self._update_layer_summary()

    def _update_size_mode(self) -> None:
        is_custom = self.custom_radio.isChecked()
        self.diameter_input.setEnabled(is_custom)
        self.predef_selector.setEnabled(not is_custom)

    def _update_sleeve_size_mode(self) -> None:
        is_custom = self.sleeve_custom_radio.isChecked()
        self.sleeve_thickness.setEnabled(is_custom)
        self.sleeve_predef_selector.setEnabled(not is_custom)

    def _color_button_style(self, color: str, selected: bool = False) -> str:
        border = "2px solid black" if selected else "1px solid #444"
        return f"background-color: {color}; border: {border}; border-radius: 10px;"

    def _set_color(self, color: str) -> None:
        self.selected_color = color
        for btn, col in zip(self.color_buttons, self.color_palette):
            is_selected = col == self.selected_color
            btn.setStyleSheet(self._color_button_style(col, is_selected))

    def _set_sleeve_color(self, color: str) -> None:
        self.selected_sleeve_color = color
        for btn, col in zip(self.sleeve_color_buttons, self.sleeve_color_palette):
            is_selected = col == self.selected_sleeve_color
            btn.setStyleSheet(self._color_button_style(col, is_selected))

    def _add_wire(self) -> None:
        count = self.count_input.value()
        if self.predef_size.isChecked():
            label = self.predef_selector.currentText()
            diameter = float(self.predefined_types[label])
        else:
            diameter = self.diameter_input.value()
            label = f"{diameter:.3f} mm"
        color = self.selected_color

        # Merge with existing identical wires (same diameter & color)
        for i, (cnt, dia, col, lbl) in enumerate(self.wire_defs):
            if abs(dia - diameter) < 1e-9 and col == color:
                new_total = cnt + count
                self.wire_defs[i] = (new_total, diameter, color, label)
                self._refresh_list()
                self._set_status(
                    f"Updated {label}: {new_total} wire{'s' if new_total != 1 else ''} in this group."
                )
                return

        self.wire_defs.append((count, diameter, color, label))
        self._refresh_list()
        self._set_status(f"Added {count} wire{'s' if count != 1 else ''} of {label}.")

    def _remove_selected_wire(self) -> None:
        row = self.wire_list.currentRow()
        if row >= 0:
            count, diameter, color, label = self.wire_defs.pop(row)
            self._refresh_list()
            self._set_status(
                f"Removed {count} wire{'s' if count != 1 else ''} of {label}."
            )

    def _refresh_list(self) -> None:
        self.wire_list.clear()
        total_wires = 0
        for cnt, dia, color, label in self.wire_defs:
            total_wires += cnt
            item = QListWidgetItem(f"{cnt} x {label}")
            item.setBackground(QColor(color))
            item.setForeground(
                QColor("white") if QColor(color).lightness() < 128 else QColor("black")
            )
            self.wire_list.addItem(item)

        if self.wire_defs:
            unique_groups = len(self.wire_defs)
            group_text = "group" if unique_groups == 1 else "groups"
            wire_text = "wire" if total_wires == 1 else "wires"
            self.wire_summary_label.setText(
                f"{total_wires} {wire_text} across {unique_groups} {group_text}."
            )
        else:
            self.wire_summary_label.setText(
                "No wires added yet. Use section 1 to add them."
            )

        if hasattr(self, "optimize_button"):
            self.optimize_button.setEnabled(bool(self.wire_defs))

    def _update_diameter_label_current(self) -> None:
        """
        Update the diameter label using the most relevant current state:
        - If a fresh solution exists (_last_R), show that.
        - Else, if we have sleeve layers, show the last layer's outer_R.
        - Else, clear the label.
        """
        if self._last_R is not None:
            R = float(self._last_R)
            self.diameter_label.setText(
                f"Outer diameter: {(R*2):.3f} mm / {(R*2)/25.4:.3f} in"
            )
        elif self.layers:
            R = float(self.layers[-1]["outer_R"])
            self.diameter_label.setText(
                f"Outer diameter (sleeves): {(R*2):.3f} mm / {(R*2)/25.4:.3f} in"
            )
        else:
            self.diameter_label.setText("")

    def _optimize(self) -> None:
        radii = [d / 2.0 for cnt, d, c, l in self.wire_defs for _ in range(cnt)]
        colors = [c for cnt, d, c, l in self.wire_defs for _ in range(cnt)]
        if not radii:
            QMessageBox.warning(
                self, "Input Error", "Add at least one wire before optimizing."
            )
            self._set_status("Add wires to run the optimizer.")
            return

        optimizer = WireBundleOptimizer(
            radii=radii,
            margin=self.margin_input.value() / 100.0,
            inner_exclusion_radius=self.frozen_core_radius,
        )

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        original_text = self.optimize_button.text()
        self.optimize_button.setText("Optimizing...")
        self.optimize_button.setEnabled(False)
        self._set_status("Running optimization...")
        QApplication.processEvents()
        total_runs = max(1, self.inits_input.value())
        self.progress_bar.setRange(0, total_runs)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Optimization progress: 0/{total_runs}")
        self.progress_bar.setVisible(True)
        self.progress_bar.repaint()
        QApplication.processEvents()

        def on_progress(done: int, total: int) -> None:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
            self.progress_bar.setFormat(f"Optimization progress: {done}/{total}")
            QApplication.processEvents()

        start = perf_counter()

        try:
            coords, radii_arr, R = optimizer.solve_multi(
                n_initializations=self.inits_input.value(),
                max_iterations=self.max_iter_input.value(),
                progress_cb=on_progress,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Solver Error",
                f"Optimization failed with an unexpected error: \n\n{exc}",
            )
            self._set_status("Optimization failed. See error message.")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.optimize_button.setText(original_text)
            self.optimize_button.setEnabled(bool(self.wire_defs))
            self.progress_bar.setVisible(False)

        elapsed = perf_counter() - start
        if coords is None or not np.isfinite(R):
            QMessageBox.warning(
                self,
                "No Feasible Solution",
                "The solver could not find a feasible wire arrangement.",
            )
            self._set_status(
                "Solver finished without a feasible layout. Adjust inputs and try again."
            )
            return

        self._last_coords = coords
        self._last_radii = radii_arr
        self._last_R = R
        self._last_colors = colors

        # Update plot (include history)
        self.plot_widget.set_layers(self.layers, self.frozen_core_radius)
        self.plot_widget.update_scene(coords, radii_arr, R, colors)

        # Correct inches conversion (diameter in in)
        self.diameter_label.setText(
            f"Outer diameter: {(R*2):.3f} mm / {(R*2)/25.4:.3f} in"
        )

        # Allow adding sleeves: either fresh solution or existing layers allow it
        if hasattr(self, "add_sleeve_btn"):
            self._update_add_sleeve_button()
            self._update_undo_button()

        self._update_layer_summary()
        self._set_status(
            f"Optimization complete in {elapsed:.2f} s: {len(radii_arr)} wire{'s' if len(radii_arr) != 1 else ''}, outer Ø {(R * 2):.3f} mm."
        )

    def _update_add_sleeve_button(self) -> None:
        can_add = (self._last_R is not None) or (self.frozen_core_radius > 0.0)
        self.add_sleeve_btn.setEnabled(bool(can_add))

    def _update_undo_button(self) -> None:
        if hasattr(self, "undo_layer_btn"):
            self.undo_layer_btn.setEnabled(bool(self.layers))

    def _update_layer_summary(self) -> None:
        if not hasattr(self, "layer_table"):
            return

        total_layers = len(self.layers)
        self.total_layers_label.setText(f"Layers: {total_layers}")

        if total_layers:
            outer_R = float(self.layers[-1].get("outer_R", 0.0))
            self.bundle_outer_label.setText(f"Bundle outer Ø: {(outer_R * 2.0):.3f} mm")
        else:
            self.bundle_outer_label.setText("Bundle outer Ø: —")

        self.layer_table.setRowCount(total_layers)
        for row, layer in enumerate(self.layers):
            inner_R = float(layer.get("inner_R", 0.0))
            outer_R = float(layer.get("outer_R", 0.0))
            thickness = max(outer_R - inner_R, 0.0)
            descriptor = layer.get("sleeve_label") or "Layer"

            inner_mm = inner_R * 2.0
            outer_mm = outer_R * 2.0
            thickness_mm = thickness

            inner_in = inner_mm / 25.4
            outer_in = outer_mm / 25.4
            thickness_in = thickness_mm / 25.4

            values = [
                str(row + 1),
                descriptor,
                f"{inner_mm:.3f} mm / {inner_in:.3f} in",
                f"{outer_mm:.3f} mm / {outer_in:.3f} in",
                f"{thickness_mm:.3f} mm / {thickness_in:.3f} in",
            ]

            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 1:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    )
                self.layer_table.setItem(row, col, item)

        if total_layers == 0:
            self.layer_table.clearContents()

        self.layer_table.setVisible(total_layers > 0)

    def _set_status(self, message: str) -> None:
        if hasattr(self, "status_label"):
            self.status_label.setText(message)

    def _add_sleeve(self) -> None:
        """
        Add a single sleeve ring. If a fresh optimization result is available,
        the ring will wrap that solution and lock it as a layer. Otherwise, the
        ring will be added on top of the existing frozen core.
        """
        # Determine thickness from UI
        if self.sleeve_custom_radio.isChecked():
            thickness = float(self.sleeve_thickness.value())
            sleeve_label = f"Custom {thickness:.3f} mm"
        else:
            sleeve_label = self.sleeve_predef_selector.currentText()
            try:
                thickness = float(self.predefined_sleeves[sleeve_label])
            except Exception:
                QMessageBox.warning(
                    self, "Invalid Selection", "Invalid predefined sleeve."
                )
                return
        if thickness <= 0:
            QMessageBox.warning(
                self, "Invalid Thickness", "Please set a positive sleeve thickness."
            )
            return

        ring_color = self.selected_sleeve_color

        # Decide inner radius
        if self._last_R is not None:
            inner_R = float(self._last_R)
            # Save layer with wires
            outer_R = inner_R + thickness
            self.layers.append(
                {
                    "coords": np.array(self._last_coords, dtype=float).copy(),
                    "radii": np.array(self._last_radii, dtype=float).copy(),
                    "colors": list(self._last_colors),
                    "inner_R": inner_R,
                    "outer_R": outer_R,
                    "ring_color": ring_color,
                    "sleeve_label": sleeve_label,
                    "wire_defs": [tuple(entry) for entry in self.wire_defs],
                }
            )

            # Clear last solution and working wires (prepare for next ring)
            self.frozen_core_radius = outer_R
            self.wire_defs.clear()
            self._refresh_list()

            self._last_coords = None
            self._last_radii = None
            self._last_R = None
            self._last_colors = None
        elif self.frozen_core_radius > 0.0:
            inner_R = float(self.frozen_core_radius)
            outer_R = inner_R + thickness
            # Sleeve-only layer
            self.layers.append(
                {
                    "coords": np.empty((0, 2)),
                    "radii": np.array([]),
                    "colors": [],
                    "inner_R": inner_R,
                    "outer_R": outer_R,
                    "ring_color": ring_color,
                    "sleeve_label": sleeve_label,
                    "wire_defs": [tuple(entry) for entry in self.wire_defs],
                }
            )
            self.frozen_core_radius = outer_R
        else:
            QMessageBox.information(
                self,
                "No Core",
                "Optimize first to create a core before adding sleeves.",
            )
            return

        # Update plot to show layers (no current solution yet)
        self.plot_widget.set_layers(self.layers, self.frozen_core_radius)
        self.plot_widget.update_scene(np.empty((0, 2)), np.array([]), 0.0, [])
        self._update_diameter_label_current()
        self._update_add_sleeve_button()
        self._update_undo_button()
        self._update_layer_summary()
        self._set_status(
            f"Added layer '{sleeve_label}' (thickness {thickness:.3f} mm)."
        )

    def _undo_last_layer(self) -> None:
        """Remove the most recently added layer and restore prior state."""
        if not self.layers:
            return

        removed_layer = self.layers.pop()
        self.frozen_core_radius = (
            float(self.layers[-1]["outer_R"]) if self.layers else 0.0
        )

        # Restore historical layers in the plot first
        self.plot_widget.set_layers(self.layers, self.frozen_core_radius)

        coords = removed_layer.get("coords")
        radii = removed_layer.get("radii")
        colors = removed_layer.get("colors")

        if coords is not None and len(coords):
            self._last_coords = np.array(coords, dtype=float)
            self._last_radii = np.array(radii, dtype=float)
            self._last_R = float(removed_layer.get("inner_R", 0.0))
            self._last_colors = list(colors) if colors is not None else []

            self.plot_widget.update_scene(
                self._last_coords,
                self._last_radii,
                self._last_R,
                self._last_colors,
            )

            saved_defs = removed_layer.get("wire_defs") or []
            if saved_defs and not self.wire_defs:
                self.wire_defs = [tuple(entry) for entry in saved_defs]
                self._refresh_list()
        else:
            self._last_coords = None
            self._last_radii = None
            self._last_R = None
            self._last_colors = None
            self.plot_widget.update_scene(np.empty((0, 2)), np.array([]), 0.0, [])

        descriptor = removed_layer.get("sleeve_label") or "layer"
        self._update_diameter_label_current()
        self._update_add_sleeve_button()
        self._update_undo_button()
        self._update_layer_summary()
        self._set_status(f"Undo: removed {descriptor}.")

    def _clear_all(self) -> None:
        """
        Clear EVERYTHING: historical layers, frozen core, current wires,
        and any pending optimization results.
        """
        self.layers.clear()
        self.frozen_core_radius = 0.0
        self.wire_defs.clear()
        self._refresh_list()

        # Reset any last solution and disable actions that require it
        self._last_coords = None
        self._last_radii = None
        self._last_R = None
        self._last_colors = None
        self._update_add_sleeve_button()
        self._update_undo_button()
        self._update_layer_summary()
        self._set_status("Cleared all layers and wires.")

        # Refresh plot to empty
        self.plot_widget.set_layers(self.layers, self.frozen_core_radius)
        self.plot_widget.update_scene(np.empty((0, 2)), np.array([]), 0.0, [])
        self._update_diameter_label_current()

    @staticmethod
    def _app_stylesheet() -> str:
        return """
        QWidget {
            font-family: Arial, sans-serif;
            font-size: 12px;
        }
        QPushButton {
            padding: 4px 10px;
            background-color: #447acc;
            color: white;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #2f5ea8;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
            border: 1px solid #999999;
        }
        QListWidget {
            border: 1px solid #bbb;
            padding: 3px;
        }
        """


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(WireBundleApp._app_stylesheet())
    window = WireBundleApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
