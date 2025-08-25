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

from joblib import parallel_backend

parallel_backend("threading")  # Fixes a Windows devices bug for joblib

import sys
import yaml
import numpy as np
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
)
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush
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

        # Draw historical layers (shield rings + their wires)
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
            painter.setBrush(QColor(136, 136, 136, 90))  # soft grey
            painter.drawPath(ring_path)

            # ring outline
            ring_pen = QPen(QColor("#888888"))
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

            # Wires of that layer
            coords = L["coords"]
            radii = L["radii"]
            colors = L["colors"]
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

        # Color palette
        self.color_palette = [
            "#007acc",
            "#cc0000",
            "#009933",
            "#ff8800",
            "#9933cc",
            "#444444",
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
        self.inits_input.setValue(8)
        self.inits_input.setFixedWidth(70)
        row1_layout.addWidget(self.inits_input)

        row1_layout.addWidget(QLabel("Max Solver Iterations:"))
        self.max_iter_input = QSpinBox()
        self.max_iter_input.setRange(1, 10000)
        self.max_iter_input.setValue(2000)
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
        layout.addWidget(QLabel("3. Defined Wires"))
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

        # ── Section 4: Shielding ──────────────────────────────────────────────
        shield_group = QGroupBox(
            "4. Shielding (optimize a bundle before adding shielding)"
        )
        shield_form = QFormLayout()
        shield_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        shield_form.setHorizontalSpacing(20)
        shield_form.setVerticalSpacing(8)

        self.shield_thickness = QDoubleSpinBox()
        self.shield_thickness.setRange(0.01, 10000.0)
        self.shield_thickness.setDecimals(3)
        self.shield_thickness.setValue(0.1)
        self.shield_thickness.setFixedWidth(100)

        self.add_shield_btn = QPushButton("Add Shielding")
        self.add_shield_btn.setToolTip(
            "Promote the current optimized bundle to a shielded core."
        )
        self.add_shield_btn.setEnabled(False)  # visually disabled via stylesheet
        self.add_shield_btn.clicked.connect(self._add_shielding)

        shield_form.addRow("Shield thickness (mm):", self.shield_thickness)
        shield_form.addRow("", self.add_shield_btn)
        shield_group.setLayout(shield_form)
        layout.addWidget(shield_group)

        # ── Section 5: Optimize ───────────────────────────────────────────────
        self.optimize_button = QPushButton("Optimize and Plot")
        self.optimize_button.setFixedHeight(32)
        self.optimize_button.clicked.connect(self._optimize)
        layout.addWidget(self.optimize_button)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # ── Section 6: Results ────────────────────────────────────────────────
        layout.addWidget(QLabel("<b>6. Results</b>"))
        self.diameter_label = QLabel("")
        layout.addWidget(self.diameter_label)

        self.plot_widget = WirePlotWidget()
        self.plot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.plot_widget)

        # Mount content into the scroll area and finish
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        self.setMinimumSize(600, 640)

    def _update_size_mode(self) -> None:
        is_custom = self.custom_radio.isChecked()
        self.diameter_input.setEnabled(is_custom)
        self.predef_selector.setEnabled(not is_custom)

    def _color_button_style(self, color: str, selected: bool = False) -> str:
        border = "2px solid black" if selected else "1px solid #444"
        return f"background-color: {color}; border: {border}; border-radius: 10px;"

    def _set_color(self, color: str) -> None:
        self.selected_color = color
        for btn, col in zip(self.color_buttons, self.color_palette):
            is_selected = col == self.selected_color
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
                self.wire_defs[i] = (cnt + count, diameter, color, label)
                self._refresh_list()
                return

        self.wire_defs.append((count, diameter, color, label))
        self._refresh_list()

    def _remove_selected_wire(self) -> None:
        row = self.wire_list.currentRow()
        if row >= 0:
            del self.wire_defs[row]
            self._refresh_list()

    def _refresh_list(self) -> None:
        self.wire_list.clear()
        for cnt, dia, color, label in self.wire_defs:
            item = QListWidgetItem(f"{cnt} x {label}")
            item.setBackground(QColor(color))
            item.setForeground(
                QColor("white") if QColor(color).lightness() < 128 else QColor("black")
            )
            self.wire_list.addItem(item)

    def _update_diameter_label_current(self) -> None:
        """
        Update the diameter label using the most relevant current state:
        - If a fresh solution exists (_last_R), show that.
        - Else, if we have shield layers, show the last layer's outer_R.
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
                f"Outer diameter (shields): {(R*2):.3f} mm / {(R*2)/25.4:.3f} in"
            )
        else:
            self.diameter_label.setText("")

    def _optimize(self) -> None:
        radii = [d / 2.0 for cnt, d, c, l in self.wire_defs for _ in range(cnt)]
        colors = [c for cnt, d, c, l in self.wire_defs for _ in range(cnt)]
        if not radii:
            QMessageBox.warning(self, "Input Error", "No wires defined.")
            return

        optimizer = WireBundleOptimizer(
            radii=radii,
            margin=self.margin_input.value() / 100.0,
            inner_exclusion_radius=self.frozen_core_radius,
        )

        coords, radii_arr, R = optimizer.solve_multi(
            n_initializations=self.inits_input.value(),
            max_iterations=self.max_iter_input.value(),
            n_jobs=-1,
        )

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

        # Allow adding a shield layer based on this result
        self.add_shield_btn.setEnabled(True)

    def _add_shielding(self) -> None:
        """Promote the current optimized bundle to a shielded layer and clear wires."""
        if (
            self._last_coords is None
            or self._last_radii is None
            or self._last_R is None
            or self._last_colors is None
        ):
            QMessageBox.information(
                self, "No Solution", "Optimize first, then add shielding."
            )
            return

        thickness = self.shield_thickness.value()
        if thickness <= 0:
            QMessageBox.warning(
                self, "Invalid Thickness", "Please set a positive shield thickness."
            )
            return

        inner_R = float(self._last_R)
        outer_R = inner_R + thickness

        # Save the solved layer
        self.layers.append(
            {
                "coords": np.array(self._last_coords, dtype=float).copy(),
                "radii": np.array(self._last_radii, dtype=float).copy(),
                "colors": list(self._last_colors),
                "inner_R": inner_R,
                "outer_R": outer_R,
            }
        )

        # Freeze this core for next runs and clear current wires for the next ring
        self.frozen_core_radius = outer_R
        self.wire_defs.clear()
        self._refresh_list()

        # Reset last solution (you must define the next wires and optimize again)
        self._last_coords = None
        self._last_radii = None
        self._last_R = None
        self._last_colors = None
        self.add_shield_btn.setEnabled(False)

        # Update plot to show layers (no current solution yet)
        self.plot_widget.set_layers(self.layers, self.frozen_core_radius)
        self.plot_widget.update_scene(np.empty((0, 2)), np.array([]), 0.0, [])
        self._update_diameter_label_current()

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
        self.add_shield_btn.setEnabled(False)

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
