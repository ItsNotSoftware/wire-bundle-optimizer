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

# This fixes multiple windows spawning issues on Windows when using multiprocessing.
from multiprocessing import freeze_support

freeze_support()
####

import sys
import yaml
import numpy as np
from typing import List
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
)
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtCore import Qt
from optimizer import WireBundleOptimizer


def load_wire_types(filepath: str = "wire_types.yaml") -> dict:
    """
    Load predefined wire types from a YAML file.

    Parameters:
        filepath (str): Path to the YAML file containing wire definitions.

    Returns:
        dict: Dictionary mapping wire type names to diameters in mm.
    """
    try:
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        QMessageBox.warning(
            None,
            "File Not Found",
            f"Wire types file '{filepath}' not found. Please create it to use predefined sizes.",
        )
        return {}


class WirePlotWidget(QWidget):
    """
    QWidget that visualizes the wire bundle layout.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the plot widget.

        Parameters:
            parent (QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self.positions = np.array([])
        self.radii = np.array([])
        self.outer_radius = 0.0
        self.colors = []
        self.setMinimumSize(300, 300)

    def update_data(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        outer_radius: float,
        colors: List[str],
    ) -> None:
        """
        Update the plot data.

        Parameters:
            positions (np.ndarray): Wire positions as an Nx2 array.
            radii (np.ndarray): Wire radii.
            outer_radius (float): Outer radius of the bundle.
            colors (List[str]): List of colors for each wire.
        """
        self.positions = positions
        self.radii = radii
        self.outer_radius = outer_radius
        self.colors = colors
        self.update()

    def paintEvent(self, event: any) -> None:
        """
        Paint event to draw the wires and outer boundary.

        Parameters:
            event: Paint event.
        """
        if self.positions.size == 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Determine scale based on widget size
        w, h = self.width(), self.height()
        scale = min(w, h) / (2 * (self.outer_radius + np.max(self.radii)))
        painter.translate(w / 2, h / 2)

        # Draw outer boundary
        outer_pen = QPen(QColor("gray"))
        outer_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(outer_pen)
        painter.drawEllipse(
            int(-self.outer_radius * scale),
            int(-self.outer_radius * scale),
            int(self.outer_radius * 2 * scale),
            int(self.outer_radius * 2 * scale),
        )

        # Draw wires
        for (x, y), r, color in zip(self.positions, self.radii, self.colors):
            painter.setPen(QPen(QColor(color)))
            painter.setBrush(QColor(color))
            painter.drawEllipse(
                int((x - r) * scale),
                int((y - r) * scale),
                int(r * 2 * scale),
                int(r * 2 * scale),
            )


class WireBundleApp(QWidget):
    """
    Main GUI application for defining wire types and optimizing layout.
    """

    def __init__(self) -> None:
        """Initialize the main application window."""
        super().__init__()
        self.setWindowTitle("Wire Bundle Optimizer")
        self.wire_defs = []  # Each wire is (count, diameter, color, label)
        self.predefined_types = load_wire_types()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface for the wire bundle optimizer."""
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # Prepare color palette attributes
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

        # ─── Section 1: Define Wire Types ────────────────────────────────────────
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

        # Custom diameter
        self.custom_radio = QRadioButton("Custom")
        self.custom_radio.toggled.connect(self._update_size_mode)
        size_layout.addWidget(self.custom_radio)

        self.diameter_input = QDoubleSpinBox()
        self.diameter_input.setRange(0.01, 1000.0)
        self.diameter_input.setDecimals(3)
        self.diameter_input.setValue(1.0)
        self.diameter_input.setFixedWidth(90)
        size_layout.addWidget(self.diameter_input)

        size_layout.addSpacing(20)

        # Predefined size
        self.predef_size = QRadioButton("Predefined Sizes")
        self.predef_size.toggled.connect(self._update_size_mode)
        size_layout.addWidget(self.predef_size)

        self.predef_selector = QComboBox()
        self.predef_selector.addItems(self.predefined_types.keys())
        self.predef_selector.setEnabled(False)
        size_layout.addWidget(self.predef_selector)

        form.addRow("Wire Diameter (mm):", size_layout)

        # Default to custom
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

        # Add Wire button
        self.add_button = QPushButton("Add Wire")
        self.add_button.setFixedHeight(28)
        self.add_button.clicked.connect(self._add_wire)
        form.addRow("", self.add_button)

        wire_group.setLayout(form)
        layout.addWidget(wire_group)

        # ─── Section 2: Optimization Parameters ─────────────────────────────────
        # ─── Section 2: Optimization Parameters ─────────────────────────────────
        opt_group = QGroupBox("2. Optimization Parameters")
        opt_main_layout = QVBoxLayout()

        # First row: initializations + max iterations
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
            QLabel(
                "Manufacturing Tolerance Margin (Extra spacing added between wires to allow for manufacturing tolerances):"
            )
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

        # ─── Section 3: Defined Wires ─────────────────────────────────────────
        layout.addWidget(QLabel("3. Defined Wires"))
        self.wire_list = QListWidget()
        self.wire_list.setFixedHeight(90)
        layout.addWidget(self.wire_list)

        remove_button = QPushButton("Remove Selected")
        remove_button.setFixedHeight(28)
        remove_button.clicked.connect(self._remove_selected_wire)
        layout.addWidget(remove_button)

        # ─── Section 4: Optimize & Results ────────────────────────────────────
        self.optimize_button = QPushButton("Optimize and Plot")
        self.optimize_button.setFixedHeight(32)
        self.optimize_button.clicked.connect(self._optimize)
        layout.addWidget(self.optimize_button)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        layout.addWidget(QLabel("<b>4. Results</b>"))
        self.diameter_label = QLabel("")
        layout.addWidget(self.diameter_label)

        self.plot_widget = WirePlotWidget()
        layout.addWidget(self.plot_widget)

        # Finalize
        self.setLayout(layout)
        self.setMinimumSize(550, 600)

    def _update_size_mode(self) -> None:
        """Enable the appropriate diameter input based on radio selection."""
        is_custom = self.custom_radio.isChecked()
        self.diameter_input.setEnabled(is_custom)
        self.predef_selector.setEnabled(not is_custom)

    def _color_button_style(self, color: str, selected: bool = False) -> str:
        """Return button style with optional highlight."""
        border = "2px solid black" if selected else "1px solid #444"
        return f"background-color: {color}; border: {border}; border-radius: 10px;"

    def _set_color(self, color: str) -> None:
        """Set the currently selected color and update UI."""
        self.selected_color = color
        for btn, col in zip(self.color_buttons, self.color_palette):
            is_selected = col == self.selected_color
            btn.setStyleSheet(self._color_button_style(col, is_selected))

    def _add_wire(self) -> None:
        """Add a new wire group based on UI input."""
        count = self.count_input.value()
        if self.predef_size.isChecked():
            label = self.predef_selector.currentText()
            diameter = self.predefined_types[label]
        else:
            diameter = self.diameter_input.value()
            label = f"{diameter:.3f} mm"
        color = self.selected_color

        # Merge with existing identical wires
        for i, (cnt, dia, col, lbl) in enumerate(self.wire_defs):
            if abs(dia - diameter) < 1e-6 and col == color:
                self.wire_defs[i] = (cnt + count, diameter, color, label)
                self._refresh_list()
                return

        self.wire_defs.append((count, diameter, color, label))
        self._refresh_list()

    def _remove_selected_wire(self) -> None:
        """Remove currently selected wire group."""
        row = self.wire_list.currentRow()
        if row >= 0:
            del self.wire_defs[row]
            self._refresh_list()

    def _refresh_list(self) -> None:
        """Update the visual list of wires."""
        self.wire_list.clear()
        for cnt, dia, color, label in self.wire_defs:
            item = QListWidgetItem(f"{cnt} x {label}")
            item.setBackground(QColor(color))
            item.setForeground(
                QColor("white") if QColor(color).lightness() < 128 else QColor("black")
            )
            self.wire_list.addItem(item)

    def _optimize(self) -> None:
        """Run layout optimization and update the plot."""
        radii = [d / 2 for cnt, d, c, l in self.wire_defs for _ in range(cnt)]
        colors = [c for cnt, d, c, l in self.wire_defs for _ in range(cnt)]
        if not radii:
            QMessageBox.warning(self, "Input Error", "No wires defined.")
            return

        optimizer = WireBundleOptimizer(radii, margin=self.margin_input.value() / 100.0)
        coords, radii_arr, R = optimizer.solve_multi(
            n_initializations=self.inits_input.value(),
            max_iterations=self.max_iter_input.value(),
            n_jobs=-1,
        )
        self.plot_widget.update_data(coords, radii_arr, R, colors)
        self.diameter_label.setText(
            f"Outer diameter: {(R*2):.3f} mm / {R / 25.4:.3f} in"
        )


def main() -> None:
    """Run the PyQt6 application."""
    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
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
        QListWidget {
            border: 1px solid #bbb;
            padding: 3px;
        }
    """
    )
    window = WireBundleApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
