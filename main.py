import sys
from typing import List, Tuple
import numpy as np
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
    QGridLayout,
    QFrame,
    QListWidgetItem,
)
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtCore import Qt
from optimizer import WireBundleOptimizer


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
        self.wire_defs = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface for the wire bundle optimizer."""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # === Wire input section ===
        layout.addWidget(QLabel("<b>1. Define Wire Types</b>"))

        wire_layout = QGridLayout()
        wire_layout.setHorizontalSpacing(8)

        # Wire input controls
        self.count_input = QSpinBox()
        self.count_input.setMinimum(1)
        self.count_input.setMaximum(999)
        self.count_input.setFixedWidth(70)
        self.diameter_input = QDoubleSpinBox()
        self.diameter_input.setMinimum(0.01)
        self.diameter_input.setMaximum(1000.0)
        self.diameter_input.setDecimals(3)
        self.diameter_input.setValue(1.0)
        self.diameter_input.setFixedWidth(90)

        # Color selection
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
        self.color_picker_layout = QHBoxLayout()
        self.color_picker_layout.setSpacing(6)
        self.color_picker_layout.addWidget(QLabel("Color:"))

        # Create color buttons
        for color in self.color_palette:
            btn = QPushButton()
            btn.setStyleSheet(
                self._color_button_style(color, selected=(color == self.selected_color))
            )
            btn.setFixedSize(20, 20)
            btn.clicked.connect(lambda _, c=color: self._set_color(c))
            self.color_buttons.append(btn)
            self.color_picker_layout.addWidget(btn)
        self.add_button = QPushButton("Add Wire")
        self.add_button.setFixedHeight(28)
        self.add_button.clicked.connect(self._add_wire)

        # Wire input layout
        count_layout = QHBoxLayout()
        count_layout.setSpacing(4)
        count_label = QLabel("Count:")
        count_layout.addWidget(count_label)
        count_layout.addWidget(self.count_input)
        diam_layout = QHBoxLayout()
        diam_layout.setSpacing(4)
        diam_label = QLabel("Diameter (mm):")
        diam_layout.addWidget(diam_label)
        diam_layout.addWidget(self.diameter_input)

        # Add wire input controls to the grid layout
        wire_layout.addLayout(count_layout, 0, 0, 1, 2)
        wire_layout.addLayout(diam_layout, 0, 2, 1, 2)
        wire_layout.addLayout(self.color_picker_layout, 0, 4, 1, 3)
        wire_layout.addWidget(self.add_button, 0, 7)

        layout.addLayout(wire_layout)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # === Optimization Parameters ===
        layout.addWidget(QLabel("<b>2. Optimization Parameters</b>"))
        opt_layout = QGridLayout()
        opt_layout.setHorizontalSpacing(8)

        self.inits_input = QSpinBox()
        self.inits_input.setMinimum(1)
        self.inits_input.setValue(8)
        self.inits_input.setFixedWidth(70)

        self.max_iter_input = QSpinBox()
        self.max_iter_input.setMinimum(1)
        self.max_iter_input.setMaximum(10000)
        self.max_iter_input.setValue(2000)
        self.max_iter_input.setFixedWidth(70)

        opt_layout.addWidget(
            QLabel("Solver Initializations (better results, more runtime):"),
            0,
            0,
        )
        opt_layout.addWidget(self.inits_input, 0, 1)
        opt_layout.addWidget(QLabel("Max Solver Iterations:"), 0, 2)
        opt_layout.addWidget(self.max_iter_input, 0, 3)

        layout.addLayout(opt_layout)

        # === Wire list + controls ===
        layout.addWidget(QLabel("3. Defined Wires"))
        self.wire_list = QListWidget()
        self.wire_list.setFixedHeight(80)
        layout.addWidget(self.wire_list)

        remove_button = QPushButton("Remove Selected")
        remove_button.setFixedHeight(28)
        remove_button.clicked.connect(self._remove_selected_wire)
        layout.addWidget(remove_button)

        # === Optimize and result ===
        self.optimize_button = QPushButton("Optimize and Plot")
        self.optimize_button.setFixedHeight(32)
        self.optimize_button.clicked.connect(self._optimize)
        layout.addWidget(self.optimize_button)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        layout.addWidget(QLabel("<b>4. Results </b>"))
        self.diameter_label = QLabel("")
        layout.addWidget(self.diameter_label)

        # === Plot area ===
        self.plot_widget = WirePlotWidget()
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)
        self.setMinimumSize(500, 580)

    def _color_button_style(self, color: str, selected: bool = False) -> str:
        """
        Generate the style for color buttons.

        Parameters:
            color (str): Color code for the button.
            selected (bool): Whether this color is currently selected.

        Returns:
            str: CSS style string for the button.
        """
        border = "2px solid black" if selected else "1px solid #444"
        return (
            f"background-color: {color}; " f"border: {border}; " f"border-radius: 10px;"
        )

    def _set_color(self, color: str) -> None:
        """
        Set the selected wire color and update button styles.

        Parameters:
            color (str): Color code to set as selected.
        """
        self.selected_color = color
        for btn, col in zip(self.color_buttons, self.color_palette):
            is_selected = col == self.selected_color
            btn.setStyleSheet(self._color_button_style(col, is_selected))

    def _add_wire(self) -> None:
        """Add a new wire definition based on user input."""
        count = self.count_input.value()
        diameter = self.diameter_input.value()
        color = self.selected_color

        # Look for existing wire definitions with the same diameter and color
        for i, (cnt, dia, col) in enumerate(self.wire_defs):
            if abs(dia - diameter) < 1e-6 and col == color:
                # If found, just update the count
                self.wire_defs[i] = (cnt + count, diameter, color)
                self._refresh_list()
                return

        self.wire_defs.append((count, diameter, color))
        self._refresh_list()

    def _remove_selected_wire(self) -> None:
        """Remove the currently selected wire definition from the list."""
        row = self.wire_list.currentRow()
        if row >= 0:
            del self.wire_defs[row]
            self._refresh_list()

    def _refresh_list(self) -> None:
        """Refresh the wire list display."""
        self.wire_list.clear()

        for cnt, dia, color in self.wire_defs:
            item = QListWidgetItem(f"{cnt} x {dia:.3f} mm")
            item.setBackground(QColor(color))
            item.setForeground(
                QColor("white") if QColor(color).lightness() < 128 else QColor("black")
            )
            self.wire_list.addItem(item)

    def _optimize(self) -> None:
        """Run the optimization and update the plot with results."""
        radii = [d / 2 for cnt, d, _ in self.wire_defs for _ in range(cnt)]
        colors = [c for cnt, _, c in self.wire_defs for _ in range(cnt)]

        if not radii:
            QMessageBox.warning(self, "Input Error", "No wires defined.")
            return

        # Launch the optimizer
        optimizer = WireBundleOptimizer(radii)
        coords, radii_arr, R = optimizer.solve_multi(
            n_initializations=self.inits_input.value(),
            max_iterations=self.max_iter_input.value(),
            n_jobs=-1,
        )

        # Update the plot with the results
        self.plot_widget.update_data(coords, radii_arr, R, colors)
        self.diameter_label.setText(
            f"Outer diameter: {(R*2):.3f} mm / {R / 25.4:.3f} in"
        )


def main() -> None:
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
