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
        super().__init__(parent)
        self.positions: np.ndarray = np.array([])
        self.radii: np.ndarray = np.array([])
        self.outer_radius: float = 0.0
        self.colors: List[str] = []
        self.setMinimumSize(300, 300)

    def update_data(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        outer_radius: float,
        colors: List[str],
    ) -> None:
        self.positions = positions
        self.radii = radii
        self.outer_radius = outer_radius
        self.colors = colors
        self.update()

    def paintEvent(self, event) -> None:
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
        super().__init__()
        self.setWindowTitle("Wire Bundle Optimizer")
        self.wire_defs: List[Tuple[int, float, str]] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # === Wire input section ===
        wire_layout = QGridLayout()
        wire_layout.setHorizontalSpacing(8)

        self.count_input = QSpinBox()
        self.count_input.setMinimum(1)
        self.count_input.setFixedWidth(70)

        self.diameter_input = QDoubleSpinBox()
        self.diameter_input.setMinimum(0.01)
        self.diameter_input.setDecimals(3)
        self.diameter_input.setValue(1.0)
        self.diameter_input.setFixedWidth(90)

        # --- Improved Color Palette ---
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

        for color in self.color_palette:
            btn = QPushButton()
            btn.setStyleSheet(
                self._color_button_style(color, selected=(color == self.selected_color))
            )
            btn.clicked.connect(lambda _, c=color: self._set_color(c))
            self.color_buttons.append(btn)
            self.color_picker_layout.addWidget(btn)

        self.add_button = QPushButton("Add Wire")
        self.add_button.setFixedHeight(28)
        self.add_button.clicked.connect(self._add_wire)

        wire_layout.addWidget(QLabel("Count:"), 0, 0)
        wire_layout.addWidget(self.count_input, 0, 1)
        wire_layout.addWidget(QLabel("Diameter:"), 0, 2)
        wire_layout.addWidget(self.diameter_input, 0, 3)
        wire_layout.addLayout(self.color_picker_layout, 0, 4, 1, 3)
        wire_layout.addWidget(self.add_button, 0, 7)

        layout.addLayout(wire_layout)

        # === Separator ===
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # === Optimization Parameters ===
        opt_layout = QGridLayout()
        opt_layout.setHorizontalSpacing(8)

        self.inits_input = QSpinBox()
        self.inits_input.setMinimum(1)
        self.inits_input.setValue(8)
        self.inits_input.setFixedWidth(70)

        self.max_iter_input = QSpinBox()
        self.max_iter_input.setMinimum(1)
        self.max_iter_input.setValue(200)
        self.max_iter_input.setFixedWidth(70)

        opt_layout.addWidget(QLabel("Initializations:"), 0, 0)
        opt_layout.addWidget(self.inits_input, 0, 1)
        opt_layout.addWidget(QLabel("Max Iterations:"), 0, 2)
        opt_layout.addWidget(self.max_iter_input, 0, 3)

        layout.addLayout(opt_layout)

        # === Wire list + controls ===
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

        self.diameter_label = QLabel("")
        layout.addWidget(self.diameter_label)

        # === Plot area ===
        self.plot_widget = WirePlotWidget()
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)
        self.setMinimumSize(500, 580)

    def _color_button_style(self, color: str, selected: bool = False) -> str:
        """
        Return style string for a color button.
        """
        border = "2px solid black" if selected else "1px solid #444"
        size = "26px" if selected else "22px"
        return (
            f"background-color: {color}; "
            f"border: {border}; "
            f"border-radius: 4px;"
            f"min-width: {size}; min-height: {size}; max-width: {size}; max-height: {size};"
        )

    def _set_color(self, color: str) -> None:
        """
        Set the selected wire color and update button styles.
        """
        self.selected_color = color
        for btn, col in zip(self.color_buttons, self.color_palette):
            is_selected = col == self.selected_color
            btn.setStyleSheet(self._color_button_style(col, is_selected))

    def _add_wire(self) -> None:
        count = self.count_input.value()
        diameter = self.diameter_input.value()
        color = self.selected_color

        for i, (cnt, dia, col) in enumerate(self.wire_defs):
            if abs(dia - diameter) < 1e-6 and col == color:
                self.wire_defs[i] = (cnt + count, diameter, color)
                self._refresh_list()
                return

        self.wire_defs.append((count, diameter, color))
        self._refresh_list()

    def _remove_selected_wire(self) -> None:
        row = self.wire_list.currentRow()
        if row >= 0:
            del self.wire_defs[row]
            self._refresh_list()

    def _refresh_list(self) -> None:
        self.wire_list.clear()
        for cnt, dia, color in self.wire_defs:
            item = QListWidgetItem(f"{cnt} x {dia:.3f} mm")
            item.setBackground(QColor(color))
            item.setForeground(
                QColor("white") if QColor(color).lightness() < 128 else QColor("black")
            )
            self.wire_list.addItem(item)

    def _optimize(self) -> None:
        radii: List[float] = [
            d / 2.0 for cnt, d, _ in self.wire_defs for _ in range(cnt)
        ]
        colors: List[str] = [c for cnt, _, c in self.wire_defs for _ in range(cnt)]

        if not radii:
            QMessageBox.warning(self, "Input Error", "No wires defined.")
            return

        optimizer = WireBundleOptimizer(radii)
        coords, radii_arr, R = optimizer.solve_multi(
            n_initializations=self.inits_input.value(),
            max_iterations=self.max_iter_input.value(),
            n_jobs=-1,
        )

        self.plot_widget.update_data(coords, radii_arr, R, colors)
        self.diameter_label.setText(f"Outer diameter: {R:.3f} mm / {R / 25.4:.3f} in")


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
