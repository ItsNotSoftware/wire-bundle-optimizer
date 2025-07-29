import numpy as np
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg
from optimizer import WireBundleOptimizer
from typing import List, Tuple, Any


class WireBundleApp:

    def __init__(self) -> None:
        self.wire_defs: List[Tuple[int, float]] = []
        self.scale_factor: float = 1.4  # default zoom
        self._setup_gui()

    def _setup_gui(self) -> None:
        """Initialize the GUI components using DearPyGui."""

        dpg.create_context()
        dpg.set_global_font_scale(self.scale_factor)
        dpg.create_viewport(title="Wire Bundle Optimizer", width=800, height=850)

        # Theme
        with dpg.theme() as self.theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(
                    dpg.mvThemeCol_WindowBg, (25, 25, 28), category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_TitleBg, (35, 35, 38), category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button, (45, 115, 245), category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    (70, 140, 255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_FrameBgHovered,
                    (60, 60, 70),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_SliderGrab,
                    (100, 200, 250),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 10, 6, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_ItemSpacing, 12, 8, category=dpg.mvThemeCat_Core
                )

        # Main window
        with dpg.window(
            label="Wire Bundle Configurator",
            tag="main_window",
            pos=(0, 0),
            width=-1,
            height=-1,
        ):
            dpg.bind_item_theme("main_window", self.theme)

            # Section: Define wires
            dpg.add_text("Define Wire Types", bullet=True)
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="Count    ",
                    default_value=1,
                    tag="input_count",
                    width=200,
                    min_value=1,
                )

                dpg.add_input_double(
                    label="Diameter    ",
                    default_value=1.0,
                    tag="input_diameter",
                    width=200,
                    min_value=0.001,
                )

                dpg.add_button(
                    label="Add Wire Type", callback=self._add_wire_callback, width=220
                )

            dpg.add_spacer(height=5)
            dpg.add_text("Current wire types:")
            dpg.add_listbox(items=[], tag="wire_listbox", num_items=6, width=-1)

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=20)

            # Section: Optimization settings
            dpg.add_text("Optimization Settings", bullet=True)
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="Initializations - ",
                    default_value=8,
                    tag="input_inits",
                    width=200,
                    min_value=1,
                )
                dpg.add_text(
                    "Number of random initializations for optimization. Higher values increase the chance of finding a better global solution but also increase computation time.",
                    wrap=1000,
                )

            with dpg.group(horizontal=True):
                dpg.add_input_int(
                    label="Max Iterations -",
                    default_value=200,
                    tag="input_maxiter",
                    width=200,
                    min_value=1,
                )
                dpg.add_text(
                    "Increasing this may help with convergence, but typically 200 iterations is sufficient. Higher values may increase runtime.",
                    wrap=1000,
                )

            dpg.add_spacer(height=40)
            dpg.add_button(
                label="Optimize & Plot", callback=self._optimize_callback, width=-1
            )

            dpg.add_spacer(height=70)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            # Section: GUI scale
            dpg.add_text("GUI Scale")
            dpg.add_slider_float(
                label="Scale Factor",
                tag="scale_slider",
                default_value=self.scale_factor,
                min_value=0.5,
                max_value=2.0,
                callback=self._scale_callback,
                width=-1,
            )

        dpg.set_primary_window("main_window", True)
        dpg.setup_dearpygui()

    def _update_wire_list(self) -> None:
        items = [f"{cnt} x {round(dia, 3)}mm" for cnt, dia in self.wire_defs]
        dpg.configure_item("wire_listbox", items=items)

    def _add_wire_callback(self, sender: str, app_data: Any, user_data: Any) -> None:
        """
        Callback to add a new wire definition based on user input.

        Args:
            sender (str): The ID of the widget that triggered the callback.
            app_data (Any): Data passed from the widget, not used here.
            user_data (Any): Additional data passed to the callback, not used here.
        """
        # Get user input values
        count = dpg.get_value("input_count")
        diameter = dpg.get_value("input_diameter")

        if count > 0 and diameter > 0:
            # If the wire diameter already exists, increment the count
            for i, (existing_count, existing_diameter) in enumerate(self.wire_defs):
                if diameter == existing_diameter:
                    self.wire_defs[i] = (existing_count + count, diameter)
                    self._update_wire_list()
                    dpg.set_value("input_count", 1)
                    dpg.set_value("input_diameter", 1.0)
                    return

            self.wire_defs.append((count, diameter))
            self._update_wire_list()
            dpg.set_value("input_count", 1)
            dpg.set_value("input_diameter", 1.0)

        else:
            print("Please enter positive count and diameter.")

    def _optimize_callback(self, sender: str, app_data: Any, user_data: Any) -> None:
        """
        Callback to perform the optimization and plot the results.

        Args:
            sender (str): The ID of the widget that triggered the callback.
            app_data (Any): Data passed from the widget, not used here.
            user_data (Any): Additional data passed to the callback, not used here.
        """

        # Convert diameters to radii
        radii = [d / 2.0 for cnt, d in self.wire_defs for _ in range(cnt)]

        if not radii:
            print("No wires defined!")
            return

        # Get optimization parameters (input)
        n_inits = dpg.get_value("input_inits")
        max_iter = dpg.get_value("input_maxiter")

        optimizer = WireBundleOptimizer(radii)
        positions, radii_arr, outer_radius = optimizer.solve_multi(
            n_inits, max_iterations=max_iter, n_jobs=-1
        )

        # Plot results
        _, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.set_title(f"Optimized Wire Bundle Diameter: {(outer_radius*2):.3f}")
        ax.add_patch(
            plt.Circle((0, 0), outer_radius, color="gray", fill=False, linestyle="--")
        )

        # Plot each wire as a circle
        for (x, y), r in zip(positions, radii_arr):
            ax.add_patch(plt.Circle((x, y), r, alpha=0.6))
        lim = outer_radius + max(radii_arr)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True)
        plt.show()

    def _scale_callback(self, sender: str, scale: float, user_data: Any) -> None:
        """Callback to adjust the GUI scale factor."""
        self.scale_factor = scale
        dpg.set_global_font_scale(self.scale_factor)

    def run(self) -> None:
        """Run the application."""
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


def main() -> None:
    app = WireBundleApp()
    app.run()


if __name__ == "__main__":
    main()
