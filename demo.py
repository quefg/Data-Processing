"""
@file demo.py
@brief Real-time plotting application with PyQt5 and Matplotlib.
@author Author: Jichuan Zhong
@date 2024-12-10

This application creates a PyQt5-based GUI for real-time plotting of data.
It displays both raw and processed data on two separate canvases.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QCheckBox
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import savgol_filter

class PlotCanvas(FigureCanvas):
    """
    @class PlotCanvas
    @brief A Matplotlib canvas for plotting data within a PyQt5 application.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        @brief Constructor for PlotCanvas.
        @param parent The parent widget.
        @param width Width of the canvas in inches.
        @param height Height of the canvas in inches.
        @param dpi Dots per inch for the canvas.
        """
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)

    def plot(self, x, y, color='blue', label=None):
        """
        @brief Plot data on the canvas.
        @param x X-axis data.
        @param y Y-axis data.
        @param color Color of the plot line.
        @param label Label for the plot line.
        """
        self.axes.plot(x, y, color=color, label=label, linewidth=4)
        if label:
            self.axes.legend()
        self.draw()

    def clear(self):
        """
        @brief Clear all plots from the canvas.
        """
        self.axes.cla()

class MainWindow(QMainWindow):
    """
    @class MainWindow
    @brief Main application window for real-time data plotting.
    """
    element_count = 100  # Number of data points in the plot.
    x_length = 5         # Length of the X-axis in seconds.
    new_x = x_length     # Current end of the X-axis.
    step = float(x_length) / element_count  # Step size for the X-axis.

    def __init__(self):
        """
        @brief Constructor for MainWindow.
        """
        super().__init__()
        self.setWindowTitle("Dual Canvas Plot")
        self.setGeometry(100, 100, 1200, 600)

        # Central widget and layout setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Real-time plot canvas
        self.real_time_canvas = PlotCanvas(self, width=5, height=4)
        self.layout.addWidget(self.real_time_canvas)

        # Processed plot canvas
        self.processed_canvas = PlotCanvas(self, width=5, height=4)
        self.layout.addWidget(self.processed_canvas)

        # Reference curve visibility control
        self.reference_checkbox = QCheckBox("Show Reference Curve (sin)", self)
        self.reference_checkbox.setChecked(True)
        self.reference_checkbox.stateChanged.connect(self.updateReferenceVisibility)
        self.layout.addWidget(self.reference_checkbox)

        # Timer for real-time data updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updatePlots)

        # Initialize data
        self.x_data = np.linspace(0, self.x_length, self.element_count)
        self.reference_y_data = np.sin(self.x_data)
        self.real_time_y_data = np.sin(self.x_data)
        self.show_reference = True
        self.timer.start(100)  # Update interval in milliseconds.

    def updatePlots(self):
        """
        @brief Update plots with new real-time and processed data.
        """
        # Shift and update data
        self.real_time_y_data = np.roll(self.real_time_y_data, -1)
        self.new_x = self.new_x + self.step
        self.real_time_y_data[-1] = np.sin(self.new_x) + np.random.normal(0, 0.1)

        # Update real-time plot
        self.real_time_canvas.clear()
        self.real_time_canvas.plot(self.x_data, self.real_time_y_data, color='blue', label='Real-time Data')

        # Process data (denoise and fit)
        filtered_y_data = savgol_filter(self.real_time_y_data, window_length=51, polyorder=3)

        # Update processed plot
        self.processed_canvas.clear()
        self.processed_canvas.plot(self.x_data, filtered_y_data, color='green', label='Processed Data')

        # Update reference curve
        self.reference_y_data = np.roll(self.reference_y_data, -1)
        self.reference_y_data[-1] = np.sin(self.new_x)
        if self.show_reference:
            self.processed_canvas.plot(self.x_data, self.reference_y_data, color='gray', label='Reference Curve')

    def updateReferenceVisibility(self):
        """
        @brief Toggle the visibility of the reference curve.
        """
        self.show_reference = self.reference_checkbox.isChecked()

if __name__ == "__main__":
    """
    @brief Main entry point for the application.
    """
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
