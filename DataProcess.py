"""
@file PavoDataProcess.py
@brief Interactive GUI for visualizing data smoothing algorithms.
@version 1.0
@date 2024-12-14
@author Jack(Jichuan Zhong)
@details Search "TODO" to find the place where you should implement your code.
"""

import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QCheckBox, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
from scipy.linalg import block_diag
from scipy.ndimage import gaussian_filter1d

# Everybody should implement your own data processing class
class JackMethods:
    """
    @class JackMethods
    @brief Jack's data processing methods. 
    @details This is the demo class for u to refer to and learn how to add ur methods into test code.
    """
    
    def movingAverage(self, data, window_size=10):
        """
        @brief Computes the moving average for the input data.
        @param data Input data as an array-like object.
        @param window_size Size of the moving average window (default is 10).
        @return Smoothed data using moving average.
        """
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def exponentialSmoothing(self, data, alpha=0.3):
        """
        @brief Applies exponential smoothing to the input data.
        @param data Input data as an array-like object.
        @param alpha Smoothing factor (0 < alpha <= 1).
        @return Smoothed data using exponential smoothing.
        """
        smoothed = [data[0]]
        for i in range(1, len(data)):
            smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
        return np.array(smoothed)
    
    def savgolFilter(self, data, window_length, polyorder):
        """
        @brief Applies Savitzky-Golay filter to the input data.
        @param data Input data as an array-like object.
        @return Smoothed data using Savitzky-Golay filter.
        """
        return savgol_filter(data, window_length=15, polyorder=2)
    
    
class PhoebeMethods:
    def gaussianFilter(self, data, sigma=2):
        """
        @brief Applies Gaussian filter to smooth the input data.
        @param data Input data as an array-like object.
        @param sigma Standard deviation of the Gaussian kernel (default is 2).
        @return Smoothed data using Gaussian filter.
        """
        return gaussian_filter1d(data, sigma)

    
class AlgerMethods:
    """
    @class AlgerMethods
    @brief Alger's data processing methods.
    @details Contains the data processing algorithms implemented by Alger.
    """
    def butterworthFilter(self, data, cutoff=0.15, order=2):
        """
        @brief Applies a Butterworth Low-Pass Filter. 
        @param data Input signal data.
        @param cutoff Normalized cutoff frequency (0 < cutoff < 1).
        @param order Order of the Butterworth filter (default is 2).
        @return Smoothed signal after filtering.
        """
        # Design the Butterworth low-pass filter
        b, a = butter(order, cutoff, btype='low', analog=False)
        # Apply zero-phase filtering to prevent phase shift
        filtered_data = filtfilt(b, a, data)
        return filtered_data



class GaoJunMethods:
    """
    @class GaoJunMethods
    @brief GaoJun's data processing methods using Extended Kalman Filter (EKF).
    """
    
    def extendedKalmanFilter(self, data):
        """
        @brief Implements Extended Kalman Filter (EKF) for smoothing data.
        @param data Input data as an array-like object.
        @return Smoothed data using EKF.
        """
        # Initialization
        n = len(data)
        smoothed_data = np.zeros(n)
        
        # Define EKF parameters
        Q = 1e-2  # Process noise covariance
        R = 1  # Measurement noise covariance
        P = 1  # Estimation error covariance
        x = data[0]  # Initial estimate
        K = 0       # Kalman gain
        
        # Kalman Filter process
        for i in range(n):
            # Prediction step
            x_pred = x  # Assuming constant velocity model (prediction = previous state)
            P_pred = P + Q  # Update covariance
            
            # Measurement update
            K = P_pred / (P_pred + R)  # Calculate Kalman gain
            x = x_pred + K * (data[i] - x_pred)  # Update state with measurement
            P = (1 - K) * P_pred  # Update error covariance
            
            smoothed_data[i] = x  # Store smoothed value
        
        return smoothed_data


class SmoothingPlotApp(QMainWindow):
    """
    @class SmoothingPlotApp
    @brief A GUI application for comparing data smoothing algorithms.
    @details This class creates an interactive PyQt5 window with checkboxes to toggle between
             original test data and the results of smoothing algorithms.
    """
    
    def __init__(self):
        """
        @brief Initializes the SmoothingPlotApp class.
        @details Sets up the GUI layout, initializes test data, and displays the plots.
        """
        super().__init__()
        self.setWindowTitle("Smoothing Plot")

        # Main window layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Matplotlib FigureCanvas
        self.figure, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # Checkbox layout
        self.checkbox_layout = QHBoxLayout()
        self.layout.addLayout(self.checkbox_layout)
        
        #TODO: Instantiate ur data processing class here
        self.jack_methods = JackMethods()

        # AlgerMethod
        self.alger_methods = AlgerMethods()
        #GaoJunmethod
        self.gaojun_methods = GaoJunMethods()

        # Initialize data
        self.initData()

        # Plot initial data
        self.plotInitialData()

        # Add checkboxes for interaction
        self.addCheckboxes()

        # Phoebe's methods
        self.phoebe_methods = PhoebeMethods()
        self.phoebe_gaussian_result = self.phoebe_methods.gaussianFilter(self.test_data, sigma=2)


    def initData(self):
        """
        @brief Loads test data or generates synthetic data.
        @details Reads the data from 'test_data.xlsx'. If the file is not found, synthetic data
                 consisting of trend, noise, and periodic segments is generated and saved.
        @exception FileNotFoundError If the data file is missing.
        """
        try:
            data = pd.read_excel('test_data.xlsx')
            self.test_data = data['Test Data(y_value)'].values
            print('Data loaded from file.')
        except FileNotFoundError:
            # Generate synthetic test data
            np.random.seed(42)
            N = 100
            trend_data = np.linspace(-50, 50, N)
            noise_data = np.linspace(50, -50, N) + np.random.normal(0, 5, N)
            periodic_data = 50 * np.sin(np.linspace(0, 8 * np.pi, N)) + np.random.normal(0, 5, N)
            self.test_data = np.concatenate([trend_data, noise_data, periodic_data])
            self.test_data[20:23] += 50
            self.test_data[35:40] -= 30
            self.test_data[60:61] -= 50
            
            # Save to Excel file
            data = pd.DataFrame({
                'Test Data(y_value)': self.test_data
            })
            data.to_excel('test_data.xlsx', index=False)

        # Define data segments
        self.segments = {
            "Trend Data": (0, 100),
            "Noise Data": (100, 200),
            "Periodic Data": (200, 300)
        }

        #TODO: Use ur methods to compute smoothing results here
        # Jack's methods
        self.jack_ma_result = self.jack_methods.movingAverage(self.test_data)
        self.jack_es_result = self.jack_methods.exponentialSmoothing(self.test_data)
        self.jack_sg_result = self.jack_methods.savgolFilter(self.test_data, window_length=15, polyorder=2)
        # Phoebe's methods
        try:
            data = pd.read_excel('test_data.xlsx')
            self.test_data = data['Test Data(y_value)'].values
            print('Data loaded from file.')
        except FileNotFoundError:
            # Generate synthetic test data
            np.random.seed(42)
            N = 100
            trend_data = np.linspace(-50, 50, N)
            noise_data = np.linspace(50, -50, N) + np.random.normal(0, 5, N)
            periodic_data = 50 * np.sin(np.linspace(0, 8 * np.pi, N)) + np.random.normal(0, 5, N)
            self.test_data = np.concatenate([trend_data, noise_data, periodic_data])
        
            # Save to Excel file
            data = pd.DataFrame({'Test Data(y_value)': self.test_data})
            data.to_excel('test_data.xlsx', index=False)

            # Instantiate method classes
            self.jack_methods = JackMethods()
            self.phoebe_methods = PhoebeMethods()  # Add PhoebeMethods here

            # Compute Jack's smoothing methods
            self.jack_ma_result = self.jack_methods.movingAverage(self.test_data)
            self.jack_es_result = self.jack_methods.exponentialSmoothing(self.test_data)
            self.jack_sg_result = self.jack_methods.savgolFilter(self.test_data, window_length=15, polyorder=2)

            # Phoebe's methods
            
        
        # Alger's methods
        self.alger_bw_result = self.alger_methods.butterworthFilter(self.test_data, cutoff=0.15, order=2)
        
        # GaoJun's methods
        self.gaojun_ekf_result = self.gaojun_methods.extendedKalmanFilter(self.test_data)

    def plotInitialData(self):
        """
        @brief Plots the initial test data segments.
        @details Creates the original data plot and initializes empty lines for smoothing results.
        """
        self.colors = ['blue', 'orange', 'green']
        self.initial_lines = []
        for idx, (label, (start, end)) in enumerate(self.segments.items()):
            line, = self.ax.plot(range(start, end), self.test_data[start:end], label=label, color=self.colors[idx])
            self.initial_lines.append(line)

        # Initialize smoothing result lines
        self.jack_ma_line, = self.ax.plot([], [], label='JackMA', linestyle='--', color='red', lw=1.5)
        self.jack_es_line, = self.ax.plot([], [], label='JackES', linestyle='--', color='purple', lw=1.5)
        self.jack_sg_line, = self.ax.plot([], [], label='JackSG', linestyle='--', color='brown', lw=1.5)
        
        #TODO: Create ur line here. The value of the line label is consistent with the label in the checkbox below
        # Alger's ButterWorth Filter line
        self.alger_bw_line, = self.ax.plot([], [], label="AlgerBW", linestyle='--', color='cyan', lw=1.5)
        # GaoJun's EKF Filter line
        self.gaojun_ekf_line, = self.ax.plot([],[],label = 'GaoJunEKF', linestyle='--', color = 'purple',lw=1.5)
        # Phoebe's Gaussian Filter line
        self.phoebe_gaussian_line, = self.ax.plot([], [], label='PhoebeGaussian', linestyle='--', color='magenta', lw=1.5)

        self.ax.set_title('Interactive Smoothing Algorithm Comparison')
        self.ax.set_xlabel('Index')
        self.ax.set_ylabel('Value')
        self.ax.legend(loc='upper left')
        self.ax.grid()

        self.canvas.draw()

    def addCheckboxes(self):
        """
        @brief Adds checkboxes for toggling data visibility.
        @details Users can toggle the visibility of original data and smoothing results.
        """
        self.check_initial = QCheckBox('Show Initial Data')
        self.check_initial.setChecked(True)  # Initially show original data
        self.jack_check_ma = QCheckBox('JackMA')
        self.jack_check_es = QCheckBox('JackES')
        self.jack_check_sg = QCheckBox('JackSG')
       
        #TODO: Add ur checkbox here
        # Alger's Method
        self.alger_check_bw = QCheckBox("AlgerBW")
        self.checkbox_layout.addWidget(self.alger_check_bw)
        self.alger_check_bw.stateChanged.connect(self.updatePlot)
        #GaoJun's Method
        self.gaojun_check_ekf = QCheckBox('GaoJunEKF')
        #Phoebe's Method
        self.phoebe_check_gaussian = QCheckBox('PhoebeGaussian')
        self.checkbox_layout.addWidget(self.phoebe_check_gaussian)
        self.phoebe_check_gaussian.stateChanged.connect(self.updatePlot)


        self.checkbox_layout.addWidget(self.check_initial)
        self.checkbox_layout.addWidget(self.jack_check_ma)
        self.checkbox_layout.addWidget(self.jack_check_es)
        self.checkbox_layout.addWidget(self.jack_check_sg)
        self.checkbox_layout.addWidget(self.gaojun_check_ekf)

        # Connect checkboxes to event handlers
        self.check_initial.stateChanged.connect(self.updatePlot)
        self.jack_check_ma.stateChanged.connect(self.updatePlot)
        self.jack_check_es.stateChanged.connect(self.updatePlot)
        self.jack_check_sg.stateChanged.connect(self.updatePlot)
        self.gaojun_check_ekf.stateChanged.connect(self.updatePlot)

    def updatePlot(self):
        """
        @brief Updates the plot based on the checkbox states.
        @details Toggles the visibility of original data and smoothing algorithm results.
        """
        # Toggle visibility of initial data lines
        if self.check_initial.isChecked():
            for line in self.initial_lines:
                line.set_visible(True)
        else:
            for line in self.initial_lines:
                line.set_visible(False)

        # Update Moving Average line
        if self.jack_check_ma.isChecked():
            self.jack_ma_line.set_data(range(len(self.jack_ma_result)), self.jack_ma_result)
            self.jack_ma_line.set_visible(True)
        else:
            self.jack_ma_line.set_visible(False)

        # Update Exponential Smoothing line
        if self.jack_check_es.isChecked():
            self.jack_es_line.set_data(range(len(self.jack_es_result)), self.jack_es_result)
            self.jack_es_line.set_visible(True)
        else:
            self.jack_es_line.set_visible(False)

        # Update Savitzky-Golay line
        if self.jack_check_sg.isChecked():
            self.jack_sg_line.set_data(range(len(self.jack_sg_result)), self.jack_sg_result)
            self.jack_sg_line.set_visible(True)
        else:
            self.jack_sg_line.set_visible(False)
            
        #TODO: Add judgement here to show ur line
        # Update EKF Filter line
        if self.gaojun_check_ekf.isChecked():
            self.gaojun_ekf_line.set_data(range(len(self.gaojun_ekf_result)), self.gaojun_ekf_result)
            self.gaojun_ekf_line.set_visible(True)
        else:
            self.gaojun_ekf_line.set_visible(False)   

        # Update Butterworth Filter line
        if self.alger_check_bw.isChecked():
            self.alger_bw_line.set_data(range(len(self.alger_bw_result)), self.alger_bw_result)
            self.alger_bw_line.set_visible(True)
        else:
            self.alger_bw_line.set_visible(False)
        # Update Gaussian Filter line
        if self.phoebe_check_gaussian.isChecked():
            self.phoebe_gaussian_line.set_data(range(len(self.phoebe_gaussian_result)), self.phoebe_gaussian_result)
            self.phoebe_gaussian_line.set_visible(True)
        else:
            self.phoebe_gaussian_line.set_visible(False)

        # Redraw the canvas
        self.canvas.draw()

if __name__ == '__main__':
    """
    @brief Main entry point of the application.
    @details Creates and runs the SmoothingPlotApp GUI.
    """
    app = QApplication(sys.argv)
    main_window = SmoothingPlotApp()
    main_window.show()
    sys.exit(app.exec_())
