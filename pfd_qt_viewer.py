#!/usr/bin/env python3
"""
Standalone PFD Viewer using Qt WebEngine
Better WebGL support than WebKitGTK
"""

import sys
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QShortcut
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from PyQt5.QtGui import QKeySequence

class PFDWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Avionics PFD - Synthetic Vision")
        self.setGeometry(100, 100, 1280, 800)

        # Create web view
        self.browser = QWebEngineView()

        # Enable WebGL and hardware acceleration
        settings = self.browser.settings()
        settings.setAttribute(QWebEngineSettings.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.PluginsEnabled, True)
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)

        # Load the synthetic vision page
        self.browser.setUrl(QUrl("http://localhost:8000/synthetic-vision"))

        self.setCentralWidget(self.browser)

        # Fullscreen shortcut (F11)
        self.fullscreen_shortcut = QShortcut(QKeySequence(Qt.Key_F11), self)
        self.fullscreen_shortcut.activated.connect(self.toggle_fullscreen)

        # Escape to exit fullscreen
        self.escape_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.escape_shortcut.activated.connect(self.showNormal)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

def main():
    # Set environment for better GPU support on Jetson
    import os
    os.environ.setdefault('QT_XCB_GL_INTEGRATION', 'xcb_egl')
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--ignore-gpu-blocklist --enable-gpu-rasterization')

    app = QApplication(sys.argv)
    app.setApplicationName("Avionics PFD")

    window = PFDWindow()
    window.show()

    print("Qt PFD Viewer started - Press F11 for fullscreen")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
