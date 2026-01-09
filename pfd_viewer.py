#!/usr/bin/env python3
"""
Standalone PFD Viewer using WebKitGTK
Uses native EGL/OpenGL on Jetson for better GPU acceleration
"""

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('WebKit2', '4.0')
from gi.repository import Gtk, WebKit2, Gdk

class PFDWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="Avionics PFD - Synthetic Vision")

        # Set window properties
        self.set_default_size(1280, 800)
        self.set_position(Gtk.WindowPosition.CENTER)

        # Enable OpenGL/hardware acceleration
        settings = WebKit2.Settings()
        settings.set_enable_webgl(True)
        settings.set_enable_accelerated_2d_canvas(True)
        settings.set_hardware_acceleration_policy(
            WebKit2.HardwareAccelerationPolicy.ALWAYS
        )
        settings.set_enable_smooth_scrolling(True)

        # Create WebView with settings
        self.webview = WebKit2.WebView.new_with_settings(settings)

        # Load the synthetic vision page
        self.webview.load_uri("http://localhost:8000/synthetic-vision")

        # Add to window
        self.add(self.webview)

        # Connect close event
        self.connect("destroy", Gtk.main_quit)

        # Fullscreen toggle with F11
        self.connect("key-press-event", self.on_key_press)

    def on_key_press(self, widget, event):
        if event.keyval == Gdk.KEY_F11:
            if self.get_window().get_state() & Gdk.WindowState.FULLSCREEN:
                self.unfullscreen()
            else:
                self.fullscreen()
        elif event.keyval == Gdk.KEY_Escape:
            self.unfullscreen()

def main():
    # Enable OpenGL
    Gdk.set_allowed_backends("x11,*")

    window = PFDWindow()
    window.show_all()

    print("PFD Viewer started - Press F11 for fullscreen, Esc to exit fullscreen")
    Gtk.main()

if __name__ == "__main__":
    main()
