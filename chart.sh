#!/usr/bin/env sh

show_chart() {
    sleep 1
    google-chrome-stable http://localhost:8000/experiments
}

show_chart &!
python3 -m http.server
