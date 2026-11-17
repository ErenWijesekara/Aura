#!/bin/zsh
# Builds Aura.app next to this script.
set -e
cd "$(dirname "$0")"
APP=Aura.app
mkdir -p $APP/Contents/MacOS
swiftc -O main.swift -o $APP/Contents/MacOS/Aura
cat > $APP/Contents/Info.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Aura</string>
  <key>CFBundleIdentifier</key><string>local.aura.visualizer</string>
  <key>CFBundleExecutable</key><string>Aura</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSUIElement</key><true/>
  <key>NSAudioCaptureUsageDescription</key><string>Aura listens to the audio your Mac plays so the line around the notch can move with it.</string>
  <key>NSAppleEventsUsageDescription</key><string>Aura asks Music for the cover of the song that's playing, to take the line's colours from it.</string>
</dict></plist>
PLIST
codesign -s - --force $APP >/dev/null 2>&1
echo "Built $(pwd)/$APP"
