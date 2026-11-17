# Aura

A thread of glass around your MacBook's notch that dances to whatever your Mac is playing.

Aura lives in the menu bar. It listens to all system audio (Music, Spotify, YouTube, games, anything) and draws a glowing line around the notch that swells with the sound. The left half follows the left channel and the right half follows the right. Where the bass hits, the line gives off a haze like the air above a flame.

## Features

- **Around the Notch** or **Around the Whole Display**: trace just the camera housing, or the entire screen edge.
- **Cursor Magnet**: the line reaches toward your pointer and wobbles when it lets go.
- **Heat Haze**: a shimmer where the bass lifts the line.
- **Response Speed**: Smooth, Fast or Fastest.
- **Colours**: Siri, Aurora, Sunset, Ice, or **Album Art**, which takes its colours from the cover of whatever Apple Music is playing.
- **Bass Feel**: the whole screen thumps with the kick drum as a subtle zoom blur. It works with or without the line. Strength can be Subtle, Balanced or Punchy.
- **Follows your volume**: full movement at full volume, none when muted.

## Requirements

- macOS 14.2 or later (Aura uses Core Audio process taps to hear system audio)
- A Mac with Metal graphics
- Best on a MacBook with a notch. On other displays, use **Around the Whole Display**.

## Install

### Download

Grab `Aura.zip` from the [Releases](../../releases) page, unzip it, and move `Aura.app` to Applications.

The app is not notarized, so the first time you open it, **right-click → Open** and confirm. If macOS still blocks it, go to **System Settings → Privacy & Security** and click **Open Anyway**.

### Build from source

You need the Xcode Command Line Tools (`xcode-select --install`).

```sh
git clone https://github.com/ErenWijesekara/Aura.git
cd Aura
./build.sh
open Aura.app
```

## Permissions

macOS asks for these the first time each feature is used:

| Permission | Why |
| --- | --- |
| **Audio Recording** (system audio) | To hear what your Mac is playing. Audio is analysed live and never saved or sent anywhere. |
| **Automation → Music** | Only for the Album Art colours, to read the current track's cover from Apple Music. |
| **Screen Recording** | Only for Bass Feel, which needs a copy of the screen to blur on each kick. |

You can change these later in **System Settings → Privacy & Security**.

## Usage

Click the waveform icon in the menu bar to toggle the visualizer, switch styles, pick colours, or turn on Bass Feel. Settings are remembered between launches.

To start Aura at login, add it in **System Settings → General → Login Items**.

## How it works

Everything is in [`main.swift`](main.swift):

1. **Audio tap**: a Core Audio process tap captures all system output without interrupting playback.
2. **Analyzer**: turns the last ~43 ms of audio into 40 log-spaced frequency bands per channel, and separately tracks the kick and bass.
3. **Line**: the outline of the notch (or display) as a chain of points. Each point is pushed outward by its band and pulled toward the cursor, both on springs, so it moves like liquid rather than a bar graph.
4. **Renderer**: draws the chain as a glass tube with a coloured glow and heat haze, using Metal in a transparent overlay window.
5. **Bass Feel**: uses ScreenCaptureKit to capture the screen and lays faint, slightly enlarged copies over it on each kick.

### Previewing without running the app

```sh
Aura.app/Contents/MacOS/Aura --snapshot frame.png
Aura.app/Contents/MacOS/Aura --snapshot frame.png --edges --palette 2 --time 2.5 --volume 1
```

This renders a single frame of a built-in demo song to a PNG. Optional flags:

| Flag | Effect |
| --- | --- |
| `--edges` | Trace the whole display instead of the notch |
| `--palette 0-4` | Siri, Aurora, Sunset, Ice, Album Art |
| `--time 2.5` | How many seconds into the demo song |
| `--volume 1` | Volume from 0 to 1 |
| `--no-haze` | Turn off the heat haze | You can also run the app with `AURA_DEMO=1` to animate it with the demo song instead of real audio.

## License

[MIT](LICENSE)
