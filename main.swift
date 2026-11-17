import AppKit
import AVFoundation
import AudioToolbox
import CoreAudio
import Accelerate
import Metal
import QuartzCore
import ScreenCaptureKit
import ImageIO
import UniformTypeIdentifiers
import os

func fbuf(_ n: Int) -> UnsafeMutablePointer<Float> {
    let p = UnsafeMutablePointer<Float>.allocate(capacity: n)
    p.initialize(repeating: 0, count: n)
    return p
}

typealias Vec = SIMD2<Float>

@inline(__always) func len(_ v: Vec) -> Float { (v.x * v.x + v.y * v.y).squareRoot() }
@inline(__always) func unit(_ v: Vec) -> Vec { let l = len(v); return l > 1e-6 ? v / l : Vec(0, 0) }
/// Quarter turn: for a path walked with the screen content on this side, the direction audio pushes the line.
@inline(__always) func perp(_ v: Vec) -> Vec { Vec(-v.y, v.x) }
@inline(__always) func smoothstep(_ a: Float, _ b: Float, _ x: Float) -> Float {
    let t = min(1, max(0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)
}

// MARK: - Core Audio helpers

struct AudioError: LocalizedError {
    let msg: String
    var errorDescription: String? { msg }
}

func check(_ err: OSStatus, _ what: String) throws {
    if err != noErr { throw AudioError(msg: "\(what) failed (\(err))") }
}

func addr(_ sel: AudioObjectPropertySelector, _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal)
    -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func getProp<T>(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector, _ value: inout T) throws {
    var a = addr(sel)
    var size = UInt32(MemoryLayout<T>.size)
    try check(AudioObjectGetPropertyData(obj, &a, 0, nil, &size, &value), "read property \(sel)")
}

func defaultOutputDevice() throws -> AudioObjectID {
    var id = AudioObjectID(kAudioObjectUnknown)
    try getProp(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultOutputDevice, &id)
    return id
}

func deviceUID(_ dev: AudioObjectID) throws -> String {
    var a = addr(kAudioDevicePropertyDeviceUID)
    var uid: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    try check(AudioObjectGetPropertyData(dev, &a, 0, nil, &size, &uid), "read device UID")
    guard let s = uid?.takeRetainedValue() else { throw AudioError(msg: "no output device UID") }
    return s as String
}

/// The default output's volume as set by the user (0…1), 0 when muted, 1 if the device has no volume control.
func systemOutputVolume() -> Float {
    guard let dev = try? defaultOutputDevice() else { return 1 }
    var mute: UInt32 = 0
    var ma = addr(kAudioDevicePropertyMute, kAudioDevicePropertyScopeOutput)
    var size = UInt32(MemoryLayout<UInt32>.size)
    if AudioObjectHasProperty(dev, &ma), AudioObjectGetPropertyData(dev, &ma, 0, nil, &size, &mute) == noErr, mute != 0 {
        return 0
    }
    var vol: Float32 = 1
    var va = addr(kAudioHardwareServiceDeviceProperty_VirtualMainVolume, kAudioDevicePropertyScopeOutput)
    size = UInt32(MemoryLayout<Float32>.size)
    guard AudioObjectHasProperty(dev, &va), AudioObjectGetPropertyData(dev, &va, 0, nil, &size, &vol) == noErr else { return 1 }
    return min(1, max(0, vol))
}

// MARK: - audio tap

struct Biquad {
    var b0: Double = 1, b1: Double = 0, b2: Double = 0, a1: Double = 0, a2: Double = 0
    var x1: Double = 0, x2: Double = 0, y1: Double = 0, y2: Double = 0

    private static func make(_ b0: Float, _ b1: Float, _ b2: Float, _ a0: Float, _ a1: Float, _ a2: Float) -> Biquad {
        let a0 = Double(a0)
        return Biquad(b0: Double(b0) / a0, b1: Double(b1) / a0, b2: Double(b2) / a0, a1: Double(a1) / a0, a2: Double(a2) / a0)
    }
    static func lowpass(_ f: Float, _ sr: Float, q: Float = 0.7071) -> Biquad {
        let w = 2 * Float.pi * f / sr, c = cos(w), al = sin(w) / (2 * q)
        return make((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + al, -2 * c, 1 - al)
    }
    static func highpass(_ f: Float, _ sr: Float, q: Float = 0.7071) -> Biquad {
        let w = 2 * Float.pi * f / sr, c = cos(w), al = sin(w) / (2 * q)
        return make((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al)
    }
    @inline(__always) mutating func run(_ xf: Float) -> Float {
        let x = Double(xf)
        let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1; x1 = x; y2 = y1; y1 = y
        return Float(y)
    }
}

/// Loudness of the kick and bass alone: everything above 120 Hz is filtered out (24 dB/oct) before measuring,
/// so vocals, snares and hats don't register at all.
final class BassMeter {
    private var f1 = Biquad(), f2 = Biquad(), f3 = Biquad()
    private(set) var energy: Float = 0   // loudest block since the main thread last read it

    func configure(sampleRate sr: Float) {
        f1 = .lowpass(120, sr); f2 = .lowpass(120, sr); f3 = .highpass(30, sr)
        energy = 0
    }

    /// Audio thread.
    func feed(_ l: UnsafePointer<Float>, _ r: UnsafePointer<Float>, stride: Int, count n: Int) {
        guard n > 0 else { return }
        var e: Float = 0
        for i in 0..<n {
            let b = f3.run(f2.run(f1.run((l[i * stride] + r[i * stride]) * 0.5)))
            e += b * b
        }
        e /= Float(n)
        if e > energy { energy = e }
    }

    /// Main thread: returns the loudest block energy since the last call.
    func take() -> Float { let e = energy; energy = 0; return e }
}

/// Listens to everything the Mac plays (without muting or changing it) and keeps the last third of a second of
/// each channel.
final class AudioTap {
    static let ringSize = 1 << 14
    /// The output device or its sample rate changed: the tap has to be rebuilt.
    var onChange: (() -> Void)?
    private(set) var running = false
    private(set) var sampleRate: Float = 48000
    let bass = BassMeter()

    private let ring = fbuf(AudioTap.ringSize), ringR = fbuf(AudioTap.ringSize)     // left, right
    private var head = 0
    private let lock: UnsafeMutablePointer<os_unfair_lock> = {
        let p = UnsafeMutablePointer<os_unfair_lock>.allocate(capacity: 1)
        p.initialize(to: os_unfair_lock())
        return p
    }()
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private let ioQueue = DispatchQueue(label: "aura.io", qos: .userInteractive)
    private var listeners: [(AudioObjectID, AudioObjectPropertyAddress, AudioObjectPropertyListenerBlock)] = []

    private func listen(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector) {
        var a = addr(sel)
        guard AudioObjectHasProperty(obj, &a) else { return }
        let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in self?.onChange?() }
        if AudioObjectAddPropertyListenerBlock(obj, &a, .main, block) == noErr { listeners.append((obj, a, block)) }
    }

    func start() throws {
        guard !running else { return }
        do {
            try startPipeline()
            running = true
        } catch {
            stop()
            throw error
        }
    }

    func stop() {
        if let p = procID {
            AudioDeviceStop(aggID, p)
            AudioDeviceDestroyIOProcID(aggID, p)
        }
        procID = nil
        for (obj, a, block) in listeners {
            var a = a
            AudioObjectRemovePropertyListenerBlock(obj, &a, .main, block)
        }
        listeners = []
        if aggID != kAudioObjectUnknown { AudioHardwareDestroyAggregateDevice(aggID) }
        aggID = AudioObjectID(kAudioObjectUnknown)
        if tapID != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tapID) }
        tapID = AudioObjectID(kAudioObjectUnknown)
        os_unfair_lock_lock(lock)
        for i in 0..<AudioTap.ringSize { ring[i] = 0; ringR[i] = 0 }
        os_unfair_lock_unlock(lock)
        running = false
    }

    private func startPipeline() throws {
        // Aura plays nothing itself, so there is no process to exclude; the tap leaves everyone audible.
        let desc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        desc.uuid = UUID()
        desc.name = "Aura"
        desc.isPrivate = true
        desc.muteBehavior = .unmuted
        try check(AudioHardwareCreateProcessTap(desc, &tapID), "create process tap")

        var asbd = AudioStreamBasicDescription()
        try getProp(tapID, kAudioTapPropertyFormat, &asbd)
        guard asbd.mFormatID == kAudioFormatLinearPCM, asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0,
              asbd.mBitsPerChannel == 32 else { throw AudioError(msg: "unsupported tap format (\(asbd.mBitsPerChannel)-bit)") }
        let nonInterleaved = asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0

        let outDev = try defaultOutputDevice()
        let outUID = try deviceUID(outDev)
        let agg: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Aura Tap",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outUID]],
            kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: true,
                                               kAudioSubTapUIDKey: desc.uuid.uuidString]],
        ]
        try check(AudioHardwareCreateAggregateDevice(agg as CFDictionary, &aggID), "create aggregate device")

        // The tap is clocked by the aggregate (i.e. the output device), which may not run at the rate the
        // tap reported before it was attached.
        var sr: Float64 = 0
        try? getProp(aggID, kAudioDevicePropertyNominalSampleRate, &sr)
        if sr <= 0 { try? getProp(outDev, kAudioDevicePropertyNominalSampleRate, &sr) }
        if sr <= 0 { sr = asbd.mSampleRate }
        sampleRate = Float(sr)
        bass.configure(sampleRate: sampleRate)
        listen(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultOutputDevice)
        listen(aggID, kAudioDevicePropertyNominalSampleRate)
        listen(outDev, kAudioDevicePropertyNominalSampleRate)

        try check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, ioQueue) { [weak self] _, inData, _, outData, _ in
            // we only listen: whatever the aggregate would play for us is silence
            for b in UnsafeMutableAudioBufferListPointer(outData) {
                if let d = b.mData { memset(d, 0, Int(b.mDataByteSize)) }
            }
            // tap streams come after any input streams of the output sub-device
            let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inData))
            guard let self, abl.count > 0 else { return }
            if nonInterleaved && abl.count >= 2 {
                let l = abl[abl.count - 2], r = abl[abl.count - 1]
                guard let lp = l.mData?.assumingMemoryBound(to: Float.self),
                      let rp = r.mData?.assumingMemoryBound(to: Float.self) else { return }
                self.write(lp, rp, stride: 1, count: Int(min(l.mDataByteSize, r.mDataByteSize)) / 4)
            } else {
                let b = abl[abl.count - 1]
                let ch = Int(b.mNumberChannels)
                guard ch > 0, let p = b.mData?.assumingMemoryBound(to: Float.self) else { return }
                self.write(p, ch >= 2 ? p + 1 : p, stride: ch, count: Int(b.mDataByteSize) / 4 / ch)
            }
        }, "create IO proc")
        try check(AudioDeviceStart(aggID, procID), "start tap")
    }

    /// Audio thread.
    private func write(_ l: UnsafePointer<Float>, _ r: UnsafePointer<Float>, stride: Int, count n: Int) {
        let mask = AudioTap.ringSize - 1
        bass.feed(l, r, stride: stride, count: n)
        os_unfair_lock_lock(lock)
        for i in 0..<n {
            ring[head] = l[i * stride]
            ringR[head] = r[i * stride]
            head = (head + 1) & mask
        }
        os_unfair_lock_unlock(lock)
    }

    /// Main thread: copies the newest `n` samples of each channel, oldest first.
    func latest(_ left: UnsafeMutablePointer<Float>, _ right: UnsafeMutablePointer<Float>, count n: Int) {
        let mask = AudioTap.ringSize - 1
        os_unfair_lock_lock(lock)
        var j = (head - n) & mask
        for i in 0..<n {
            left[i] = ring[j]
            right[i] = ringR[j]
            j = (j + 1) & mask
        }
        os_unfair_lock_unlock(lock)
    }
}

// MARK: - analysis

/// How quickly the line follows the music.
enum Response: Int, CaseIterable {
    case smooth, fast, fastest

    var title: String { ["Smooth", "Fast", "Fastest"][rawValue] }
    /// Seconds a band takes to rise, and to fall back.
    var attack: Float { [0.018, 0.010, 0.005][rawValue] }
    var release: Float { [0.25, 0.11, 0.05][rawValue] }
    /// Natural frequency of the spring each point of the line rides on, rad/s (3.5, 6.4 and 11 Hz).
    var spring: Float { [22, 40, 70][rawValue] }
}

/// What the music is doing right now, refreshed once per display frame.
///  - left, right: each channel as 40 log-spaced bands from 40 Hz to 16 kHz, each 0…1. Levels are relative to the
///    loudest band of either channel lately (a reference that jumps up at once and falls 3 dB/s), so quiet and
///    loud music move the line alike, while a sound panned to one side still only shows on that side. How fast
///    a band rises and falls is up to `response`: ~20 ms and ~250 ms unless the user wants it quicker.
///  - bass: 0…1, how hard the kick and bass are hitting (see `BassMeter`), mapped to the top
///    15 dB under the loudest the low end has recently been, rising in ~10 ms and falling over ~250 ms. Bass
///    below −39 dBFS never counts, so leakage from a bass-less track doesn't register.
///  - activity: 1 while something plays, fading to 0 after ~1 s of silence (or when the Mac is muted).
///  - gain: the Mac's volume, 0…1, eased so a press of a volume key glides. The levels above don't depend on
///    the volume; whatever moves with the music multiplies by this, so it moves furthest at full volume.
final class Analyzer {
    static let bands = 40
    private let n = 2048
    private let setup = vDSP_create_fftsetup(11, FFTRadix(kFFTRadix2))!
    private let window = fbuf(2048), samples = fbuf(2048), samplesR = fbuf(2048)
    private let re = fbuf(1024), im = fbuf(1024), power = fbuf(1024)
    private var db = [Float](repeating: -120, count: Analyzer.bands), dbR = [Float](repeating: -120, count: Analyzer.bands)
    private var edges: [Float] = []     // band edges in FFT bins, one more than there are bands
    private var tilt: [Float] = []      // dB added per band: music carries far less energy up high
    private var rate: Float = 0
    /// The Mac's output volume as last read (see `systemOutputVolume`).
    var volume: Float = 1
    /// The Mac's volume is at zero: nothing audible, so nothing to show.
    var muted: Bool { volume <= 0 }
    var response = Response.smooth

    private(set) var left = [Float](repeating: 0, count: Analyzer.bands)
    private(set) var right = [Float](repeating: 0, count: Analyzer.bands)
    private(set) var level: Float = 0, bass: Float = 0, activity: Float = 0, gain: Float = 1
    private var ref: Float = -48, bassRef: Float = -24, quiet: Float = 10

    init() {
        vDSP_hann_window(window, vDSP_Length(n), Int32(vDSP_HANN_DENORM))
    }

    private func configure(_ sr: Float) {
        rate = sr
        let bin = sr / Float(n)
        let b = Analyzer.bands
        edges = (0...b).map { 40 * pow(400, Float($0) / Float(b)) / bin }
        tilt = (0..<b).map { max(0, 2.2 * log2(40 * pow(400, (Float($0) + 0.5) / Float(b)) / 200)) }
    }

    /// Power between two (fractional) bins; bin k covers k − ½ … k + ½. A band narrower than a bin reads that bin.
    private func energy(_ lo: Float, _ hi: Float) -> Float {
        let k0 = max(1, Int((lo + 0.5).rounded(.down))), k1 = min(n / 2 - 1, Int((hi + 0.5).rounded(.down)))
        guard k0 <= k1 else { return 0 }
        var e: Float = 0
        for k in k0...k1 {
            e += power[k] * max(0, min(hi, Float(k) + 0.5) - max(lo, Float(k) - 0.5))
        }
        return e / min(1, hi - lo)
    }

    /// Turns 2048 samples (overwritten) into the level of each band in dB. Returns their total power.
    private func spectrum(_ x: UnsafeMutablePointer<Float>, into db: inout [Float]) -> Float {
        vDSP_vmul(x, 1, window, 1, x, 1, vDSP_Length(n))
        var split = DSPSplitComplex(realp: re, imagp: im)
        x.withMemoryRebound(to: DSPComplex.self, capacity: n / 2) {
            vDSP_ctoz($0, 2, &split, 1, vDSP_Length(n / 2))
        }
        vDSP_fft_zrip(setup, &split, 1, 11, FFTDirection(FFT_FORWARD))
        vDSP_zvmags(&split, 1, power, 1, vDSP_Length(n / 2))
        var k = 4 / Float(n * n)        // a full-scale sine comes out as 1 (0 dB)
        vDSP_vsmul(power, 1, &k, power, 1, vDSP_Length(n / 2))
        var total: Float = 0
        vDSP_sve(power + 1, 1, &total, vDSP_Length(n / 2 - 1))
        for b in 0..<Analyzer.bands {
            db[b] = 10 * log10(energy(edges[b], edges[b + 1]) + 1e-12) + tilt[b]
        }
        return total
    }

    func update(_ tap: AudioTap, dt: Float) {
        gain += (volume - gain) * (1 - exp(-dt / 0.15))
        if rate != tap.sampleRate { configure(tap.sampleRate) }
        tap.latest(samples, samplesR, count: n)
        if muted {
            vDSP_vclr(samples, 1, vDSP_Length(n))
            vDSP_vclr(samplesR, 1, vDSP_Length(n))
        }
        let total = spectrum(samples, into: &db) + spectrum(samplesR, into: &dbR)
        let silent = total < 2e-7
        quiet = silent ? quiet + dt : 0
        let awake: Float = quiet < 1.0 ? 1 : 0
        activity += (awake - activity) * (1 - exp(-dt / (awake > activity ? 0.12 : 0.7)))

        let top = max(db.max() ?? -120, dbR.max() ?? -120)
        ref = max(-48, top > ref ? ref + (top - ref) * (1 - exp(-dt / 0.05)) : ref - 3 * dt)
        let up = 1 - exp(-dt / response.attack), down = 1 - exp(-dt / response.release)
        var sum: Float = 0
        for b in 0..<Analyzer.bands {
            var l = min(1, max(0, 1 - (ref - db[b]) / 30)), r = min(1, max(0, 1 - (ref - dbR[b]) / 30))
            l *= l
            r *= r
            left[b] += (l - left[b]) * (l > left[b] ? up : down)
            right[b] += (r - right[b]) * (r > right[b] ? up : down)
            sum += left[b] + right[b]
        }
        level += (sum / Float(2 * Analyzer.bands) - level) * (1 - exp(-dt / 0.08))

        let low = muted ? -120 : 10 * log10(tap.bass.take() + 1e-12)
        // follow the track's loud peaks so kicks register whatever the mastering level
        bassRef = max(-24, max(low, bassRef - 3 * dt))      // faint leakage never counts as "loud"
        let hit = low < -60 ? 0 : pow(min(1, max(0, (low - (bassRef - 15)) / 15)), 1.6)
        bass += (hit - bass) * (1 - exp(-dt / (hit > bass ? 0.01 : 0.25)))
    }

    /// A made-up song (a kick under slowly moving mids that differ between the channels, with neighbouring bands
    /// as uneven as real music's) for `AURA_DEMO=1` and `--snapshot`. `right`: how loud its right channel is.
    func demo(time t: Float, dt: Float, right loud: Float = 1) {
        let kick = pow(max(0, sin(t * 2 * .pi)), 8)
        var sum: Float = 0
        for b in 0..<Analyzer.bands {
            let x = Float(b) / Float(Analyzer.bands - 1)
            for side in 0..<2 {
                let o = Float(side) * 1.9
                let rough = sin(Float(b) * 12.9898 + (t * 10).rounded(.down) * 78.233 + o) * 43758.5453
                let v = 0.2 * (1 - x) + 0.3 + 0.3 * sin(t * 1.7 + x * 9 + o) * sin(t * 0.9 + x * 4)
                    + kick * 0.7 * max(0, 1 - x * 4) + 0.35 * (rough - rough.rounded(.down) - 0.5)
                let level = min(1, max(0, v)) * (side == 0 ? 1 : loud)
                if side == 0 { left[b] = level } else { right[b] = level }
                sum += level
            }
        }
        level = sum / Float(2 * Analyzer.bands)
        bass = kick
        activity = 1
        gain += (volume - gain) * (1 - exp(-dt / 0.15))
    }
}

// MARK: - the line

/// Builds an outline as closely spaced points, in a window's coordinates (origin top left, y down).
/// Walk it with the screen content on the `perp` side: left to right under the notch, clockwise around a display.
struct Outline {
    var pts: [Vec] = []
    let step: Float

    mutating func move(_ p: Vec) { pts = [p] }

    mutating func line(_ p: Vec) {
        let a = pts[pts.count - 1]
        let n = max(1, Int((len(p - a) / step).rounded(.up)))
        for k in 1...n { pts.append(a + (p - a) * (Float(k) / Float(n))) }
    }

    /// An arc that starts where the outline currently ends.
    mutating func arc(_ c: Vec, _ r: Float, _ a0: Float, _ a1: Float) {
        let n = max(4, Int((abs(a1 - a0) * r / min(step, 1.5)).rounded(.up)))
        for k in 1...n {
            let a = a0 + (a1 - a0) * Float(k) / Float(n)
            pts.append(c + Vec(cos(a), sin(a)) * r)
        }
    }
}

/// The camera housing, in points from the screen's top left corner.
struct Notch {
    let x0: Float, x1: Float, h: Float
}

func notch(of s: NSScreen) -> Notch? {
    guard s.safeAreaInsets.top > 0, let l = s.auxiliaryTopLeftArea, let r = s.auxiliaryTopRightArea else { return nil }
    return Notch(x0: Float(l.maxX - s.frame.minX), x1: Float(r.minX - s.frame.minX), h: Float(s.safeAreaInsets.top))
}

/// The chain of points the renderer draws.
/// Every point rests on the outline. Audio pushes it along its normal by its band's level, in the left channel
/// for points on the left half of the display and the right channel on the right half; the cursor pulls it
/// sideways on a slightly underdamped spring, so the line reaches for the pointer and wobbles when it lets go.
///
/// To move like a liquid rather than a bar graph, the push is smoothed twice. Along the line, the band levels
/// are blurred (see `blur`) so no band stands out from its neighbours as a spike. In time, each point follows
/// its level on a spring of its own, so the line swells and settles instead of jumping.
///
/// Where the bass lifts the middle of the line it also gives off `heat`, which the renderer draws as haze.
final class Line {
    let closed: Bool
    let count: Int
    let rest: [Vec], normal: [Vec]
    /// Where each point reads the spectrum: 0 = lowest band … 1 = highest.
    let u: [Float]
    /// Which channel each point plays: 0 = left … 1 = right, crossing over within a few pt of the display's middle.
    let pan: [Float]
    /// 0 at the loose ends of an open line, 1 elsewhere.
    let fade: [Float]
    /// The furthest audio may push a point. Inside a bend the normals run into each other, so the line would
    /// fold over itself if it moved further than the bend's radius. It may move half of that there, and more
    /// the further from the bend it is (slowly at first), so the line tucks into corners like a liquid's edge.
    /// Half way between two bends, where "the nearest bend" changes sides, the limit is rounded off: a peak in
    /// it would show as a corner in the line.
    let cap: [Float]
    let arc: [Float]
    let total: Float
    let maxAmp: Float
    /// 1 along the middle of the notch's underside (where the bass plays), falling to 0 by its corners.
    let centre: [Float]
    /// Whether the line gives off haze at all.
    var haze = true

    private(set) var pos: [Vec]
    private(set) var energy: [Float], vis: [Float]
    /// How much haze each point gives off, 0…1: how far the bass has lifted it, lingering for a moment after.
    private(set) var heat: [Float]
    private var off: [Vec], vel: [Vec]      // the cursor's pull
    private var push: [Float], rate: [Float], want: [Float]     // the audio's push, its speed, and its target
    private let mix: [Float]    // blur weight of each point: how much of itself replaces the running average
    private let ripple: Float   // the travelling ripple's wavenumber, rad/pt

    /// `wing`: how much of each end of an open line fades out. `smooth`: blur length along the line, pt.
    /// `hot`: how far either side of the line's middle (its start, if closed) the haze reaches, pt.
    /// `mid`: x of the display's middle, where the left channel hands over to the right.
    init(_ pts: [Vec], closed: Bool, wing: Float, maxAmp: Float, smooth: Float, hot: Float, mid: Float) {
        let n = pts.count
        self.closed = closed
        self.maxAmp = maxAmp
        count = n
        rest = pts
        func at(_ i: Int) -> Vec { closed ? pts[(i + n) % n] : pts[min(n - 1, max(0, i))] }

        var arc = [Float](repeating: 0, count: n)
        for i in 1..<n { arc[i] = arc[i - 1] + len(pts[i] - pts[i - 1]) }
        let total = arc[n - 1] + (closed ? len(pts[0] - pts[n - 1]) : 0)
        self.arc = arc
        self.total = total

        // bend: how far along the line the nearest inside bend is; room: half that bend's radius
        var normal = [Vec](repeating: Vec(0, 1), count: n)
        var bend = [Float](repeating: 1e6, count: n), room = [Float](repeating: 1e6, count: n)
        for i in 0..<n {
            normal[i] = perp(unit(at(i + 1) - at(i - 1)))
            let turn = unit(at(i + 1) - at(i)) - unit(at(i) - at(i - 1))
            let k = 2 * (turn.x * normal[i].x + turn.y * normal[i].y) / max(1e-3, len(at(i + 1) - at(i - 1)))
            if k > 0.004 {
                bend[i] = 0
                room[i] = 0.5 / k
            }
        }
        func spread(from a: Int, to b: Int) {
            let d = bend[a] + len(pts[b] - pts[a])
            if d < bend[b] {
                bend[b] = d
                room[b] = room[a]
            } else if bend[a] == 0 && bend[b] == 0 {
                room[b] = min(room[b], room[a])     // one bend, one radius: where it starts it only seems wider
            }
        }
        for _ in 0..<2 {
            for i in 1..<(closed ? n + 1 : n) { spread(from: i - 1, to: i % n) }
            for i in stride(from: closed ? n - 1 : n - 2, through: 0, by: -1) { spread(from: (i + 1) % n, to: i) }
        }
        self.normal = normal
        mix = (0..<n).map { 1 - exp(-len(pts[$0] - at($0 - 1)) / smooth) }
        var cap = room
        for i in 0..<n {
            let d = bend[i]
            cap[i] += 0.35 * d * d / (d + 24)
        }
        // blurring lowers peaks and raises dips: keep the lower of the two, so only the peaks change
        var soft = cap
        Line.blur(&soft, mix: mix, closed: closed)
        self.cap = (0..<n).map { min(cap[$0], soft[$0]) }
        pan = pts.map { smoothstep(mid - 18, mid + 18, $0.x) }

        if closed {
            // bass at the top centre (where the line starts) and bottom centre, the highs half way down each side
            u = arc.map { 1 - abs(2 * min($0, total - $0) / (total / 2) - 1) }
            fade = [Float](repeating: 1, count: n)
        } else {
            // bass under the middle, the highs out on the wings
            u = arc.map { abs($0 - total / 2) / (total / 2) }
            fade = arc.map { smoothstep(0, wing, min($0, total - $0)) }
        }
        // ~0.05 rad/pt; around a closed line a whole number of waves, or they wouldn't meet where it closes
        ripple = closed ? 2 * .pi * (total * 0.05 / (2 * .pi)).rounded() / total : 0.05
        centre = arc.map { 1 - smoothstep(0.4 * hot, hot, closed ? min($0, total - $0) : abs($0 - total / 2)) }
        pos = pts
        energy = [Float](repeating: 0, count: n)
        heat = energy
        vis = [Float](repeating: 0, count: n)
        off = [Vec](repeating: Vec(0, 0), count: n)
        vel = off
        push = energy
        rate = energy
        want = energy
    }

    /// Blurs `x` along the line: a running average swept forward, then backward, twice over. One sweep each way
    /// would leave a crease where a lone peak was; two give a bell shape about four blur lengths wide.
    private static func blur(_ x: inout [Float], mix: [Float], closed: Bool) {
        let n = x.count
        for _ in 0..<2 {
            // a closed line has no first point: go round once to warm the average up, then once for real
            var y = x[closed ? n - 1 : 0]
            if closed { for i in 0..<n { y += mix[i] * (x[i] - y) } }
            for i in 0..<n {
                y += mix[i] * (x[i] - y)
                x[i] = y
            }
            if closed { for i in stride(from: n - 1, through: 0, by: -1) { y += mix[(i + 1) % n] * (x[i] - y) } }
            for i in stride(from: n - 1, through: 0, by: -1) {
                y += mix[(i + 1) % n] * (x[i] - y)
                x[i] = y
            }
        }
    }

    /// Catmull-Rom through the bands, so neighbouring points read a smooth curve rather than 40 steps.
    private func sample(_ b: [Float], _ x: Float) -> Float {
        let f = min(1, max(0, x)) * Float(b.count - 1)
        let i = min(Int(f), b.count - 2)
        let t = f - Float(i)
        let p0 = b[max(i - 1, 0)], p1 = b[i], p2 = b[i + 1], p3 = b[min(i + 2, b.count - 1)]
        return max(0, p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0))))
    }

    /// Advances the line by `dt`. `cursor` is nil when the magnet is off.
    /// Returns how visible the most visible point is, so the caller can skip drawing a line nobody can see.
    func step(_ a: Analyzer, cursor: Vec?, time: Float, dt: Float) -> Float {
        let n = count
        for i in 0..<n { want[i] = sample(a.left, u[i]) * (1 - pan[i]) + sample(a.right, u[i]) * pan[i] }
        Line.blur(&want, mix: mix, closed: closed)
        // the audio's spring: ~3.5 Hz unless a faster response is chosen, damping ratio 0.8 (a touch of overshoot,
        // like a surface settling). A stiff spring needs smaller steps than a frame to stay stable.
        let w = a.response.spring
        let ka = w * w, ca = 1.6 * w
        let steps = max(1, Int((w * dt / 0.4).rounded(.up)))
        let h = dt / Float(steps)
        // the cursor's spring: ~2.4 Hz, damping ratio 0.45
        let k: Float = 220, c: Float = 13
        // The cursor takes hold of the nearest point of the line and draws it most of the way over (four fifths
        // when close, less the further away it is, up to ~19 pt); the points around follow in the same direction,
        // less and less. Because they all move the same way the line rises as a soft mound, never to a point.
        var hold = Vec(0, 0), anchor = Vec(0, 0)
        if let cur = cursor {
            var best = Float.infinity
            for i in 0..<n {
                let to = cur - rest[i]
                let d2 = to.x * to.x + to.y * to.y
                if d2 < best {
                    best = d2
                    anchor = rest[i]
                }
            }
            hold = (cur - anchor) * (0.8 * exp(-best / (55 * 55)))
        }
        // haze gathers within a few frames of a kick and takes about half a second to clear
        let warm = 1 - exp(-dt / 0.05), cool = 1 - exp(-dt / 0.45)
        let kick = haze ? (0.3 + 0.7 * a.bass) * a.activity : 0
        var seen: Float = 0
        for i in 0..<n {
            let v = want[i]
            // a slow ripple travelling along the line keeps held notes alive; the lower the volume, the less it all moves
            let raw = fade[i] * (v * maxAmp * (1 + 0.1 * sin(arc[i] * ripple - time * 2.6)) + a.bass * 2) * a.gain
            // tanh: eases into the limit instead of hitting it, which would show as a kink in the line
            let target = cap[i] * tanh(raw / cap[i]) * a.activity
            for _ in 0..<steps {
                rate[i] += (ka * (target - push[i]) - ca * rate[i]) * h
                push[i] = max(0, push[i] + rate[i] * h)
            }
            let hot = centre[i] * min(1, push[i] / maxAmp) * kick
            heat[i] += (hot - heat[i]) * (hot > heat[i] ? warm : cool)

            var pull = Vec(0, 0), near: Float = 0
            if let cur = cursor {
                let from = rest[i] - anchor, to = cur - rest[i]
                pull = hold * exp(-(from.x * from.x + from.y * from.y) / (40 * 40))
                near = exp(-(to.x * to.x + to.y * to.y) / (70 * 70))
            }
            vel[i] += (k * (pull - off[i]) - c * vel[i]) * dt
            off[i] += vel[i] * dt
            pos[i] = rest[i] + normal[i] * push[i] + off[i]
            // the cursor wakes the line where it is, even in silence
            vis[i] = fade[i] * max(a.activity, 0.8 * near)
            energy[i] = max(v * a.activity * (0.4 + 0.6 * a.gain), 0.55 * near)
            seen = max(seen, vis[i])
        }
        return seen
    }
}

/// Everything that depends on the display and the chosen style: where the overlay window sits and the line in it.
final class Stage {
    let screen: NSScreen
    let edges: Bool
    let frame: CGRect       // the overlay window, in screen coordinates
    let line: Line
    let radius: Float       // how far the glow reaches from the line, pt
    let extent: Float       // how far anything is drawn from the line (the haze drifts past the glow), pt
    let width: Float        // the glass tube's half-width at rest, pt
    let cycles: Float       // times the palette repeats along the line
    let notchRect: CGRect?  // the notch in the window's coordinates (origin top left)

    init(screen: NSScreen, edges: Bool) {
        self.screen = screen
        self.edges = edges
        let w = Float(screen.frame.width), h = Float(screen.frame.height)
        let nt = notch(of: screen)
        let gap: Float = 1.6        // the line rests this far outside the notch / inside the display's edge
        let join: Float = 9         // radius of the bend where the notch meets the top edge
        let chin: Float = 10        // radius of the notch's bottom corners
        let corner: Float = 9       // radius of the display's corners

        /// From the top edge left of the notch, down around it, back up to the top edge on its right.
        func aroundNotch(_ o: inout Outline, _ nt: Notch, dx: Float) {
            let l = nt.x0 - gap - dx, r = nt.x1 + gap - dx, bot = nt.h + gap
            o.line(Vec(l - join, gap))
            o.arc(Vec(l - join, gap + join), join, -.pi / 2, 0)
            o.line(Vec(l, bot - chin))
            o.arc(Vec(l + chin, bot - chin), chin, .pi, .pi / 2)
            o.line(Vec(r - chin, bot))
            o.arc(Vec(r - chin, bot - chin), chin, .pi / 2, 0)
            o.line(Vec(r, gap + join))
            o.arc(Vec(r + join, gap + join), join, .pi, 1.5 * .pi)
        }

        // the haze rises from the middle of the notch's underside
        let hot = nt.map { ($0.x1 - $0.x0) / 2 } ?? 95
        if edges {
            radius = 36
            extent = 48
            width = 1.5
            cycles = 3
            frame = screen.frame
            var o = Outline(step: 5)
            // clockwise from the top centre; with a notch that's the middle of its underside
            if let nt = nt {
                let r = nt.x1 + gap, bot = nt.h + gap
                o.move(Vec(w / 2, bot))
                o.line(Vec(r - chin, bot))
                o.arc(Vec(r - chin, bot - chin), chin, .pi / 2, 0)
                o.line(Vec(r, gap + join))
                o.arc(Vec(r + join, gap + join), join, .pi, 1.5 * .pi)
            } else {
                o.move(Vec(w / 2, gap))
            }
            o.line(Vec(w - gap - corner, gap))
            o.arc(Vec(w - gap - corner, gap + corner), corner, -.pi / 2, 0)
            o.line(Vec(w - gap, h - gap - corner))
            o.arc(Vec(w - gap - corner, h - gap - corner), corner, 0, .pi / 2)
            o.line(Vec(gap + corner, h - gap))
            o.arc(Vec(gap + corner, h - gap - corner), corner, .pi / 2, .pi)
            o.line(Vec(gap, gap + corner))
            o.arc(Vec(gap + corner, gap + corner), corner, .pi, 1.5 * .pi)
            if let nt = nt {
                let l = nt.x0 - gap, bot = nt.h + gap
                o.line(Vec(l - join, gap))
                o.arc(Vec(l - join, gap + join), join, -.pi / 2, 0)
                o.line(Vec(l, bot - chin))
                o.arc(Vec(l + chin, bot - chin), chin, .pi, .pi / 2)
            }
            o.line(Vec(w / 2, o.pts[o.pts.count - 1].y))
            o.pts.removeLast()      // it's the starting point again
            line = Line(o.pts, closed: true, wing: 0, maxAmp: 24, smooth: 13, hot: hot, mid: w / 2)
            notchRect = nt.map { CGRect(x: CGFloat($0.x0), y: 0, width: CGFloat($0.x1 - $0.x0), height: CGFloat($0.h)) }
        } else {
            radius = 22
            extent = 36
            width = 1.3
            cycles = 1
            let amp: Float = 17, wing: Float = 110
            let pad = extent + 4
            var o = Outline(step: 2)
            var x0: Float, span: Float, top: Float
            if let nt = nt {
                x0 = nt.x0 - gap - join - wing - pad
                span = nt.x1 - nt.x0 + 2 * (gap + join + wing + pad)
                top = nt.h
                o.move(Vec(pad, gap))
                aroundNotch(&o, nt, dx: x0)
                o.line(Vec(span - pad, gap))
                notchRect = CGRect(x: CGFloat(nt.x0 - x0), y: 0, width: CGFloat(nt.x1 - nt.x0), height: CGFloat(nt.h))
            } else {
                // no notch: a straight line under the middle of the top edge
                span = 2 * (230 + pad)
                x0 = w / 2 - span / 2
                top = 0
                o.move(Vec(pad, gap))
                o.line(Vec(span - pad, gap))
                notchRect = nil
            }
            let height = top + gap + amp + 26 + pad      // 26: room for the cursor to pull the line down
            frame = CGRect(x: screen.frame.minX + CGFloat(x0), y: screen.frame.maxY - CGFloat(height),
                           width: CGFloat(span), height: CGFloat(height))
            line = Line(o.pts, closed: false, wing: wing * 0.9, maxAmp: amp, smooth: 7, hot: hot, mid: w / 2 - x0)
        }
    }
}

// MARK: - rendering

/// Four colours the line cycles through along its length (and slowly over time).
enum Palette: Int, CaseIterable {
    case siri, aurora, sunset, ice
    case album      // the colours of the cover of what Apple Music is playing (see `AlbumArt`)

    var title: String { ["Siri", "Aurora", "Sunset", "Ice", "Album Art (Apple Music)"][rawValue] }

    var colors: [SIMD4<Float>] {
        switch self {
        // album: what the line wears until a cover has been read
        case .siri, .album: return [SIMD4(1.00, 0.18, 0.56, 1), SIMD4(0.62, 0.32, 1.00, 1),
                              SIMD4(0.16, 0.56, 1.00, 1), SIMD4(1.00, 0.52, 0.16, 1)]
        case .aurora: return [SIMD4(0.10, 1.00, 0.70, 1), SIMD4(0.14, 0.68, 1.00, 1),
                              SIMD4(0.56, 0.40, 1.00, 1), SIMD4(0.24, 1.00, 0.45, 1)]
        case .sunset: return [SIMD4(1.00, 0.36, 0.20, 1), SIMD4(1.00, 0.16, 0.50, 1),
                              SIMD4(1.00, 0.72, 0.20, 1), SIMD4(0.76, 0.26, 0.90, 1)]
        case .ice:    return [SIMD4(0.62, 0.84, 1.00, 1), SIMD4(0.92, 0.96, 1.00, 1),
                              SIMD4(0.50, 0.74, 1.00, 1), SIMD4(0.80, 0.90, 1.00, 1)]
        }
    }
}

struct Vertex {
    var pos: Vec        // this corner of the segment's quad, pt
    var a: Vec, b: Vec  // the segment's two ends, pt
    var t: Vec          // palette position at a / b
    var e: Vec          // energy at a / b
    var v: Vec          // visibility at a / b
    var z: Vec          // heat at a / b
}

struct Uniforms {
    var size: Vec       // the target's size, pt
    var time: Float, radius: Float, width: Float, level: Float
    var px: Float       // half a pixel, in pt
    var perPt: Float    // palette position per pt of line
    var extent: Float   // half the width of a segment's quad, pt
    var wrap: Float     // the length of a closed line, pt; 0 for an open one
    var pad: (Float, Float) = (0, 0)
    var pal: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
}

/// Each segment of the line is drawn as a quad reaching `extent` to either side; the fragment shader measures the
/// distance from the pixel to the segment and shades by it. Where quads overlap (inside tight bends) the depth
/// buffer — which holds that distance — keeps whichever segment is nearest, so the line looks like one stroke.
///
/// Within `width` of the line is the tube: a tinted body, bright rims, a specular streak on the side facing the
/// bezel and a little shade on the other. Beyond it the glow: longer and stronger toward the bezel, so the space
/// the audio opens up between the line and the edge fills with colour, and a hair of shadow right at the tube so
/// it stays readable over white windows.
///
/// Where the line is hot (see `Line.heat`) it also gives off haze, like the air over a flame: the glow spreads
/// and its edge wavers, and wisps of the line's colour drift away from it into the screen, thinning as they go.
/// The wisps are noise sampled in the line's own coordinates (pt along it, pt out from it) and scrolled outward.
/// A closed line starts and ends in the middle of the haze, so there "along it" counts back from its end.
let shaderSource = """
#include <metal_stdlib>
using namespace metal;

struct Vertex { float2 pos, a, b, t, e, v, z; };
struct Uniforms {
    float2 size;
    float time, radius, width, level, px, perPt, extent, wrap;
    float4 pal[4];
};
struct VOut {
    float4 position [[position]];
    float2 p;
    float2 a [[flat]];
    float2 b [[flat]];
    float2 t [[flat]];
    float2 e [[flat]];
    float2 v [[flat]];
    float2 z [[flat]];
};
struct FOut {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

vertex VOut vert(uint vid [[vertex_id]], const device Vertex *verts [[buffer(0)]],
                 constant Uniforms &u [[buffer(1)]]) {
    Vertex v = verts[vid];
    VOut o;
    o.position = float4(v.pos.x / u.size.x * 2.0 - 1.0, 1.0 - v.pos.y / u.size.y * 2.0, 0.5, 1.0);
    o.p = v.pos; o.a = v.a; o.b = v.b; o.t = v.t; o.e = v.e; o.v = v.v; o.z = v.z;
    return o;
}

static float hash(float2 p) {
    p = fract(p * float2(0.1031, 0.1030));
    p += dot(p, p.yx + 33.33);
    return fract((p.x + p.y) * p.y);
}

static float noise(float2 p) {
    float2 i = floor(p), f = p - i;
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(mix(hash(i), hash(i + float2(1.0, 0.0)), f.x),
               mix(hash(i + float2(0.0, 1.0)), hash(i + float2(1.0, 1.0)), f.x), f.y);
}

// soft clouds, 0…1, around 0.5
static float cloud(float2 p) {
    return 0.65 * noise(p) + 0.35 * noise(p * 2.03 + 17.0);
}

static float3 palette(float x, constant Uniforms &u) {
    x = fract(x) * 4.0;
    int i = int(x);
    float f = x - float(i);
    f = f * f * (3.0 - 2.0 * f);
    return mix(u.pal[i & 3].rgb, u.pal[(i + 1) & 3].rgb, f);
}

fragment FOut frag(VOut in [[stage_in]], constant Uniforms &u [[buffer(1)]]) {
    FOut o;
    float2 ab = in.b - in.a;
    float h = saturate(dot(in.p - in.a, ab) / max(dot(ab, ab), 1e-6));
    float2 dv = in.p - (in.a + h * ab);
    float dist = length(dv);
    float vis = mix(in.v.x, in.v.y, h);
    if (dist >= u.extent || vis < 0.003) discard_fragment();
    float side = ab.x * dv.y - ab.y * dv.x >= 0.0 ? 1.0 : -1.0;     // +1: toward the screen content
    float along = mix(in.t.x, in.t.y, h);
    float t = along + u.time * 0.06;
    float e = mix(in.e.x, in.e.y, h);
    float z = mix(in.z.x, in.z.y, h);

    float w = u.width * (1.0 + 0.8 * e);
    float x = side * dist / w;
    float cover = smoothstep(w + u.px, w - u.px, dist);

    // the glass tube
    float3 col = palette(t, u);
    float rim = smoothstep(0.45, 1.0, abs(x));
    float spec = exp(-pow((x + 0.38) / 0.3, 2.0));
    float3 tube = mix(col, float3(1.0), 0.5) * (1.0 - 0.25 * smoothstep(-0.2, 1.0, x)) + rim * 0.3 + spec * 0.7;
    float tubeA = saturate(0.6 + 0.25 * e + 0.25 * rim + 0.3 * spec);

    // the glow around it
    float g = max(dist - w, 0.0);
    float gg = g, wisp = 0.0;
    if (z > 0.003) {
        // pt along the line (swaying, as rising air does), and pt out from it less how far the haze has drifted
        float s = along / u.perPt;
        if (s > u.wrap * 0.5) s -= u.wrap;
        float2 q = float2(s + 3.0 * sin(dist * 0.13 - u.time * 4.0), side * dist - u.time * 16.0);
        float shimmer = cloud(float2(q.x * 0.09, q.y * 0.08 - u.time * 0.7)) - 0.5;
        gg = max(g + shimmer * 10.0 * z * smoothstep(0.0, 6.0, g), 0.0);
        wisp = smoothstep(0.15, 0.85, cloud(float2(q.x * 0.04 + 31.0, q.y * 0.03)));    // drawn out along the drift
    }
    float reach = u.radius * (side < 0.0 ? 0.26 : 0.16) * (0.55 + 0.6 * e) * (1.0 + 1.3 * z);
    float glow = (0.3 + 0.7 * e) * (0.7 + 0.6 * u.level) * exp(-gg / reach)
        * smoothstep(u.radius * (1.0 + 0.4 * z), u.radius * 0.5, dist);
    float3 glowCol = palette(t + 0.08 + 0.3 * g / u.radius, u);
    float shadow = 0.2 * exp(-g / 1.1);

    float3 rgb = glowCol * glow;
    float a = glow * 0.8;               // less alpha than colour: the glow adds light rather than paint
    if (z > 0.003) {
        float haze = z * (0.4 + 0.6 * wisp) * exp(-gg / (u.extent * (0.16 + 0.24 * z)))
            * (side < 0.0 ? 0.5 : 1.0) * smoothstep(u.extent, u.extent * 0.55, dist);
        rgb += mix(palette(t + 0.2 + 0.4 * g / u.extent, u), float3(1.0), 0.3) * haze * 0.6;
        a += haze * 0.45;
    }
    a += shadow * (1.0 - a);
    float k = tubeA * cover;
    rgb = tube * k + rgb * (1.0 - k);
    a = k + a * (1.0 - k);

    o.color = float4(min(rgb, float3(1.0)), min(a, 1.0)) * vis;
    o.depth = dist / u.extent;
    return o;
}
"""

final class Renderer {
    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private var buffers: [MTLBuffer?] = [nil, nil, nil]     // rotated so a frame still on the GPU isn't rewritten
    private var frame = 0
    private var depth: MTLTexture?
    private var mitre: [Vec] = []
    var palette = Palette.siri
    /// The current cover's colours, for the album palette; nil until one has been read.
    var album: [SIMD4<Float>]?
    private var shown: [SIMD4<Float>] = []      // the colours on screen, easing toward the chosen ones
    private var shownAt: Float = 0

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.queue = queue
        do {
            let lib = try device.makeLibrary(source: shaderSource, options: nil)
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = lib.makeFunction(name: "vert")
            d.fragmentFunction = lib.makeFunction(name: "frag")
            d.colorAttachments[0].pixelFormat = .bgra8Unorm
            d.depthAttachmentPixelFormat = .depth16Unorm
            pipeline = try device.makeRenderPipelineState(descriptor: d)
        } catch {
            NSLog("Aura: couldn't build the shader: \(error)")
            return nil
        }
        let ds = MTLDepthStencilDescriptor()
        ds.depthCompareFunction = .less
        ds.isDepthWriteEnabled = true
        guard let s = device.makeDepthStencilState(descriptor: ds) else { return nil }
        depthState = s
    }

    /// Fills the next vertex buffer with one quad per segment. Returns the buffer and its vertex count.
    private func build(_ stage: Stage) -> (MTLBuffer, Int)? {
        let line = stage.line, n = line.count, pos = line.pos
        let segs = line.closed ? n : n - 1
        let bytes = segs * 6 * MemoryLayout<Vertex>.stride
        frame = (frame + 1) % buffers.count
        if buffers[frame] == nil || buffers[frame]!.length < bytes {
            buffers[frame] = device.makeBuffer(length: bytes, options: .storageModeShared)
        }
        guard let buf = buffers[frame] else { return nil }

        // corners shared by neighbouring quads sit on the mitre of the two segments, so quads meet without gaps
        if mitre.count != n { mitre = [Vec](repeating: Vec(0, 0), count: n) }
        for i in 0..<n {
            let a = pos[i > 0 ? i - 1 : (line.closed ? n - 1 : 0)], b = pos[i < n - 1 ? i + 1 : (line.closed ? 0 : n - 1)]
            var t0 = unit(pos[i] - a), t1 = unit(b - pos[i])
            if t0 == Vec(0, 0) { t0 = t1 }
            if t1 == Vec(0, 0) { t1 = t0 }
            let n1 = perp(t1)
            var m = perp(t0) + n1
            let l = len(m)
            if l < 0.2 { m = n1 } else {
                m /= l
                m /= max(m.x * n1.x + m.y * n1.y, 0.35)
            }
            mitre[i] = m * stage.extent
        }

        let v = buf.contents().bindMemory(to: Vertex.self, capacity: segs * 6)
        let perPt = stage.cycles / line.total
        for i in 0..<segs {
            let j = (i + 1) % n
            let ta = line.arc[i] * perPt
            // b's palette position continues from a's, so the segment that closes a loop doesn't run backwards
            let t = Vec(ta, ta + len(line.rest[j] - line.rest[i]) * perPt)
            let e = Vec(line.energy[i], line.energy[j]), vis = Vec(line.vis[i], line.vis[j])
            let z = Vec(line.heat[i], line.heat[j])
            // two triangles: a−, a+, b− and b−, a+, b+
            var q = Vertex(pos: pos[i] - mitre[i], a: pos[i], b: pos[j], t: t, e: e, v: vis, z: z)
            let o = i * 6
            v[o] = q
            q.pos = pos[i] + mitre[i]
            v[o + 1] = q
            v[o + 4] = q
            q.pos = pos[j] - mitre[j]
            v[o + 2] = q
            v[o + 3] = q
            q.pos = pos[j] + mitre[j]
            v[o + 5] = q
        }
        return (buf, segs * 6)
    }

    /// Draws the stage's line into `target` (cleared to transparent first). With `stage` nil it only clears.
    /// Returns the command buffer, uncommitted, so the caller can present a drawable or wait for it.
    func draw(_ stage: Stage?, into target: MTLTexture, scale: Float, time: Float, level: Float) -> MTLCommandBuffer? {
        if depth == nil || depth!.width != target.width || depth!.height != target.height {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth16Unorm, width: target.width,
                                                             height: target.height, mipmapped: false)
            d.usage = .renderTarget
            d.storageMode = device.hasUnifiedMemory ? .memoryless : .private
            depth = device.makeTexture(descriptor: d)
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.clearDepth = 1
        pass.depthAttachment.storeAction = .dontCare
        guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return nil }
        if let stage = stage, let (buf, count) = build(stage) {
            // a change of palette (or of song) melts into the new colours over about a second
            let want = palette == .album ? album ?? palette.colors : palette.colors
            let ease = shown.isEmpty ? 1 : 1 - exp(-max(0, time - shownAt) / 0.4)
            if shown.isEmpty { shown = want }
            for i in 0..<4 { shown[i] += (want[i] - shown[i]) * ease }
            shownAt = time
            let p = shown
            var u = Uniforms(size: Vec(Float(target.width) / scale, Float(target.height) / scale), time: time,
                             radius: stage.radius, width: stage.width, level: level, px: 0.5 / scale,
                             perPt: stage.cycles / stage.line.total, extent: stage.extent,
                             wrap: stage.line.closed ? stage.line.total : 0,
                             pal: (p[0], p[1], p[2], p[3]))
            enc.setRenderPipelineState(pipeline)
            enc.setDepthStencilState(depthState)
            enc.setVertexBuffer(buf, offset: 0, index: 0)
            enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: count)
        }
        enc.endEncoding()
        return cmd
    }
}

final class MetalView: NSView {
    let metal = CAMetalLayer()

    init(device: MTLDevice) {
        super.init(frame: .zero)
        metal.device = device
        metal.pixelFormat = .bgra8Unorm
        metal.isOpaque = false
        metal.framebufferOnly = true
        layer = metal
        wantsLayer = true
    }
    required init?(coder: NSCoder) { fatalError() }

    func fit(scale: CGFloat) {
        metal.contentsScale = scale
        metal.drawableSize = CGSize(width: bounds.width * scale, height: bounds.height * scale)
    }
}

/// A see-through window that floats over everything (menu bar and full-screen apps included) and never takes a click.
func makeOverlay(above: Int) -> NSWindow {
    let w = NSWindow(contentRect: .zero, styleMask: .borderless, backing: .buffered, defer: false)
    w.isOpaque = false
    w.backgroundColor = .clear
    w.hasShadow = false
    w.ignoresMouseEvents = true
    w.level = NSWindow.Level(rawValue: NSWindow.Level.statusBar.rawValue + above)
    w.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary, .ignoresCycle]
    w.isReleasedWhenClosed = false
    w.animationBehavior = .none
    return w
}

// MARK: - album colours

/// A colour from a cover: its hue and how saturated it is, both 0…1.
struct Swatch {
    let hue: Float, sat: Float
}

/// The colours of whatever Apple Music is playing, read from the track's cover.
/// Music announces every track change to the whole system, and listening to that needs no permission. The cover
/// itself is asked for over AppleScript, which macOS allows after a one-time "Aura wants to control Music" prompt.
final class AlbumArt {
    /// The cover's most prominent colours, up to four, strongest first. Empty until a cover has been read.
    private(set) var swatches: [Swatch] = []
    var onChange: (() -> Void)?
    /// macOS won't let us ask Music (Privacy & Security → Automation).
    var onDenied: (() -> Void)?
    private var token: NSObjectProtocol?
    private let queue = DispatchQueue(label: "aura.album", qos: .utility)
    private var round = 0       // bumped by every track change, so a slow read of the previous cover is dropped

    func start() {
        guard token == nil else { return }
        token = DistributedNotificationCenter.default().addObserver(
            forName: NSNotification.Name("com.apple.Music.playerInfo"), object: nil, queue: .main) { [weak self] _ in
            self?.read()
        }
        read()
    }

    func stop() {
        if let t = token { DistributedNotificationCenter.default().removeObserver(t) }
        token = nil
        round += 1
    }

    /// A streamed track's cover can arrive a moment after the track starts, so an empty answer is asked for again.
    private func read(attempt: Int = 0) {
        if attempt == 0 { round += 1 }
        let mine = round
        queue.asyncAfter(deadline: .now() + (attempt == 0 ? 0.2 : 1.5)) {
            let (found, denied) = AlbumArt.fetch()
            DispatchQueue.main.async {
                guard mine == self.round, self.token != nil else { return }
                if denied {
                    self.onDenied?()
                } else if let found {
                    self.swatches = found
                    self.onChange?()
                } else if attempt < 2 {
                    self.read(attempt: attempt + 1)
                }
            }
        }
    }

    /// Asks Music for the current track's cover and picks its colours. nil when there is no cover to be had.
    static func fetch() -> (swatches: [Swatch]?, denied: Bool) {
        // asking would launch Music if it weren't running
        guard !NSRunningApplication.runningApplications(withBundleIdentifier: "com.apple.Music").isEmpty else { return (nil, false) }
        let source = """
        tell application id "com.apple.Music"
            try
                return raw data of artwork 1 of current track
            on error number n
                return n
            end try
        end tell
        """
        var error: NSDictionary?
        guard let answer = NSAppleScript(source: source)?.executeAndReturnError(&error), error == nil else { return (nil, false) }
        if answer.descriptorType == typeSInt32 { return (nil, answer.int32Value == -1743) }     // -1743: not permitted
        let options = [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: 64] as CFDictionary
        guard let src = CGImageSourceCreateWithData(answer.data as CFData, nil),
              let cover = CGImageSourceCreateThumbnailAtIndex(src, 0, options) else { return (nil, false) }
        return (swatches(of: cover), false)
    }

    /// The colours a cover is made of. Its pixels vote for their hue (in 24 bins of 15°), the more saturated and
    /// brighter the pixel the stronger its vote, greys and near-blacks not at all; the winning hues are returned
    /// with their average saturation. Neighbouring bins can't both win, so the colours are at least 15° apart.
    /// A cover with no colour to speak of (black and white) gives one swatch with no saturation.
    static func swatches(of cover: CGImage) -> [Swatch] {
        let n = 32, bins = 24
        var px = [UInt8](repeating: 0, count: n * n * 4)
        guard let ctx = CGContext(data: &px, width: n, height: n, bitsPerComponent: 8, bytesPerRow: n * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return [] }
        ctx.draw(cover, in: CGRect(x: 0, y: 0, width: n, height: n))
        var votes = [Float](repeating: 0, count: bins), x = votes, y = votes, sats = votes
        for i in 0..<(n * n) {
            let r = Float(px[i * 4]) / 255, g = Float(px[i * 4 + 1]) / 255, b = Float(px[i * 4 + 2]) / 255
            let v = max(r, g, b), d = v - min(r, g, b)
            guard v > 0.12, d / v > 0.15 else { continue }
            var h = v == r ? (g - b) / d : v == g ? 2 + (b - r) / d : 4 + (r - g) / d
            h = (h / 6).truncatingRemainder(dividingBy: 1)
            if h < 0 { h += 1 }
            let w = d / v * (0.3 + 0.7 * v), bin = min(bins - 1, Int(h * Float(bins)))
            votes[bin] += w
            x[bin] += w * cos(h * 2 * .pi)
            y[bin] += w * sin(h * 2 * .pi)
            sats[bin] += w * d / v
        }
        guard votes.reduce(0, +) > 0.02 * Float(n * n) else { return [Swatch(hue: 0, sat: 0)] }
        var found: [Swatch] = []
        var first: Float = 0
        while found.count < 4 {
            guard let top = votes.indices.max(by: { votes[$0] < votes[$1] }), votes[top] > 0.08 * first, votes[top] > 0 else { break }
            if found.isEmpty { first = votes[top] }
            var h = atan2(y[top], x[top]) / (2 * .pi)
            if h < 0 { h += 1 }
            found.append(Swatch(hue: h, sat: sats[top] / votes[top]))
            for b in [top - 1, top, top + 1] { votes[(b + bins) % bins] = 0 }
        }
        return found
    }
}

/// Four colours for the line from a cover's swatches: each hue at full brightness (a glow can't be dark), kept
/// vivid unless the cover itself is grey; with fewer than four swatches, paler tints of them fill in.
func lineColors(_ swatches: [Swatch]) -> [SIMD4<Float>]? {
    guard !swatches.isEmpty else { return nil }
    var c = swatches.map { s -> SIMD4<Float> in
        let sat = s.sat < 0.15 ? s.sat : min(0.95, max(0.55, s.sat))
        let k = s.hue * 6, f = k - k.rounded(.down)
        let p = 1 - sat, q = 1 - sat * f, t = 1 - sat * (1 - f)
        switch Int(k) % 6 {
        case 0: return SIMD4(1, t, p, 1)
        case 1: return SIMD4(q, 1, p, 1)
        case 2: return SIMD4(p, 1, t, 1)
        case 3: return SIMD4(p, q, 1, 1)
        case 4: return SIMD4(t, p, 1, 1)
        default: return SIMD4(1, p, q, 1)
        }
    }
    let k = c.count
    var i = 0
    while c.count < 4 {
        let base = c[i % k]
        c.append(i % 2 == 0 ? base + (SIMD4(0.82, 0.86, 0.9, 1) - base) * 0.5 : base)
        i += 1
    }
    return c
}

// MARK: - bass feel

/// The whole display thumps with the kick, like looking at a speaker cone.
///
/// Another app's pixels can't be moved, but they can be copied: ScreenCaptureKit hands us the screen (minus
/// Aura's own windows and the cursor) at half resolution, and on each kick a few faint copies of it are laid
/// over the real screen, each a little larger than the last. Stacked copies scaled about the centre are a zoom
/// blur: nothing moves in the middle, the smear grows toward the edges. The real screen stays live and sharp
/// underneath, so text is still readable mid-kick and there is no lag to notice.
final class BassFeel: NSObject, SCStreamOutput, SCStreamDelegate {
    /// 0 subtle, 1 balanced, 2 punchy
    var strength = 1
    private(set) var running = false
    /// The stream died on its own (display unplugged, permission revoked, …).
    var onStop: (() -> Void)?

    private let window = makeOverlay(above: 1)
    private let root = CALayer()
    private var ghosts: [CALayer] = []
    private var stream: SCStream?
    private let queue = DispatchQueue(label: "aura.capture", qos: .userInteractive)
    private var latest: IOSurfaceRef?
    private var fresh = false
    private var session = 0     // bumped by every stop, so a start that finishes late knows it was cancelled

    override init() {
        super.init()
        let view = NSView()
        view.layer = root
        view.wantsLayer = true
        root.masksToBounds = true
        root.isHidden = true
        for _ in 0..<4 {
            let g = CALayer()
            g.contentsGravity = .resize
            root.addSublayer(g)
            ghosts.append(g)
        }
        window.contentView = view
    }

    func start(on screen: NSScreen, done: @escaping (Error?) -> Void) {
        stop()
        let mine = session
        window.setFrame(screen.frame, display: false)
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        for g in ghosts { g.frame = CGRect(origin: .zero, size: screen.frame.size) }
        CATransaction.commit()
        window.orderFrontRegardless()

        let id = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID
        SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: false) { [weak self] content, error in
            DispatchQueue.main.async {
                guard let self, self.session == mine else { return }    // stopped or restarted meanwhile
                let fail: (Error) -> Void = { self.stop(); done($0) }
                guard let content, let display = content.displays.first(where: { $0.displayID == id }) else {
                    return fail(error ?? AudioError(msg: "couldn't find the display to capture"))
                }
                // Capturing our own copies of the screen would feed them back into themselves, an endless tunnel.
                let me = content.applications.filter { $0.processID == getpid() }
                guard !me.isEmpty else { return fail(AudioError(msg: "couldn't exclude Aura from the capture")) }
                let cfg = SCStreamConfiguration()
                cfg.width = display.width       // points, i.e. half the Retina resolution: the copies are blurred anyway
                cfg.height = display.height
                cfg.minimumFrameInterval = CMTime(value: 1, timescale: 60)
                cfg.pixelFormat = kCVPixelFormatType_32BGRA
                cfg.showsCursor = false
                cfg.queueDepth = 5
                let filter = SCContentFilter(display: display, excludingApplications: me, exceptingWindows: [])
                let s = SCStream(filter: filter, configuration: cfg, delegate: self)
                do {
                    try s.addStreamOutput(self, type: .screen, sampleHandlerQueue: self.queue)
                } catch {
                    return fail(error)
                }
                self.stream = s
                s.startCapture { error in
                    DispatchQueue.main.async {
                        guard self.session == mine else { return }
                        if let error { return fail(error) }
                        self.running = true
                        done(nil)
                    }
                }
            }
        }
    }

    func stop() {
        session += 1
        stream?.stopCapture(completionHandler: nil)
        stream = nil
        running = false
        latest = nil
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        root.isHidden = true
        for g in ghosts { g.contents = nil }
        CATransaction.commit()
        window.orderOut(nil)
    }

    /// Capture queue. A still screen sends no new frames, so the first one has to be kept even if it arrives
    /// before the stream reports that it started.
    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sb.isValid,
              let info = CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let raw = info.first?[.status] as? Int, SCFrameStatus(rawValue: raw) == .complete,
              let pixels = sb.imageBuffer, let surface = CVPixelBufferGetIOSurface(pixels)?.takeUnretainedValue() else { return }
        DispatchQueue.main.async {
            guard stream === self.stream else { return }
            self.latest = surface
            self.fresh = true
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        DispatchQueue.main.async {
            guard stream === self.stream else { return }
            self.stop()
            self.onStop?()
        }
    }

    /// Once per display frame. `amount` is the kick, 0…1 (1 only at full volume).
    func update(amount a: Float, time: Float) {
        guard running else { return }
        let on = a > 0.004 && latest != nil
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        root.isHidden = !on
        if on {
            if fresh {
                for g in ghosts { g.contents = latest }
                fresh = false
            }
            // the outermost copy grows by `zoom` at a full kick: 0.6 / 1.1 / 1.9 % (4 / 8 / 14 pt at the edge of a 13" display)
            let zoom = CGFloat([0.006, 0.011, 0.019][strength] * a)
            let each: Float = [0.2, 0.26, 0.3][strength]
            // a cone doesn't only pump, it rattles: a point or so of jitter that dies with the kick
            let shake = CGFloat([0.6, 1.2, 2.0][strength] * a)
            let dx = shake * CGFloat(sin(time * 83)), dy = shake * CGFloat(sin(time * 67 + 1.3))
            root.opacity = smoothstep(0.004, 0.2, a)
            for (i, g) in ghosts.enumerated() {
                let f = CGFloat(i + 1) / CGFloat(ghosts.count)
                let s = 1 + zoom * f
                g.opacity = each
                g.setAffineTransform(CGAffineTransform(a: s, b: 0, c: 0, d: s, tx: dx * f, ty: dy * f))
            }
        }
        CATransaction.commit()
    }
}

// MARK: - menu bar app

final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    let defaults = UserDefaults.standard
    let tap = AudioTap()
    let analyzer = Analyzer()
    let bass = BassFeel()
    let album = AlbumArt()
    var deniedShown = false
    let demo = ProcessInfo.processInfo.environment["AURA_DEMO"] != nil
    var renderer: Renderer!
    var window: NSWindow!
    var view: MetalView!
    var stage: Stage?
    var link: CADisplayLink?
    var statusItem: NSStatusItem!
    var audioError: String?
    var restartPending = false
    var lastTick: CFTimeInterval = 0
    var clock: Float = 0
    var blank = false       // the overlay already shows nothing, so there's nothing to redraw

    var lineOn: Bool { defaults.bool(forKey: "enabled") }
    var feelOn: Bool { defaults.bool(forKey: "bassFeel") }
    var drawing = false     // the line is on: cached for the frame callback

    func applicationDidFinishLaunching(_ n: Notification) {
        defaults.register(defaults: ["enabled": true, "edges": false, "magnet": true, "haze": true, "bassFeel": false,
                                     "bassStrength": 1, "palette": 0, "speed": 0])
        guard let r = Renderer() else {
            let alert = NSAlert()
            alert.messageText = "Aura couldn't start"
            alert.informativeText = "This Mac's graphics couldn't be set up for drawing the line."
            alert.runModal()
            NSApp.terminate(nil)
            return
        }
        renderer = r
        renderer.palette = Palette(rawValue: defaults.integer(forKey: "palette")) ?? .siri
        analyzer.response = Response(rawValue: defaults.integer(forKey: "speed")) ?? .smooth
        bass.strength = min(2, max(0, defaults.integer(forKey: "bassStrength")))
        bass.onStop = { [weak self] in self?.bassFailed() }
        album.onChange = { [weak self] in
            guard let self else { return }
            self.renderer.album = lineColors(self.album.swatches)
        }
        album.onDenied = { [weak self] in self?.explainAutomation() }
        if renderer.palette == .album { album.start() }

        window = makeOverlay(above: 2)
        view = MetalView(device: renderer.device)
        window.contentView = view

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        statusItem.button?.image = NSImage(systemSymbolName: "waveform", accessibilityDescription: "Aura")
        let menu = NSMenu()
        menu.delegate = self
        statusItem.menu = menu

        tap.onChange = { [weak self] in self?.restartAudio() }
        NotificationCenter.default.addObserver(self, selector: #selector(screensChanged),
                                               name: NSApplication.didChangeScreenParametersNotification, object: nil)
        // the tap hears audio before the volume control, so the volume has to be read separately
        analyzer.volume = systemOutputVolume()
        Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            self?.analyzer.volume = systemOutputVolume()
        }
        apply()
    }

    func applicationWillTerminate(_ n: Notification) {
        tap.stop()
        bass.stop()
    }

    /// The display the line lives on: the one with the notch, else the main one.
    func targetScreen() -> NSScreen? {
        NSScreen.screens.first { notch(of: $0) != nil } ?? NSScreen.screens.first
    }

    /// Brings windows, audio and capture in line with the settings. The line and Bass Feel are independent:
    /// either can run without the other, and with both off Aura stops listening altogether.
    /// `asking`: the user just switched Bass Feel on (see `startBass`).
    func apply(asking: Bool = false) {
        link?.invalidate()
        link = nil
        drawing = lineOn
        guard lineOn || feelOn, let screen = targetScreen() else {
            window.orderOut(nil)
            tap.stop()
            bass.stop()
            stage = nil
            return
        }
        let s = Stage(screen: screen, edges: defaults.bool(forKey: "edges"))
        s.line.haze = defaults.bool(forKey: "haze")
        stage = s
        if drawing {
            window.setFrame(s.frame, display: false)
            view.fit(scale: screen.backingScaleFactor)
            window.orderFrontRegardless()
            blank = false
        } else {
            window.orderOut(nil)
        }
        startAudio()
        if feelOn {
            if !bass.running { startBass(asking: asking) }
        } else {
            bass.stop()
        }
        lastTick = 0
        link = screen.displayLink(target: self, selector: #selector(tick(_:)))
        link?.add(to: .main, forMode: .common)
    }

    func startAudio() {
        guard !demo, !tap.running else { return }
        do {
            try tap.start()
            audioError = nil
        } catch {
            audioError = error.localizedDescription
            NSLog("Aura: \(error.localizedDescription)")
        }
    }

    func restartAudio() {
        guard tap.running, !restartPending else { return }
        restartPending = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            self.restartPending = false
            self.tap.stop()
            if self.stage != nil { self.startAudio() }
        }
    }

    @objc func screensChanged() {
        bass.stop()
        apply()
    }

    @objc func tick(_ link: CADisplayLink) {
        guard let stage = stage else { return }
        let now = link.targetTimestamp
        let dt = Float(lastTick == 0 ? 1.0 / 60 : min(0.05, max(0.001, now - lastTick)))
        lastTick = now
        clock += dt
        if demo { analyzer.demo(time: clock, dt: dt) } else { analyzer.update(tap, dt: dt) }

        bass.update(amount: analyzer.bass * analyzer.activity * analyzer.gain, time: clock)
        guard drawing else { return }

        var cursor: Vec?
        if defaults.bool(forKey: "magnet") {
            let m = NSEvent.mouseLocation
            cursor = Vec(Float(m.x - stage.frame.minX), Float(stage.frame.maxY - m.y))
        }
        let seen = stage.line.step(analyzer, cursor: cursor, time: clock, dt: dt)

        let show = seen > 0.004
        if !show && blank { return }
        guard let drawable = view.metal.nextDrawable(),
              let cmd = renderer.draw(show ? stage : nil, into: drawable.texture,
                                      scale: Float(view.metal.contentsScale), time: clock, level: analyzer.level) else { return }
        cmd.present(drawable)
        cmd.commit()
        blank = !show
    }

    // MARK: menu

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        func add(_ title: String, _ action: Selector?, on: Bool = false, tag: Int = 0, to m: NSMenu = menu) {
            let item = NSMenuItem(title: title, action: action, keyEquivalent: "")
            item.target = self
            item.state = on ? .on : .off
            item.tag = tag
            m.addItem(item)
        }
        if let e = audioError, stage != nil {
            add("Can't hear the Mac: \(e)", nil)
            menu.addItem(.separator())
        }
        add("Visualizer", #selector(toggleLine), on: lineOn)
        let edges = defaults.bool(forKey: "edges")
        add("Around the Notch", #selector(setStyle(_:)), on: !edges, tag: 0)
        add("Around the Whole Display", #selector(setStyle(_:)), on: edges, tag: 1)
        menu.addItem(.separator())
        add("Cursor Magnet", #selector(toggleMagnet), on: defaults.bool(forKey: "magnet"))
        add("Heat Haze", #selector(toggleHaze), on: defaults.bool(forKey: "haze"))
        let speed = NSMenu()
        for r in Response.allCases {
            add(r.title, #selector(setSpeed(_:)), on: analyzer.response == r, tag: r.rawValue, to: speed)
        }
        let speedItem = NSMenuItem(title: "Response Speed", action: nil, keyEquivalent: "")
        speedItem.submenu = speed
        menu.addItem(speedItem)
        menu.addItem(.separator())
        add("Bass Feel", #selector(toggleBass), on: feelOn)
        let strength = NSMenu()
        for (i, name) in ["Subtle", "Balanced", "Punchy"].enumerated() {
            add(name, #selector(setStrength(_:)), on: bass.strength == i, tag: i, to: strength)
        }
        let strengthItem = NSMenuItem(title: "Bass Feel Strength", action: nil, keyEquivalent: "")
        strengthItem.submenu = strength
        menu.addItem(strengthItem)
        let colours = NSMenu()
        for p in Palette.allCases {
            if p == .album { colours.addItem(.separator()) }
            add(p.title, #selector(setPalette(_:)), on: renderer.palette == p, tag: p.rawValue, to: colours)
        }
        let coloursItem = NSMenuItem(title: "Colours", action: nil, keyEquivalent: "")
        coloursItem.submenu = colours
        menu.addItem(coloursItem)
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Quit Aura", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
    }

    @objc func toggleLine() {
        defaults.set(!lineOn, forKey: "enabled")
        apply()
    }

    @objc func setStyle(_ item: NSMenuItem) {
        defaults.set(item.tag == 1, forKey: "edges")
        apply()
    }

    @objc func toggleMagnet() {
        defaults.set(!defaults.bool(forKey: "magnet"), forKey: "magnet")
    }

    @objc func toggleHaze() {
        let on = !defaults.bool(forKey: "haze")
        defaults.set(on, forKey: "haze")
        stage?.line.haze = on
    }

    @objc func setSpeed(_ item: NSMenuItem) {
        analyzer.response = Response(rawValue: item.tag) ?? .smooth
        defaults.set(item.tag, forKey: "speed")
    }

    @objc func setStrength(_ item: NSMenuItem) {
        bass.strength = item.tag
        defaults.set(item.tag, forKey: "bassStrength")
    }

    @objc func setPalette(_ item: NSMenuItem) {
        renderer.palette = Palette(rawValue: item.tag) ?? .siri
        defaults.set(item.tag, forKey: "palette")
        deniedShown = false
        if renderer.palette == .album { album.start() } else { album.stop() }
    }

    /// Once per time the album palette is chosen, not once per song.
    func explainAutomation() {
        guard !deniedShown else { return }
        deniedShown = true
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = "Aura can't ask Music for the cover"
        alert.informativeText = "To take its colours from the album you're playing, Aura asks Music for the cover. "
            + "Allow Aura to control Music under Privacy & Security → Automation; until then the line keeps its Siri colours."
        alert.addButton(withTitle: "Open System Settings")
        alert.addButton(withTitle: "Not Now")
        if alert.runModal() == .alertFirstButtonReturn,
           let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc func toggleBass() {
        defaults.set(!feelOn, forKey: "bassFeel")
        apply(asking: true)
    }

    /// Bass Feel has to see the screen to copy it, which macOS only allows with Screen Recording permission.
    /// `asking`: the user just switched it on, so prompt for the permission and explain if it's missing.
    func startBass(asking: Bool) {
        guard let screen = stage?.screen else { return }
        if !CGPreflightScreenCaptureAccess() {
            bassFailed()
            guard asking else { return }
            // macOS shows its own prompt the first time only; after that the explaining is on us
            let asked = defaults.bool(forKey: "askedScreen")
            defaults.set(true, forKey: "askedScreen")
            if !CGRequestScreenCaptureAccess() && asked { explainPermission() }
            return
        }
        bass.start(on: screen) { [weak self] error in
            guard let self, let error else { return }
            self.bassFailed()
            NSLog("Aura: bass feel: \(error.localizedDescription)")
            if asking { self.explainPermission() }
        }
    }

    /// Bass Feel couldn't start or stopped by itself: switch it off, and everything with it if the line is off too.
    func bassFailed() {
        defaults.set(false, forKey: "bassFeel")
        if !lineOn { DispatchQueue.main.async { self.apply() } }
    }

    func explainPermission() {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = "Bass Feel needs to see your screen"
        alert.informativeText = "To make the display thump, Aura copies what's on screen and blurs it for a moment. "
            + "Nothing is saved or sent anywhere.\n\nAllow Aura under Privacy & Security → Screen & System Audio "
            + "Recording, quit and reopen Aura, then turn Bass Feel on again."
        alert.addButton(withTitle: "Open System Settings")
        alert.addButton(withTitle: "Not Now")
        if alert.runModal() == .alertFirstButtonReturn,
           let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") {
            NSWorkspace.shared.open(url)
        }
    }
}

// MARK: - snapshot

/// `Aura --snapshot out.png [--edges] [--cursor x,y] [--palette n] [--volume v] [--time s] [--speed n] [--no-haze]
/// [--left] [--backdrop dark|light]`: draws one frame of the made-up song (`s` seconds in, at volume `v`, only its
/// left channel with `--left`) to a PNG, over a half dark, half light backdrop with the notch painted in.
/// For looking at the line without running the app.
func snapshot(to path: String, args: [String]) -> Bool {
    guard let screen = NSScreen.screens.first(where: { notch(of: $0) != nil }) ?? NSScreen.screens.first,
          let renderer = Renderer() else { return false }
    func value(_ flag: String) -> String? {
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }
    let stage = Stage(screen: screen, edges: args.contains("--edges"))
    renderer.palette = Palette(rawValue: Int(value("--palette") ?? "0") ?? 0) ?? .siri
    if renderer.palette == .album { renderer.album = lineColors(AlbumArt.fetch().swatches ?? []) }
    var cursor: Vec?
    if let c = value("--cursor")?.split(separator: ",").compactMap({ Float($0) }), c.count == 2 { cursor = Vec(c[0], c[1]) }

    stage.line.haze = !args.contains("--no-haze")
    let analyzer = Analyzer()
    analyzer.volume = min(1, max(0, Float(value("--volume") ?? "1") ?? 1))
    analyzer.response = Response(rawValue: Int(value("--speed") ?? "0") ?? 0) ?? .smooth
    let right: Float = args.contains("--left") ? 0 : 1
    var t: Float = 0
    for _ in 0..<Int((Float(value("--time") ?? "2.5") ?? 2.5) * 60) {
        t += 1.0 / 60
        analyzer.demo(time: t, dt: 1.0 / 60, right: right)
        _ = stage.line.step(analyzer, cursor: cursor, time: t, dt: 1.0 / 60)
    }

    let scale: Float = 2
    let w = Int(Float(stage.frame.width) * scale), h = Int(Float(stage.frame.height) * scale)
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    d.usage = [.renderTarget, .shaderRead]
    d.storageMode = .shared
    guard let tex = renderer.device.makeTexture(descriptor: d),
          let cmd = renderer.draw(stage, into: tex, scale: scale, time: t, level: analyzer.level) else { return false }
    cmd.commit()
    cmd.waitUntilCompleted()

    var px = [UInt8](repeating: 0, count: w * h * 4)
    tex.getBytes(&px, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    let nr = stage.notchRect
    let backdrop = value("--backdrop")
    for y in 0..<h {
        for x in 0..<w {
            let i = (y * w + x) * 4
            let inNotch = nr?.contains(CGPoint(x: CGFloat(Float(x) / scale), y: CGFloat(Float(y) / scale))) ?? false
            let dark = backdrop == "dark" || (backdrop != "light" && x < w / 2)
            let bg: [Float] = inNotch ? [0, 0, 0] : dark ? [0.16, 0.12, 0.10] : [0.96, 0.96, 0.96]     // b, g, r
            let a = Float(px[i + 3]) / 255
            for c in 0..<3 { px[i + c] = UInt8(min(255, Float(px[i + c]) + bg[c] * 255 * (1 - a))) }
            px[i + 3] = 255
        }
    }
    let info = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
    guard let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info),
          let image = ctx.makeImage(),
          let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { return false }
    CGImageDestinationAddImage(dest, image, nil)
    return CGImageDestinationFinalize(dest)
}

// MARK: - main

let arguments = CommandLine.arguments
if let i = arguments.firstIndex(of: "--snapshot"), i + 1 < arguments.count {
    exit(snapshot(to: arguments[i + 1], args: arguments) ? 0 : 1)
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
