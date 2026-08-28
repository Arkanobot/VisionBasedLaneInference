import { useCallback, useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { AlertTriangle, Camera, CameraOff, RefreshCw, SwitchCamera } from 'lucide-react'

const SESSION = Math.random().toString(36).slice(2, 10)

function Panel({ title, aside, children, className = '' }) {
  return (
    <section className={`border border-rule bg-surface ${className}`}>
      {(title || aside) && (
        <header className="flex items-center gap-3 border-b border-rule px-4 py-2.5">
          <h3 className="label">{title}</h3>
          {aside && <div className="ml-auto font-mono text-[11px] text-faint">{aside}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

/**
 * A head-up readout laid over the frame. On a phone the camera is held up and
 * pointed at the road, so the numbers have to be legible without looking away
 * from the image, a panel below the fold would never be read.
 */
function Hud({ out }) {
  const cells = [
    ['Lanes', out.lanes, null],
    ['Width', out.widthM, 'm'],
    ['Lane', out.laneWidthM, 'm'],
    ['Ego', out.egoLane ? `${out.egoLane}/${out.lanes}` : '—', null],
  ]
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0">
      <div className="grid grid-cols-4 gap-px bg-strong/15 backdrop-blur-md">
        {cells.map(([l, v, u]) => (
          <div key={l} className="bg-page/70 px-2 py-2 text-center">
            <div className="label !text-[8.5px] !tracking-[.12em]">{l}</div>
            <div className="mt-0.5 font-mono text-[clamp(1rem,4.5vw,1.35rem)] font-medium leading-none text-strong">
              {v}{u && <span className="ml-0.5 text-[10px] text-faint">{u}</span>}
            </div>
          </div>
        ))}
      </div>
      <div className="h-1 w-full bg-page/70">
        <div className={`h-full transition-all duration-300 ${out.confidence >= .7 ? 'bg-lane'
                          : out.confidence >= .5 ? 'bg-signal' : 'bg-road'}`}
             style={{ width: `${Math.round((out.confidence ?? 0) * 100)}%` }} />
      </div>
    </div>
  )
}

export default function Live({ cameraHeight, maxRange }) {
  const [on, setOn]           = useState(false)
  const [facing, setFacing]   = useState('environment')
  const [err, setErr]         = useState('')
  const [out, setOut]         = useState(null)
  const [fps, setFps]         = useState(0)
  const [quality, setQuality] = useState(640)

  const videoRef  = useRef()
  const canvasRef = useRef()
  const streamRef = useRef()
  const busyRef   = useRef(false)
  const stopRef   = useRef(false)
  const timesRef  = useRef([])

  const stop = useCallback(() => {
    stopRef.current = true
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    setOn(false)
  }, [])

  const start = useCallback(async (mode = facing) => {
    setErr('')
    if (!window.isSecureContext && location.hostname !== 'localhost') {
      setErr('Browsers only allow camera access over HTTPS or on localhost. ' +
             'Open this page on the machine running it, or serve it over HTTPS.')
      return
    }
    try {
      streamRef.current?.getTracks().forEach(t => t.stop())
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: mode }, width: { ideal: 1280 } }, audio: false,
      })
      streamRef.current = stream
      stopRef.current = false
      setOn(true)
      requestAnimationFrame(() => { if (videoRef.current) videoRef.current.srcObject = stream })
      fetch('/api/live/reset', { method: 'POST', body: new URLSearchParams({ session: SESSION }) })
    } catch (e) { setErr(`Camera unavailable: ${e.message}`) }
  }, [facing])

  useEffect(() => () => stop(), [stop])

  useEffect(() => {
    if (!on) return
    let raf
    const tick = async () => {
      if (stopRef.current) return
      const v = videoRef.current
      if (v && v.videoWidth && !busyRef.current) {
        busyRef.current = true
        const c = canvasRef.current
        const scale = quality / v.videoWidth
        c.width = quality; c.height = Math.round(v.videoHeight * scale)
        c.getContext('2d').drawImage(v, 0, 0, c.width, c.height)
        const t0 = performance.now()
        c.toBlob(async (blob) => {
          try {
            const fd = new FormData()
            fd.append('file', blob, 'f.jpg')
            fd.append('session', SESSION)
            fd.append('cameraHeight', String(cameraHeight))
            fd.append('maxRange', String(maxRange))
            fd.append('width', String(quality))
            const res = await fetch('/api/live', { method: 'POST', body: fd })
            if (res.ok) {
              setOut(await res.json())
              const t = timesRef.current
              t.push(performance.now() - t0)
              if (t.length > 10) t.shift()
              setFps(1000 / (t.reduce((a, b) => a + b, 0) / t.length))
            }
          } catch { /* frame dropped; the next one will do */ }
          finally { busyRef.current = false }
        }, 'image/jpeg', 0.7)
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [on, quality, cameraHeight, maxRange])

  const flip = () => {
    const next = facing === 'environment' ? 'user' : 'environment'
    setFacing(next); start(next)
  }

  const reset = () => fetch('/api/live/reset', {
    method: 'POST', body: new URLSearchParams({ session: SESSION }) })

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <canvas ref={canvasRef} className="hidden" />

      <div className="border border-rule bg-surface">
        <div className="relative bg-black">
          <video ref={videoRef} autoPlay playsInline muted
                 className={`w-full bg-black ${on && out ? 'hidden' : ''}`} />
          {on && out && <img src={out.frame} alt="live inference" className="w-full bg-black" />}

          {!on && !err && (
            <div className="grid aspect-video place-items-center bg-surface px-6 text-center">
              <div>
                <Camera className="mx-auto mb-4 text-rule2" size={32} strokeWidth={1.2} />
                <p className="display text-[24px] text-body">Point the camera at a road</p>
                <p className="prose-serif mx-auto mt-2 max-w-sm text-[14px] leading-[1.6] text-muted">
                  Lane structure, road users and distances update continuously.
                  Temporal smoothing is on, so the overlay holds steady rather than
                  flickering frame to frame.
                </p>
              </div>
            </div>
          )}

          {on && (
            <div className="pointer-events-none absolute left-0 top-0 flex gap-px">
              <span className="bg-page/75 px-2.5 py-1.5 font-mono text-[11px] text-lane backdrop-blur">
                ● {fps.toFixed(1)} fps
              </span>
              {out?.ms != null && (
                <span className="bg-page/75 px-2.5 py-1.5 font-mono text-[11px] text-body backdrop-blur">
                  {out.ms} ms
                </span>
              )}
            </div>
          )}

          {on && out?.ok && <Hud out={out} />}
        </div>

        <div className="flex gap-px border-t border-rule bg-rule">
          {!on ? (
            <button onClick={() => start()}
              className="flex flex-1 items-center justify-center gap-2 bg-accent px-4 py-3.5
                         text-page transition-opacity hover:opacity-90">
              <Camera size={17} />
              <span className="label !text-[10.5px] !text-inherit">Start camera</span>
            </button>
          ) : (
            <>
              <button onClick={stop}
                className="flex flex-1 items-center justify-center gap-2 bg-surface px-4 py-3.5
                           text-body transition-colors hover:bg-sunken">
                <CameraOff size={17} />
                <span className="label !text-[10.5px] !text-inherit">Stop</span>
              </button>
              <button onClick={flip} aria-label="Switch camera"
                className="grid w-16 place-items-center bg-surface text-muted
                           transition-colors hover:bg-sunken hover:text-strong">
                <SwitchCamera size={17} />
              </button>
              <button onClick={reset} aria-label="Reset the temporal filter"
                className="grid w-16 place-items-center bg-surface text-muted
                           transition-colors hover:bg-sunken hover:text-strong">
                <RefreshCw size={17} />
              </button>
            </>
          )}
        </div>
      </div>

      {err && (
        <div className="border-l-2 border-signal bg-signal/[.12] px-4 py-3
                        text-[12.5px] leading-relaxed text-signal">
          {err}
        </div>
      )}

      {out && !out.isRoad && (
        <div className="flex gap-3 border-l-2 border-signal bg-signal/[.12] px-4 py-3.5">
          <AlertTriangle className="mt-0.5 shrink-0 text-signal" size={17} />
          <div>
            <div className="label !text-[10px] !text-signal">Not a road scene</div>
            <div className="mt-1 text-[13px] text-body">{out.reason}.</div>
          </div>
        </div>
      )}

      {out?.ok && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
          <Panel title="Detail" aside={`confidence ${out.confidence}`}>
            <dl className="divide-y divide-rule text-[13px]">
              {[
                ['Road type', out.roadType],
                ['Road users', out.users > 0 ? out.users : '—'],
                ['Lead vehicle', out.leadM != null ? `${out.leadM} m` : '—'],
                ['Nearest', out.nearest?.length ? `${out.nearest.join(' · ')} m` : '—'],
              ].map(([k, v]) => (
                <div key={k} className="flex items-center justify-between px-4 py-2.5">
                  <dt className="text-muted">{k}</dt>
                  <dd className="font-mono text-body">{v}</dd>
                </div>
              ))}
            </dl>
          </Panel>
        </motion.div>
      )}

      <Panel title="Capture width" aside={`${quality} px`}>
        <div className="p-4">
          <input type="range" min="320" max="960" step="80" value={quality}
                 onChange={e => setQuality(+e.target.value)}
                 className="w-full accent-accent" />
          <p className="mt-2.5 text-[11.5px] leading-relaxed text-faint">
            Lower is faster. The model runs at 512 px internally, so beyond about
            640 px you are paying for upload bandwidth rather than accuracy.
          </p>
        </div>
      </Panel>
    </div>
  )
}
