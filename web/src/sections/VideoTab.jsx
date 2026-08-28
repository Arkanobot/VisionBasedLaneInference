import { useState } from 'react'
import { motion } from 'framer-motion'
import { Film } from 'lucide-react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import Dropzone from '../components/Dropzone'
import { fadeUp } from '../components/ui'
import { api } from '../lib/api'

const AXIS = { tick: { fill: '#6c7682', fontSize: 11, fontFamily: '"IBM Plex Mono", monospace' },
               stroke: '#383f49' }
const TIP = { contentStyle: { background: '#12151a', border: '1px solid #383f49', borderRadius: 0,
                              fontSize: 12, fontFamily: '"IBM Plex Sans", system-ui' },
              labelStyle: { color: '#eef1f5', fontWeight: 600 } }

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

function Readout({ label, value, hint, tone }) {
  return (
    <div className="bg-page px-4 py-4">
      <div className="label !text-[9.5px]">{label}</div>
      <div className={`mt-2 font-mono text-[clamp(1.4rem,3.6vw,1.9rem)] font-medium leading-none
                       ${tone === 'good' ? 'text-lane' : tone === 'warn' ? 'text-signal' : 'text-strong'}`}>
        {value}
      </div>
      {hint && <div className="mt-1.5 text-[11px] leading-tight text-faint">{hint}</div>}
    </div>
  )
}

export default function VideoTab({ cameraHeight, maxRange }) {
  const [busy, setBusy] = useState(false)
  const [d, setD] = useState(null)
  const [err, setErr] = useState('')
  const [temporal, setTemporal] = useState(true)

  const run = async (file) => {
    setBusy(true); setErr(''); setD(null)
    try { setD(await api.video(file, { cameraHeight, maxRange, temporal })) }
    catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[22rem,minmax(0,1fr)]">
      <div className="space-y-4">
        <Dropzone kind="video" onFile={run} busy={busy} />

        <Panel title="Temporal smoothing">
          <div className="flex items-center justify-between gap-4 p-4">
            <div className="min-w-0">
              <div className="text-[13px] text-body">Kalman filter + lane-count vote</div>
              <div className="mt-0.5 text-[11.5px] text-faint">
                Applied over [c₀, c₁, c₂, w]
              </div>
            </div>
            <button onClick={() => setTemporal(v => !v)} role="switch" aria-checked={temporal}
              aria-label="Temporal smoothing"
              className={`relative h-6 w-11 shrink-0 border transition-colors
                ${temporal ? 'border-accent bg-accent/20' : 'border-rule2 bg-sunken'}`}>
              <motion.span layout
                className={`absolute top-[3px] h-4 w-4 ${temporal ? 'bg-accent' : 'bg-muted'}`}
                animate={{ left: temporal ? 24 : 3 }}
                transition={{ type: 'spring', stiffness: 500, damping: 32 }} />
            </button>
          </div>
        </Panel>

        {err && (
          <div className="border-l-2 border-road bg-road/[.12] px-4 py-3 text-[13px] text-road">
            {err}
          </div>
        )}

        <Panel title="Why it helps">
          <p className="p-4 text-[12.5px] leading-[1.65] text-muted">
            Measured over 150 IDD Temporal sequences (4,464 frames): smoothing cuts
            carriageway width jitter by <strong className="text-body">89%</strong> and
            lane-count changes by <strong className="text-body">79%</strong>, while
            raising the solution rate from 85.1% to 88.9%. It holds through bad
            frames rather than ignoring the input.
          </p>
        </Panel>
      </div>

      <div className="min-w-0 space-y-5">
        {!d && !busy && (
          <Panel className="relative overflow-hidden">
            <div className="pointer-events-none absolute inset-0 grid-bg opacity-40" />
            <div className="relative grid place-items-center px-6 py-20 text-center">
              <Film className="mb-5 text-rule2" size={32} strokeWidth={1.2} />
              <h3 className="display text-[27px] text-body">Process a clip</h3>
              <p className="prose-serif mt-2.5 max-w-md text-[15px] leading-[1.6] text-muted">
                Every frame is inferred independently, then the lane geometry is
                tracked across time. Toggle smoothing to see the difference.
              </p>
            </div>
          </Panel>
        )}

        {busy && (
          <Panel title="Processing" aside="frame by frame">
            <div className="grid place-items-center gap-4 py-16">
              <div className="h-0.5 w-56 overflow-hidden bg-sunken">
                <div className="h-full w-1/3 animate-shimmer bg-gradient-to-r
                                from-transparent via-accent to-transparent bg-[length:200%_100%]" />
              </div>
              <p className="label">Processing frames…</p>
            </div>
          </Panel>
        )}

        {d && (
          <motion.div {...fadeUp} className="space-y-5">
            <Panel>
              <video src={d.videoUrl} controls autoPlay loop muted playsInline
                     className="w-full bg-black" />
            </Panel>

            <div className="border border-rule">
              <div className="grid grid-cols-2 gap-px bg-rule sm:grid-cols-4">
                <Readout label="Frames" value={d.frames} hint={`${d.medianMs} ms each`} />
                <Readout label="Solved" value={`${Math.round(d.solutionRate * 100)}%`}
                         hint={d.temporal ? 'smoothing on' : 'smoothing off'} />
                <Readout label="Median carriageway" value={d.medianWidthM?.toFixed(2)}
                         hint="metres" />
                <Readout label="Lane changes" value={d.laneCountChanges}
                         tone={d.laneCountChanges === 0 ? 'good'
                               : d.laneCountChanges > 4 ? 'warn' : undefined}
                         hint={d.modalLanes ? `modal ${d.modalLanes} lanes` : 'fewer is steadier'} />
              </div>
            </div>

            {d.series?.length > 1 && (
              <Panel title="Carriageway width per frame"
                     aside={`${d.series.length} samples`}>
                <div className="p-4">
                  <ResponsiveContainer width="100%" height={180}>
                    <LineChart data={d.series} margin={{ left: -18, right: 6, top: 4 }}>
                      <CartesianGrid strokeDasharray="2 3" stroke="#232830" />
                      <XAxis dataKey="f" {...AXIS} />
                      <YAxis domain={['dataMin - 1', 'dataMax + 1']} {...AXIS} />
                      <Tooltip {...TIP} />
                      <Line type="monotone" dataKey="width" stroke="#d0483c" strokeWidth={2}
                            dot={false} connectNulls name="width (m)" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="border-t border-rule px-4 py-3 text-[11.5px] leading-relaxed text-faint">
                  A flat trace is the goal, the road ahead barely changes over a few
                  seconds, so swings are estimator noise rather than signal.
                </p>
              </Panel>
            )}
          </motion.div>
        )}
      </div>
    </div>
  )
}
