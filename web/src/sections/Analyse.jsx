import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AlertTriangle, Layers, Users } from 'lucide-react'
import Dropzone from '../components/Dropzone'
import { Gauge, ScanFrame, fadeUp } from '../components/ui'
import { api } from '../lib/api'

/* Each view is a stage of the pipeline made visible. The note says what the
   viewer is looking at, so the demo teaches the method rather than only
   displaying its output. */
const VIEWS = [
  ['lanes',     'Lane inference', 'Inferred lane boundaries projected back into the camera frame. Count comes from carriageway width ÷ IRC design width, not from paint.'],
  ['semantics', 'Semantics',      'The seven level-1 IDD classes. Drivable surface in red is the only one the geometry stage consumes.'],
  ['mask',      'Road mask',      'The drivable class after cleanup: largest bottom-connected component, holes filled.'],
  ['bev',       "Bird's-eye",     'The road plane rectified to metric top-down using the pitch recovered from the vanishing point. Lane widths are measured here.'],
  ['input',     'Input',          'The frame as supplied, before any processing.'],
]

/** One reading on the instrument cluster. */
function Readout({ label, value, unit, hint, wide = false }) {
  return (
    <div className={`bg-page px-4 py-4 ${wide ? 'col-span-2' : ''}`}>
      <div className="label !text-[9.5px]">{label}</div>
      <div className="mt-2 flex items-baseline gap-1">
        <span className="font-mono text-[clamp(1.5rem,4vw,2rem)] font-medium leading-none text-strong">
          {value}
        </span>
        {unit && <span className="font-mono text-[13px] text-faint">{unit}</span>}
      </div>
      {hint && <div className="mt-1.5 text-[11px] leading-tight text-faint">{hint}</div>}
    </div>
  )
}

function Panel({ title, aside, children, className = '' }) {
  return (
    <section className={`border border-rule bg-surface ${className}`}>
      {(title || aside) && (
        <header className="flex items-center gap-3 border-b border-rule px-4 py-2.5">
          <h3 className="label">{title}</h3>
          {aside && <div className="ml-auto font-mono text-[11.5px] text-muted">{aside}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

export default function Analyse({ cameraHeight, maxRange }) {
  const [busy, setBusy] = useState(false)
  const [res, setRes]   = useState(null)
  const [err, setErr]   = useState('')
  const [view, setView] = useState('lanes')

  const run = async (file) => {
    setBusy(true); setErr(''); setRes(null)
    try {
      setRes(await api.infer(file, { cameraHeight, maxRange }))
    } catch (e) { setErr(e.message) }
    finally { setBusy(false) }
  }

  const lane = res?.lane
  const notRoad = res && res.scene && res.scene.isRoad === false
  const conf = lane?.confidence ?? 0
  const active = VIEWS.find(([k]) => k === view)

  return (
    <div className="grid gap-5 lg:grid-cols-[22rem,minmax(0,1fr)]">
      <div className="space-y-4">
        <Dropzone onFile={run} busy={busy} />
        {err && (
          <div className="border-l-2 border-road bg-road/[.12] px-4 py-3 text-[13px] text-road">
            {err}
          </div>
        )}
        {res && !notRoad && (
          <Panel title="Assumptions">
            <dl className="divide-y divide-rule text-[12.5px]">
              {[
                ['Camera height', `${res.cameraHeightM} m`],
                ['Measured pitch', `${res.pitchDeg > 0 ? '+' : ''}${res.pitchDeg}°`],
                ['Markings', lane?.roadType ?? '—'],
                ['Partition basis', lane ? (lane.widthM >= 6 ? '3.5 m urban' : '3.0 m rural') : '—'],
                ['Boundary source', lane?.source ?? '—'],
              ].map(([k, v]) => (
                <div key={k} className="flex items-center justify-between px-4 py-2.5">
                  <dt className="text-muted">{k}</dt>
                  <dd className="font-mono text-body">{v}</dd>
                </div>
              ))}
            </dl>
            <p className="border-t border-rule px-4 py-3 text-[11px] leading-relaxed text-faint">
              Every metric value scales linearly with camera height. Lateral scale
              is independent of focal length, it cancels in the ground-plane
              projection.
            </p>
          </Panel>
        )}
      </div>

      <div className="min-w-0 space-y-5">
        <AnimatePresence mode="wait">
          {!res && !busy && (
            <motion.div key="empty" {...fadeUp}>
              <Panel className="relative overflow-hidden">
                <div className="pointer-events-none absolute inset-0 grid-bg opacity-40" />
                <div className="relative grid place-items-center px-6 py-20 text-center">
                  <Layers className="mb-5 text-rule2" size={32} strokeWidth={1.2} />
                  <h3 className="display text-[27px] text-body">Choose an image</h3>
                  <p className="prose-serif mt-2.5 max-w-md text-[15px] leading-[1.6] text-muted">
                    Lane structure is inferred from the geometry of the drivable
                    surface. The road does not need markings.
                  </p>
                </div>
              </Panel>
            </motion.div>
          )}

          {busy && (
            <motion.div key="busy" {...fadeUp}>
              <Panel title="Inferring" aside="running">
                <ScanFrame busy />
              </Panel>
            </motion.div>
          )}

          {res && (
            <motion.div key="res" {...fadeUp} className="space-y-5">
              {notRoad ? (
                <Panel title="Scene rejected" className="border-signal/50">
                  <div className="flex items-start gap-3.5 p-5">
                    <AlertTriangle className="mt-0.5 shrink-0 text-signal" size={20} />
                    <div>
                      <h3 className="display text-[23px] text-signal">Not a road scene</h3>
                      <p className="prose-serif mt-1.5 text-[15px] text-body">{res.scene.reason}.</p>
                      <p className="mt-3.5 text-[12px] leading-[1.65] text-muted">
                        Lane inference was not attempted. A network trained only on
                        roads cannot abstain on its own, this model labels a blank
                        black frame 26% drivable, so the pipeline tests the structure
                        of the scene before trusting it. Semantic classes present:{' '}
                        <span className="font-mono text-body">{res.scene.classes}</span>{' '}
                        (a road scene shows 5 to 7); drivable above the horizon:{' '}
                        <span className="font-mono text-body">
                          {(res.scene.roadAboveHorizon * 100).toFixed(0)}%
                        </span>{' '}
                        (should be 0%).
                      </p>
                    </div>
                  </div>
                </Panel>
              ) : lane ? (
                <div className="border border-rule">
                  <div className="grid grid-cols-2 gap-px bg-rule sm:grid-cols-4">
                    <Readout label="Lanes" value={lane.count}
                             hint={lane.egoLane ? `ego lane ${lane.egoLane}` : null} />
                    <Readout label="Carriageway" value={lane.widthM?.toFixed(2)} unit="m"
                             hint={lane.widthIsLowerBound ? 'lower bound' : null} />
                    <Readout label="Lane width" value={lane.laneWidthM?.toFixed(2)} unit="m"
                             hint="carriageway / lanes" />
                    <Readout label="Latency" value={res.totalMs} unit="ms"
                             hint={`${res.fps} fps · ${res.device}`} />
                  </div>
                  <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-t border-rule
                                  bg-page px-4 py-4">
                    <Gauge value={conf} size={82} />
                    <div className="min-w-0 flex-1">
                      <div className="label">Interpretation</div>
                      <p className="prose-serif mt-1.5 text-[14px] leading-[1.55] text-body">
                        {conf >= .7
                          ? 'High confidence, boundary error is typically under 0.6 m at this level.'
                          : conf >= .5
                          ? 'Moderate confidence, the geometry is usable but the carriageway edges are less certain.'
                          : 'Low confidence, treat the lane count and widths as unreliable for this frame.'}
                      </p>
                      {res.traffic?.users > 0 && (
                        <p className="mt-2 flex items-center gap-1.5 font-mono text-[11.5px] text-faint">
                          <Users size={11} />
                          {res.traffic.users} road users
                          {res.traffic.leadDistanceM != null &&
                            ` · lead vehicle ${res.traffic.leadDistanceM} m`}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <Panel title="No solution">
                  <div className="p-5">
                    <h3 className="display text-[23px] text-body">Carriageway not measurable</h3>
                    <p className="prose-serif mt-2 max-w-column text-[15px] leading-[1.6] text-muted">
                      The system refuses 39 of 204 validation frames by design: 20 where
                      the carriageway measures below the 2.5 m plausibility floor, 11 where
                      the road edges are not visible, and 8 where no vanishing point is
                      recovered. A wrong lane model is worse than none.
                    </p>
                  </div>
                </Panel>
              )}

              <Panel>
                <div className="flex gap-5 overflow-x-auto border-b border-rule px-4">
                  {VIEWS.filter(([k]) => res.views[k]).map(([k, label]) => (
                    <button key={k} onClick={() => setView(k)}
                      className={`relative shrink-0 py-2.5 transition-colors
                        ${view === k ? 'text-strong' : 'text-muted hover:text-body'}`}>
                      <span className="label !text-[10px] !text-inherit">{label}</span>
                      {view === k && (
                        <motion.span layoutId="viewrule"
                          className="absolute inset-x-0 -bottom-px h-0.5 bg-accent"
                          transition={{ type: 'spring', stiffness: 420, damping: 34 }} />
                      )}
                    </button>
                  ))}
                </div>
                <motion.img key={view} src={res.views[view] ?? res.views.input}
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: .25 }}
                  alt={active?.[1] ?? view} className="w-full" />
                {active && (
                  <p className="border-t border-rule px-4 py-3 text-[12.5px] leading-[1.6] text-muted">
                    <span className="label mr-2 !text-[9.5px] !text-body">{active[1]}</span>
                    {active[2]}
                  </p>
                )}
              </Panel>

              <Panel title="Stage timings"
                     aside={`${res.totalMs} ms · ${res.fps} fps · ${res.device}`}>
                <div className="space-y-2 p-4">
                  {Object.entries(res.timings).map(([k, v]) => (
                    <div key={k} className="flex items-center gap-3">
                      <span className="w-32 shrink-0 text-[12px] text-muted">{k}</span>
                      <div className="h-1.5 flex-1 bg-sunken">
                        <motion.div className="h-full bg-accent"
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min(100, (v / res.totalMs) * 100)}%` }}
                          transition={{ duration: .6, ease: 'easeOut' }} />
                      </div>
                      <span className="w-16 shrink-0 text-right font-mono text-[12px] text-body">
                        {v.toFixed(1)} ms
                      </span>
                    </div>
                  ))}
                </div>
              </Panel>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
