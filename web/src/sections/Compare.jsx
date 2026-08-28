import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { GitCompare } from 'lucide-react'
import Dropzone from '../components/Dropzone'
import { ScanFrame, fadeUp } from '../components/ui'
import { api } from '../lib/api'

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

/** Drag-to-reveal comparison. Both images are the same size by construction. */
function Reveal({ left, right, labelLeft, labelRight }) {
  const [x, setX] = useState(50)
  const box = useRef()
  const move = (clientX) => {
    const r = box.current?.getBoundingClientRect()
    if (!r) return
    setX(Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100)))
  }
  return (
    <div ref={box}
      className="relative touch-none select-none overflow-hidden bg-page"
      onMouseMove={e => e.buttons === 1 && move(e.clientX)}
      onTouchMove={e => move(e.touches[0].clientX)}
      onClick={e => move(e.clientX)}>
      <img src={right} alt={labelRight} className="w-full" draggable={false} />
      <div className="absolute inset-0 overflow-hidden" style={{ width: `${x}%` }}>
        <img src={left} alt={labelLeft} draggable={false}
             className="h-full max-w-none object-cover"
             style={{ width: box.current?.getBoundingClientRect().width || '100%' }} />
      </div>
      <div className="absolute inset-y-0 w-px bg-accent" style={{ left: `${x}%` }}>
        <div className="absolute top-1/2 grid h-9 w-9 -translate-x-1/2 -translate-y-1/2
                        place-items-center border border-accent bg-page">
          <GitCompare size={13} className="text-accent" />
        </div>
      </div>
      <span className="label absolute left-0 top-0 bg-page/85 px-2.5 py-1.5 !text-[9.5px] !text-body">
        {labelLeft}
      </span>
      <span className="label absolute right-0 top-0 bg-accent px-2.5 py-1.5 !text-[9.5px] !text-page">
        {labelRight}
      </span>
    </div>
  )
}

export default function Compare({ cameraHeight, maxRange }) {
  const [busy, setBusy] = useState(false)
  const [d, setD] = useState(null)
  const [err, setErr] = useState('')
  const [mode, setMode] = useState('lanes')
  const [measured, setMeasured] = useState(null)

  useEffect(() => { api.results().then(setMeasured).catch(() => {}) }, [])

  const run = async (file) => {
    setBusy(true); setErr(''); setD(null)
    try { setD(await api.compare(file, { cameraHeight, maxRange })) }
    catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  const rows = measured?.tables?.evaluation_val
  const find = pred => rows?.find(pred) ?? null
  const p4 = find(r => r.key === 'phase4_dl' || r.name.startsWith('Phase 4: IDD'))
  const p3 = find(r => r.name.includes('HSV OR YOLOP') || r.name.startsWith('Phase 3: HSV'))
  const trap = find(r => r.name.includes('trapezoid'))
  const f4 = v => (v == null ? '—' : v.toFixed(4))

  const ROWS = [
    ['Road detection', 'fixed HSV threshold', 'network trained on IDD'],
    ['Drivable IoU',   f4(p3?.drivable_iou),  f4(p4?.drivable_iou)],
    ['Boundary F1',    f4(p3?.boundary_f1),   f4(p4?.boundary_f1)],
    ['Lane method',    'widest row, bisected', 'metric partition in BEV'],
    ['Lane count',     'always 2',            'inferred from width'],
  ]

  return (
    <div className="grid gap-5 lg:grid-cols-[22rem,minmax(0,1fr)]">
      <div className="space-y-4">
        <Dropzone onFile={run} busy={busy} />
        {err && (
          <div className="border-l-2 border-road bg-road/[.12] px-4 py-3 text-[13px] text-road">
            {err}
          </div>
        )}
        <Panel title="What is being compared">
          <p className="p-4 text-[12.5px] leading-[1.65] text-muted">
            The Phase 3 prototype is reproduced exactly, including its{' '}
            <code className="bg-sunken px-1 font-mono text-[11.5px] text-body">bitwise_or</code>{' '}
            fusion and its 120%-gain compositing. On the validation split it scores{' '}
            <strong className="text-body">below a fixed trapezoid that never looks
            at the image</strong>{' '}
            ({f4(p3?.drivable_iou)} against {f4(trap?.drivable_iou)}), the finding
            that made a rebuild the right call rather than tuning.
          </p>
        </Panel>
      </div>

      <div className="min-w-0 space-y-5">
        {!d && !busy && (
          <Panel className="relative overflow-hidden">
            <div className="pointer-events-none absolute inset-0 grid-bg opacity-40" />
            <div className="relative grid place-items-center px-6 py-20 text-center">
              <GitCompare className="mb-5 text-rule2" size={32} strokeWidth={1.2} />
              <h3 className="display text-[27px] text-body">Compare both pipelines</h3>
              <p className="prose-serif mt-2.5 max-w-md text-[15px] leading-[1.6] text-muted">
                The same frame is run through Phase 3 and Phase 4 side by side.
                Drag the handle to reveal one under the other.
              </p>
            </div>
          </Panel>
        )}
        {busy && <Panel title="Running both" aside="processing"><ScanFrame busy /></Panel>}

        {d && (
          <motion.div {...fadeUp} className="space-y-5">
            <Panel aside="drag the handle">
              <div className="flex gap-5 border-b border-rule px-4">
                {[['lanes', 'Lane output'], ['mask', 'Road mask']].map(([k, l]) => (
                  <button key={k} onClick={() => setMode(k)}
                    className={`relative py-2.5 transition-colors
                      ${mode === k ? 'text-strong' : 'text-muted hover:text-body'}`}>
                    <span className="label !text-[10px] !text-inherit">{l}</span>
                    {mode === k && (
                      <motion.span layoutId="cmprule"
                        className="absolute inset-x-0 -bottom-px h-0.5 bg-accent"
                        transition={{ type: 'spring', stiffness: 420, damping: 34 }} />
                    )}
                  </button>
                ))}
              </div>
              <Reveal left={d.phase3[mode]} right={d.phase4[mode]}
                      labelLeft="Phase 3" labelRight="Phase 4" />
            </Panel>

            <Panel title="Measured differences"
                   aside={rows ? `${p4?.frames ?? 204}-frame validation split` : 'loading…'}>
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="border-b border-rule">
                    <th className="label px-4 py-2.5 text-left !text-[9.5px]"></th>
                    <th className="label px-4 py-2.5 text-left !text-[9.5px]">Phase 3</th>
                    <th className="label px-4 py-2.5 text-left !text-[9.5px] !text-accent">Phase 4</th>
                  </tr>
                </thead>
                <tbody>
                  {ROWS.map(([k, a, b]) => (
                    <tr key={k} className="border-b border-rule last:border-0">
                      <td className="px-4 py-2.5 text-muted">{k}</td>
                      <td className="px-4 py-2.5 font-mono text-[12px] text-body">{a}</td>
                      <td className="px-4 py-2.5 font-mono text-[12px] font-semibold text-strong">{b}</td>
                    </tr>
                  ))}
                  <tr className="border-t border-rule2 bg-sunken/50">
                    <td className="px-4 py-2.5 text-muted">This frame</td>
                    <td className="px-4 py-2.5 font-mono text-[12px] text-body">{d.phase3.ms} ms</td>
                    <td className="px-4 py-2.5 font-mono text-[12px] font-semibold text-strong">
                      {d.phase4.ms} ms · {d.result.lane ? `${d.result.lane.count} lanes` : 'no solution'}
                    </td>
                  </tr>
                </tbody>
              </table>
            </Panel>
          </motion.div>
        )}
      </div>
    </div>
  )
}
