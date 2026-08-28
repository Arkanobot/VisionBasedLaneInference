import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { FileText } from 'lucide-react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { fadeUp } from '../components/ui'
import { api } from '../lib/api'

const AXIS = { tick: { fill: '#6c7682', fontSize: 11, fontFamily: '"IBM Plex Mono", monospace' },
               stroke: '#383f49' }
const GRID = '#232830'
const TIP = { contentStyle: { background: '#12151a', border: '1px solid #383f49', borderRadius: 0,
                              fontSize: 12, fontFamily: '"IBM Plex Sans", system-ui' },
              labelStyle: { color: '#eef1f5', fontWeight: 600 } }

const SHORT = {
  'trivial: everything is road': 'all road',
  'trivial: fixed trapezoid prior': 'trapezoid prior',
  'Phase 3: HSV threshold (CV only)': 'Phase 3 HSV',
  'Phase 3: YOLOP as wired in Phase 3': 'YOLOP as wired',
  'Phase 3: HSV OR YOLOP (the Phase 3 pipeline)': 'Phase 3 pipeline',
  'YOLOP, correctly wired': 'YOLOP fixed',
  'YOLOP (correct) + confidence-weighted CV fusion': 'YOLOP + blend',
  'Phase 4: adaptive CIELAB + texture (CV only)': 'Phase 4 classical',
  'Phase 4: IDD-trained segmentation': 'Phase 4 learned',
  'Phase 4: learned + cleanup (no CV fusion)': '+ cleanup',
  'Phase 4: learned + confidence-weighted CV fusion': '+ blend',
  'Phase 4: learned + CV fusion + cleanup': '+ blend + cleanup',
}

function Panel({ title, aside, children, className = '' }) {
  return (
    <section className={`border border-rule bg-surface ${className}`}>
      {(title || aside) && (
        <header className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-rule px-4 py-2.5">
          <h3 className="label">{title}</h3>
          {aside && <div className="ml-auto font-mono text-[11px] text-faint">{aside}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

/** Renders the measured markdown reports without pulling in a markdown library. */
function Doc({ text }) {
  const lines = useMemo(() => text.split('\n'), [text])
  const strip = s => s.replace(/\*\*(.+?)\*\*/g, '$1').replace(/`(.+?)`/g, '$1')
  return (
    <div className="space-y-2 text-[13.5px] leading-[1.65] text-body">
      {lines.map((ln, i) => {
        if (/^#{1,3}\s/.test(ln)) {
          const lvl = ln.match(/^#+/)[0].length
          return (
            <h4 key={i} className={`pt-4 first:pt-0 ${lvl === 1
              ? 'display text-[19px] text-strong' : 'label !text-[10px]'}`}>
              {strip(ln.replace(/^#+\s/, ''))}
            </h4>
          )
        }
        if (/^\|/.test(ln)) {
          const cells = ln.split('|').slice(1, -1).map(c => c.trim())
          if (cells.every(c => /^:?-+:?$/.test(c.replace(/\s/g, '')))) return null
          return (
            <div key={i} className="grid gap-3 border-b border-rule py-1.5"
                 style={{ gridTemplateColumns: `minmax(0,2fr) repeat(${Math.max(0, cells.length - 1)}, minmax(0,1fr))` }}>
              {cells.map((c, j) => (
                <span key={j} className={j === 0
                  ? 'text-[12.5px] text-body' : 'text-right font-mono text-[12px] text-muted'}>
                  {strip(c)}
                </span>
              ))}
            </div>
          )
        }
        if (/^[-*]\s/.test(ln)) {
          return (
            <p key={i} className="flex gap-2.5 pl-1">
              <span className="text-accent">·</span>
              <span>{strip(ln.replace(/^[-*]\s/, ''))}</span>
            </p>
          )
        }
        if (!ln.trim()) return <div key={i} className="h-2" />
        return <p key={i}>{strip(ln)}</p>
      })}
    </div>
  )
}

function Readout({ label, value, hint, tone }) {
  return (
    <div className="bg-page px-4 py-4">
      <div className="label !text-[9.5px]">{label}</div>
      <div className={`mt-2 font-mono text-[clamp(1.4rem,3.6vw,1.9rem)] font-medium leading-none
                       ${tone === 'good' ? 'text-lane' : 'text-strong'}`}>
        {value}
      </div>
      {hint && <div className="mt-1.5 text-[11px] leading-tight text-faint">{hint}</div>}
    </div>
  )
}

export default function Results() {
  const [d, setD] = useState(null)
  const [doc, setDoc] = useState(null)

  useEffect(() => { api.results().then(setD).catch(() => {}) }, [])

  const rows = d?.tables?.evaluation_val
  const ours = useMemo(
    () => rows?.find(r => r.key === 'phase4_dl' || r.name.startsWith('Phase 4: IDD')) ?? null,
    [rows])
  const phase3 = useMemo(
    () => rows?.find(r => r.name.includes('HSV OR YOLOP') || r.name.startsWith('Phase 3: HSV')) ?? null,
    [rows])
  const summary = d?.tables?.lane_inference_summary

  const ablation = useMemo(() => (rows ?? []).map(r => ({
    name: SHORT[r.name] ?? r.name,
    iou: +r.drivable_iou.toFixed(4),
    ours: r.name.startsWith('Phase 4: IDD'),
    trivial: r.name.startsWith('trivial'),
  })), [rows])

  const training = d?.tables?.idd20k_history ?? d?.tables?.training_history ?? []
  const bestEpoch = useMemo(() => training.length
    ? training.reduce((a, b) => (b.val_miou ?? 0) > (a.val_miou ?? 0) ? b : a) : null, [training])

  if (!d) {
    return (
      <div className="grid place-items-center border border-rule bg-surface p-16">
        <span className="label">Loading measured results…</span>
      </div>
    )
  }

  const f4 = v => (v == null ? '—' : v.toFixed(4))
  const solvedPct = summary ? (100 * summary.solved / (ours?.frames ?? 204)) : null

  return (
    <motion.div {...fadeUp} className="space-y-5">
      <div className="border border-rule">
        <div className="grid grid-cols-2 gap-px bg-rule sm:grid-cols-4">
          <Readout label="Drivable IoU" value={f4(ours?.drivable_iou)} tone="good"
                   hint={phase3 ? `Phase 3, ${f4(phase3.drivable_iou)}` : null} />
          <Readout label="Boundary F1" value={f4(ours?.boundary_f1)} tone="good"
                   hint={phase3 ? `Phase 3, ${f4(phase3.boundary_f1)}` : null} />
          <Readout label="Semantic mIoU" value={f4(ours?.miou)}
                   hint="7 classes, same split" />
          <Readout label="Lane solutions"
                   value={solvedPct != null ? `${solvedPct.toFixed(1)}%` : '—'}
                   hint={summary ? `${summary.solved} of ${ours?.frames ?? 204} frames` : null} />
        </div>
        <p className="border-t border-rule bg-surface px-4 py-3 text-[11.5px] leading-relaxed text-faint">
          All four figures are read live from{' '}
          <code className="bg-sunken px-1 font-mono text-body">outputs/results/</code>, measured
          by the evaluation harness on the same {ours?.frames ?? 204}-frame IDD-Lite
          validation split. Nothing on this page is typed in by hand.
        </p>
      </div>

      {ablation.length > 0 && (
        <Panel title="Drivable-area ablation"
               aside={`${ablation.length} configurations · ${ours?.frames ?? 204} frames`}>
          <p className="border-b border-rule px-4 py-3 text-[12.5px] leading-[1.6] text-muted">
            The two grey bars are trivial baselines that never look at the image.
            The Phase 3 pipeline scores below the fixed trapezoid, the finding
            that motivated rebuilding rather than tuning.
          </p>
          <div className="p-4">
            <ResponsiveContainer width="100%" height={Math.max(260, ablation.length * 32)}>
              <BarChart data={ablation} layout="vertical" margin={{ left: 96, right: 24 }}>
                <CartesianGrid strokeDasharray="2 3" stroke={GRID} horizontal={false} />
                <XAxis type="number" domain={[0, 1]} {...AXIS} />
                <YAxis type="category" dataKey="name" width={92} {...AXIS} />
                <Tooltip {...TIP} formatter={v => [v, 'drivable IoU']} cursor={{ fill: '#ffffff08' }} />
                <Bar dataKey="iou" radius={0}>
                  {ablation.map((e, i) => (
                    <Cell key={i} fill={e.ours ? '#4ec27a' : e.trivial ? '#4a545e' : '#d0483c'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}

      {training.length > 0 && (
        <Panel title="Training · IDD-20k"
               aside={bestEpoch ? `best mIoU ${bestEpoch.val_miou.toFixed(4)} at epoch ${bestEpoch.epoch}` : null}>
          <p className="border-b border-rule px-4 py-3 text-[12.5px] leading-[1.6] text-muted">
            6,993 images at 512×288, seven classes plus a distilled lane-line head.{' '}
            {training.length} epochs on Apple MPS in 127 minutes. Measured on the
            981-frame IDD-20k validation split, a different, larger split from the
            204-frame one used for the comparative evaluation above.
          </p>
          <div className="p-4">
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={training} margin={{ left: -14, right: 8 }}>
                <CartesianGrid strokeDasharray="2 3" stroke={GRID} />
                <XAxis dataKey="epoch" {...AXIS} />
                <YAxis domain={[0, 1]} {...AXIS} />
                <Tooltip {...TIP} />
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: '"IBM Plex Sans", system-ui' }} />
                <Line type="monotone" dataKey="val_drivable_iou" name="drivable IoU"
                      stroke="#4ec27a" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="val_miou" name="mIoU"
                      stroke="#d0483c" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="val_lane_iou" name="lane IoU"
                      stroke="#4d90e0" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}

      <Panel title="Measurement records" aside={`${Object.keys(d.docs).length} reports`}>
        <div className="flex flex-wrap gap-x-5 gap-y-1 border-b border-rule px-4">
          {Object.keys(d.docs).map(k => (
            <button key={k} onClick={() => setDoc(doc === k ? null : k)}
              className={`relative shrink-0 py-2.5 transition-colors
                ${doc === k ? 'text-strong' : 'text-muted hover:text-body'}`}>
              <span className="label !text-[10px] !text-inherit">{k.replace(/_/g, ' ')}</span>
              {doc === k && (
                <motion.span layoutId="docrule"
                  className="absolute inset-x-0 -bottom-px h-0.5 bg-accent"
                  transition={{ type: 'spring', stiffness: 420, damping: 34 }} />
              )}
            </button>
          ))}
        </div>
        <div className="max-h-[30rem] overflow-y-auto p-5">
          {doc
            ? <Doc text={d.docs[doc]} />
            : (
              <div className="grid place-items-center py-12 text-center">
                <FileText className="mb-3 text-rule2" size={28} strokeWidth={1.3} />
                <p className="prose-serif max-w-md text-[14.5px] leading-[1.6] text-muted">
                  Choose a record above. Each one is the written account of a
                  measurement, including the four that failed and were kept in
                  rather than deleted.
                </p>
              </div>
            )}
        </div>
      </Panel>
    </motion.div>
  )
}
