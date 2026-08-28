import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

const NOTES = {
  structured: {
    title: 'Marked highway, the lane head fires',
    body: `The distilled lane-line head finds the painted dividers and they are used
           directly as boundaries, so the partition is read off the road rather than
           inferred from its width. Confidence 0.93 is near the top of the observed
           range, which the calibration in §06 associates with a median boundary error
           below 0.6 m. This is the easy case, and the one conventional lane detection
           already handles.`,
  },
  multilane: {
    title: 'Three lanes with no paint, inference proper',
    body: `No usable markings, so there is nothing to detect. The carriageway measures
           8.88 m in the rectified view, which divided by the IRC urban lane width and
            gives three lanes of 2.96 m. Nothing in the image says "three
           lanes"; the count comes from metric width and a design standard. This is the
           case the project exists for.`,
  },
  rural: {
    title: 'Single-lane rural road',
    body: `A 4.87 m carriageway rounds to one lane, correctly, the alternative,
           forcing two 2.4 m lanes, is rejected by the minimum-width bound. Confidence
           0.92 despite the road being narrow and unmarked, because the edges are clean
           and the fit residual is low. The earlier pipeline would have reported two
           lanes here, as it did on every road.`,
  },
  dense: {
    title: 'Dense traffic, degraded but honest',
    body: `Seven road users, 21% of the frame is vehicles. A solution is still produced
           at 7.47 m and two lanes, but confidence drops to 0.67 and the calibrated
           score is doing its job: this frame sits in the band where median boundary
           error roughly doubles. The width measurement survives because occlusions are
           bridged before measuring, without that, the gap between two vehicles is
           recorded as the width of the road.`,
  },
  declined: {
    title: 'Declined, and why that is correct',
    body: `No solution. The carriageway runs off both sides of the frame, so neither
           observed boundary is a road edge; they are the edges of the photograph.
           Fitting lanes to them would produce a confident, wrong answer. Fifteen
           percent of validation frames end here, and the failure analysis in §09 shows
           the dominant cause is road geometry relative to the frame, not traffic.`,
  },
  dashcam: {
    title: 'Out of distribution, a consumer dashcam',
    body: `A 70mai dashcam at dusk: wide-angle lens, heavy vignetting, the car's own
           bonnet across the bottom of the frame. None of these appear in the training
           data. It reports 9.03 m and three lanes, which matches the road, but
           confidence falls to 0.58 and the segmentation is visibly poorer than on
           in-distribution frames. Two defects were found and fixed on this image
           alone; the remaining gap is the training distribution, not the method.`,
  },
}
const ORDER = ['structured', 'multilane', 'rural', 'dense', 'declined', 'dashcam']

function Metric({ k, v }) {
  return (
    <div>
      <div className="stat-label">{k}</div>
      <div className="font-mono text-[15px] tabular-nums text-strong">{v ?? '—'}</div>
    </div>
  )
}

export default function CaseStudies() {
  const [cases, setCases] = useState(null)
  const [sel, setSel] = useState('structured')
  useEffect(() => {
    fetch('/figures/cases.json').then(r => r.json()).then(setCases).catch(() => {})
  }, [])
  if (!cases) return null
  const c = cases[sel], n = NOTES[sel]

  return (
    <div className="mt-7">
      <div className="flex flex-wrap gap-1.5">
        {ORDER.filter(k => cases[k]).map(k => (
          <button key={k} onClick={() => setSel(k)}
            className={` px-2.5 py-1.5 font-sans text-[12px] font-medium capitalize
                        transition ${sel === k ? 'bg-road text-white'
                        : 'border border-rule text-muted hover:border-rule2 hover:text-body'}`}>
            {k}
          </button>
        ))}
      </div>

      <motion.figure key={sel} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
        transition={{ duration: .35 }} className="mt-4">
        <img src={c.file} alt={n.title}
             className="w-full border border-rule bg-page" />
        <div className="mt-4 border border-rule bg-surface/50 p-5">
          <h4 className="font-sans text-[14px] font-semibold text-strong">{n.title}</h4>
          <div className="mt-4 grid grid-cols-3 gap-4 sm:grid-cols-6">
            <Metric k="lanes" v={c.lanes} />
            <Metric k="carriageway" v={c.width != null ? `${c.width} m` : null} />
            <Metric k="lane width" v={c.laneWidth != null ? `${c.laneWidth} m` : null} />
            <Metric k="ego lane" v={c.ego ? `${c.ego}/${c.lanes}` : null} />
            <Metric k="confidence" v={c.conf} />
            <Metric k="road users" v={c.users} />
          </div>
          <p className="prose-serif mt-4 border-t border-sunken pt-4 text-[15.5px] leading-[1.65] text-body">{n.body}</p>
          <p className="mt-3 font-mono text-[11px] text-muted">
            {c.frame != null ? `IDD-Lite validation frame #${c.frame}` : 'user-supplied frame'}
            {c.source && ` · boundary source: ${c.source}`} · {c.ms} ms
          </p>
        </div>
      </motion.figure>
    </div>
  )
}
