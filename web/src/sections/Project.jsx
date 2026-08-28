import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { ArrowLeft, Radar } from 'lucide-react'
import { Eq, Fig, H2, H3, Margin, Note, Numbering, P, Plate, Refs, Table } from '../project/parts'
import { GeometryDiagram, PartitionDiagram, ProcessFlow, SystemArchitecture,
         TrainingDiagram } from '../project/diagrams'
import CaseStudies from '../project/CaseStudies'
import { Section } from '../components/ui'
import { api } from '../lib/api'

const AXIS = { tick: { fill: '#5a646e', fontSize: 11, fontFamily: '"IBM Plex Mono", monospace' },
               stroke: '#bcc2c8' }
const GRID = '#d8dce0'
const TIP = { contentStyle: { background: '#ffffff', border: '1px solid #bcc2c8',
              borderRadius: 0, fontSize: 12, fontFamily: '"IBM Plex Sans", system-ui',
              boxShadow: '0 2px 10px rgb(14 18 22 / .10)' },
              labelStyle: { color: '#0e1216', fontWeight: 600 } }

const TOC = [
  ['abstract', '00', 'Abstract'],
  ['problem', '01', 'Detection is not inference'],
  ['prior', '02', 'Where the project stood'],
  ['method', '03', 'Method'],
  ['data', '04', 'Data'],
  ['training', '05', 'Training'],
  ['results', '06', 'Results'],
  ['calibration', '07', 'Metric calibration'],
  ['cases', '08', 'Case studies'],
  ['applications', '09', 'Where this is useful'],
  ['negative', '10', 'What did not work'],
  ['failure', '11', 'Failure analysis'],
  ['limits', '12', 'Limitations'],
  ['refs', '13', 'References'],
]

const CLASS_DIST = [
  { name: 'drivable', v: 32.4 }, { name: 'far objects', v: 26.0 },
  { name: 'sky', v: 18.9 }, { name: 'roadside', v: 11.2 },
  { name: 'vehicles', v: 8.1 }, { name: 'non-drivable', v: 2.2 },
  { name: 'living things', v: 1.3 },
]
const PIE_COLORS = ['#c0342a','#2f7d4f','#7fa8d8','#2f6dbd','#2a4fa8','#9a7106','#a445b8']

const PERCLASS = [
  ['drivable',         '0.9326', '0.9536', '+2.3%'],
  ['sky',              '0.9430', '0.9523', '+1.0%'],
  ['vehicles',         '0.7562', '0.8448', '+11.7%'],
  ['far objects',      '0.7411', '0.8148', '+9.9%'],
  ['living things',    '0.5144', '0.6101', '+18.6%'],
  ['roadside objects', '0.4794', '0.5697', '+18.8%'],
  ['non-drivable',     '0.4167', '0.5486', '+31.7%'],
  ['mean',             '0.6833', '0.7563', '+10.7%'],
]

const REFERENCES = [
  { authors: 'Varma, G., Subramanian, A., Namboodiri, A., Chandraker, M., Jawahar, C.V.',
    title: 'IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments',
    venue: 'WACV 2019', url: 'https://idd.insaan.iiit.ac.in/' },
  { authors: 'Wu, D., Liao, M., Zhang, W., Wang, X.',
    title: 'YOLOP: You Only Look Once for Panoptic Driving Perception',
    venue: 'Machine Intelligence Research, 2022', url: 'https://github.com/hustvl/YOLOP' },
  { authors: 'Howard, A., Sandler, M., Chu, G., Chen, L.-C., et al.',
    title: 'Searching for MobileNetV3', venue: 'ICCV 2019', url: 'https://arxiv.org/abs/1905.02244' },
  { authors: 'Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.',
    title: 'Feature Pyramid Networks for Object Detection', venue: 'CVPR 2017',
    url: 'https://arxiv.org/abs/1612.03144' },
  { authors: 'Mallot, H., Bülthoff, H., Little, J., Bohrer, S.',
    title: 'Inverse perspective mapping simplifies optical flow computation and obstacle detection',
    venue: 'Biological Cybernetics, 1991' },
  { authors: 'Csurka, G., Larlus, D., Perronnin, F.',
    title: 'What is a good evaluation measure for semantic segmentation?',
    venue: 'BMVC 2013' },
  { authors: 'Indian Roads Congress',
    title: 'IRC:86, Geometric Design Standards for Urban Roads in Plains', venue: 'IRC, New Delhi' },
  { authors: 'Kalman, R.E.',
    title: 'A New Approach to Linear Filtering and Prediction Problems',
    venue: 'Journal of Basic Engineering, 1960' },
  { authors: 'Zuiderveld, K.',
    title: 'Contrast Limited Adaptive Histogram Equalisation', venue: 'Graphics Gems IV, 1994' },
]

export default function Project() {
  const [d, setD] = useState(null)
  const [active, setActive] = useState('abstract')
  useEffect(() => { api.results().then(setD).catch(() => {}) }, [])

  useEffect(() => {
    const io = new IntersectionObserver(
      es => es.forEach(e => e.isIntersecting && setActive(e.target.id)),
      { rootMargin: '-20% 0px -70% 0px' })
    TOC.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el) })
    return () => io.disconnect()
  }, [d])

  const ablation = useMemo(() => {
    const rows = d?.tables?.evaluation_val
    if (!rows) return []
    const short = {
      'trivial: everything is road': 'all road',
      'trivial: fixed trapezoid prior': 'trapezoid prior',
      'Phase 3: HSV threshold (CV only)': 'Phase 3 HSV',
      'Phase 3: YOLOP as wired in Phase 3': 'YOLOP as wired',
      'Phase 3: HSV OR YOLOP (the Phase 3 pipeline)': 'Phase 3 pipeline',
      'YOLOP, correctly wired': 'YOLOP corrected',
      'YOLOP (correct) + confidence-weighted CV fusion': 'YOLOP + blend',
      'Phase 4: adaptive CIELAB + texture (CV only)': 'Phase 4 classical',
      'Phase 4: IDD-trained segmentation': 'Phase 4 learned',
      'Phase 4: learned + cleanup (no CV fusion)': '+ cleanup',
      'Phase 4: learned + confidence-weighted CV fusion': '+ blend',
      'Phase 4: learned + CV fusion + cleanup': '+ blend + cleanup',
    }
    return rows.map(r => ({ name: short[r.name] ?? r.name,
      iou: +r.drivable_iou.toFixed(4),
      kind: r.name.startsWith('Phase 4: IDD') ? 'ours'
          : r.name.startsWith('trivial') ? 'trivial' : 'other' }))
  }, [d])

  const training = d?.tables?.idd20k_history ?? []

  const sweep = useMemo(() => {
    const rows = d?.tables?.sensitivity
    if (!Array.isArray(rows) || !rows.length) return null
    return rows.map(r => [
      `${r.camera_height_m.toFixed(2)} m${r.camera_height_m === 1.75 ? ' (adopted)' : ''}`,
      `${r.solved}/${r.frames}`,
      `${r.median_carriageway_m.toFixed(2)} m`,
      `${r.median_lane_width_m.toFixed(2)} m`,
      r.mean_lane_count.toFixed(2),
    ])
  }, [d])

  return (
   <Numbering>
    <div className="min-h-screen">
      <header className="no-print sticky top-0 z-40 border-b border-rule bg-page/95 backdrop-blur">
        <div className="mx-auto flex h-12 max-w-plate items-center gap-5 px-5 sm:px-8">
          <Link to="/" className="flex items-center gap-1.5 text-muted transition-colors hover:text-strong">
            <ArrowLeft size={14} />
            <span className="label !text-[10px]">Index</span>
          </Link>
          <span className="hidden h-3.5 w-px bg-rule2 sm:block" />
          <span className="label hidden !text-[10px] sm:block">
            Vision-Based Lane Inference
          </span>
          <span className="ml-auto font-mono text-[11px] text-faint">
            {active !== 'abstract' && `§${TOC.find(([id]) => id === active)?.[1] ?? ''}`}
          </span>
          <Link to="/demo"
            className="flex items-center gap-1.5 border border-strong px-2.5 py-1 text-strong transition-colors hover:bg-strong hover:text-page">
            <Radar size={13} />
            <span className="label !text-[10px] !text-inherit">Live demo</span>
          </Link>
        </div>
      </header>

      <div className="border-b border-strong">
        <div className="mx-auto max-w-plate px-5 pb-10 pt-14 sm:px-8 sm:pt-20">
          <div className="label mb-6 !text-accent">Technical Report &middot; Phase 4</div>

          <h1 className="display max-w-[19ch] text-[clamp(2.6rem,8.5vw,6.2rem)] text-strong">
            Vision-Based<br />Lane Inference
          </h1>

          <p className="prose-serif mt-7 max-w-column text-[19px] leading-[1.5] text-muted">
            Inferring lane structure on unstructured and structured Indian roads,
            where the paint that lane detectors look for is frequently absent.
          </p>

          <dl className="mt-12 grid grid-cols-2 gap-x-8 gap-y-6 border-t border-rule pt-6 sm:grid-cols-4">
            {[
              ['Authors', <>Harshwardhan&nbsp;M.&nbsp;Mohadikar<br />
                           <span className="text-faint">2023EBCS353</span><br />
                           <span className="mt-1 block">Shreyas&nbsp;Bhat&nbsp;K</span>
                           <span className="text-faint">2023EBCS460</span></>],
              ['Supervisor', <>Dr.&nbsp;Ashok&nbsp;Yemineni</>],
              ['Institution', <>BITS&nbsp;Pilani<br />
                           <span className="text-faint">BSc Computer Science</span></>],
              ['Dataset', <>IDD &middot; 20k<br />
                           <span className="text-faint">7,974 annotated frames</span></>],
            ].map(([k, v]) => (
              <div key={k}>
                <dt className="label mb-1.5">{k}</dt>
                <dd className="prose-serif text-[14px] leading-[1.5] text-body">{v}</dd>
              </div>
            ))}
          </dl>

          <div className="mt-10 grid grid-cols-2 gap-px border border-rule bg-rule sm:grid-cols-4">
            {[
              ['0.9403', 'Drivable IoU', 'val, 981 frames'],
              ['0.8343', 'Boundary F1', 'val, 981 frames'],
              ['0.46 m', 'Median edge error', '2,430 boundary samples'],
              ['31 fps', 'End-to-end', 'laptop GPU'],
            ].map(([v, k, sub]) => (
              <div key={k} className="bg-page px-4 py-5">
                <div className="font-mono text-[clamp(1.4rem,3.4vw,2rem)] font-medium leading-none text-strong">
                  {v}
                </div>
                <div className="label mt-2.5 !text-[10px] !text-body">{k}</div>
                <div className="mt-1 text-[11.5px] text-faint">{sub}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mx-auto grid max-w-plate gap-12 px-5 pb-32 pt-14 sm:px-8 lg:grid-cols-[13rem,minmax(0,1fr)] xl:gap-20">
        <nav className="no-print hidden lg:block">
          <div className="sticky top-[4.5rem]">
            <div className="label mb-3 border-b border-rule pb-2">Contents</div>
            {TOC.map(([id, n, label]) => {
              const on = active === id
              return (
                <a key={id} href={`#${id}`}
                  className={`group flex gap-3 border-l-2 py-[7px] pl-3 text-[13px] leading-snug
                    transition-colors ${on
                      ? 'border-accent text-strong'
                      : 'border-transparent text-muted hover:border-rule2 hover:text-body'}`}>
                  <span className={`font-mono text-[11px] ${on ? 'text-accent' : 'text-faint'}`}>
                    {n}
                  </span>
                  <span className="prose-serif">{label}</span>
                </a>
              )
            })}
          </div>
        </nav>

        <article className="relative min-w-0">
          <H2 id="abstract" n="00">Abstract</H2>
          <P lead>
            Lane-detection systems locate painted markings. On the unstructured
            roads that make up much of the Indian network there is often no paint,
            yet the road still carries a lane structure that drivers agree on and
            navigate by. This project infers that structure rather than detecting it.
          </P>
          <P>
            A semantic segmentation network trained on the Indian Driving Dataset
            produces a drivable-surface mask. The vanishing point of the two road
            boundaries gives the camera pitch, which rectifies the ground plane to a
            metric bird's-eye view. In that view the carriageway is measured in
            metres and divided by the IRC design lane width, so the output is a lane
            <em> count</em> rather than an arbitrary subdivision. A Kalman filter
            over the lane geometry stabilises the result across video.
          </P>
          <P>
            On the IDD-Lite validation split the system reaches{' '}
            <strong className="text-strong">0.9403 drivable IoU</strong> and{' '}
            <strong className="text-strong">0.8343 boundary F1</strong>, against
            0.4270 and 0.2048 for the pipeline this replaces, which, we show, scores
            below a fixed trapezoid that never looks at the image. Lane inference
            produces a metric solution on 83.8% of frames with a median carriageway
            boundary error of 0.46 m. The complete pipeline runs at 31 fps on a
            laptop GPU.
          </P>
          <Note tone="warn">
            Three experiments in this report failed and are reported as such: fusing
            the classical and learned branches, bridging vehicle occlusions for
            vanishing-point estimation, and inferring lanes from observed traffic
            trajectories. Each was implemented, measured, and rejected on the
            evidence. They are kept because a method section that reports only what
            worked is not a record of what was learned.
          </Note>

          <H2 id="problem" n="01">Detection is not inference</H2>
          <P>
            The distinction is the whole problem. A marked carriageway can be handled
            by finding the paint: a lane-line segmentation head, a Hough transform, a
            row-wise classifier. None of that transfers to a road with no markings,
            which describes a large fraction of the Indian network and essentially
            all of its rural and peri-urban roads.
          </P>
          <P>
            What remains on an unmarked road is geometry. The drivable surface has a
            measurable width, vehicles occupy it in loosely-ordered streams, and
            design standards specify how wide a lane should be. Lane structure can
            therefore be <em>inferred</em> from the shape of the road rather than
            read off its surface.
          </P>
          <H3>Why image space is the wrong place to do it</H3>
          <P>
            The previous iteration of this project estimated lanes by taking the
            widest run of road pixels in each image row and splitting it at the
            midpoint. That has two structural faults. It returns exactly two lanes on
            every road regardless of width, and it reasons in image space, where the
            number of pixels per metre falls away sharply with distance.
          </P>
          <Eq where="a constant 7 m carriageway, through the calibrated camera model">
            385 px at 5 m &nbsp;&rarr;&nbsp; 55 px at 35 m
          </Eq>
          <P>
            The same seven metres of road is worth seven times fewer pixels at
            35 m than at 5 m. Any per-row rule is fighting that factor, which is why
            the far field was the least stable part of the earlier system.
          </P>

          <H2 id="prior" n="02">Where the project stood</H2>
          <P>
            The prior pipeline combined a fixed HSV threshold for asphalt with a
            pretrained YOLOP drivable-area head, fused with a bitwise OR. It was
            evaluated qualitatively, on seven web images. Re-implementing it exactly
            and scoring it on labelled Indian data produced three findings.
          </P>
          <Table
            head={['Configuration', 'Drivable IoU', 'Precision', 'Recall', 'Boundary F1']}
            rows={[
              ['Trivial: everything is road', '0.3179', '0.3179', '1.0000', '—'],
              ['Trivial: fixed trapezoid, ignores image', '0.6450', '0.8246', '0.7476', '0.1957'],
              ['HSV threshold', '0.4131', '0.4682', '0.7783', '0.2093'],
              ['YOLOP, as it was wired', '0.1667', '0.9119', '0.1694', '0.1382'],
              ['HSV OR YOLOP, the prior pipeline', '0.4270', '0.4747', '0.8093', '0.2048'],
              ['YOLOP, preprocessing corrected', '0.7243', '0.9923', '0.7284', '0.2205'],
            ]}
            caption="IDD-Lite validation split, 204 frames. The two trivial baselines establish the floor any method must clear."
          />
          <Note tone="bad">
            <strong className="text-strong">The classical detector scored below a
            fixed triangle.</strong> At 0.4131 IoU the HSV threshold performs worse
            than a hard-coded trapezoid that never looks at the image (0.6450). Its
            precision of 0.4682 means more than half the pixels it called road were
            not road: the bounds accept any desaturated, mid-brightness pixel, which
            describes concrete, dust, overcast sky and white vehicles as readily as
            asphalt.
          </Note>
          <Note tone="good">
            <strong className="text-strong">YOLOP was not failing, it was
            mis-wired.</strong> As used, it scored 0.1667 IoU at 0.1694 recall: it
            was finding about one sixth of the road. Two preprocessing steps were
            missing, the input was stretched to 640×384 rather than letterboxed, and
            ImageNet normalisation was omitted entirely. Restoring both, with
            identical weights and no retraining, raises the same model to{' '}
            <strong className="text-strong">0.7243 IoU, a 4.3× improvement</strong>.
          </Note>
          <Note tone="bad">
            <strong className="text-strong">The fusion could not have worked.</strong>{' '}
            A bitwise OR can only <em>add</em> pixels, so it cannot suppress a false
            positive from either branch, and the gate in front of it
            (<code className="bg-sunken px-1 font-mono text-[12px]">dl_ratio &lt; 0.01</code>)
            almost never fires because a drivable head typically claims 20 to 30% of the
            frame. Measured: HSV alone 0.4131, HSV OR YOLOP 0.4270. The deep-learning
            branch contributed <strong className="text-strong">0.0138 IoU</strong>.
          </Note>

          <H2 id="method" n="03">Method</H2>
          <P>
            The pipeline is a chain of five stages, each of which can be disabled
            independently so that its contribution can be attributed.
          </P>
          <SystemArchitecture />
          <Fig src="/figures/qualitative_phase4.png"
            caption="Validation frames through the full pipeline: input, inferred lanes with the ego lane highlighted, seven-class semantics, and the rectified ground plane with its metre grid." />

          <H3>3.1 Semantic segmentation</H3>
          <P>
            A MobileNetV3-Large encoder with an LR-ASPP context module and a
            lightweight FPN decoder, 3.33 M parameters. The encoder is chosen for
            depthwise-separable efficiency; the FPN decoder recovers the boundary
            detail a plain LR-ASPP head loses, which matters because the rectification
            downstream is sensitive to road-edge accuracy. Seven classes, plus a
            binary lane-line head described in §3.4.
          </P>

          <H3>3.2 Vanishing point and camera pitch</H3>
          <P>
            For each scanline the longest contiguous run of road pixels gives a left
            and right boundary point. Two robust line fits intersect at the vanishing
            point. Camera pitch follows directly:
          </P>
          <Eq where="cy = principal point, v_vp = vanishing point row, f = focal length">
            θ = arctan( (c<sub>y</sub> − v<sub>vp</sub>) / f )
          </Eq>
          <P>
            Pitch is therefore <em>measured every frame</em> rather than assumed.
            Boundary points lying on the frame border are excluded: there the observed
            edge is the edge of the photograph, not of the road, and including them
            drives the fit vertical. On one validation frame this produced a 25.1°
            pitch estimate and a rectification containing only sky; excluding them
            corrects it to −0.9°.
          </P>

          <H3>3.3 Rectification to a metric ground plane</H3>
          <P>
            With pitch known, the ground plane maps to a bird's-eye view by a
            homography. Substituting the measured pitch back into the projection and
            simplifying gives the lateral scale directly:
          </P>
          <GeometryDiagram />
          <Eq where="du = pixel width at image row v, H = camera height">
            dX = du · H / (v − v<sub>horizon</sub>)
          </Eq>
          <Note tone="good">
            <strong className="text-strong">The focal length cancels.</strong> It
            cancels because the horizon position already encodes f·tan θ, and that
            combination is exactly what sets the ground-plane scale. Verified by
            sweeping the assumed field of view from 40° to 140° with the vanishing
            point fixed: recovered lateral span moves 3.467 m → 3.503 m, about{' '}
            <strong className="text-strong">1% across a 100° range</strong>, while
            forward distance changes by a factor of nine. Lane inference is a purely
            lateral operation, so it is insensitive to field of view, and calibration
            reduces to a single parameter.
          </Note>

          <H3>3.4 Lane partitioning</H3>
          <P>
            In the rectified view the carriageway is measured in metres at 0.5 m
            intervals of forward distance. Where painted markings are found they are
            used directly as lane boundaries. Where they are not, the unstructured
            case, the carriageway is divided:
          </P>
          <Eq where="IRC:86, 3.5 m urban, 3.0 m rural carriageway">
            n = round( W<sub>carriageway</sub> / W<sub>lane</sub> )
          </Eq>
          <PartitionDiagram />
          <P>
            The integer rounding is what makes this a lane <em>count</em> rather than
            an arbitrary subdivision, and it is why the method needs a metric ground
            plane rather than pixels. The count is then bounded so that no lane falls
            below the minimum usable width.
          </P>
          <P>
            Marking detection uses a lane-line head distilled from YOLOP, which the
            prior pipeline never read, it used only the drivable-area output. On
            validation frames YOLOP's lane head returns lane pixels on 67% of frames
            against 9.4% for a hand-written top-hat filter. Distilling it into a
            second head on our own network gives lane detection in the same 13 ms
            forward pass rather than YOLOP's separate ~150 ms.
          </P>

          <H3>3.5 Temporal stabilisation</H3>
          <P>
            A Kalman filter tracks the carriageway centreline and width. The state is
            parameterised as centreline-plus-width rather than two independent edges,
            which stops the filter from letting the inferred road breathe. The
            discrete lane count is stabilised separately by a majority vote, since
            smoothing an integer would be meaningless.
          </P>

          <H3>3.6 The decision path</H3>
          <P>
            Four stages can decline to answer. That is deliberate: on 15% of
            validation frames the carriageway is not visible enough to measure, and a
            confident wrong answer there is worse than an honest refusal.
          </P>
          <ProcessFlow />

          <H2 id="data" n="04">Data</H2>
          <P>
            All training and evaluation uses the Indian Driving Dataset [1], captured
            on Indian roads with a forward-facing vehicle camera. Three subsets are
            used, for three different purposes.
          </P>
          <Table
            head={['Subset', 'Images', 'Resolution', 'Used for']}
            rows={[
              ['IDD-Lite', '1,403 / 204', '320×227', 'initial training, ablation baseline'],
              ['IDD Segmentation 20k, Part I', '6,993 / 981', '→ 512×288', 'the shipped model'],
              ['IDD Temporal Val', '150 sequences, 4,650 frames', '960×540', 'temporal evaluation'],
            ]}
            caption="Train / validation counts. IDD ships ground truth as polygon JSON; label images were rasterised with the dataset's own tooling at the level-1 (7-class) hierarchy."
          />
          <div className="mt-6 grid gap-6 sm:grid-cols-2">
            <div>
              <div className="stat-label mb-2">Class distribution, training split</div>
              <ResponsiveContainer width="100%" height={210}>
                <PieChart>
                  <Pie data={CLASS_DIST} dataKey="v" nameKey="name" innerRadius={42}
                       outerRadius={80} paddingAngle={2}>
                    {CLASS_DIST.map((e, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
                  </Pie>
                  <Tooltip {...TIP} formatter={v => [`${v}%`, 'of pixels']} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="self-center">
              <P>
                The distribution is heavily imbalanced. Two classes —{' '}
                <span className="text-strong">non-drivable at 2.2%</span> and{' '}
                <span className="text-strong">living things at 1.3%</span>, are rare
                enough that they were the weakest by IoU on the smaller subset, and
                they are precisely the classes that improved most when the training
                set grew. Loss weighting uses inverse-square-root frequency, which
                lifts the rare classes without the instability of full inverse
                weighting.
              </P>
            </div>
          </div>
          <Note tone="warn">
            <strong className="text-strong">Only Part I of IDD-20k is used.</strong>{' '}
            The dataset ships in two parts; the preparation script located the first
            directory containing <code className="bg-sunken px-1 font-mono text-[13px]">leftImg8bit/</code>{' '}
            and <code className="bg-sunken px-1 font-mono text-[13px]">gtFine/</code> and
            stopped there, so Part II was downloaded, extracted, and silently never
            merged. The shipped model is therefore trained on 6,993 of roughly 14,000
            available annotated frames. The bug is fixed, the scan now collects every
            dataset root rather than the first, but retraining on the full set is
            outstanding, and on the evidence of §5, where 5× the data bought +10.8%
            mIoU, it is the single highest-value item left.
          </Note>
          <Note>
            <strong className="text-strong">Contamination check.</strong> IDD-Lite is
            a subsample of IDD, so its validation frames could in principle appear in
            the 20k training split. Checked by drive and frame identifier:{' '}
            <strong className="text-strong">0 of 204</strong> IDD-Lite validation
            frames occur in IDD-20k training. Every number reported here is on data
            the model never saw.
          </Note>

          <H2 id="training" n="05">Training</H2>
          <P>
            AdamW with cosine decay and linear warm-up, weighted cross-entropy plus
            Dice loss, auxiliary deep supervision from the stride-16 feature level,
            and a binary lane term with positive weighting. 40 epochs at 288×512,
            batch 16, on a single laptop GPU, 127 minutes. An earlier attempt at the
            same schedule on Colab was destroyed at epoch 26 when the free GPU quota
            expired; §10 records why the resume mechanism did not save it.
          </P>
          {training.length > 0 && (
            <figure className="mt-6">
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={training} margin={{ left: -12, right: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#d8dce0" />
                  <XAxis dataKey="epoch" {...AXIS} />
                  <YAxis domain={[0, 1]} {...AXIS} />
                  <Tooltip {...TIP} /><Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="val_drivable_iou" name="drivable IoU"
                        stroke="#2f7d4f" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_miou" name="mIoU"
                        stroke="#c0342a" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_lane_iou" name="lane-line IoU"
                        stroke="#2f6dbd" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <figcaption className="mt-2.5 text-[13px] text-muted">
                Validation metrics per epoch on the 981-frame IDD-20k validation
                split. The schedule ran to completion and converged well before it
                ended, the final five epochs span 0.0006 mIoU, and the best epoch
                (35) reached 0.7574 mIoU, 0.9538 drivable IoU and 0.6311 lane-line
                IoU.
              </figcaption>
            </figure>
          )}

          <TrainingDiagram />

          <H3>Augmentation</H3>
          <P>
            The augmentation suite was chosen against the failure modes the earlier
            phase had documented qualitatively, over-exposure, cast shadows, motion
            blur, low resolution, rather than from a standard recipe. A second group
            targets conditions the source dataset does not contain at all: IDD is
            daytime footage from a normal lens, so synthetic vignetting, non-linear
            low-light response and barrel distortion were added to cover consumer
            dashcam optics.
          </P>

          <H3>Scaling the training set</H3>
          <Table
            head={['Class', 'IDD-Lite (1,403)', 'IDD-20k (6,993)', 'change']}
            rows={PERCLASS} highlight={7}
            caption="Per-class validation IoU. The rare classes gained most, as predicted: non-drivable +31.7%, roadside +18.8%. Drivable IoU, already near its ceiling, moved least."
          />

          <H2 id="results" n="06">Results</H2>
          {ablation.length > 0 && (
            <figure className="mt-6">
              <ResponsiveContainer width="100%" height={Math.max(300, ablation.length * 30)}>
                <BarChart data={ablation} layout="vertical" margin={{ left: 100, right: 24 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#d8dce0" horizontal={false} />
                  <XAxis type="number" domain={[0, 1]} {...AXIS} />
                  <YAxis type="category" dataKey="name" width={96} {...AXIS} />
                  <Tooltip {...TIP} formatter={v => [v, 'drivable IoU']} />
                  <Bar dataKey="iou" radius={0}>
                    {ablation.map((e, i) => (
                      <Cell key={i} fill={e.kind === 'ours' ? '#2f7d4f'
                        : e.kind === 'trivial' ? '#9aa3ac' : '#c0342a'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <figcaption className="mt-2.5 text-[13px] text-muted">
                Full ablation on the IDD-Lite validation split. Grey are the trivial
                baselines; green is the method as delivered.
              </figcaption>
            </figure>
          )}
          <Fig src="/figures/phase3_vs_phase4.png"
            caption="Identical frames through both pipelines. Columns three and four contrast road masks against ground truth; five and six contrast the lane output. The prior mask routinely claims sky and vegetation." />

          <H3>Lane inference accuracy</H3>
          <P>
            No dataset provides ground-truth lane geometry for Indian roads, that
            absence is why the problem is inference. But whatever the lane count, the
            outermost inferred boundaries must coincide with the real edges of the
            road, and the drivable ground truth gives those. Comparing them in the
            rectified plane at 1 m intervals over 6 to 22 m:
          </P>
          <Table
            head={['Quantity', 'median', 'mean', 'p90']}
            rows={[
              ['Left edge error', '0.461 m', '1.225 m', '3.772 m'],
              ['Right edge error', '0.457 m', '1.239 m', '3.863 m'],
              ['Centreline error', '0.540 m', '1.010 m', '2.469 m'],
              ['Carriageway width error', '1.107 m', '1.976 m', '4.921 m'],
            ]}
            caption="2,430 boundary samples over 169 frames. Half of all boundaries fall within 46 cm of the true road edge at distances up to 22 m; the mean sitting far above the median indicates a heavy tail."
          />

          <H3>A calibrated confidence</H3>
          <P>
            A heavy error tail matters only if the system cannot tell which frames are
            in it. As first written it could not, the confidence score correlated{' '}
            <strong className="text-strong">−0.187</strong> with measured error.
            Correlating candidate signals against error showed why: the strongest
            predictor was unused, while a component indistinguishable from noise
            carried 40% of the weight.
          </P>
          <Table
            head={['Signal', 'corr. with error', 'in old score?']}
            rows={[
              ['Edge polynomial fit residual', '+0.435', 'no'],
              ['Road users detected', '+0.270', 'no'],
              ['Vanishing-point confidence', '−0.251', 'no'],
              ['Fraction of samples with both edges valid', '−0.248', 'no'],
              ['Width coefficient of variation', '+0.188', 'yes'],
              ['Per-edge coverage', '−0.054', 'yes, 40% weight'],
            ]}
          />
          <P>
            Rebuilt from the four predictive signals, weighted by strength, the score
            now correlates <strong className="text-strong">−0.508</strong> with
            error, verified at −0.540 on a held-out half. Gating at 0.7 keeps 71% of
            frames and removes 25% of the median error and 29% of the p90, the tail
            is now identifiable in advance rather than only in hindsight.
          </P>

          <H3>Temporal stability</H3>
          <Table
            head={['Measure', 'per-frame', '+ Kalman', 'change']}
            rows={[
              ['Width jitter, p50', '0.486 m', '0.054 m', '−89%'],
              ['Width jitter, p90', '4.066 m', '0.342 m', '−92%'],
              ['Centreline offset jitter', '0.876 m', '0.176 m', '−80%'],
              ['Heading jitter', '0.1248', '0.0226', '−82%'],
              ['Lane-count changes / 100 frames', '28.29', '5.86', '−79%'],
              ['Solution rate', '85.1%', '88.9%', '+3.8 pp'],
            ]}
            caption="150 sequences, 4,464 frames of IDD Temporal Val. The solution rate rising alongside the jitter falling is the control: a filter that had stopped tracking its input would improve jitter alone."
          />

          <H2 id="calibration" n="07">Metric calibration</H2>
          <P>
            Because §3.3 shows the focal length cancels, the pipeline has exactly one
            metric free parameter: the camera height. IDD ships no camera metadata, so
            it cannot be measured and is stated as a prior of 1.75 m. Every metric
            output scales linearly with it.
          </P>
          <Table
            head={['Camera height', 'frames solved', 'median carriageway', 'median lane width', 'mean lane count']}
            rows={sweep ?? []}
            highlight={sweep ? sweep.findIndex(r => r[0].includes('adopted')) : -1}
            caption="Sensitivity sweep over the single free parameter."
          />
          <Note tone="good">
            Median inferred lane width stays within{' '}
            <strong className="text-strong">3.16 to 3.45 m across the entire sweep</strong>,
            against the IRC design range of 3.0 to 3.5 m. This is a real robustness
            property of integer partitioning: as the assumed scale grows, the inferred
            count grows with it and the quotient stays near the design width. The
            reported <em>structure</em> degrades gracefully under scale error even
            though absolute widths do not.
          </Note>

          <H2 id="cases" n="08">Case studies</H2>
          <P>
            Six frames, chosen to span what the system does well, what it does
            poorly, and where it declines. Every number below is what the pipeline
            actually returned for that frame.
          </P>
          <CaseStudies />

          <H2 id="applications" n="09">Where this is useful</H2>
          <P>
            A metric lane structure with a calibrated confidence is a different
            output from a set of detected lane markings, and it supports things
            marking detection cannot.
          </P>
          <div className="mt-6 grid gap-4 sm:grid-cols-2">
            {[
              ['Driver assistance on unmarked roads',
               'Lane-departure and lane-keeping systems fail silently where there is no paint. A width-derived lane structure degrades to "one wide lane" rather than to nothing, and reports how much it should be trusted.'],
              ['Road inventory and asset survey',
               'Carriageway width at metric scale, per frame, from a dashcam. Surveying width currently means either a site visit or manual measurement from imagery; this measures it at 31 fps with a stated error of 0.46 m median.'],
              ['Traffic analysis on heterogeneous roads',
               'Road users are placed on a metric ground plane and assigned to inferred lanes, giving per-lane occupancy and headway on roads where induction loops and lane-based counting do not apply.'],
              ['Dataset annotation assistance',
               'Lane annotation on unstructured roads is slow because annotators must decide where lanes are. A metric proposal with a confidence score gives them a starting point and flags the frames worth checking.'],
              ['Wrong-side and lane-discipline monitoring',
               'Implemented but unvalidated here for want of ground truth. Given an inferred lane structure and tracked road users, the geometry is straightforward; what is missing is labelled footage, not method.'],
              ['A teaching artefact',
               'The pipeline is deliberately stage-separable, and each stage can be disabled to see what it contributes. The demo exposes all eleven stages on any frame the user supplies.'],
            ].map(([h, b], i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-60px' }} transition={{ duration: .4, delay: i * .05 }}
                className="border border-rule bg-surface/40 p-5">
                <h4 className="font-sans text-[13px] font-semibold text-strong">{h}</h4>
                <p className="prose-serif mt-2 text-[15px] leading-[1.6] text-muted">{b}</p>
              </motion.div>
            ))}
          </div>

          <H2 id="negative" n="10">What did not work</H2>
          <P>
            Four plausible ideas were implemented, measured, and rejected. They are
            reported because the measurements are the contribution: a negative result
            that is measured is worth more than a positive one that is assumed.
          </P>

          <H3>Fusing the classical and learned branches</H3>
          <P>
            Replacing the prior bitwise OR with a principled confidence-weighted sum
            still <em>reduces</em> accuracy: 0.9403 → 0.9066 with the in-domain model,
            and 0.7243 → 0.6979 with out-of-domain YOLOP. The classical branch is much
            weaker than either learned branch (0.5697 IoU), so any fixed-weight blend
            drags the result toward it. The hybrid CV+DL premise the project carried
            from its earliest phase is <strong className="text-strong">not supported
            by the data</strong>. The classical branch is retained as a gated fallback, used only if the learned branch collapses, which is exactly equivalent
            to the learned branch on this data while still covering model failure.
          </P>

          <H3>Bridging vehicle occlusions for the vanishing point</H3>
          <P>
            Vehicles fragment the road mask, so unioning them back in seems natural.
            Measured, it drops vanishing-point recovery from 90.2% to 74.5%: parked
            roadside vehicles extend the region sideways into the frame border, which
            is exactly the failure mode the border-exclusion rule removes. A more
            targeted version, bridging only gaps genuinely covered by occluder pixels, avoids that but still degrades width consistency in dense traffic
            (0.309 → 0.372), because the bridged boundary is noisier to fit than the
            smaller unoccluded fragment.
          </P>
          <Note tone="good">
            The same operation applied to the <em>width-measurement</em> stage is
            essential, and finding that required separating the two uses. The per-row
            longest run is interrupted by any vehicle standing on the carriageway, so
            a pair of parked vans causes the gap between them to be recorded as the
            width of the road, one frame reported a 2.75 m "carriageway" that was
            exactly that. Bridging for width raises coverage from 150 to 171 frames
            and the median carriageway from 5.24 m to 6.55 m. The two stages need
            different masks because they ask different questions.
          </Note>

          <H3>Inferring lanes from observed traffic</H3>
          <P>
            The most interesting failure. On a road without paint, the lanes might be
            wherever traffic actually flows, so accumulating observed vehicle
            positions and finding modes in their lateral distribution should recover
            lane structure empirically. Sweeping the evidence threshold makes the
            estimator fire on 75% of clips rather than 3.4% of frames, but{' '}
            <strong className="text-strong">agreement with the geometric estimate
            stays flat at 34 to 40%</strong>, and recovered spacing holds at a 4.8 m
            median against the IRC 3.0 to 3.5 m range without tightening as more evidence
            is required.
          </P>
          <P>
            That pattern rules out insufficient evidence as the explanation. If the
            modes were lanes seen through noise, demanding more observations would pull
            the spacing toward the design width. It does not. The modes are stable and
            reproducible, they are simply not lanes. A plausible reading, offered as
            interpretation rather than measurement, is that over a three-second window
            the two dominant clusters are a vehicle ahead and an oncoming vehicle,
            about 5 m apart on a two-lane road. The estimator does recover 2, 3 and
            4-lane layouts from synthetic sequences with centres accurate to better
            than 0.5 m, so what has been shown is narrower and more interesting:{' '}
            <strong className="text-strong">three-second windows of real Indian
            traffic do not exhibit lane-shaped lateral modes</strong>.
          </P>

          <H3>Closing the consumer-dashcam domain gap at test time</H3>
          <P>
            A frame from a 70mai dashcam, dusk, wide-angle lens, heavy vignetting,
            the vehicle's own bonnet across the bottom, produced a visibly poor
            result. Two genuine defects surfaced first and were fixed. Bottom
            connectivity assumed the road reaches the last row, which is true on IDD
            and false on any dashcam that sees its own bonnet: 48,461 px of correctly
            segmented carriageway were discarded while a 972-px false positive on the
            bonnet was kept. Rewriting the rule as <em>low in the frame and large</em>{' '}
            rather than <em>touching the last row</em> improved IDD as a side effect,
            0.9067 → 0.9258. Separately, only one of the two carriageway-width
            estimates was bounds-checked, so where the fitted edges cross the second
            collapsed to its 1e-6 clamp and a zero-width carriageway propagated
            downstream, 0.00 m became 9.03 m, three lanes, matching the road.
          </P>
          <P>
            What remained is not a bug but the training distribution. IDD is daylight
            footage from a normal lens; nothing in it resembles a wide-angle sensor at
            dusk. Four remedies were measured and all four were rejected.
          </P>
          <Table
            head={['Approach', 'on the dashcam frame', 'on IDD']}
            rows={[
              ['CLAHE applied unconditionally', 'drivable 13.0% → 29.7%', 'IoU 0.9437 → 0.8790'],
              ['Adaptive CLAHE by image statistics', 'no trigger exists', '—'],
              ['Dual hypothesis, select by lane confidence', 'picks the wrong one', 'IoU 0.9329 → 0.9062'],
              ['Dual hypothesis, select by classical agreement', 'picks correctly', 'IoU 0.9402 → 0.9002'],
              ['Fine-tune with synthetic dashcam augmentation', '13.6% → 13.5%', '0.9329 → 0.9303'],
            ]}
            caption="Every remedy either costs more on the evaluation set than it recovers on the target case, or cannot be triggered. Contrast enhancement works and cannot be applied."
          />
          <P>
            The reason no statistic can gate it is the interesting part. Lightness
            standard deviation, percentile spread, median and dark-pixel fraction were
            measured across the validation split against the dashcam frame, and it sits{' '}
            <em>inside</em> IDD's normal range on every one.
          </P>
          <Table
            head={['Statistic', 'IDD p05', 'IDD p50', 'IDD p95', 'dashcam']}
            rows={[
              ['Lightness std', '49.5', '66.4', '86.9', '53.7'],
              ['Lightness spread (p95−p05)', '152.7', '215.5', '248.7', '178.0'],
              ['Lightness median', '50.0', '92.5', '126.2', '87.0'],
              ['Dark fraction', '0.1', '0.2', '0.4', '0.2'],
            ]}
            caption="The failure is therefore not low contrast. CLAHE helps for a different reason, it breaks up a large, smoothly varying hazy region the network otherwise labels sky."
          />
          <P>
            A prediction-side trigger was also tested. Sky predicted below the horizon
            is physically impossible, and it does separate, 0.47% on the dashcam
            against a maximum of 0.43% across the validation split, but by a margin
            far too narrow to gate on. Fine-tuning on synthetic vignetting, non-linear
            low-light response and barrel distortion for 12 epochs moved recovered
            drivable area from 13.6% to 13.5%, no change, while costing 0.0026 IoU.
          </P>
          <Note tone="warn">
            <strong className="text-strong">The honest statement of the
            limitation:</strong> the system is trained on daylight, normal-lens
            footage and degrades on wide-angle sensors at dusk. The fix is training
            data of that kind, not a test-time transform. <code className="bg-sunken px-1 font-mono text-[13px]">seg_idd20k</code>{' '}
            remains the shipped checkpoint; the fine-tuned variant is retained as{' '}
            <code className="bg-sunken px-1 font-mono text-[13px]">seg_robust</code>{' '}
            but is not used, because it costs accuracy on the evaluation set and buys
            nothing on the case it was built for. IDD-AW, the adverse-weather subset
            with low-light captures, is the obvious next source and is the one
            substantive experiment this analysis leaves open.
          </Note>

          <H2 id="failure" n="11">Failure analysis</H2>
          <P>
            Lane inference produces no solution on 31 of 204 validation frames.
            Attributing each to a stage, and characterising what separates accurate
            frames from inaccurate ones, overturned an assumption the project had
            carried from its qualitative phase.
          </P>
          <Table
            head={['Cause', 'frames', 'road pixels', 'road touching frame edge']}
            rows={[
              ['Road edges not visible', '10', '37.0%', '28.6%'],
              ['No vanishing point', '10', '29.2%', '21.2%'],
              ['Carriageway too narrow', '10', '24.3%', '13.7%'],
              ['(solved, for reference)', '173', '32.1%', '20.9%'],
            ]}
          />
          <Note tone="warn">
            <strong className="text-strong">Traffic density does not explain
            failure.</strong> Unsolved frames average 8.8% vehicle pixels against 8.5%
            for solved ones, a ratio of 1.04. The earlier documentation described
            failures as "predominantly dense traffic", inherited from a qualitative
            impression and never measured. The real signature is road geometry
            relative to the frame: the largest failure mode is a carriageway too wide
            to fit, where neither observed boundary is a road edge and declining to
            answer is the correct behaviour.
          </Note>
          <Fig src="/figures/failure_gallery.png"
            caption="Worst-performing frames and frames with no solution, against ground truth." />

          <H2 id="limits" n="12">Limitations</H2>
          <ul className="mt-5 space-y-3 text-[15px] leading-relaxed text-body">
            {[
              ['Metric scale rests on an assumed camera height of 1.75 m.',
               'All metric outputs scale linearly with it. One known width at deployment removes the assumption entirely.'],
              ['Lane inference refuses 39 of 204 validation frames.',
               'By design: 20 where the carriageway falls below the 2.5 m floor, 11 where the road edges are not visible, 8 where no vanishing point is recovered.'],
              ['The ground plane is assumed flat.',
               'Crests, dips and banked curves violate the homography, and the error grows with distance.'],
              ['Interior lane divisions on unmarked roads are unvalidated.',
               'The carriageway extent is measured against ground truth; the divisions within it cannot be, because no such ground truth exists.'],
              ['The training data is daytime, normal-lens footage.',
               'Consumer dashcams are wide-angle and often used at dusk. Synthetic vignetting, low-light and barrel distortion are in the augmentation suite, but real night footage is not in the training set.'],
              ['Wrong-side detection is implemented but unvalidated.',
               'It requires motion, so it is video-only, and IDD carries no ground truth for it.'],
              ['Half of the available annotated data is unused.',
               'Part II of IDD-20k was never merged into the training set, so the model sees 6,993 of about 14,000 annotated frames. See §4.'],
              ['No test-split evaluation.',
               "IDD's test split is unlabelled, so all results are on validation. Selecting the best epoch by validation mIoU is a mild optimistic bias."],
            ].map(([h, b], i) => (
              <li key={i} className="border-l-2 border-rule pl-4">
                <strong className="text-strong">{h}</strong> {b}
              </li>
            ))}
          </ul>

          <H2 id="refs" n="13">References</H2>
          <Refs items={REFERENCES} />

          <div className="mt-20 border-t-2 border-strong pt-6">
            <p className="prose-serif max-w-column text-[13.5px] leading-[1.6] text-muted">
              Every figure and table in this report is produced by the evaluation
              harness in the project's{' '}
              <code className="bg-sunken px-1 font-mono text-[12.5px] text-body">eval/</code>{' '}
              directory and can be regenerated from source. No number here is
              estimated,  up, or carried over from an earlier phase.
            </p>
            <div className="no-print mt-8 flex flex-wrap gap-3">
              <Link to="/demo" className="btn-primary"><Radar size={16} />Run the demo</Link>
              <Link to="/" className="btn-ghost"><ArrowLeft size={15} />Index</Link>
            </div>
          </div>
        </article>
      </div>
    </div>
   </Numbering>
  )
}
