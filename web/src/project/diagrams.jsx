import { motion } from 'framer-motion'
import { Plate } from './parts'

/* Shared ink. Colours come from the project's own segmentation palette so the
   diagrams and the model output speak the same language. */
const C = {
  line: '#4a545e', dim: '#6d7784', text: '#0e1216', faint: '#5a646e',
  road: '#c0342a', lane: '#2f7d4f', sky: '#2f6dbd', warn: '#9a7106',
  panel: '#f1f3f5', panelEdge: '#b6bdc4',
}
const FS = 'IBM Plex Sans, system-ui, sans-serif'
const FM = 'IBM Plex Mono, ui-monospace, monospace'

const draw = (i = 0) => ({
  initial: { pathLength: 0, opacity: 0 },
  whileInView: { pathLength: 1, opacity: 1 },
  viewport: { once: true, margin: '-80px' },
  transition: { duration: .8, delay: i * .07, ease: 'easeInOut' },
})
const fade = (i = 0) => ({
  initial: { opacity: 0 },
  whileInView: { opacity: 1 },
  viewport: { once: true, margin: '-80px' },
  transition: { duration: .45, delay: .25 + i * .05 },
})

function Box({ x, y, w, h, label, sub, tone = 'default' }) {
  const stroke = { default: C.panelEdge, road: C.road, lane: C.lane, sky: C.sky }[tone]
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} fill={C.panel} stroke={stroke} strokeWidth="1" />
      <text x={x + w / 2} y={y + (sub ? h / 2 - 4 : h / 2 + 4)} textAnchor="middle"
            fill={C.text} fontSize="11.5" fontFamily={FS} fontWeight="500">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 11} textAnchor="middle"
            fill={C.faint} fontSize="9.5" fontFamily={FM}>{sub}</text>}
    </g>
  )
}
function Arrow({ from, to, label, dashed = false, i = 0 }) {
  const [x1, y1] = from, [x2, y2] = to
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2
  return (
    <g>
      <motion.line {...draw(i)} x1={x1} y1={y1} x2={x2} y2={y2}
        stroke={C.line} strokeWidth="1.4" markerEnd="url(#ah)"
        strokeDasharray={dashed ? '4 3' : undefined} />
      {label && (
        <motion.text {...fade(i)} x={mx} y={my - 5} textAnchor="middle"
          fill={C.dim} fontSize="9" fontFamily={FM}>{label}</motion.text>
      )}
    </g>
  )
}
const Defs = () => (
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill={C.line} />
    </marker>
    <marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill={C.road} />
    </marker>
  </defs>
)
/* Diagrams are plates: hairlined, on drawing-grid paper, and figure-numbered
   from the same counter as the images so the sequence is continuous. */
const Frame = ({ children, caption, vb }) => (
  <Plate caption={caption}>
    <svg viewBox={vb} className="w-full" style={{ minWidth: 560 }}>{children}</svg>
  </Plate>
)

/* ── 1. System architecture ─────────────────────────────────────────────── */
export function SystemArchitecture() {
  return (
    <Frame vb="0 0 900 400" caption={
      <>End-to-end architecture. The classical branch is retained only as a gated
      fallback, measured, blending it into the learned branch costs accuracy in
      both domains tested (§08). Dashed paths are inactive by default.</>}>
      <Defs />
      <text x="12" y="18" fill={C.faint} fontSize="10" fontFamily={FS} letterSpacing="1.6">PERCEPTION</text>
      <text x="330" y="18" fill={C.faint} fontSize="10" fontFamily={FS} letterSpacing="1.6">GEOMETRY</text>
      <text x="640" y="18" fill={C.faint} fontSize="10" fontFamily={FS} letterSpacing="1.6">INFERENCE</text>
      <line x1="318" y1="26" x2="318" y2="380" stroke={C.panelEdge} strokeDasharray="3 5" />
      <line x1="628" y1="26" x2="628" y2="380" stroke={C.panelEdge} strokeDasharray="3 5" />

      <Box x={12} y={150} w={92} h={44} label="frame" sub="RGB" />
      <Box x={140} y={92} w={150} h={48} label="segmentation" sub="MobileNetV3 · FPN" tone="road" />
      <Box x={140} y={160} w={150} h={44} label="lane-line head" sub="distilled" tone="lane" />
      <Box x={140} y={228} w={150} h={44} label="classical CIELAB" sub="fallback only" />

      <Arrow from={[104, 165]} to={[138, 116]} i={0} />
      <Arrow from={[104, 172]} to={[138, 180]} i={1} />
      <Arrow from={[104, 180]} to={[138, 246]} dashed i={2} />

      <Box x={340} y={92} w={140} h={44} label="road mask" sub="7 classes" tone="road" />
      <Box x={340} y={160} w={140} h={48} label="vanishing point" sub="robust edge fit" />
      <Box x={340} y={230} w={140} h={44} label="camera pitch" sub="θ = atan(Δv/f)" />
      <Box x={505} y={155} w={105} h={58} label="rectify" sub="metric BEV" tone="sky" />

      <Arrow from={[290, 116]} to={[338, 114]} i={3} />
      <Arrow from={[290, 250]} to={[338, 250]} dashed i={4} />
      <Arrow from={[410, 136]} to={[410, 158]} i={5} />
      <Arrow from={[410, 208]} to={[410, 228]} i={6} />
      <Arrow from={[480, 252]} to={[556, 215]} i={7} label="H, θ" />
      <Arrow from={[480, 114]} to={[556, 154]} i={8} />

      <Box x={650} y={70} w={150} h={46} label="carriageway width" sub="metres" tone="sky" />
      <Box x={650} y={136} w={150} h={46} label="lane partition" sub="÷ IRC width" tone="lane" />
      <Box x={650} y={202} w={150} h={46} label="Kalman + vote" sub="video only" />
      <Box x={650} y={268} w={150} h={46} label="road users" sub="lane · distance" />
      <Box x={828} y={136} w={60} h={46} label="output" tone="lane" />

      <Arrow from={[610, 178]} to={[648, 96]} i={9} />
      <Arrow from={[725, 116]} to={[725, 134]} i={10} />
      <Arrow from={[725, 182]} to={[725, 200]} i={11} />
      <Arrow from={[610, 190]} to={[648, 290]} i={12} />
      <Arrow from={[800, 160]} to={[826, 160]} i={13} />
      <motion.path {...draw(14)} d="M725 248 L725 330 L560 330 L560 214" fill="none"
        stroke={C.road} strokeWidth="1.2" strokeDasharray="4 3" markerEnd="url(#ahr)" />
      <motion.text {...fade(14)} x="640" y="344" textAnchor="middle" fill={C.road}
        fontSize="9" fontFamily={FM}>state carried to next frame</motion.text>
    </Frame>
  )
}

/* ── 2. The geometry that makes it metric ───────────────────────────────── */
export function GeometryDiagram() {
  return (
    <Frame vb="0 0 900 340" caption={
      <>Side elevation. The camera sits at height <em>H</em> with pitch <em>θ</em>.
      Rays through the image plane meet the ground at distances that compress with
      range, which is why image-space reasoning fails. The horizon row is the
      image of infinitely distant ground, so measuring it recovers θ directly.</>}>
      <Defs />
      <motion.line {...draw(0)} x1="60" y1="270" x2="860" y2="270" stroke={C.line} strokeWidth="1.6" />
      <text x="800" y="288" fill={C.faint} fontSize="10" fontFamily={FS}>ground plane</text>
      <motion.line {...draw(1)} x1="110" y1="270" x2="110" y2="150" stroke={C.dim} strokeDasharray="3 3" />
      <rect x="96" y="132" width="30" height="20" rx="3" fill={C.panel} stroke={C.road} />
      <text x="111" y="146" textAnchor="middle" fill={C.text} fontSize="9" fontFamily={FS}>cam</text>
      <motion.text {...fade(1)} x="86" y="215" textAnchor="end" fill={C.text}
        fontSize="12" fontFamily={FM}>H</motion.text>
      <motion.line {...draw(2)} x1="110" y1="152" x2="110" y2="268" stroke={C.road} strokeWidth="1.2" />
      <motion.line {...draw(3)} x1="126" y1="142" x2="860" y2="142" stroke={C.sky}
        strokeWidth="1.2" strokeDasharray="6 4" />
      <text x="770" y="134" fill={C.sky} fontSize="10" fontFamily={FS}>horizon · v_horizon</text>
      {[[250, '5 m'], [400, '10 m'], [560, '20 m'], [740, '35 m']].map(([x, l], i) => (
        <g key={x}>
          <motion.line {...draw(4 + i)} x1="126" y1="142" x2={x} y2="270"
            stroke={C.lane} strokeWidth="1" opacity=".75" />
          <circle cx={x} cy="270" r="3" fill={C.lane} />
          <motion.text {...fade(4 + i)} x={x} y="288" textAnchor="middle"
            fill={C.faint} fontSize="9.5" fontFamily={FM}>{l}</motion.text>
        </g>
      ))}
      <motion.line {...draw(8)} x1="180" y1="96" x2="180" y2="215" stroke={C.warn} strokeWidth="1.6" />
      <text x="188" y="92" fill={C.warn} fontSize="10" fontFamily={FS}>image plane</text>
      <motion.path {...draw(9)} d="M150 142 A 40 40 0 0 1 156 156" fill="none" stroke={C.road} strokeWidth="1.2" />
      <text x="163" y="160" fill={C.road} fontSize="11" fontFamily={FM}>θ</text>
      <motion.g {...fade(10)}>
        <rect x="470" y="176" width="390" height="66" rx="5" fill={C.panel} stroke={C.panelEdge} />
        <text x="665" y="200" textAnchor="middle" fill={C.text} fontSize="14" fontFamily={FM}>
          dX = du · H / (v − v_horizon)
        </text>
        <text x="665" y="221" textAnchor="middle" fill={C.faint} fontSize="10" fontFamily={FS}>
          lateral scale, the focal length cancels
        </text>
        <text x="665" y="235" textAnchor="middle" fill={C.lane} fontSize="9.5" fontFamily={FM}>
          1% variation across 40 to 140° assumed FOV
        </text>
      </motion.g>
    </Frame>
  )
}

/* ── 3. Lane partitioning ───────────────────────────────────────────────── */
export function PartitionDiagram() {
  const lanes = 3, W = 10.5, x0 = 150, wpx = 520
  return (
    <Frame vb="0 0 900 250" caption={
      <>Partitioning in the rectified view. The carriageway is measured in metres,
      divided by the IRC design lane width, and  to an integer, which is
      what makes the output a lane <em>count</em>. The count is then bounded so no
      lane falls below the minimum usable width.</>}>
      <Defs />
      <text x="20" y="60" fill={C.faint} fontSize="10" fontFamily={FS}>measured</text>
      <motion.rect {...fade(0)} x={x0} y="42" width={wpx} height="44" rx="3"
        fill="rgba(220,60,60,.13)" stroke={C.road} />
      <motion.line {...draw(0)} x1={x0} y1="104" x2={x0 + wpx} y2="104"
        stroke={C.text} strokeWidth="1" markerEnd="url(#ah)" />
      <motion.text {...fade(1)} x={x0 + wpx / 2} y="122" textAnchor="middle"
        fill={C.text} fontSize="12" fontFamily={FM}>W = {W} m</motion.text>

      <motion.g {...fade(2)}>
        <text x={x0 + wpx / 2} y="152" textAnchor="middle" fill={C.faint}
          fontSize="12" fontFamily={FM}>round( 10.5 / 3.5 ) = 3</text>
      </motion.g>

      <text x="20" y="205" fill={C.faint} fontSize="10" fontFamily={FS}>inferred</text>
      {[...Array(lanes)].map((_, i) => (
        <motion.g key={i} {...fade(3 + i)}>
          <rect x={x0 + (i * wpx) / lanes + 2} y="184" width={wpx / lanes - 4} height="44" rx="3"
            fill={i === 1 ? 'rgba(90,222,138,.16)' : 'rgba(168,179,196,.06)'}
            stroke={i === 1 ? C.lane : C.panelEdge} />
          <text x={x0 + (i * wpx) / lanes + wpx / lanes / 2} y="211" textAnchor="middle"
            fill={i === 1 ? C.lane : C.faint} fontSize="10.5" fontFamily={FM}>3.50 m</text>
        </motion.g>
      ))}
      <motion.text {...fade(6)} x={x0 + wpx + 14} y="211" fill={C.lane}
        fontSize="9.5" fontFamily={FS}>← ego</motion.text>
    </Frame>
  )
}

/* ── 4. End-to-end process flow, with the decisions ─────────────────────── */
export function ProcessFlow() {
  const rows = [
    ['Frame arrives', 'any forward-facing view', null],
    ['Scene plausible?', 'classes present · road above horizon', 'no → decline, state why'],
    ['Segment', '7 classes + lane lines, 13 ms', null],
    ['Learned branch collapsed?', 'coverage below floor', 'yes → classical fallback'],
    ['Fit road edges', 'border-clipped points excluded', 'too few → decline'],
    ['Rectify', 'pitch from vanishing point', null],
    ['Measure carriageway', 'occlusions bridged', 'implausible width → decline'],
    ['Partition', '÷ IRC lane width, bounded', null],
    ['Smooth', 'Kalman + count vote (video)', null],
    ['Report', 'geometry, users, calibrated confidence', null],
  ]
  return (
    <Frame vb={`0 0 900 ${rows.length * 46 + 30}`} caption={
      <>The decision path for one frame. Four stages can decline to answer, and
      each says which. Refusing where the carriageway is not visible is correct
      behaviour, a wrong lane model is worse than none.</>}>
      <Defs />
      {rows.map(([t, s, bail], i) => {
        const y = 18 + i * 46
        return (
          <g key={i}>
            <motion.g {...fade(i)}>
              <rect x="150" y={y} width="330" height="34" fill={C.panel}
                stroke={bail ? C.warn : C.panelEdge} />
              <text x="164" y={y + 15} fill={C.text} fontSize="11.5"
                fontFamily={FS} fontWeight="500">{t}</text>
              <text x="164" y={y + 27} fill={C.faint} fontSize="9.5" fontFamily={FM}>{s}</text>
              <text x="138" y={y + 22} textAnchor="end" fill={C.dim}
                fontSize="10" fontFamily={FM}>{String(i + 1).padStart(2, '0')}</text>
            </motion.g>
            {i < rows.length - 1 && (
              <motion.line {...draw(i)} x1="315" y1={y + 34} x2="315" y2={y + 46}
                stroke={C.line} strokeWidth="1.2" markerEnd="url(#ah)" />
            )}
            {bail && (
              <motion.g {...fade(i)}>
                <line x1="480" y1={y + 17} x2="520" y2={y + 17} stroke={C.warn}
                  strokeWidth="1.2" strokeDasharray="3 3" markerEnd="url(#ah)" />
                <text x="528" y={y + 21} fill={C.warn} fontSize="10" fontFamily={FS}>{bail}</text>
              </motion.g>
            )}
          </g>
        )
      })}
    </Frame>
  )
}

/* ── 5. Training pipeline ───────────────────────────────────────────────── */
export function TrainingDiagram() {
  return (
    <Frame vb="0 0 900 300" caption={
      <>Training. Lane-line supervision comes from distillation because IDD has no
      marking annotation; the teacher runs once, offline. The augmentation suite
      is split between failure modes observed in the data and conditions the
      dataset does not contain at all.</>}>
      <Defs />
      <Box x={20} y={40} w={130} h={46} label="IDD Segmentation" sub="6,993 train" tone="road" />
      <Box x={20} y={112} w={130} h={46} label="polygon JSON" sub="rasterised" />
      <Box x={20} y={190} w={130} h={46} label="YOLOP teacher" sub="offline, once" tone="sky" />

      <Box x={210} y={40} w={150} h={46} label="observed failures" sub="shadow · glare · blur" />
      <Box x={210} y={112} w={150} h={46} label="absent conditions" sub="dusk · vignette · barrel" tone="warn" />
      <Box x={210} y={190} w={150} h={46} label="lane pseudo-labels" sub="7,974 · 94% positive" tone="sky" />

      <Arrow from={[150, 63]} to={[208, 63]} i={0} />
      <Arrow from={[150, 135]} to={[208, 135]} i={1} />
      <Arrow from={[150, 213]} to={[208, 213]} i={2} />

      <Box x={425} y={95} w={130} h={70} label="augment" sub="per sample" />
      <Arrow from={[360, 63]} to={[423, 112]} i={3} />
      <Arrow from={[360, 135]} to={[423, 130]} i={4} />

      <Box x={610} y={60} w={140} h={50} label="MobileNetV3" sub="LR-ASPP + FPN" tone="road" />
      <Box x={610} y={140} w={140} h={44} label="semantic head" sub="7 classes" />
      <Box x={610} y={200} w={140} h={44} label="lane head" sub="binary" tone="sky" />
      <Arrow from={[555, 130]} to={[608, 90]} i={5} />
      <Arrow from={[680, 110]} to={[680, 138]} i={6} />
      <Arrow from={[680, 110]} to={[680, 198]} dashed i={7} />
      <Arrow from={[360, 213]} to={[608, 222]} dashed i={8} label="distillation target" />

      <Box x={790} y={140} w={95} h={44} label="CE + Dice" tone="lane" />
      <Box x={790} y={200} w={95} h={44} label="BCE + Dice" tone="lane" />
      <Arrow from={[750, 162]} to={[788, 162]} i={9} />
      <Arrow from={[750, 222]} to={[788, 222]} i={10} />
    </Frame>
  )
}
