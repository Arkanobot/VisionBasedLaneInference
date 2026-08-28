import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { Activity, BarChart3, BookOpen, Camera, Film, GitCompare, Radar, Settings2, Workflow } from 'lucide-react'
import Analyse from './Analyse'
import Live from './Live'
import Compare from './Compare'
import Pipeline from './Pipeline'
import Results from './Results'
import VideoTab from './VideoTab'
import { Card, Chip, Section } from '../components/ui'
import { api } from '../lib/api'

const TABS = [
  { id: 'analyse',  label: 'Analyse',  icon: Radar,       el: Analyse },
  { id: 'pipeline', label: 'How it works', icon: Workflow, el: Pipeline },
  { id: 'live',     label: 'Live camera', icon: Camera,   el: Live },
  { id: 'video',    label: 'Video',    icon: Film,        el: VideoTab },
  { id: 'compare',  label: 'Phase 3 vs 4', icon: GitCompare, el: Compare },
  { id: 'results',  label: 'Results',  icon: BarChart3,   el: Results },
]

export default function Demo() {
  const [tab, setTab] = useState('analyse')
  const [health, setHealth] = useState(null)
  const [settings, setSettings] = useState(false)
  const [cameraHeight, setCameraHeight] = useState(1.75)
  const [maxRange, setMaxRange] = useState(22)

  useEffect(() => { api.health().then(setHealth).catch(() => setHealth({ status: 'down' })) }, [])

  const Active = TABS.find(t => t.id === tab).el
  const live = health?.status === 'ok'

  return (
    <div className="min-h-screen pb-20">
      <header className="sticky top-0 z-40 border-b border-rule bg-page/90 backdrop-blur">
        <Section className="flex h-16 items-center gap-4">
          <Link to="/" className="flex min-w-0 items-center gap-3 transition hover:opacity-80">
            <div className="grid h-9 w-9 shrink-0 place-items-center border border-strong">
              <Radar size={17} className="text-strong" />
            </div>
            <div className="min-w-0">
              <h1 className="display truncate text-[20px] leading-none text-strong">
                Lane Inference
              </h1>
              <p className="label mt-1 truncate !text-[9.5px]">
                Unstructured &amp; structured Indian roads
              </p>
            </div>
          </Link>

          <div className="ml-auto flex items-center gap-2">
            <Chip tone={live ? 'good' : 'bad'} className="hidden sm:inline-flex">
              <span className={`h-1.5 w-1.5 rounded-full ${live ? 'animate-pulse-dot bg-lane' : 'bg-road'}`} />
              {live ? `${health.device} · ${health.model?.run ?? 'classical'}` : 'backend down'}
            </Chip>
            <Link to="/project" className="btn-ghost !min-h-0 !px-2.5 !py-2 hidden sm:inline-flex"
                  title="Technical report"><BookOpen size={16}/></Link>
            <button onClick={() => setSettings(s => !s)}
              aria-label="Settings"
              className={`btn-ghost !min-h-0 !px-2.5 !py-2 ${settings ? 'border-accent text-accent' : ''}`}>
              <Settings2 size={16}/>
            </button>
          </div>
        </Section>

        <AnimatePresence>
          {settings && (
            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }} transition={{ duration: .22 }}
              className="overflow-hidden border-t border-rule bg-surface">
              <Section className="grid gap-5 py-4 sm:grid-cols-2">
                <label className="block">
                  <div className="mb-1 flex items-baseline justify-between">
                    <span className="stat-label">Camera height</span>
                    <span className="font-mono text-sm text-body">{cameraHeight.toFixed(2)} m</span>
                  </div>
                  <input type="range" min="1.0" max="2.5" step="0.05" value={cameraHeight}
                    onChange={e => setCameraHeight(+e.target.value)}
                    className="w-full accent-accent" />
                  <p className="mt-1 text-[11px] text-muted">
                    The pipeline's only metric free parameter. Every distance scales
                    linearly with it; lateral scale is independent of focal length.
                  </p>
                </label>
                <label className="block">
                  <div className="mb-1 flex items-baseline justify-between">
                    <span className="stat-label">Overlay reach</span>
                    <span className="font-mono text-sm text-body">{maxRange} m</span>
                  </div>
                  <input type="range" min="10" max="30" step="1" value={maxRange}
                    onChange={e => setMaxRange(+e.target.value)}
                    className="w-full accent-accent" />
                  <p className="mt-1 text-[11px] text-muted">
                    How far ahead lane boundaries are drawn. Beyond ~22 m the
                    rectification is driven by very few pixels.
                  </p>
                </label>
              </Section>
            </motion.div>
          )}
        </AnimatePresence>

        <Section className="-mb-px flex gap-6 overflow-x-auto">
          {TABS.map(t => {
            const Icon = t.icon
            const on = tab === t.id
            return (
              <button key={t.id} onClick={() => setTab(t.id)}
                className={`relative flex shrink-0 items-center gap-2 pb-2.5 pt-1 transition-colors
                            ${on ? 'text-strong' : 'text-muted hover:text-body'}`}>
                <Icon size={14} />
                <span className="label !text-[10.5px] !text-inherit">{t.label}</span>
                {on && <motion.span layoutId="tabrule"
                        className="absolute inset-x-0 -bottom-px h-0.5 bg-accent"
                        transition={{ type: 'spring', stiffness: 420, damping: 34 }} />}
              </button>
            )
          })}
        </Section>
      </header>

      <main className="pt-6">
        <Section>
          <AnimatePresence mode="wait">
            <motion.div key={tab}
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }} transition={{ duration: .2 }}>
              <Active cameraHeight={cameraHeight} maxRange={maxRange} />
            </motion.div>
          </AnimatePresence>
        </Section>
      </main>

      <footer className="mt-16 border-t border-rule py-6">
        <Section className="flex flex-wrap items-center gap-x-6 gap-y-2 text-[11.5px] text-muted">
          <span className="flex items-center gap-1.5">
            <Activity size={12}/>BITS Pilani · BSc Computer Science
          </span>
          <span>Shreyas Bhat K · Harshwardhan Mukund Mohadikar</span>
          <span className="sm:ml-auto">
            Trained on the Indian Driving Dataset · IIIT Hyderabad
          </span>
        </Section>
      </footer>
    </div>
  )
}
