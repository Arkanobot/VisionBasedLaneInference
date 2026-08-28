import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronLeft, ChevronRight, Workflow } from 'lucide-react'
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

export default function Pipeline({ cameraHeight, maxRange }) {
  const [busy, setBusy] = useState(false)
  const [data, setData] = useState(null)
  const [i, setI] = useState(0)
  const [err, setErr] = useState('')

  const run = async (file) => {
    setBusy(true); setErr(''); setData(null); setI(0)
    try { setData(await api.stages(file, { cameraHeight, maxRange })) }
    catch (e) { setErr(e.message) }
    finally { setBusy(false) }
  }

  const stages = data?.stages ?? []
  const cur = stages[i]

  return (
    <div className="grid gap-5 lg:grid-cols-[22rem,minmax(0,1fr)]">
      <div className="space-y-4">
        <Dropzone onFile={run} busy={busy} />
        {err && (
          <div className="border-l-2 border-road bg-road/[.12] px-4 py-3 text-[13px] text-road">
            {err}
          </div>
        )}
        <Panel title="Why show the stages">
          <p className="p-4 text-[12.5px] leading-[1.65] text-muted">
            A system that turns a photograph into a number is hard to trust and
            harder to debug. Stepping through each stage shows where an answer
            comes from, and, when it is wrong, which stage is responsible.
          </p>
        </Panel>

        {stages.length > 0 && (
          <Panel title="Stages" aside={`${i + 1} / ${stages.length}`}>
            <ol className="divide-y divide-rule">
              {stages.map((s, k) => (
                <li key={s.key}>
                  <button onClick={() => setI(k)}
                    className={`flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors
                      ${k === i ? 'bg-sunken' : 'hover:bg-sunken/60'}`}>
                    <span className={`font-mono text-[11px] ${k === i ? 'text-accent' : 'text-faint'}`}>
                      {String(k + 1).padStart(2, '0')}
                    </span>
                    <span className={`text-[13px] ${k === i ? 'text-strong' : 'text-muted'}`}>
                      {s.title}
                    </span>
                    {k === i && <span className="ml-auto h-1.5 w-1.5 bg-accent" />}
                  </button>
                </li>
              ))}
            </ol>
          </Panel>
        )}
      </div>

      <div className="min-w-0">
        {!stages.length && !busy && (
          <Panel className="relative overflow-hidden">
            <div className="pointer-events-none absolute inset-0 grid-bg opacity-40" />
            <div className="relative grid place-items-center px-6 py-20 text-center">
              <Workflow className="mb-5 text-rule2" size={32} strokeWidth={1.2} />
              <h3 className="display text-[27px] text-body">Walk through the pipeline</h3>
              <p className="prose-serif mt-2.5 max-w-md text-[15px] leading-[1.6] text-muted">
                Choose an image and every intermediate result is kept, from the raw
                frame to the final metric partition.
              </p>
            </div>
          </Panel>
        )}
        {busy && <Panel title="Processing" aside="running"><ScanFrame busy /></Panel>}

        {cur && (
          <motion.div {...fadeUp}>
            <Panel>
              <header className="flex items-center gap-3 border-b border-rule px-4 py-2.5">
                <span className="font-mono text-[12px] text-accent">
                  {String(i + 1).padStart(2, '0')} / {String(stages.length).padStart(2, '0')}
                </span>
                <h3 className="display text-[17px] text-strong">{cur.title}</h3>
                <div className="ml-auto flex gap-1">
                  <button onClick={() => setI(Math.max(0, i - 1))} disabled={i === 0}
                    aria-label="Previous stage"
                    className="btn-ghost !min-h-0 !px-2 !py-1.5"><ChevronLeft size={15} /></button>
                  <button onClick={() => setI(Math.min(stages.length - 1, i + 1))}
                    disabled={i === stages.length - 1} aria-label="Next stage"
                    className="btn-ghost !min-h-0 !px-2 !py-1.5"><ChevronRight size={15} /></button>
                </div>
              </header>

              <div className="flex h-0.5 gap-px bg-rule">
                {stages.map((s, k) => (
                  <span key={s.key}
                    className={`flex-1 ${k <= i ? 'bg-accent' : 'bg-transparent'}`} />
                ))}
              </div>

              <AnimatePresence mode="wait">
                <motion.img key={cur.key} src={cur.image} alt={cur.title}
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  transition={{ duration: .25 }} className="w-full bg-page" />
              </AnimatePresence>

              <p className="border-t border-rule px-4 py-4 text-[13.5px] leading-[1.65] text-body">
                {cur.note}
              </p>
            </Panel>
          </motion.div>
        )}
      </div>
    </div>
  )
}
