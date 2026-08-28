import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowUpRight, BookOpen, Radar } from 'lucide-react'

const CARDS = [
  { to: '/project', icon: BookOpen, title: 'Project',
    body: 'The full technical report, method, geometry, datasets, training, every ablation, the experiments that failed, and the limitations.',
    meta: '14 sections · 8 figures · 8 tables' },
  { to: '/demo', icon: Radar, title: 'Demo',
    body: 'Run the system on a photograph, a video clip, or your live camera. Step through every stage of the pipeline on a frame you choose.',
    meta: 'Upload · Video · Live camera · Stages' },
]

const rise = (d = 0) => ({
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: .6, delay: d, ease: [.16, 1, .3, 1] },
})

export default function Landing() {
  return (
    <div className="relative flex min-h-screen flex-col justify-center overflow-hidden px-5 py-16 sm:px-8">
      <div aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{ background:
          'radial-gradient(60rem 40rem at 22% -10%, rgb(var(--rule2) / .55), transparent 65%)' }} />

      <main className="relative mx-auto w-full max-w-4xl">
        <motion.p {...rise(0)} className="label">
          BITS Pilani · BSc Computer Science
        </motion.p>

        <motion.h1 {...rise(.06)}
          className="display mt-5 text-[clamp(2.1rem,6vw,3.4rem)] text-strong">
          Vision-Based Lane Inference
        </motion.h1>

        <motion.p {...rise(.12)}
          className="prose-serif mt-4 max-w-column text-[17px] leading-[1.6] text-muted">
          Lane structure inferred, not detected, on unstructured and structured
          Indian roads, where the paint that lane detectors look for is often absent.
        </motion.p>

        <motion.div {...rise(.16)}
          className="mt-9 flex flex-wrap items-end gap-x-10 gap-y-5 border-y border-rule py-6">
          <div>
            <div className="label mb-2.5">Authors</div>
            <p className="prose-serif text-[19px] leading-[1.35] text-strong sm:text-[21px]">
              Harshwardhan Mukund Mohadikar
              <span className="ml-2 font-mono text-[12px] text-faint">2023EBCS353</span>
              <br />
              Shreyas Bhat K
              <span className="ml-2 font-mono text-[12px] text-faint">2023EBCS460</span>
            </p>
          </div>
          <div>
            <div className="label mb-2.5">Supervisor</div>
            <p className="prose-serif text-[16px] leading-[1.35] text-body">
              Dr. Ashok Yemineni
            </p>
          </div>
        </motion.div>

        <div className="mt-11 grid gap-4 sm:grid-cols-2">
          {CARDS.map(({ to, icon: Icon, title, body, meta }, i) => (
            <motion.div key={to} {...rise(.24 + i * .07)}>
              <Link to={to}
                className="group relative flex h-full flex-col overflow-hidden border border-rule2
                           bg-surface/50 p-7 backdrop-blur-xl transition-all duration-300
                           hover:-translate-y-0.5 hover:border-muted hover:bg-surface/80">
                <span aria-hidden="true"
                  className="pointer-events-none absolute inset-x-0 top-0 h-px
                             bg-gradient-to-r from-transparent via-strong/25 to-transparent" />
                <div className="flex items-start justify-between">
                  <Icon size={24} className="text-muted transition-colors group-hover:text-strong" />
                  <ArrowUpRight size={19}
                    className="text-faint transition-all duration-300
                               group-hover:-translate-y-0.5 group-hover:translate-x-0.5
                               group-hover:text-strong" />
                </div>
                <h2 className="display mt-7 text-[27px] text-strong">{title}</h2>
                <p className="prose-serif mt-2.5 text-[15px] leading-[1.6] text-muted">{body}</p>
                <p className="label mt-6 !text-[9.5px]">{meta}</p>
              </Link>
            </motion.div>
          ))}
        </div>

        <motion.p {...rise(.36)}
          className="mt-10 border-t border-rule pt-5 text-[12.5px] leading-relaxed text-faint">
          Trained on the Indian Driving Dataset · IIIT Hyderabad.
          0.9403 drivable IoU · 0.8343 boundary F1 · 31 fps end to end.
        </motion.p>
      </main>
    </div>
  )
}
