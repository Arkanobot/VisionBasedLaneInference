import { createContext, useContext, useMemo, useRef } from 'react'
import { motion } from 'framer-motion'

/* ───────────────────────────────────────────────────────────────────────────
   Numbering

   A journal numbers its figures, tables and equations, and the body text
   refers to them by number. Hard-coding those numbers means every insertion
   silently renumbers the prose, so they are allocated at render time: each
   element asks the counter for the next number in its own sequence, keyed by
   a stable id so React's double-invoke in development cannot double-count.
   ────────────────────────────────────────────────────────────────────────── */
const Counter = createContext(null)

export function Numbering({ children }) {
  const store = useRef({ fig: new Map(), tab: new Map(), eq: new Map() })
  const api = useMemo(() => ({
    next(kind, key) {
      const m = store.current[kind]
      if (!m.has(key)) m.set(key, m.size + 1)
      return m.get(key)
    },
  }), [])
  return <Counter.Provider value={api}>{children}</Counter.Provider>
}

/**
 * `enabled` is false for an uncaptioned block. Such a block has nowhere to show
 * a number, so allocating one would burn it and leave a visible gap in the
 * sequence -- the reason Table 5 was missing from the report.
 */
function useNumber(kind, key, enabled = true) {
  const ctx = useContext(Counter)
  const idRef = useRef(null)
  if (idRef.current === null) idRef.current = key ?? `${kind}-${Math.random()}`
  return ctx && enabled ? ctx.next(kind, idRef.current) : null
}

/* ── headings ───────────────────────────────────────────────────────────── */
export function H2({ id, n, children }) {
  return (
    <header className="mt-24 first:mt-0">
      <div className="flex items-baseline gap-4">
        <span className="display shrink-0 text-[13px] tracking-[.18em] text-accent">
          §{n}
        </span>
        <span className="h-px flex-1 bg-rule2" />
      </div>
      <h2 id={id}
        className="prose-serif mt-4 max-w-column scroll-mt-28 text-[30px] font-normal leading-[1.18] tracking-[-.01em] text-strong sm:text-[35px]"
        style={{ textWrap: 'balance' }}>
        {children}
      </h2>
    </header>
  )
}

export function H3({ children }) {
  return (
    <h3 className="label mt-12 !text-[11.5px] !text-muted">{children}</h3>
  )
}

/* ── prose ──────────────────────────────────────────────────────────────── */
export function P({ children, lead = false, drop = false }) {
  return (
    <p className={`prose-serif mt-5 max-w-column ${drop ? 'drop' : ''} ${lead
      ? 'text-[20px] leading-[1.58] text-strong'
      : 'text-[17px] leading-[1.68] text-body'}`}>
      {children}
    </p>
  )
}

/**
 * A marginal note. On a wide viewport it sits in the outer margin beside the
 * text it annotates, as in a printed offprint; below that it folds inline
 * behind a rule, because a 40-character column in the margin is unreadable.
 */
export function Margin({ children, label }) {
  return (
    <aside className="my-6 border-l-2 border-accent pl-4 text-[14px] leading-[1.6] text-muted xl:absolute xl:-right-[19rem] xl:my-0 xl:w-[16rem] xl:border-l-0 xl:border-t xl:border-rule2 xl:pl-0 xl:pt-3">
      {label && <div className="label mb-1.5">{label}</div>}
      <div className="prose-serif">{children}</div>
    </aside>
  )
}

export function Note({ children, tone = 'neutral' }) {
  const tones = {
    neutral: 'border-rule2 bg-sunken',
    good:    'border-lane/50 bg-lane/[.06]',
    warn:    'border-signal/50 bg-signal/[.07]',
    bad:     'border-road/50 bg-road/[.06]',
  }
  return (
    <div className={`prose-serif mt-7 max-w-column border-l-2 px-5 py-4 text-[15.5px]
                     leading-[1.62] text-body ${tones[tone]}`}>{children}</div>
  )
}

/* ── numbered blocks ────────────────────────────────────────────────────── */
export function Fig({ src, caption, alt, wide = false, id }) {
  const n = useNumber('fig', id ?? src, Boolean(caption))
  return (
    <figure className={`mt-8 ${wide ? '' : 'max-w-column'}`}>
      <motion.img
        initial={{ opacity: 0, y: 8 }} whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-60px' }} transition={{ duration: .5 }}
        src={src} alt={alt ?? caption}
        className="w-full border border-rule bg-surface" />
      <Caption kind="Fig." n={n}>{caption}</Caption>
    </figure>
  )
}

/** A diagram drawn in the page rather than loaded as an image. */
export function Plate({ children, caption, wide = true, id }) {
  const n = useNumber('fig', id ?? caption, Boolean(caption))
  return (
    <figure className={`mt-8 ${wide ? '' : 'max-w-column'}`}>
      <div className="overflow-x-auto border border-rule bg-surface">
        <div className="grid-bg min-w-[38rem] p-5 sm:p-7">{children}</div>
      </div>
      <Caption kind="Fig." n={n}>{caption}</Caption>
    </figure>
  )
}

export function Eq({ children, where, id }) {
  const n = useNumber('eq', id ?? String(children))
  return (
    <div className="mt-7 max-w-column">
      <div className="flex items-center gap-4 border-y border-rule py-5">
        <div className="flex-1 text-center font-mono text-[15.5px] text-strong">{children}</div>
        {n && <span className="shrink-0 font-mono text-[13px] text-faint">({n})</span>}
      </div>
      {where && (
        <div className="mt-2.5 text-[13px] leading-relaxed text-faint">{where}</div>
      )}
    </div>
  )
}

export function Table({ head, rows, caption, highlight, wide = false, id }) {
  const n = useNumber('tab', id ?? caption, Boolean(caption))
  return (
    <figure className={`mt-8 ${wide ? '' : 'max-w-column'}`}>
      <Caption kind="Table" n={n} above>{caption}</Caption>
      <div className="overflow-x-auto border-y-2 border-strong">
        <table className="w-full border-collapse text-[13.5px]">
          <thead>
            <tr className="border-b border-rule2">
              {head.map((h, i) => (
                <th key={i}
                  className={`label whitespace-nowrap px-3 py-2.5 !text-[10px] !text-muted
                              ${i ? 'text-right' : 'text-left'}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}
                className={`border-b border-rule last:border-0
                            ${highlight === i ? 'bg-accent/[.10]' : ''}`}>
                {r.map((c, j) => (
                  <td key={j}
                    className={`whitespace-nowrap px-3 py-2 ${j ? 'text-right font-mono' : 'prose-serif'}
                      ${highlight === i ? 'font-semibold text-strong' : 'text-body'}`}>{c}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  )
}

function Caption({ kind, n, children, above = false }) {
  if (!children) return null
  return (
    <figcaption className={`max-w-column text-[13px] leading-[1.55] text-muted
                            ${above ? 'mb-2.5' : 'mt-3'}`}>
      {n != null && (
        <span className="label mr-2 !text-[10.5px] !text-strong">{kind}&nbsp;{n}</span>
      )}
      <span className="prose-serif">{children}</span>
    </figcaption>
  )
}

export function Refs({ items }) {
  return (
    <ol className="mt-6 max-w-column space-y-3.5 text-[14px] leading-[1.6] text-body">
      {items.map((r, i) => (
        <li key={i} className="flex gap-3">
          <span className="shrink-0 font-mono text-[12.5px] text-faint">[{i + 1}]</span>
          <span className="prose-serif">
            {r.authors}. <span className="italic text-strong">{r.title}</span>. {r.venue}.
            {r.url && (
              <a href={r.url} target="_blank" rel="noreferrer"
                className="ml-1.5 border-b border-accent/50 text-accent hover:border-accent">
                link
              </a>
            )}
          </span>
        </li>
      ))}
    </ol>
  )
}
