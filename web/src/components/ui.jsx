import { motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'

export function Section({ id, children, className = '' }) {
  return <section id={id} className={`mx-auto w-full max-w-7xl px-4 sm:px-6 ${className}`}>{children}</section>
}

export function Card({ children, className = '', hover = false, ...rest }) {
  return (
    <div className={`card ${hover ? 'card-hover' : ''} ${className}`} {...rest}>
      {children}
    </div>
  )
}

/** Counts up to a value. Purely decorative, so it respects reduced motion. */
export function Counter({ value, decimals = 0, suffix = '', duration = 900 }) {
  const [shown, setShown] = useState(0)
  const raf = useRef()
  useEffect(() => {
    if (value == null || Number.isNaN(value)) return
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduce) { setShown(value); return }
    const t0 = performance.now()
    const tick = (t) => {
      const p = Math.min(1, (t - t0) / duration)
      setShown(value * (1 - Math.pow(1 - p, 3)))
      if (p < 1) raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [value, duration])
  if (value == null || Number.isNaN(value)) return <span className="text-muted">—</span>
  return <span>{shown.toFixed(decimals)}{suffix}</span>
}

export function Stat({ label, value, decimals = 0, suffix = '', hint, tone = 'default' }) {
  const tones = {
    default: 'text-strong',
    good: 'text-lane',
    warn: 'text-signal',
    bad: 'text-road',
  }
  return (
    <div className="min-w-0">
      <div className="stat-label truncate">{label}</div>
      <div className={`stat-value ${tones[tone]}`}>
        {typeof value === 'number'
          ? <Counter value={value} decimals={decimals} suffix={suffix} />
          : (value ?? <span className="text-muted">—</span>)}
      </div>
      {hint && <div className="mt-0.5 text-xs text-muted">{hint}</div>}
    </div>
  )
}

/** Confidence arc. Colour encodes the measured error bands, not taste:
 *  at >=0.7 median boundary error is 0.59 m, below 0.5 it is 0.79 m. */
export function Gauge({ value = 0, size = 108, label = 'confidence' }) {
  const r = (size - 14) / 2
  const c = 2 * Math.PI * r
  const v = Math.max(0, Math.min(1, value))
  const tone = v >= 0.7 ? '#5ade8a' : v >= 0.5 ? '#ffb020' : '#dc3c3c'
  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size/2} cy={size/2} r={r} strokeWidth="7" fill="none" className="stroke-rule" />
        <motion.circle
          cx={size/2} cy={size/2} r={r} strokeWidth="7" fill="none" stroke={tone}
          strokeLinecap="round" strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - v) }}
          transition={{ duration: .9, ease: [.16,1,.3,1] }}
        />
      </svg>
      <div className="absolute px-1 text-center" style={{ width: size - 16 }}>
        <div className="font-mono font-semibold tabular-nums"
             style={{ color: tone, fontSize: size >= 96 ? 20 : 17 }}>
          {v.toFixed(2)}
        </div>
        {size >= 96 && (
          <div className="truncate text-[9.5px] uppercase tracking-[.12em] text-muted">
            {label}
          </div>
        )}
      </div>
    </div>
  )
}

export function Chip({ children, tone = 'neutral', className = '' }) {
  const tones = {
    neutral: 'border-rule2 bg-sunken text-body',
    good:    'border-lane/40 bg-lane/[.12] text-lane',
    warn:    'border-signal/40 bg-signal/[.12] text-signal',
    bad:     'border-road/40 bg-road/[.12] text-road',
    info:    'border-sky/40 bg-sky/[.12] text-sky',
  }
  return <span className={`chip ${tones[tone]} ${className}`}>{children}</span>
}

export function Spinner({ className = '' }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none" width="18" height="18">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.5" className="opacity-20" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  )
}

/** Image with a scanning line while the backend is thinking. */
export function ScanFrame({ src, busy, alt = '', className = '' }) {
  return (
    <div className={`relative overflow-hidden  bg-surface ${className}`}>
      {src
        ? <img src={src} alt={alt} className="w-full" />
        : <div className="aspect-video w-full skeleton" />}
      {busy && (
        <>
          <div className="absolute inset-0 bg-page/40" />
          <div className="absolute inset-x-0 h-[2px] animate-scan bg-gradient-to-r from-transparent via-road to-transparent shadow-[0_0_14px_2px] shadow-road/50" />
        </>
      )}
    </div>
  )
}

export const fadeUp = {
  initial: { opacity: 0, y: 14 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: .45, ease: [.16, 1, .3, 1] },
}
