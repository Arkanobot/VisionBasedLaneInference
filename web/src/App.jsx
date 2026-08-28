import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { useLayoutEffect } from 'react'
import Landing from './sections/Landing'
import Project from './sections/Project'
import Demo from './sections/Demo'

/**
 * The report is a document and the demo is an instrument, so each route owns
 * its ground. Setting the class on <html> (rather than a wrapper) means the
 * body background, scrollbars and overscroll area all follow the route.
 */
const GROUND = { '/project': 'theme-paper' }

function Ground() {
  const { pathname, hash } = useLocation()
  useLayoutEffect(() => {
    const cls = GROUND[pathname] ?? 'theme-instrument'
    const root = document.documentElement
    root.classList.remove('theme-paper', 'theme-instrument')
    root.classList.add(cls)
    const meta = document.querySelector('meta[name="theme-color"]')
    if (meta) meta.setAttribute('content', cls === 'theme-paper' ? '#f6f7f8' : '#0b0d0f')
  }, [pathname])

  useLayoutEffect(() => { if (!hash) window.scrollTo(0, 0) }, [pathname, hash])
  return null
}

export default function App() {
  return (
    <BrowserRouter>
      <Ground />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/project" element={<Project />} />
        <Route path="/demo" element={<Demo />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
