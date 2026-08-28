import { renderToString } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import Landing from './src/sections/Landing'
import Project from './src/sections/Project'
import Demo from './src/sections/Demo'

const PAGES = { '/': Landing, '/project': Project, '/demo': Demo }
let bad = 0
for (const [path, C] of Object.entries(PAGES)) {
  try {
    const html = renderToString(
      <MemoryRouter initialEntries={[path]}><C /></MemoryRouter>)
    const text = html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim()
    console.log(`  OK   ${path.padEnd(9)} ${String(html.length).padStart(7)} bytes html, ${String(text.length).padStart(6)} chars text`)
  } catch (e) {
    bad++
    console.log(`  FAIL ${path.padEnd(9)} ${e.constructor.name}: ${e.message}`)
    console.log('       ' + (e.stack || '').split('\n').slice(1, 5).join('\n       '))
  }
}
console.log(bad ? `\n${bad} page(s) crash on render` : '\nall pages render')
