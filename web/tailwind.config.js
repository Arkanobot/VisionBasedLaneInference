/** @type {import('tailwindcss').Config} */

// Semantic colours resolve through CSS variables so one component set renders
// correctly on either ground: `.theme-paper` for the report, `.theme-instrument`
// for the demo. See index.css for the variable definitions.
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`

export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        page:    v('page'),      // page ground
        surface: v('surface'),   // figure / table ground
        sunken:  v('sunken'),    // fills, code, wells
        rule:    v('rule'),      // hairlines
        rule2:   v('rule2'),     // stronger rules
        strong:  v('strong'),    // headings
        body:    v('body'),      // running text
        muted:   v('muted'),     // secondary
        faint:   v('faint'),     // captions, labels
        accent:  v('accent'),    // the one signal colour
        accentQ: v('accent-quiet'),

        // The project's own segmentation palette. Data only -- never chrome.
        road:   { DEFAULT:'#dc3c3c', dark:'#a82a2a' },
        lane:   { DEFAULT:'#2f9e5c', dark:'#227944' },
        signal: { DEFAULT:'#c8860c' },
        sky:    { DEFAULT:'#3d7fd0' },
      },
      fontFamily: {
        // Source Serif 4    long-form prose. A journal text face, drawn for
        //                   reading at length on screen.
        // IBM Plex Sans     interface. Institutional, engineered, not a startup
        //                   grotesque.
        // ...Condensed      signage-style headings and eyebrows.
        // IBM Plex Mono     every number, tabular by default.
        serif: ['"Source Serif 4"','Charter','Georgia','serif'],
        sans:  ['"IBM Plex Sans"','system-ui','-apple-system','sans-serif'],
        cond:  ['"IBM Plex Sans Condensed"','"IBM Plex Sans"','system-ui','sans-serif'],
        mono:  ['"IBM Plex Mono"','ui-monospace','SFMono-Regular','Menlo','monospace'],
      },
      maxWidth: { measure: '34rem', column: '43rem', plate: '86rem' },
      animation: {
        'fade-up':  'fadeUp .55s cubic-bezier(.16,1,.3,1) both',
        'scan':     'scan 2.4s cubic-bezier(.4,0,.2,1) infinite',
        'pulse-dot':'pulseDot 2s ease-in-out infinite',
        'shimmer':  'shimmer 1.8s linear infinite',
        'draw':     'draw 1.4s cubic-bezier(.4,0,.2,1) both',
      },
      keyframes: {
        fadeUp:  { '0%':{opacity:0,transform:'translateY(10px)'}, '100%':{opacity:1,transform:'none'} },
        scan:    { '0%':{top:'0%',opacity:0}, '10%':{opacity:1}, '90%':{opacity:1}, '100%':{top:'100%',opacity:0} },
        pulseDot:{ '0%,100%':{opacity:1}, '50%':{opacity:.35} },
        shimmer: { '0%':{backgroundPosition:'-200% 0'}, '100%':{backgroundPosition:'200% 0'} },
        draw:    { '0%':{strokeDashoffset:'var(--len,1000)'}, '100%':{strokeDashoffset:'0'} },
      },
    },
  },
  plugins: [],
}
