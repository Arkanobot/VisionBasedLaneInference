const base = ''

async function jsonOrThrow(res) {
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch { /* not json */ }
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  health: () => fetch(`${base}/api/health`).then(jsonOrThrow),
  results: () => fetch(`${base}/api/results`).then(jsonOrThrow),
  samples: (kind = 'image', n = 8) =>
    fetch(`${base}/api/samples?kind=${kind}&n=${n}`).then(jsonOrThrow),

  post(path, file, fields = {}) {
    const fd = new FormData()
    fd.append('file', file, file.name || 'upload.jpg')
    Object.entries(fields).forEach(([k, v]) => fd.append(k, String(v)))
    return fetch(`${base}${path}`, { method: 'POST', body: fd }).then(jsonOrThrow)
  },

  infer:   (f, o) => api.post('/api/infer', f, o),
  stages:  (f, o) => api.post('/api/stages', f, o),
  compare: (f, o) => api.post('/api/compare', f, o),
  video:   (f, o) => api.post('/api/video', f, o),
}

/** Fetch a bundled sample and hand it back as a File, so samples and uploads
 *  travel through exactly the same code path. */
export async function sampleAsFile(url, name = 'sample.jpg') {
  const blob = await fetch(url).then(r => r.blob())
  return new File([blob], name, { type: blob.type || 'image/jpeg' })
}
