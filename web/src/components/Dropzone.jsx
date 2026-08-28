import { useCallback, useEffect, useRef, useState } from 'react'
import { Camera, Image as ImageIcon, Upload, X, Video } from 'lucide-react'
import { api, sampleAsFile } from '../lib/api'
import { Card, Spinner } from './ui'

export default function Dropzone({ onFile, kind = 'image', busy = false }) {
  const [drag, setDrag] = useState(false)
  const [samples, setSamples] = useState([])
  const [preview, setPreview] = useState(null)
  const [camOpen, setCamOpen] = useState(false)
  const [camError, setCamError] = useState('')
  const inputRef = useRef()
  const videoRef = useRef()
  const streamRef = useRef()

  useEffect(() => {
    api.samples(kind, kind === 'video' ? 6 : 8)
      .then(d => setSamples(d.samples)).catch(() => {})
  }, [kind])

  const accept = kind === 'video' ? 'video/*' : 'image/*'

  const take = useCallback((file) => {
    if (!file) return
    if (kind === 'image') {
      const url = URL.createObjectURL(file)
      setPreview(url)
    } else setPreview(null)
    onFile(file)
  }, [kind, onFile])

  const onDrop = (e) => {
    e.preventDefault(); setDrag(false)
    take(e.dataTransfer.files?.[0])
  }

  const openCam = async () => {
    setCamError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } },
        audio: false,
      })
      streamRef.current = stream
      setCamOpen(true)
      requestAnimationFrame(() => { if (videoRef.current) videoRef.current.srcObject = stream })
    } catch (err) {
      setCamError(
        window.isSecureContext
          ? `Camera unavailable: ${err.message}`
          : 'Browsers only allow camera access over HTTPS or on localhost. ' +
            'Open this page on the machine running it, or serve it over HTTPS.')
    }
  }

  const closeCam = () => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    setCamOpen(false)
  }

  const shoot = () => {
    const v = videoRef.current
    if (!v) return
    const c = document.createElement('canvas')
    c.width = v.videoWidth; c.height = v.videoHeight
    c.getContext('2d').drawImage(v, 0, 0)
    c.toBlob(b => {
      take(new File([b], 'camera.jpg', { type: 'image/jpeg' }))
      closeCam()
    }, 'image/jpeg', 0.92)
  }

  useEffect(() => () => closeCam(), [])

  return (
    <div className="space-y-4">
      {camOpen ? (
        <div className="border border-rule bg-surface">
          <video ref={videoRef} autoPlay playsInline muted className="w-full bg-black" />
          <div className="flex gap-px border-t border-rule bg-rule">
            <button onClick={shoot}
              className="flex flex-1 items-center justify-center gap-2 bg-accent px-4 py-3
                         text-page transition-opacity hover:opacity-90">
              <Camera size={16} />
              <span className="label !text-[10px] !text-inherit">Capture</span>
            </button>
            <button onClick={closeCam}
              className="flex items-center justify-center gap-2 bg-surface px-5 py-3 text-muted
                         transition-colors hover:bg-sunken hover:text-strong">
              <X size={16} />
              <span className="label !text-[10px] !text-inherit">Cancel</span>
            </button>
          </div>
        </div>
      ) : (
        <div
          onDragOver={e => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={`relative cursor-pointer border border-dashed p-7 text-center
                      transition-colors duration-200
                      ${drag ? 'border-accent bg-accent/[.08]'
                             : 'border-rule2 bg-surface/50 hover:border-muted hover:bg-sunken/60'}`}
        >
          <input ref={inputRef} type="file" accept={accept} className="hidden"
                 onChange={e => take(e.target.files?.[0])} />
          {preview ? (
            <img src={preview} alt="selected" className="mx-auto max-h-56" />
          ) : (
            <>
              <div className="mx-auto mb-3.5 grid h-11 w-11 place-items-center border border-rule2 text-muted">
                {busy ? <Spinner /> : kind === 'video' ? <Video size={19}/> : <Upload size={19}/>}
              </div>
              <p className="text-[14px] text-body">
                Drop {kind === 'video' ? 'a clip' : 'a road photo'} here, or tap to choose
              </p>
              <p className="mt-1.5 text-[11.5px] text-faint">
                {kind === 'video' ? 'mp4 · processed with temporal smoothing'
                                  : 'jpg or png · a forward-facing view of a road'}
              </p>
            </>
          )}
        </div>
      )}

      {kind === 'image' && !camOpen && (
        <button onClick={openCam} className="btn-ghost w-full">
          <Camera size={16}/>
          <span className="label !text-[10px] !text-inherit">Use camera</span>
        </button>
      )}
      {camError && (
        <p className="border border-signal/40 bg-signal/[.12] px-3 py-2 text-xs text-signal">
          {camError}
        </p>
      )}

      {samples.length > 0 && (
        <div>
          <div className="label mb-2.5 flex items-center gap-2">
            <ImageIcon size={11}/> IDD validation {kind === 'video' ? 'sequences' : 'frames'}
          </div>
          <div className={`grid gap-2 ${kind === 'video' ? 'grid-cols-2 sm:grid-cols-3' : 'grid-cols-4'}`}>
            {samples.map(s => (
              <button key={s.id}
                onClick={async () => {
                  const f = await sampleAsFile(s.url, s.name)
                  if (kind === 'image') setPreview(s.thumb || null)
                  onFile(f)
                }}
                className="group overflow-hidden border border-rule transition-colors hover:border-accent focus:outline-none focus:border-accent">
                {s.thumb
                  ? <img src={s.thumb} alt={s.name} className="w-full transition-transform duration-300 group-hover:scale-105" />
                  : <div className="flex aspect-video items-center justify-center bg-sunken px-1 text-[9px] text-muted">{s.name.slice(0, 18)}</div>}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
