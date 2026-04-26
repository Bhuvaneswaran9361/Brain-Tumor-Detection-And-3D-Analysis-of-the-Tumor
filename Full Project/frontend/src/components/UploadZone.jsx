import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { downloadReport } from '../utils/api'

const MAX_SIZE = 50 * 1024 * 1024
const ACCEPTED = { 'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'] }

export default function UploadZone({ onFileSelect, isLoading }) {
  const [preview,     setPreview]     = useState(null)
  const [file,        setFile]        = useState(null)
  const [dropError,   setDropError]   = useState(null)
  const [patientId,   setPatientId]   = useState('')
  const [patientName, setPatientName] = useState('')
  const [generating,  setGenerating]  = useState(false)
  const [reportError, setReportError] = useState(null)

  const onDrop = useCallback((accepted, rejected) => {
    setDropError(null); setReportError(null)
    if (rejected.length > 0) {
      const e = rejected[0].errors[0]
      setDropError(e.code === 'file-too-large' ? 'File exceeds 50 MB limit.' : e.message)
      return
    }
    if (!accepted.length) return
    const f = accepted[0]
    setFile(f)
    setPreview(URL.createObjectURL(f))
    onFileSelect(f, {
      patientId:   patientId   || 'Unknown',
      patientName: patientName || 'N/A',
    })
  }, [onFileSelect, patientId, patientName])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: ACCEPTED, maxSize: MAX_SIZE,
    multiple: false, disabled: isLoading,
  })

  const handleDownloadReport = async () => {
    if (!file) return
    setGenerating(true); setReportError(null)
    try {
      await downloadReport(file, {
        patientId:   patientId   || 'Unknown',
        patientName: patientName || 'N/A',
      })
    } catch (e) {
      setReportError(e.message)
    } finally {
      setGenerating(false)
    }
  }

  const inputStyle = {
    width: '100%', padding: '0.5rem 0.75rem',
    background: 'var(--bg-deep)', border: '1px solid var(--border)',
    borderRadius: 'var(--r-sm)', color: 'var(--text-primary)',
    fontFamily: 'var(--font-mono)', fontSize: '0.72rem',
    outline: 'none', transition: 'border-color 0.2s',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* Patient fields */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
        {[
          { label: 'Patient ID', value: patientId, set: setPatientId, ph: 'e.g. P-001' },
          { label: 'Patient Name', value: patientName, set: setPatientName, ph: 'e.g. Jane Doe' },
        ].map(({ label, value, set, ph }) => (
          <div key={label}>
            <label style={{
              display: 'block', marginBottom: 4,
              fontFamily: 'var(--font-display)', fontSize: '0.58rem',
              color: 'var(--text-muted)', letterSpacing: '0.08em', textTransform: 'uppercase',
            }}>{label}</label>
            <input
              value={value} onChange={e => set(e.target.value)}
              placeholder={ph} style={inputStyle} disabled={isLoading}
              onFocus={e => e.target.style.borderColor = 'var(--cyan)'}
              onBlur={e  => e.target.style.borderColor = 'var(--border)'}
            />
          </div>
        ))}
      </div>

      {/* Drop zone */}
      <motion.div
        {...getRootProps()}
        whileHover={!isLoading ? { borderColor: 'var(--cyan)', boxShadow: '0 0 28px var(--cyan-glow)' } : {}}
        style={{
          border: `2px dashed ${isDragActive ? 'var(--cyan)' : 'var(--border-bright)'}`,
          borderRadius: 'var(--r-lg)',
          padding: preview ? '0.9rem' : '2.5rem 1.5rem',
          textAlign: 'center',
          cursor: isLoading ? 'not-allowed' : 'pointer',
          background: isDragActive ? 'rgba(0,229,255,0.04)' : 'var(--bg-card)',
          transition: 'all 0.25s ease',
          position: 'relative', overflow: 'hidden',
        }}
      >
        <input {...getInputProps()} />
        <AnimatePresence>
          {isDragActive && (
            <motion.div
              initial={{ top: '-5%' }} animate={{ top: '110%' }} exit={{ opacity: 0 }}
              transition={{ duration: 1.4, repeat: Infinity, ease: 'linear' }}
              style={{
                position: 'absolute', left: 0, right: 0, height: 2,
                background: 'linear-gradient(90deg, transparent, var(--cyan), transparent)',
              }}
            />
          )}
        </AnimatePresence>

        {preview ? (
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <div style={{
              width: 72, height: 72, borderRadius: 8,
              border: '1px solid var(--border-bright)', overflow: 'hidden', flexShrink: 0,
            }}>
              <img src={preview} alt="MRI preview"
                style={{ width: '100%', height: '100%', objectFit: 'cover', filter: 'grayscale(1)' }} />
            </div>
            <div style={{ flex: 1, textAlign: 'left' }}>
              <p style={{ fontFamily: 'var(--font-display)', fontSize: '0.72rem', color: 'var(--cyan)', letterSpacing: '0.05em' }}>
                {isLoading ? '⚡ ANALYSING…' : '✓ MRI LOADED'}
              </p>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.62rem', color: 'var(--text-muted)', marginTop: 4 }}>
                {file?.name} · {(file?.size / 1024).toFixed(0)} KB
              </p>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: 2 }}>
                {isLoading ? 'HADF → YOLO-NAS → En-DeNet pipeline running' : 'Click or drag to replace'}
              </p>
            </div>
            {!isLoading && (
              <button onClick={e => { e.stopPropagation(); setPreview(null); setFile(null) }}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.3rem', padding: 4 }}>×</button>
            )}
          </div>
        ) : (
          <div>
            <motion.div
              animate={{ y: [0, -6, 0] }}
              transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}
              style={{ fontSize: '3.2rem', marginBottom: '0.8rem' }}
            >🧠</motion.div>
            <p style={{
              fontFamily: 'var(--font-display)', fontSize: '0.85rem',
              color: isDragActive ? 'var(--cyan)' : 'var(--text-secondary)',
              letterSpacing: '0.07em', marginBottom: '0.4rem',
            }}>
              {isDragActive ? 'RELEASE TO SCAN' : 'DROP MRI SCAN HERE'}
            </p>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
              JPG · PNG · BMP · TIFF · WebP  ·  max 50 MB
            </p>
          </div>
        )}
      </motion.div>

      {/* Errors */}
      <AnimatePresence>
        {(dropError || reportError) && (
          <motion.p
            initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            style={{ color: 'var(--red-tumor)', fontFamily: 'var(--font-mono)', fontSize: '0.72rem', textAlign: 'center' }}
          >
            ⚠ {dropError || reportError}
          </motion.p>
        )}
      </AnimatePresence>

      {/* Download report button */}
      {file && !isLoading && (
        <motion.button
          initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
          onClick={handleDownloadReport} disabled={generating}
          style={{
            width: '100%', padding: '0.65rem',
            background: generating ? 'var(--bg-elevated)' : 'linear-gradient(135deg,#005f73,#0096c7)',
            border: '1px solid var(--border-bright)', borderRadius: 'var(--r-sm)',
            color: generating ? 'var(--text-muted)' : 'white',
            fontFamily: 'var(--font-display)', fontSize: '0.68rem',
            letterSpacing: '0.09em', cursor: generating ? 'wait' : 'pointer',
            transition: 'all 0.2s', boxShadow: generating ? 'none' : '0 2px 16px rgba(0,150,200,0.25)',
          }}
        >
          {generating ? '⟳  GENERATING PDF…' : '⬇  DOWNLOAD CLINICAL REPORT'}
        </motion.button>
      )}
    </div>
  )
}
