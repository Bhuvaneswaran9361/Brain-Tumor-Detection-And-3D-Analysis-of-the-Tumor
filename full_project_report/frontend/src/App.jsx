import { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

import UploadZone        from './components/UploadZone'
import TumorViewer3D     from './components/TumorViewer3D'
import ProbabilityChart3D from './components/ProbabilityChart3D'
import SegmentationPanel from './components/SegmentationPanel'
import MetricsGrid       from './components/MetricsGrid'
import LoadingOverlay    from './components/LoadingOverlay'
import ModelInfoPanel    from './components/ModelInfoPanel'
import { predictTumour, getHealth } from './utils/api'

// ── Tiny helpers ──────────────────────────────────────────────────────────────
function SectionLabel({ icon, text, sub, accent = 'var(--cyan)' }) {
  return (
    <div style={{ marginBottom: '0.65rem' }}>
      <h2 style={{
        fontFamily: 'var(--font-display)', fontSize: '0.68rem',
        color: 'var(--text-secondary)', letterSpacing: '0.12em', textTransform: 'uppercase',
      }}>
        <span style={{ color: accent }}>{icon} </span>{text}
      </h2>
      {sub && (
        <p style={{
          fontFamily: 'var(--font-mono)', fontSize: '0.58rem',
          color: 'var(--text-muted)', marginTop: 2,
        }}>{sub}</p>
      )}
    </div>
  )
}

// ── Header ────────────────────────────────────────────────────────────────────
function Header({ status, demoMode }) {
  const online = status === 'healthy'
  return (
    <header style={{
      padding: '1rem 2rem',
      borderBottom: '1px solid var(--border)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      background: 'linear-gradient(180deg,rgba(2,6,8,0.96) 0%,transparent 100%)',
      backdropFilter: 'blur(12px)',
      position: 'sticky', top: 0, zIndex: 20,
    }}>
      {/* Brand */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.9rem' }}>
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 14, repeat: Infinity, ease: 'linear' }}
          style={{ fontSize: '1.5rem' }}
        >🧠</motion.div>
        <div>
          <h1 style={{
            fontFamily: 'var(--font-display)', fontSize: '0.95rem',
            fontWeight: 700, letterSpacing: '0.13em', color: 'var(--text-primary)',
          }}>
            NEURO<span style={{ color: 'var(--cyan)' }}>SCAN</span> AI
          </h1>
          <p style={{
            fontFamily: 'var(--font-mono)', fontSize: '0.56rem',
            color: 'var(--text-muted)', letterSpacing: '0.07em', marginTop: 1,
          }}>
            HADF · YOLO-NAS · EN-DENET v3 · GRAD-CAM · 3D RECON
          </p>
        </div>
      </div>

      {/* Status chips */}
      <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
        {demoMode && (
          <div className="badge badge-warning">DEMO</div>
        )}
        <div className={`badge ${online ? 'badge-ok' : 'badge-warning'}`}>
          <span style={{
            width: 6, height: 6, borderRadius: '50%', display: 'inline-block',
            background: online ? 'var(--green-ok)' : 'var(--amber)',
            animation: 'pulse-dot 2s infinite',
          }} />
          {online ? 'API ONLINE' : status === 'checking' ? 'CONNECTING…' : 'OFFLINE'}
        </div>
      </div>
    </header>
  )
}

// ── Classification banner ─────────────────────────────────────────────────────
function ClassBanner({ cls, demoMode }) {
  if (!cls) return null
  const isTumor = cls.class_name !== 'No Tumor'
  const color   = cls.class_color || 'var(--text-primary)'

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      style={{
        background: `linear-gradient(135deg, var(--bg-card), ${color}0e)`,
        border: `1px solid ${color}44`,
        borderRadius: 'var(--r-lg)', padding: '1rem 1.5rem',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexWrap: 'wrap', gap: '0.75rem',
        boxShadow: `0 4px 30px ${color}1a`,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {/* Status ring */}
        <motion.div
          animate={{ boxShadow: [`0 0 12px ${color}40`, `0 0 28px ${color}70`, `0 0 12px ${color}40`] }}
          transition={{ repeat: Infinity, duration: 2 }}
          style={{
            width: 46, height: 46, borderRadius: '50%',
            border: `2px solid ${color}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: `${color}12`, fontSize: '1.4rem',
          }}
        >
          {isTumor ? '⚠' : '✓'}
        </motion.div>

        <div>
          <p style={{
            fontFamily: 'var(--font-mono)', fontSize: '0.62rem',
            color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em',
          }}>Primary Diagnosis</p>
          <p style={{
            fontFamily: 'var(--font-display)', fontSize: '1.15rem',
            color, letterSpacing: '0.08em', fontWeight: 700,
          }}>{cls.class_name}</p>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap', alignItems: 'center' }}>
        {[
          { l: 'Confidence', v: `${cls.confidence?.toFixed(1)}%` },
          { l: 'Classifier',  v: 'YOLO-NAS' },
          { l: 'Segmenter',   v: 'En-DeNet v3' },
        ].map(({ l, v }) => (
          <div key={l} style={{ textAlign: 'center' }}>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.58rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>{l}</p>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color }}>{v}</p>
          </div>
        ))}
        {demoMode && <div className="badge badge-warning">DEMO MODE</div>}
      </div>
    </motion.div>
  )
}

// ── Disclaimer footer ─────────────────────────────────────────────────────────
function Disclaimer() {
  return (
    <div style={{
      textAlign: 'center', padding: '1.5rem 2rem',
      borderTop: '1px solid var(--border)',
      fontFamily: 'var(--font-mono)', fontSize: '0.6rem',
      color: 'var(--text-muted)', lineHeight: 1.6,
    }}>
      ⚠ This system is for <strong style={{ color: 'var(--amber)' }}>research and decision-support only</strong> —
      not a substitute for professional medical diagnosis.
      All findings must be reviewed by a qualified radiologist or neuro-oncologist.
      <br />
      YOLO-NAS + En-DeNet v3 · trained on BraTS 2020 · Grad-CAM++ · EigenCAM
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [result,    setResult]   = useState(null)
  const [loading,   setLoading]  = useState(false)
  const [progress,  setProgress] = useState(0)
  const [error,     setError]    = useState(null)
  const [status,    setStatus]   = useState('checking')
  const [demoMode,  setDemoMode] = useState(false)

  // Health poll
  useEffect(() => {
    const check = async () => {
      try {
        const h = await getHealth()
        setStatus('healthy')
        setDemoMode(h.demo_mode)
      } catch { setStatus('offline') }
    }
    check()
    const t = setInterval(check, 15_000)
    return () => clearInterval(t)
  }, [])

  const handleFileSelect = useCallback(async (file, opts = {}) => {
    setResult(null); setError(null)
    setLoading(true); setProgress(8)

    // Fake smooth progress while waiting
    const timer = setInterval(() => setProgress(p => Math.min(88, p + 12)), 700)

    try {
      const data = await predictTumour(file, {
        patientId:   opts.patientId,
        patientName: opts.patientName,
        onUploadProgress: (pct) => setProgress(8 + pct * 0.12),
      })
      setProgress(100)
      setResult(data)
      setDemoMode(data.demo_mode)
    } catch (e) {
      setError(e.message || 'Prediction failed')
    } finally {
      clearInterval(timer)
      setTimeout(() => { setLoading(false); setProgress(0) }, 500)
    }
  }, [])

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <LoadingOverlay visible={loading} progress={progress} />
      <Header status={status} demoMode={demoMode} />

      <main style={{ flex: 1, padding: '1.5rem 2rem 0', maxWidth: 1680, margin: '0 auto', width: '100%' }}>

        {/* ── Top strip: upload + model info ── */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '440px 1fr',
          gap: '1.25rem',
          marginBottom: '1.5rem',
          alignItems: 'start',
        }}>
          {/* Upload */}
          <div className="glass-card" style={{ padding: '1.25rem' }}>
            <SectionLabel icon="▸" text="MRI Scan Input" sub="Upload a brain MRI for AI analysis" />
            <UploadZone onFileSelect={handleFileSelect} isLoading={loading} />
          </div>

          {/* Model info */}
          <div className="glass-card" style={{ padding: '1.25rem' }}>
            <SectionLabel icon="▸" text="Model Architecture" sub="Live data from /api/model-info · /api/classes" accent="var(--amber)" />
            <ModelInfoPanel />
          </div>
        </div>

        {/* ── Error ── */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              style={{
                background: 'rgba(255,51,51,0.07)',
                border: '1px solid rgba(255,51,51,0.35)',
                borderRadius: 'var(--r-md)', padding: '0.85rem 1.25rem',
                color: 'var(--red-tumor)',
                fontFamily: 'var(--font-mono)', fontSize: '0.72rem',
                marginBottom: '1.25rem',
              }}
            >⚠ {error}</motion.div>
          )}
        </AnimatePresence>

        {/* ── Results ── */}
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, ease: 'easeOut' }}
            >
              {/* Classification banner */}
              <div style={{ marginBottom: '1.25rem' }}>
                <ClassBanner cls={result.classification} demoMode={result.demo_mode} />
              </div>

              {/* Main 3-column layout */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: '340px 1fr 300px',
                gap: '1.25rem',
                alignItems: 'start',
                marginBottom: '1.25rem',
              }}>

                {/* Left: Segmentation + CAMs panel */}
                <div className="glass-card" style={{ padding: '1.25rem' }}>
                  <SectionLabel icon="▸" text="Imaging Analysis"
                    sub="Detection · Mask · HADF · Grad-CAM · SegGrad-CAM · EigenCAM" />
                  <SegmentationPanel result={result} />
                </div>

                {/* Centre: 3D viewer */}
                <div className="glass-card" style={{ padding: '1.25rem' }}>
                  <SectionLabel icon="▸" text="3D Tumour Reconstruction"
                    sub="Marching cubes · drag to rotate · zoom enabled" />
                  <TumorViewer3D
                    tumorData={result.tumor_3d}
                    classification={result.classification}
                    style={{ height: 420 }}
                  />
                </div>

                {/* Right: Metrics */}
                <div className="glass-card" style={{ padding: '1.25rem' }}>
                  <SectionLabel icon="▸" text="Measurements" sub="Volume · shape · spatial data" accent="var(--amber)" />
                  <MetricsGrid
                    tumorData={result.tumor_3d}
                    classification={result.classification}
                    processingTime={result.processing_time}
                  />
                </div>
              </div>

              {/* Bottom: 3D probability chart */}
              <div className="glass-card" style={{ padding: '1.25rem', marginBottom: '1.5rem' }}>
                <SectionLabel icon="▸" text="Classification Probabilities"
                  sub="YOLO-NAS softmax · auto-rotates · drag to explore" accent="var(--purple)" />
                <ProbabilityChart3D probabilities={result.classification?.probabilities} />
              </div>

              {/* Processing metadata */}
              <div style={{
                display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '2rem',
              }}>
                {[
                  { l: 'Processing Time', v: `${result.processing_time?.toFixed(3)}s` },
                  { l: 'Demo Mode',       v: result.demo_mode ? 'YES' : 'NO (Live weights)' },
                  { l: 'Classification',  v: result.classification?.class_name },
                  { l: 'Segmenter',       v: result.segmentation?.tumor_detected ? 'Tumour found' : 'Clear' },
                ].map(({ l, v }) => (
                  <div key={l} style={{
                    background: 'var(--bg-card)', border: '1px solid var(--border)',
                    borderRadius: 'var(--r-sm)', padding: '0.5rem 0.9rem',
                  }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-muted)' }}>{l}: </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--cyan)' }}>{v}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Empty state */}
        {!result && !loading && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
            style={{ textAlign: 'center', padding: '4rem 2rem', color: 'var(--text-muted)' }}
          >
            <motion.div
              initial={{ opacity: 0.2 }}        // ← add this
              animate={{ opacity: [0.2, 0.4, 0.2] }}
              transition={{ repeat: Infinity, duration: 3 }}
              style={{ fontSize: '5rem', marginBottom: '1rem' }}
            ></motion.div>
            <p style={{
              fontFamily: 'var(--font-display)', fontSize: '0.75rem',
              letterSpacing: '0.12em', color: 'var(--text-muted)',
            }}>
              UPLOAD AN MRI SCAN TO BEGIN ANALYSIS
            </p>
          </motion.div>
        )}
      </main>

      <Disclaimer />
    </div>
  )
}
