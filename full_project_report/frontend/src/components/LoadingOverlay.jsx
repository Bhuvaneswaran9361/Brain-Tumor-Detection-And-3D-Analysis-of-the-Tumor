import { motion, AnimatePresence } from 'framer-motion'

const STEPS = [
  { id: 'hadf',   pct: 0,  label: 'HADF Preprocessing',      sub: 'Anisotropic diffusion · kurtosis stopping' },
  { id: 'cls',    pct: 25, label: 'YOLO-NAS Classification',  sub: 'QARepVGG · SPPF · CBAM attention' },
  { id: 'seg',    pct: 50, label: 'En-DeNet Segmentation',    sub: 'ASPP bottleneck · attention gates · v3' },
  { id: 'cam',    pct: 68, label: 'Grad-CAM Explainability',  sub: 'Grad-CAM++ · SegGrad-CAM · EigenCAM' },
  { id: '3d',     pct: 82, label: '3D Reconstruction',        sub: 'Marching cubes · Gaussian depth extrusion' },
  { id: 'report', pct: 94, label: 'Finalising Results',       sub: 'Shape metrics · volume analysis' },
]

function Step({ label, sub, state }) {
  const colors = { done: 'var(--green-ok)', active: 'var(--cyan)', pending: 'var(--text-muted)' }
  const c = colors[state]
  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      style={{ display: 'flex', gap: '0.85rem', alignItems: 'flex-start' }}
    >
      <div style={{ paddingTop: 2, width: 18, flexShrink: 0 }}>
        {state === 'done' ? (
          <span style={{ color: 'var(--green-ok)', fontSize: '0.8rem' }}>✓</span>
        ) : state === 'active' ? (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ repeat: Infinity, duration: 0.75, ease: 'linear' }}
            style={{
              width: 14, height: 14, borderRadius: '50%',
              border: '2px solid var(--cyan)',
              borderTopColor: 'transparent', marginTop: 1,
            }}
          />
        ) : (
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            border: '1px solid var(--text-muted)', margin: '3px 0 0 3px',
          }} />
        )}
      </div>
      <div>
        <p style={{
          fontFamily: 'var(--font-display)', fontSize: '0.68rem',
          color: c, letterSpacing: '0.05em',
        }}>{label}</p>
        <p style={{
          fontFamily: 'var(--font-mono)', fontSize: '0.58rem',
          color: 'var(--text-muted)', marginTop: 2, lineHeight: 1.4,
        }}>{sub}</p>
      </div>
    </motion.div>
  )
}

export default function LoadingOverlay({ visible, progress }) {
  const activeIdx = STEPS.findLastIndex(s => (progress || 0) >= s.pct)

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          style={{
            position: 'fixed', inset: 0, zIndex: 100,
            background: 'rgba(2,6,8,0.92)',
            backdropFilter: 'blur(10px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          {/* Horizontal scan line */}
          <motion.div
            animate={{ top: ['0%', '100%'] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: 'linear' }}
            style={{
              position: 'absolute', left: 0, right: 0, height: 2, zIndex: 101,
              background: 'linear-gradient(90deg, transparent 0%, var(--cyan) 50%, transparent 100%)',
              opacity: 0.5, pointerEvents: 'none',
            }}
          />

          <motion.div
            initial={{ scale: 0.94, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.3 }}
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-bright)',
              borderRadius: 'var(--r-xl)',
              padding: '2.5rem 2.25rem',
              width: 400,
              boxShadow: '0 0 80px rgba(0,229,255,0.08), 0 20px 60px rgba(0,0,0,0.7)',
            }}
          >
            {/* Brain icon */}
            <div style={{ textAlign: 'center', marginBottom: '1.75rem' }}>
              <motion.div
                animate={{ scale: [1, 1.07, 1] }}
                transition={{ repeat: Infinity, duration: 1.8, ease: 'easeInOut' }}
                style={{ fontSize: '2.8rem', marginBottom: '0.6rem' }}
              >🧠</motion.div>
              <h2 style={{
                fontFamily: 'var(--font-display)', fontSize: '0.82rem',
                color: 'var(--cyan)', letterSpacing: '0.14em',
              }}>
                ANALYSING MRI SCAN
              </h2>
            </div>

            {/* Steps */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem', marginBottom: '1.75rem' }}>
              {STEPS.map((s, i) => (
                <Step
                  key={s.id}
                  label={s.label}
                  sub={s.sub}
                  state={i < activeIdx ? 'done' : i === activeIdx ? 'active' : 'pending'}
                />
              ))}
            </div>

            {/* Progress bar */}
            <div style={{
              height: 3, borderRadius: 2,
              background: 'var(--bg-deep)', overflow: 'hidden',
            }}>
              <motion.div
                animate={{ width: `${Math.max(4, progress || 4)}%` }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
                style={{
                  height: '100%', borderRadius: 2,
                  background: 'linear-gradient(90deg, var(--cyan-dim), var(--cyan))',
                  boxShadow: '0 0 10px var(--cyan)',
                }}
              />
            </div>
            <p style={{
              textAlign: 'right', marginTop: 6,
              fontFamily: 'var(--font-mono)', fontSize: '0.6rem', color: 'var(--text-muted)',
            }}>
              {Math.round(progress || 0)}%
            </p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}