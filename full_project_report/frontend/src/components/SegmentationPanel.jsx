import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { b64Src } from '../utils/api'

const VIEWS = [
  { id: 'overlay',   label: 'Detection',      key: 'overlay_image_b64',      desc: 'Tumour contour overlay' },
  { id: 'mask',      label: 'Mask',           key: 'mask',                   desc: 'Binary segmentation mask' },
  { id: 'hadf',      label: 'HADF',           key: 'hadf_image_b64',         desc: 'Preprocessed input' },
  { id: 'gradcam',   label: 'Grad-CAM',       key: 'gradcam_overlay_b64',    desc: 'Classifier attention · JET' },
  { id: 'seggcam',   label: 'SegGrad-CAM',    key: 'seg_gradcam_overlay_b64', desc: 'Segmenter attention · HOT' },
  { id: 'eigencam',  label: 'EigenCAM',       key: 'eigencam_overlay_b64',   desc: 'Gradient-free · VIRIDIS' },
]

function Corner({ pos }) {
  const s = {
    position: 'absolute', width: 11, height: 11,
    borderTop:    pos.includes('t') ? '2px solid var(--cyan)' : 'none',
    borderBottom: pos.includes('b') ? '2px solid var(--cyan)' : 'none',
    borderLeft:   pos.includes('l') ? '2px solid var(--cyan)' : 'none',
    borderRight:  pos.includes('r') ? '2px solid var(--cyan)' : 'none',
    top:    pos.includes('t') ? 6 : 'auto',
    bottom: pos.includes('b') ? 6 : 'auto',
    left:   pos.includes('l') ? 6 : 'auto',
    right:  pos.includes('r') ? 6 : 'auto',
    opacity: 0.6,
  }
  return <div style={s} />
}

export default function SegmentationPanel({ result }) {
  const [view, setView] = useState('overlay')

  if (!result) return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: 200, color: 'var(--text-muted)',
      fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
    }}>
      Waiting for scan…
    </div>
  )

  // Build image src for every view
  const imgSrcs = {
    overlay:  b64Src(result.overlay_image_b64),
    mask:     b64Src(result.segmentation?.mask_base64),
    hadf:     b64Src(result.hadf_image_b64),
    gradcam:  b64Src(result.gradcam_overlay_b64),
    seggcam:  b64Src(result.seg_gradcam_overlay_b64),
    eigencam: b64Src(result.eigencam_overlay_b64),
  }

  const current    = imgSrcs[view]
  const currentDef = VIEWS.find(v => v.id === view)
  const detected   = result.segmentation?.tumor_detected

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* View toggle — 2 rows of 3 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px' }}>
        {VIEWS.map((v) => (
          <button key={v.id} onClick={() => setView(v.id)} style={{
            padding: '5px 4px', border: 'none', cursor: 'pointer',
            borderRadius: 'var(--r-sm)',
            background: view === v.id ? 'var(--bg-elevated)' : 'transparent',
            color: view === v.id ? 'var(--cyan)' : 'var(--text-muted)',
            fontFamily: 'var(--font-display)', fontSize: '0.56rem',
            letterSpacing: '0.06em', textTransform: 'uppercase',
            borderBottom: view === v.id ? '1px solid var(--cyan)' : '1px solid transparent',
            transition: 'all 0.2s',
          }}>
            {v.label}
          </button>
        ))}
      </div>

      {/* Image frame */}
      <div style={{
        position: 'relative', width: '100%', aspectRatio: '1',
        background: '#000', borderRadius: 'var(--r-md)',
        overflow: 'hidden', border: '1px solid var(--border)',
      }}>
        <AnimatePresence mode="wait">
          {current ? (
            <motion.img
              key={view}
              src={current}
              alt={currentDef?.label}
              initial={{ opacity: 0, scale: 1.02 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25 }}
              style={{
                width: '100%', height: '100%', objectFit: 'contain',
                filter: view === 'mask' ? 'invert(1)' : 'none',
              }}
            />
          ) : (
            <motion.div
              key="no-img"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              style={{
                width: '100%', height: '100%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.65rem',
              }}
            >
              No image
            </motion.div>
          )}
        </AnimatePresence>

        {/* Corner brackets */}
        {['tl','tr','bl','br'].map(c => <Corner key={c} pos={c} />)}

        {/* Tumour badge */}
        {detected && (
          <div style={{
            position: 'absolute', top: 9, right: 9,
            fontFamily: 'var(--font-mono)', fontSize: '0.58rem',
            color: 'var(--red-tumor)',
            background: 'rgba(255,51,51,0.1)',
            border: '1px solid rgba(255,51,51,0.4)',
            borderRadius: 4, padding: '2px 7px', letterSpacing: '0.05em',
          }}>
            ⚠ TUMOUR
          </div>
        )}

        {/* View label */}
        <div style={{
          position: 'absolute', bottom: 7, left: 9,
          fontFamily: 'var(--font-mono)', fontSize: '0.56rem',
          color: 'rgba(0,229,255,0.5)', letterSpacing: '0.05em',
        }}>
          {currentDef?.desc}
        </div>
      </div>
    </div>
  )
}
