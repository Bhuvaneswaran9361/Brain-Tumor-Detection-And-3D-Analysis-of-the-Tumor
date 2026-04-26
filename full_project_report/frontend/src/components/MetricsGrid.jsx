import { motion } from 'framer-motion'

function Card({ label, value, unit, icon, highlight, delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
      style={{
        background: 'var(--bg-card)',
        border: `1px solid ${highlight ? 'rgba(0,229,255,0.28)' : 'var(--border)'}`,
        borderRadius: 'var(--r-md)',
        padding: '0.85rem 1rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{
          fontFamily: 'var(--font-display)', fontSize: '0.58rem',
          color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em',
        }}>{label}</span>
        <span style={{ fontSize: '0.9rem' }}>{icon}</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.3rem' }}>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: '1.25rem',
          color: highlight ? 'var(--cyan)' : 'var(--text-primary)',
          fontWeight: 500, lineHeight: 1,
        }}>{value ?? '—'}</span>
        {unit && (
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.62rem', color: 'var(--text-muted)' }}>
            {unit}
          </span>
        )}
      </div>
    </motion.div>
  )
}

function ShapeBar({ label, value, color, delay = 0 }) {
  const pct = Math.min(100, Math.round((value ?? 0) * 100))
  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay }}
      style={{ display: 'flex', flexDirection: 'column', gap: 4 }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: '0.62rem',
          color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em',
        }}>{label}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color }}>{value?.toFixed(3) ?? '—'}</span>
      </div>
      <div style={{ height: 4, background: 'var(--bg-deep)', borderRadius: 2, overflow: 'hidden' }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9, ease: 'easeOut', delay }}
          style={{ height: '100%', background: color, borderRadius: 2 }}
        />
      </div>
    </motion.div>
  )
}

function DataRow({ label, value }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '0.45rem 0', borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.62rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
        {label}
      </span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--cyan)' }}>
        {value}
      </span>
    </div>
  )
}

export default function MetricsGrid({ tumorData, classification, processingTime }) {
  const d3    = tumorData    || {}
  const shape = d3.shape_metrics
  const dims  = d3.dimensions

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Primary cards grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.6rem' }}>
        <Card label="Diagnosis"  value={classification?.class_name} highlight icon="🔬" delay={0} />
        <Card label="Confidence" value={classification?.confidence?.toFixed(1)} unit="%" icon="📊" delay={0.05} />
        <Card label="Area"       value={d3.area_pixels?.toLocaleString()} unit="px²" icon="📐" delay={0.1} />
        <Card label="Volume"     value={d3.volume_mm2?.toFixed(1)} unit="mm²" icon="📦" delay={0.15} />
        {dims && <>
          <Card label="Width"    value={dims.width_mm?.toFixed(1)}  unit="mm" icon="↔" delay={0.2} />
          <Card label="Height"   value={dims.height_mm?.toFixed(1)} unit="mm" icon="↕" delay={0.25} />
          <Card label="W (px)"   value={dims.width_px}              icon="⬜"  delay={0.3} />
          <Card label="H (px)"   value={dims.height_px}             icon="⬛"  delay={0.35} />
        </>}
        <Card label="Proc. Time" value={processingTime?.toFixed(2)} unit="s" icon="⚡" delay={0.4} />
        {d3.volume_mm3 > 0 && (
          <Card label="Vol 3D" value={d3.volume_mm3?.toFixed(1)} unit="mm³" icon="🧊" delay={0.45} />
        )}
      </div>

      {/* Shape metrics */}
      {shape && (
        <div style={{
          background: 'var(--bg-card)', border: '1px solid var(--border)',
          borderRadius: 'var(--r-md)', padding: '1rem',
        }}>
          <p style={{
            fontFamily: 'var(--font-display)', fontSize: '0.6rem',
            color: 'var(--text-muted)', letterSpacing: '0.1em',
            textTransform: 'uppercase', marginBottom: '0.8rem',
          }}>Shape Analysis</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            <ShapeBar label="Sphericity"  value={shape.sphericity}  color="var(--cyan)"    delay={0.1} />
            <ShapeBar label="Circularity" value={shape.circularity} color="#ff6b9d"        delay={0.15} />
            <ShapeBar label="Compactness" value={shape.compactness} color="var(--amber)"   delay={0.2} />
            <ShapeBar label="Elongation"  value={shape.elongation}  color="var(--purple)"  delay={0.25} />
          </div>
        </div>
      )}

      {/* Centroid + bounding box */}
      {(d3.centroid || d3.bounding_box) && (
        <div style={{
          background: 'var(--bg-card)', border: '1px solid var(--border)',
          borderRadius: 'var(--r-md)', padding: '0.85rem 1rem',
        }}>
          <p style={{
            fontFamily: 'var(--font-display)', fontSize: '0.6rem',
            color: 'var(--text-muted)', letterSpacing: '0.1em',
            textTransform: 'uppercase', marginBottom: '0.6rem',
          }}>Spatial Data</p>
          {d3.centroid && (
            <DataRow label="Centroid"
              value={`(${d3.centroid[0]}px, ${d3.centroid[1]}px)`} />
          )}
          {d3.bounding_box && (() => {
            const [x,y,w,h] = d3.bounding_box
            return (
              <DataRow label="Bounding Box"
                value={`x${x} y${y} · ${w}×${h}px`} />
            )
          })()}
        </div>
      )}
    </div>
  )
}
