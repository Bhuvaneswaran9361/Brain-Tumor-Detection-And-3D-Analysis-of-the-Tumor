import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { getModelInfo, getClasses } from '../utils/api'

function Row({ label, value, mono = true }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
      gap: '0.75rem', padding: '0.4rem 0',
      borderBottom: '1px solid var(--border)',
    }}>
      <span style={{
        fontFamily: 'var(--font-display)', fontSize: '0.6rem',
        color: 'var(--text-muted)', textTransform: 'uppercase',
        letterSpacing: '0.07em', flexShrink: 0,
      }}>{label}</span>
      <span style={{
        fontFamily: mono ? 'var(--font-mono)' : 'var(--font-body)',
        fontSize: '0.66rem', color: 'var(--text-secondary)',
        textAlign: 'right',
      }}>{value}</span>
    </div>
  )
}

function ClassChip({ name, color, description }) {
  const [open, setOpen] = useState(false)
  return (
    <div>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', textAlign: 'left', background: 'none', border: 'none', cursor: 'pointer',
          display: 'flex', alignItems: 'center', gap: '0.6rem', padding: '0.4rem 0',
        }}
      >
        <div style={{ width: 10, height: 10, borderRadius: '50%', background: color, flexShrink: 0, boxShadow: `0 0 6px ${color}` }} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-primary)' }}>{name}</span>
        <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: '0.7rem' }}>{open ? '▾' : '▸'}</span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.p
            initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.2 }}
            style={{
              fontFamily: 'var(--font-body)', fontSize: '0.65rem',
              color: 'var(--text-secondary)', paddingLeft: '1.6rem',
              lineHeight: 1.5, overflow: 'hidden', marginBottom: 4,
            }}
          >{description}</motion.p>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function ModelInfoPanel() {
  const [info,    setInfo]    = useState(null)
  const [classes, setClasses] = useState(null)
  const [tab,     setTab]     = useState('arch')

  useEffect(() => {
    getModelInfo().then(setInfo).catch(() => {})
    getClasses().then(setClasses).catch(() => {})
  }, [])

  const tabs = [
    { id: 'arch',    label: 'Architecture' },
    { id: 'classes', label: 'Classes' },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* Tab bar */}
      <div style={{
        display: 'flex', gap: 4, background: 'var(--bg-deep)',
        borderRadius: 'var(--r-sm)', padding: 4,
      }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            flex: 1, padding: '5px', border: 'none', cursor: 'pointer',
            borderRadius: 4,
            background: tab === t.id ? 'var(--bg-elevated)' : 'transparent',
            color: tab === t.id ? 'var(--cyan)' : 'var(--text-muted)',
            fontFamily: 'var(--font-display)', fontSize: '0.6rem',
            letterSpacing: '0.07em', textTransform: 'uppercase',
            borderBottom: tab === t.id ? '1px solid var(--cyan)' : '1px solid transparent',
            transition: 'all 0.2s',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'arch' && info && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {/* Classifier */}
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--r-md)', padding: '0.85rem 1rem' }}>
            <p style={{ fontFamily: 'var(--font-display)', fontSize: '0.6rem', color: 'var(--cyan)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6 }}>Classifier</p>
            <Row label="Model"  value={info.classifier?.name} />
            <Row label="Input"  value={info.classifier?.input} />
            <Row label="Norm"   value={info.classifier?.normalisation} />
            <Row label="Loss"   value={info.classifier?.loss} />
          </div>
          {/* Segmenter */}
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--r-md)', padding: '0.85rem 1rem' }}>
            <p style={{ fontFamily: 'var(--font-display)', fontSize: '0.6rem', color: 'var(--amber)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6 }}>Segmenter</p>
            <Row label="Model" value={info.segmenter?.name} />
            <Row label="Input" value={info.segmenter?.input} />
            {info.segmenter?.features?.map(f => (
              <div key={f} style={{ display: 'flex', gap: 6, padding: '2px 0' }}>
                <span style={{ color: 'var(--amber)', fontSize: '0.6rem' }}>▸</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.62rem', color: 'var(--text-secondary)' }}>{f}</span>
              </div>
            ))}
          </div>
          {/* Preprocessing */}
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--r-md)', padding: '0.85rem 1rem' }}>
            <p style={{ fontFamily: 'var(--font-display)', fontSize: '0.6rem', color: 'var(--green-ok)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6 }}>Preprocessing</p>
            <Row label="Filter" value={info.preprocessing?.name} />
            <Row label="Method" value={info.preprocessing?.method} />
          </div>
        </div>
      )}

      {tab === 'classes' && classes && (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--r-md)', padding: '0.85rem 1rem' }}>
          {Object.entries(classes).map(([name, info]) => (
            <ClassChip key={name} name={name} color={info.color} description={info.description} />
          ))}
        </div>
      )}

      {!info && !classes && (
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-muted)', textAlign: 'center', padding: '1rem 0' }}>
          Loading model info…
        </p>
      )}
    </div>
  )
}
