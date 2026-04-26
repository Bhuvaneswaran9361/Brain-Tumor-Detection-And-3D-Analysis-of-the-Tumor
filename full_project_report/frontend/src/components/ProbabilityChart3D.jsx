import { useRef, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Text, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

const CLASSES = [
  { key: 'Glioma',     color: '#ff3333', emit: '#cc0000' },
  { key: 'Meningioma', color: '#ff8c00', emit: '#cc5500' },
  { key: 'Pituitary',  color: '#ffd700', emit: '#cc9900' },
  { key: 'No Tumor',   color: '#00ff88', emit: '#00aa44' },
]

function Bar({ x, height, color, emit, label, value }) {
  const ref = useRef()
  const h   = Math.max(0.015, height)

  useFrame(({ clock }) => {
    if (!ref.current) return
    const t = clock.elapsedTime
    ref.current.scale.y = 1 + Math.sin(t * 1.3 + x) * 0.012
  })

  return (
    <group position={[x, 0, 0]}>
      <mesh ref={ref} position={[0, h / 2, 0]}>
        <boxGeometry args={[0.36, h, 0.36]} />
        <meshPhongMaterial color={color} emissive={emit} emissiveIntensity={0.35} transparent opacity={0.88} />
      </mesh>
      {/* Cap */}
      <mesh position={[0, h + 0.005, 0]}>
        <boxGeometry args={[0.38, 0.018, 0.38]} />
        <meshBasicMaterial color={color} transparent opacity={0.9} />
      </mesh>
      {/* Value label */}
      <Text position={[0, h + 0.1, 0]} fontSize={0.085} color={color} anchorX="center" anchorY="middle">
        {`${value.toFixed(1)}%`}
      </Text>
      {/* Class label */}
      <Text position={[0, -0.14, 0]} fontSize={0.06} color="#7aadce" anchorX="center" anchorY="middle">
        {label.split(' ')[0]}
      </Text>
    </group>
  )
}

function GridH() {
  return (
    <>
      {[0, 0.5, 1.0, 1.5, 2.0].map((y) => (
        <mesh key={y} position={[0, y, 0.15]}>
          <planeGeometry args={[5, 0.004]} />
          <meshBasicMaterial color="#00e5ff" transparent opacity={0.06} />
        </mesh>
      ))}
    </>
  )
}

function Scene({ probabilities }) {
  const spacing = 1.1
  const offset  = -(CLASSES.length - 1) * spacing / 2
  return (
    <>
      <ambientLight intensity={0.35} />
      <pointLight position={[2, 3, 2]}  intensity={1.4} color="#ffffff" />
      <pointLight position={[-2, 2, -1]} intensity={0.5} color="#00e5ff" />
      <mesh position={[0, -0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[6, 3.5]} />
        <meshBasicMaterial color="#091525" transparent opacity={0.45} />
      </mesh>
      <GridH />
      {CLASSES.map((cls, i) => (
        <Bar
          key={cls.key}
          x={offset + i * spacing}
          height={(probabilities?.[cls.key] ?? 0) / 100 * 2.1}
          color={cls.color} emit={cls.emit}
          label={cls.key}
          value={probabilities?.[cls.key] ?? 0}
        />
      ))}
      <OrbitControls
        enableZoom={false} enablePan={false}
        minPolarAngle={0.25} maxPolarAngle={Math.PI / 2.05}
        autoRotate autoRotateSpeed={1.1}
      />
    </>
  )
}

export default function ProbabilityChart3D({ probabilities }) {
  return (
    <div style={{
      width: '100%', height: 240,
      borderRadius: 'var(--r-md)', overflow: 'hidden',
      background: 'radial-gradient(ellipse at bottom, #091525 0%, #020608 100%)',
      border: '1px solid var(--border)',
    }}>
      <Canvas camera={{ position: [0, 1.6, 3.4], fov: 52 }}>
        <Suspense fallback={null}>
          <Scene probabilities={probabilities} />
        </Suspense>
      </Canvas>
    </div>
  )
}
