import { useRef, useMemo, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Stars } from '@react-three/drei'
import * as THREE from 'three'

// ── Point cloud from sampled_points ──────────────────────────────────────────
function PointCloud({ points, color }) {
  const ref    = useRef()
  const count  = points.length

  const { positions, colors } = useMemo(() => {
    const positions = new Float32Array(points.flat())
    const colors    = new Float32Array(count * 3)
    const base      = new THREE.Color(color)
    for (let i = 0; i < count; i++) {
      const z = points[i][2] ?? 0
      const t = Math.max(0, Math.min(1, z * 2 + 0.4))
      const c = base.clone().lerp(new THREE.Color('#ffffff'), t * 0.4)
      colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b
    }
    return { positions, colors }
  }, [points, color, count])

  useFrame(({ clock }) => {
    if (ref.current) ref.current.rotation.y = clock.elapsedTime * 0.22
  })

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color"    args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.022} vertexColors transparent opacity={0.88} sizeAttenuation />
    </points>
  )
}

// ── Pulsing tumour ellipsoid ──────────────────────────────────────────────────
function Ellipsoid({ bbox, color }) {
  const ref = useRef()
  const [bw, bh] = bbox ? [bbox[2] / 512, bbox[3] / 512] : [0.28, 0.24]
  const rx = Math.max(0.12, bw * 1.5)
  const ry = Math.max(0.12, bh * 1.5)
  // rz: approximate depth as 75% of mean radius — gives a proper 3-D blob
  // instead of a flat disc when only 2-D bbox data is available.
  const rz = Math.max(0.10, (rx + ry) / 2 * 0.75)

  useFrame(({ clock }) => {
    if (!ref.current) return
    const t = clock.elapsedTime
    ref.current.rotation.y = t * 0.28
    ref.current.rotation.x = Math.sin(t * 0.18) * 0.09
    const pulse = 1 + Math.sin(t * 1.4) * 0.035
    ref.current.scale.setScalar(pulse)
  })

  return (
    <group ref={ref}>
      <mesh scale={[rx, ry, rz]}>
        <sphereGeometry args={[1, 48, 48]} />
        <meshPhongMaterial color={color} transparent opacity={0.3} side={THREE.DoubleSide} />
      </mesh>
      <mesh scale={[rx * 1.04, ry * 1.04, rz * 1.04]}>
        <sphereGeometry args={[1, 24, 24]} />
        <meshBasicMaterial color={color} wireframe transparent opacity={0.22} />
      </mesh>
      <mesh scale={[rx * 0.65, ry * 0.65, rz * 0.65]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshPhongMaterial color={color} emissive={color} emissiveIntensity={0.55} transparent opacity={0.48} />
      </mesh>
    </group>
  )
}

// ── Brain reference shell ─────────────────────────────────────────────────────
function BrainShell() {
  const ref = useRef()
  useFrame(({ clock }) => {
    if (ref.current) ref.current.rotation.y = clock.elapsedTime * 0.07
  })
  return (
    <group ref={ref}>
      <mesh>
        <sphereGeometry args={[1.12, 48, 24, 0, Math.PI * 2, 0, Math.PI * 0.56]} />
        <meshPhongMaterial color="#122844" transparent opacity={0.1} side={THREE.BackSide} />
      </mesh>
      <mesh>
        <sphereGeometry args={[1.12, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.56]} />
        <meshBasicMaterial color="#00e5ff" wireframe transparent opacity={0.05} />
      </mesh>
    </group>
  )
}

// ── Floating label ────────────────────────────────────────────────────────────
function Label({ tumorType, confidence }) {
  const ref = useRef()
  useFrame(({ clock }) => {
    if (ref.current) ref.current.position.y = 0.9 + Math.sin(clock.elapsedTime * 0.75) * 0.04
  })
  return (
    <group ref={ref}>
      <Text fontSize={0.1} color="#ff3333" anchorX="center" anchorY="middle">
        {tumorType?.toUpperCase()}
      </Text>
      <Text fontSize={0.065} color="#00e5ff" anchorX="center" anchorY="middle" position={[0, -0.14, 0]}>
        {`CONF: ${confidence?.toFixed(1)}%`}
      </Text>
    </group>
  )
}

// ── XYZ axes ──────────────────────────────────────────────────────────────────
function Axes() {
  const axis = (pos, rot, color, lbl, lpos) => (
    <group>
      <mesh position={pos} rotation={rot}>
        <cylinderGeometry args={[0.003, 0.003, 1.1, 6]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <Text position={lpos} fontSize={0.07} color={color} anchorX="center">{lbl}</Text>
    </group>
  )
  return (
    <group>
      {axis([0.55,0,0],[0,0,-Math.PI/2],'#ff5555','X',[1.0,0,0])}
      {axis([0,0.55,0],[0,0,0],         '#55ff55','Y',[0,1.0,0])}
      {axis([0,0,0.55],[Math.PI/2,0,0], '#5599ff','Z',[0,0,1.0])}
    </group>
  )
}

// ── Full scene ────────────────────────────────────────────────────────────────
function Scene({ tumorData, classification }) {
  const color    = classification?.class_color || '#ff3333'
  const has      = tumorData?.has_tumor
  const points   = tumorData?.sampled_points || []
  const bbox     = tumorData?.bounding_box

  return (
    <>
      <ambientLight intensity={0.28} />
      <pointLight position={[3, 3, 3]}   intensity={1.1} color="#00e5ff" />
      <pointLight position={[-3,-2,-3]}   intensity={0.7} color={color} />
      <Stars radius={9} depth={50} count={700} factor={0.38} fade speed={0.4} />
      <BrainShell />
      {has && (
        <>
          <Ellipsoid bbox={bbox} color={color} />
          {points.length > 0 && <PointCloud points={points} color={color} />}
          <Label tumorType={classification?.class_name} confidence={classification?.confidence} />
        </>
      )}
      {!has && (
        <Text position={[0,0,0]} fontSize={0.11} color="#00ff88" anchorX="center" anchorY="middle">
          NO TUMOUR DETECTED
        </Text>
      )}
      <Axes />
      <OrbitControls autoRotate={false} enableZoom enablePan={false} minDistance={1.5} maxDistance={5} />
    </>
  )
}

export default function TumorViewer3D({ tumorData, classification, style }) {
  return (
    <div style={{
      width: '100%', height: '100%', minHeight: 380,
      borderRadius: 'var(--r-lg)', overflow: 'hidden',
      background: 'radial-gradient(ellipse at center, #091525 0%, #020608 100%)',
      ...style,
    }}>
      <Canvas camera={{ position: [0, 0, 3], fov: 46 }} gl={{ antialias: true }} dpr={[1, 2]}>
        <Suspense fallback={null}>
          <Scene tumorData={tumorData} classification={classification} />
        </Suspense>
      </Canvas>
    </div>
  )
}