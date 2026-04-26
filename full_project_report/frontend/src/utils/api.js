/**
 * api.js — HTTP client for the Brain Tumour Detection API
 *
 * Matches every endpoint defined in main.py:
 *   GET  /health
 *   GET  /api/model-info
 *   GET  /api/classes
 *   POST /api/predict
 *   POST /api/report        (blob download)
 *   POST /api/batch
 *   GET  /api/predict/stream  (SSE)
 */
import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 180_000,       // 3 min for large MRI + report generation
})

// ── Intercept → attach request-id to every call ──────────────────────────────
http.interceptors.request.use((cfg) => {
  cfg.headers['X-Request-ID'] = crypto.randomUUID().slice(0, 8)
  return cfg
})

// ── Response: propagate server error message to caller ───────────────────────
http.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail =
      err?.response?.data?.detail ||
      err?.response?.data?.error  ||
      err?.message                ||
      'Unknown error'
    return Promise.reject(new Error(detail))
  }
)

// ── Health ────────────────────────────────────────────────────────────────────
/**
 * GET /health
 * @returns {{ status, models_ready, demo_mode, device, uptime_s }}
 */
export async function getHealth() {
  const { data } = await axios.get('/health', { timeout: 5_000 })
  return data
}

// ── Model info ────────────────────────────────────────────────────────────────
/**
 * GET /api/model-info
 */
export async function getModelInfo() {
  const { data } = await http.get('/model-info')
  return data
}

// ── Class definitions ─────────────────────────────────────────────────────────
/**
 * GET /api/classes
 */
export async function getClasses() {
  const { data } = await http.get('/classes')
  return data
}

// ── Predict ───────────────────────────────────────────────────────────────────
/**
 * POST /api/predict
 *
 * @param {File}     file
 * @param {object}   opts
 * @param {boolean}  opts.generateReport   include report_b64 in response
 * @param {string}   opts.patientId
 * @param {string}   opts.patientName
 * @param {string}   opts.radiologist
 * @param {function} opts.onUploadProgress (pct: 0-100) => void
 *
 * Response keys (all base64 PNG unless noted):
 *   classification        { class_name, class_color, confidence, probabilities }
 *   segmentation          { mask_base64, overlay_base64, tumor_detected }
 *   tumor_3d              { has_tumor, area_pixels, volume_mm2, volume_mm3,
 *                           centroid, bounding_box, shape_metrics, dimensions,
 *                           sampled_points, figure_3d }
 *   hadf_image_b64
 *   overlay_image_b64
 *   gradcam_b64           / gradcam_overlay_b64
 *   seg_gradcam_b64       / seg_gradcam_overlay_b64
 *   eigencam_b64          / eigencam_overlay_b64
 *   demo_mode             boolean
 *   processing_time       seconds
 *   report_b64            (only when generateReport=true)
 */
export async function predictTumour(file, {
  generateReport  = false,
  patientId       = 'Unknown',
  patientName     = 'N/A',
  radiologist     = 'AI System',
  onUploadProgress,
} = {}) {
  const form = new FormData()
  form.append('file',            file)
  form.append('generate_report', String(generateReport))
  form.append('patient_id',      patientId)
  form.append('patient_name',    patientName)
  form.append('radiologist',     radiologist)

  const { data } = await http.post('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onUploadProgress && e.total) {
        onUploadProgress(Math.round((e.loaded / e.total) * 100))
      }
    },
  })
  return data
}

// ── Download PDF report ───────────────────────────────────────────────────────
/**
 * POST /api/report — triggers browser download of the PDF
 *
 * @param {File}   file
 * @param {object} opts  { patientId, patientName, radiologist }
 */
export async function downloadReport(file, {
  patientId   = 'Unknown',
  patientName = 'N/A',
  radiologist = 'AI System',
} = {}) {
  const form = new FormData()
  form.append('file',         file)
  form.append('patient_id',   patientId)
  form.append('patient_name', patientName)
  form.append('radiologist',  radiologist)

  const response = await http.post('/report', form, {
    headers:      { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob',
    timeout:      240_000,
  })

  // Trigger browser download
  const url  = URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }))
  const link = document.createElement('a')
  link.href     = url
  link.download = `brain_tumor_report_${patientId}.pdf`
  link.click()
  URL.revokeObjectURL(url)
}

// ── Batch predict ─────────────────────────────────────────────────────────────
/**
 * POST /api/batch
 * @param {File[]}   files        up to 10
 * @param {string[]} patientIds   positional, optional
 */
export async function batchPredict(files, patientIds = []) {
  const form = new FormData()
  files.forEach((f) => form.append('files', f))
  if (patientIds.length) form.append('patient_ids', patientIds.join(','))

  const { data } = await http.post('/batch', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 600_000,
  })
  return data
}

// ── SSE progress stream ────────────────────────────────────────────────────────
/**
 * GET /api/predict/stream — returns an EventSource for progress updates
 * @param {string} filename  display name for the scan
 */
export function openProgressStream(filename = 'scan') {
  return new EventSource(`/api/predict/stream?filename=${encodeURIComponent(filename)}`)
}

// ── Helpers ────────────────────────────────────────────────────────────────────
/** Convert a base64 PNG string to a data-URL src. */
export const b64Src = (b64) => (b64 ? `data:image/png;base64,${b64}` : null)
