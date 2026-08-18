export type JobKind = 'image' | 'video'

export type JobStatus = 'queued' | 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled'

interface EdgeDitJobBase {
  object: 'edge_dit.job'
  id: string
  kind: JobKind
  model: string
  status: JobStatus
  created_ms: number
  started_ms: number | null
  finished_ms: number | null
  expires_ms: number | null
  cancel_requested: boolean
  parameters: Record<string, unknown>
  error: string | null
  status_url: string
  cancel_url: string
  result_url: string
  request_id: string
}

export interface EdgeDitJobSummary extends EdgeDitJobBase {}

export interface EdgeDitJob extends EdgeDitJobBase {
  progress: {
    current_step: number
    total_steps: number
  }
}

export interface EdgeDitJobListResponse {
  object: 'list'
  data: EdgeDitJobSummary[]
  has_more: boolean
  request_id: string
}

export interface EdgeDitJobDeletedResponse {
  object: 'edge_dit.job_deleted'
  id: string
  kind: JobKind
  status: JobStatus
  request_id: string
}

export interface EdgeDitJobCleanupResponse {
  object: 'edge_dit.job_cleanup'
  removed_count: number
  removed_ids: string[]
  request_id: string
}

export interface EdgeDitImageResultItem {
  b64_png: string
  metadata: {
    index: number
    width: number
    height: number
    channels: number
    format: string
  }
}

export interface EdgeDitImageGenerationResult {
  object: 'edge_dit.image_generation'
  id: string
  model: string
  created_ms: number
  completed_ms: number
  parameters: Record<string, unknown>
  data: EdgeDitImageResultItem[]
  request_id: string
}

export interface EdgeDitVideoGenerationResult {
  object: 'edge_dit.video_generation'
  id: string
  model: string
  created_ms: number
  completed_ms: number
  parameters: Record<string, unknown>
  frame_format: string
  frames: EdgeDitImageResultItem[]
  audio?: {
    b64_f32le: string
    sample_rate: number
    channels: number
    sample_count: number
  }
  request_id: string
}

export type EdgeDitGenerationResult = EdgeDitImageGenerationResult | EdgeDitVideoGenerationResult

export function isImageGenerationResult(result: EdgeDitGenerationResult): result is EdgeDitImageGenerationResult {
  return result.object === 'edge_dit.image_generation'
}

export function isVideoGenerationResult(result: EdgeDitGenerationResult): result is EdgeDitVideoGenerationResult {
  return result.object === 'edge_dit.video_generation'
}

export function isActiveJobStatus(status: JobStatus) {
  return status === 'queued' || status === 'running' || status === 'cancelling'
}

export function hasJobProgress(job: EdgeDitJob | EdgeDitJobSummary | null | undefined): job is EdgeDitJob {
  return Boolean(job && 'progress' in job)
}
