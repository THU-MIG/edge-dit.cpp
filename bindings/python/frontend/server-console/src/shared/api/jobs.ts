import type { ConnectionTarget } from '@/shared/model/server'

import type {
  EdgeDitGenerationResult,
  EdgeDitJobCleanupResponse,
  EdgeDitJobDeletedResponse,
  EdgeDitJob,
  EdgeDitJobListResponse,
  JobKind,
  JobStatus,
} from '../model/jobs'
import { requestBlob, requestJson } from './client'

interface ListJobsOptions {
  kind?: JobKind
  limit?: number
  status?: JobStatus
}

function buildListJobsSuffix({ kind, limit, status }: ListJobsOptions) {
  const params = new URLSearchParams()

  if (status) {
    params.set('status', status)
  }
  if (kind) {
    params.set('kind', kind)
  }
  if (typeof limit === 'number') {
    params.set('limit', String(limit))
  }

  const query = params.toString()
  return `/jobs${query ? `?${query}` : ''}`
}

export function createImageJob(target: ConnectionTarget, payload: Record<string, unknown>) {
  return requestJson<EdgeDitJob>(target, '/images/generations', {
    body: JSON.stringify(payload),
    method: 'POST',
  })
}

export function createVideoJob(target: ConnectionTarget, payload: Record<string, unknown>) {
  return requestJson<EdgeDitJob>(target, '/videos/generations', {
    body: JSON.stringify(payload),
    method: 'POST',
  })
}

export function listJobs(target: ConnectionTarget, options: ListJobsOptions = {}) {
  return requestJson<EdgeDitJobListResponse>(target, buildListJobsSuffix(options))
}

export function getJob(target: ConnectionTarget, jobId: string) {
  return requestJson<EdgeDitJob>(target, `/jobs/${jobId}`)
}

export function getJobResult(target: ConnectionTarget, jobId: string) {
  return requestJson<EdgeDitGenerationResult>(target, `/jobs/${jobId}/result`)
}

export async function downloadVideo(target: ConnectionTarget, jobId: string, fps: number) {
  const response = await requestBlob(
    target,
    `/jobs/${encodeURIComponent(jobId)}/video?fps=${encodeURIComponent(String(fps))}`,
  )
  const filenameMatch = response.contentDisposition?.match(/filename="?([^";]+)"?/i)
  return {
    blob: response.blob,
    filename: filenameMatch?.[1] ?? `${jobId}.mp4`,
  }
}

export function cancelJob(target: ConnectionTarget, jobId: string) {
  return requestJson<EdgeDitJob>(target, `/jobs/${jobId}/cancel`, {
    method: 'POST',
  })
}

export function deleteJob(target: ConnectionTarget, jobId: string) {
  return requestJson<EdgeDitJobDeletedResponse>(target, `/jobs/${jobId}`, {
    method: 'DELETE',
  })
}

export function cleanupJobs(target: ConnectionTarget, options?: { nowMs?: number }) {
  return requestJson<EdgeDitJobCleanupResponse>(
    target,
    '/jobs/cleanup',
    options?.nowMs === undefined
      ? {
          method: 'POST',
        }
      : {
          body: JSON.stringify({ now_ms: options.nowMs }),
          method: 'POST',
        },
  )
}
