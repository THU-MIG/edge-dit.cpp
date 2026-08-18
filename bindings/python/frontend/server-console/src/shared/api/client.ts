import { DEFAULT_CONNECTION_TARGET, normalizeConnectionTarget } from '@/shared/model/connection-target'
import type { CapabilitiesResponse, ConnectionTarget, HealthResponse } from '@/shared/model/server'

import { ApiNetworkError, parseApiError } from './errors'

function requestId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }

  return `req_${Date.now()}_${Math.random().toString(16).slice(2)}`
}

function shouldUseDevProxy(target: ConnectionTarget) {
  const normalized = normalizeConnectionTarget(target)
  return import.meta.env.DEV && normalized.baseUrl === DEFAULT_CONNECTION_TARGET.baseUrl
}

export function buildApiUrl(target: ConnectionTarget, suffix: string) {
  const normalized = normalizeConnectionTarget(target)
  const path = `${normalized.prefix}${suffix.startsWith('/') ? suffix : `/${suffix}`}`

  if (shouldUseDevProxy(normalized)) {
    return path
  }

  return `${normalized.baseUrl}${path}`
}

export async function requestJson<T>(target: ConnectionTarget, suffix: string, init?: RequestInit): Promise<T> {
  const url = buildApiUrl(target, suffix)
  const headers = new Headers(init?.headers)
  headers.set('Accept', 'application/json')
  headers.set('X-Request-ID', requestId())

  if (init?.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  let response: Response
  try {
    response = await fetch(url, {
      ...init,
      headers,
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Network request failed'
    throw new ApiNetworkError(message)
  }

  const responseRequestId = response.headers.get('X-Request-ID')
  const payload = (await response.json().catch(() => null)) as unknown

  if (!response.ok) {
    throw parseApiError(response.status, payload, responseRequestId)
  }

  return payload as T
}

export async function requestBlob(target: ConnectionTarget, suffix: string, init?: RequestInit) {
  const url = buildApiUrl(target, suffix)
  const headers = new Headers(init?.headers)
  headers.set('Accept', 'video/mp4, application/json')
  headers.set('X-Request-ID', requestId())

  let response: Response
  try {
    response = await fetch(url, {
      ...init,
      headers,
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Network request failed'
    throw new ApiNetworkError(message)
  }

  if (!response.ok) {
    const responseRequestId = response.headers.get('X-Request-ID')
    const payload = (await response.json().catch(() => null)) as unknown
    throw parseApiError(response.status, payload, responseRequestId)
  }

  return {
    blob: await response.blob(),
    contentDisposition: response.headers.get('Content-Disposition'),
  }
}

export function getHealth(target: ConnectionTarget) {
  return requestJson<HealthResponse>(target, '/health')
}

export function getCapabilities(target: ConnectionTarget) {
  return requestJson<CapabilitiesResponse>(target, '/capabilities')
}
