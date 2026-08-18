import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Activity,
  Bell,
  Box,
  Braces,
  CircleDot,
  ClipboardList,
  Copy,
  Download,
  Image,
  Layers3,
  MonitorUp,
  Pause,
  Play,
  RefreshCcw,
  RotateCcw,
  Server,
  SkipBack,
  SkipForward,
  Sparkles,
  TimerReset,
  Trash2,
  Video,
} from 'lucide-react'
import {
  startTransition,
  useEffect,
  useEffectEvent,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type Dispatch,
  type KeyboardEvent,
  type ReactNode,
  type SetStateAction,
} from 'react'
import { toast } from 'sonner'

import {
  DEFAULT_EDIT_IMAGE_DRAFT,
  buildEditImageDraftFromPayload,
  buildEditImagePayload,
  validateEditImageDraft,
  type EditImageBindingField,
  type EditImageDraft,
} from '@/features/composer/edit-image-draft'
import {
  DEFAULT_IMAGE_DRAFT,
  buildImagePayload,
  stringifyPayload,
  validateImageDraft,
  type ImageDraft,
} from '@/features/composer/image-draft'
import {
  DEFAULT_VIDEO_DRAFT,
  buildVideoPayload,
  validateVideoDraft,
  type VideoDraft,
} from '@/features/composer/video-draft'
import { useConnectionTarget } from '@/features/connection/use-connection-target'
import { decodeImageResult } from '@/features/result-viewer/image-decoder'
import { decodeVideoResult } from '@/features/result-viewer/video-decoder'
import { getCapabilities, getHealth } from '@/shared/api/client'
import { ApiClientError, describeApiFailure } from '@/shared/api/errors'
import {
  cancelJob,
  cleanupJobs,
  createImageJob,
  createVideoJob,
  deleteJob,
  downloadVideo,
  getJob,
  getJobResult,
  listJobs,
} from '@/shared/api/jobs'
import {
  getManagedRuntimeStatus,
  restartManagedRuntime,
  startManagedRuntimeProfile,
  stopManagedRuntime,
} from '@/shared/api/runtime'
import { cn } from '@/shared/lib/cn'
import type {
  EdgeDitGenerationResult,
  EdgeDitJob,
  EdgeDitJobListResponse,
  EdgeDitJobSummary,
  JobStatus,
} from '@/shared/model/jobs'
import {
  hasJobProgress,
  isActiveJobStatus,
  isImageGenerationResult,
  isVideoGenerationResult,
} from '@/shared/model/jobs'
import {
  API_PREFIXES,
  type ActivityLogEntry,
  type CapabilitiesResponse,
  type ConnectionStatus,
  type ConnectionTarget,
} from '@/shared/model/server'
import type {
  ManagedBackendStatus,
  ManagedRuntimeProfile,
  ManagedRuntimeStatusResponse,
} from '@/shared/model/runtime'
import { JsonCodeEditor } from '@/shared/ui/json-code-editor'
import { LogTailViewer } from '@/shared/ui/log-tail-viewer'

const placeholderLogRows = [
  { time: 'local', method: 'GET', path: '/jobs', status: 'later' },
  { time: 'local', method: 'POST', path: '/images/generations', status: 'later' },
  { time: 'local', method: 'GET', path: '/jobs/{id}/result', status: 'later' },
] as const

const EMPTY_MANAGED_RUNTIME_PROFILES: ManagedRuntimeProfile[] = []

type ComposerMode = 'image' | 'edit_image' | 'video'
type EditImageTransport = EditImageBindingField | null
type BrowserNotificationState = NotificationPermission | 'unsupported'

export function ConsoleShell() {
  const queryClient = useQueryClient()
  const composerColumnRef = useRef<HTMLElement | null>(null)
  const { setTarget, target: persistedTarget } = useConnectionTarget()
  const [draftTarget, setDraftTarget] = useState<ConnectionTarget>(persistedTarget)
  const [activeTarget, setActiveTarget] = useState<ConnectionTarget>(persistedTarget)
  const [attempt, setAttempt] = useState(0)
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [composerMode, setComposerMode] = useState<ComposerMode>('image')

  const [imageDraft, setImageDraft] = useState<ImageDraft>(DEFAULT_IMAGE_DRAFT)
  const [imageDraftErrors, setImageDraftErrors] = useState<Partial<Record<keyof ImageDraft, string>>>({})
  const [editImageDraft, setEditImageDraft] = useState<EditImageDraft>(DEFAULT_EDIT_IMAGE_DRAFT)
  const [editImageDraftErrors, setEditImageDraftErrors] = useState<Partial<Record<keyof EditImageDraft, string>>>({})
  const [videoDraft, setVideoDraft] = useState<VideoDraft>(DEFAULT_VIDEO_DRAFT)
  const [videoDraftErrors, setVideoDraftErrors] = useState<Partial<Record<keyof VideoDraft, string>>>({})

  const [imageRawJsonBuffer, setImageRawJsonBuffer] = useState(() =>
    stringifyPayload(buildImagePayload(DEFAULT_IMAGE_DRAFT)),
  )
  const [editImageRawJsonBuffer, setEditImageRawJsonBuffer] = useState(() =>
    stringifyPayload(buildEditImagePayload(DEFAULT_EDIT_IMAGE_DRAFT, 'init_image_b64')),
  )
  const [videoRawJsonBuffer, setVideoRawJsonBuffer] = useState(() =>
    stringifyPayload(buildVideoPayload(DEFAULT_VIDEO_DRAFT)),
  )
  const [imageRawDetached, setImageRawDetached] = useState(false)
  const [editImageRawDetached, setEditImageRawDetached] = useState(false)
  const [videoRawDetached, setVideoRawDetached] = useState(false)
  const [submitErrorText, setSubmitErrorText] = useState<string | null>(null)

  const [selectedMediaIndex, setSelectedMediaIndex] = useState(0)
  const [isVideoPlaying, setIsVideoPlaying] = useState(false)
  const [videoLoop, setVideoLoop] = useState(true)
  const [videoFps, setVideoFps] = useState(4)

  const [actionLogRows, setActionLogRows] = useState<ActivityLogEntry[]>([])
  const [selectedManagedProfileSlug, setSelectedManagedProfileSlug] = useState('flux-dev')
  const [notificationPermission, setNotificationPermission] = useState<BrowserNotificationState>(
    getBrowserNotificationPermission,
  )
  const loggedTerminalStatuses = useRef(new Set<string>())
  const loggedResultRequestIds = useRef(new Set<string>())
  const seenRuntimeEventIds = useRef(new Set<string>())
  const runtimeEventsPrimed = useRef(false)
  const runtimeAvailability = useRef<'offline' | 'online' | 'unknown'>('unknown')

  const managedRuntimeQuery = useQuery({
    queryKey: ['managed-runtime-status'],
    queryFn: getManagedRuntimeStatus,
    refetchInterval: (query) => {
      const data = query.state.data as ManagedRuntimeStatusResponse | undefined
      const status = data?.backend.status
      return status === 'starting' || status === 'restarting' ? 1_000 : 2_500
    },
    retry: 0,
  })

  const healthQuery = useQuery({
    queryKey: ['connection', 'health', activeTarget.baseUrl, activeTarget.prefix, attempt],
    queryFn: () => getHealth(activeTarget),
  })

  const capabilitiesQuery = useQuery({
    queryKey: ['connection', 'capabilities', activeTarget.baseUrl, activeTarget.prefix, attempt],
    queryFn: () => getCapabilities(activeTarget),
    enabled: healthQuery.isSuccess,
  })

  const connectionStatus: ConnectionStatus = healthQuery.isPending
    ? 'connecting'
    : healthQuery.isError
      ? 'health_failed'
      : capabilitiesQuery.isPending
        ? 'connecting'
        : capabilitiesQuery.isError
          ? 'capabilities_failed'
          : capabilitiesQuery.isSuccess
            ? 'connected'
            : 'idle'

  const composerEnabled = connectionStatus === 'connected'
  const supportsImage = capabilitiesQuery.data?.supports.image ?? false
  const supportsVideo = capabilitiesQuery.data?.supports.video ?? false
  const managedRuntime = managedRuntimeQuery.data ?? null
  const managedRuntimeBackend = managedRuntime?.backend ?? null
  const managedRuntimeProfiles = managedRuntime?.profiles ?? EMPTY_MANAGED_RUNTIME_PROFILES
  const selectedManagedProfile =
    managedRuntimeProfiles.find((profile) => profile.slug === selectedManagedProfileSlug) ??
    managedRuntimeBackend?.profile ??
    managedRuntimeProfiles[0] ??
    null
  const managedRuntimeIsOnline = managedRuntimeQuery.isSuccess
  const managedRuntimeRecommendedTarget = managedRuntime?.recommended_connection_target ?? null
  const isUsingManagedTarget = managedRuntimeRecommendedTarget
    ? isSameConnectionTarget(activeTarget, managedRuntimeRecommendedTarget)
    : false
  const managedRuntimeHint = buildManagedRuntimeHint(managedRuntimeBackend, managedRuntimeQuery.error)
  const editImageBinding = useMemo(
    () => deriveEditImageBinding(selectedManagedProfile, managedRuntimeBackend?.profile ?? null, capabilitiesQuery.data),
    [capabilitiesQuery.data, managedRuntimeBackend?.profile, selectedManagedProfile],
  )
  const supportsEditImage = editImageBinding.transport !== null
  const lastAutoEditProfileSlug = useRef<string | null>(null)

  function refreshConnectionSession(nextTarget: ConnectionTarget, options?: { clearActivity?: boolean }) {
    startTransition(() => {
      setTarget(nextTarget)
      setDraftTarget(nextTarget)
      setActiveTarget(nextTarget)
      setAttempt((value) => value + 1)
      setSelectedJobId(null)
      setSubmitErrorText(null)
      if (options?.clearActivity ?? true) {
        setActionLogRows([])
      }
      loggedTerminalStatuses.current.clear()
      loggedResultRequestIds.current.clear()
      setSelectedMediaIndex(0)
      setIsVideoPlaying(false)
    })
  }

  useEffect(() => {
    const backendSlug = managedRuntimeBackend?.profile?.slug
    setSelectedManagedProfileSlug((current) => {
      if (backendSlug && managedRuntimeProfiles.some((profile) => profile.slug === backendSlug)) {
        return backendSlug
      }
      if (current && managedRuntimeProfiles.some((profile) => profile.slug === current)) {
        return current
      }
      return managedRuntimeProfiles[0]?.slug ?? current
    })
  }, [managedRuntimeBackend?.profile?.slug, managedRuntimeProfiles])

  useEffect(() => {
    if (composerMode === 'edit_image' && !supportsEditImage) {
      setComposerMode(supportsImage ? 'image' : supportsVideo ? 'video' : 'image')
      return
    }
    if (composerMode === 'video' && !supportsVideo && supportsImage) {
      setComposerMode(supportsEditImage ? 'edit_image' : 'image')
      return
    }
    if ((composerMode === 'image' || composerMode === 'edit_image') && !supportsImage && supportsVideo) {
      setComposerMode('video')
    }
  }, [composerMode, supportsEditImage, supportsImage, supportsVideo])

  useEffect(() => {
    const nextProfileSlug = managedRuntimeBackend?.profile?.slug ?? null
    if (!nextProfileSlug) {
      return
    }
    if (lastAutoEditProfileSlug.current === nextProfileSlug) {
      return
    }
    lastAutoEditProfileSlug.current = nextProfileSlug
    if (supportsEditImage && composerMode === 'image') {
      setComposerMode('edit_image')
    }
  }, [composerMode, managedRuntimeBackend?.profile?.slug, supportsEditImage])

  const resolvedEditTransport = editImageBinding.transport ?? 'init_image_b64'
  const imagePayload = useMemo(() => buildImagePayload(imageDraft), [imageDraft])
  const imagePayloadText = useMemo(() => stringifyPayload(imagePayload), [imagePayload])
  const editImagePayload = useMemo(
    () => buildEditImagePayload(editImageDraft, resolvedEditTransport),
    [editImageDraft, resolvedEditTransport],
  )
  const editImagePayloadText = useMemo(() => stringifyPayload(editImagePayload), [editImagePayload])
  const videoPayload = useMemo(() => buildVideoPayload(videoDraft), [videoDraft])
  const videoPayloadText = useMemo(() => stringifyPayload(videoPayload), [videoPayload])

  useEffect(() => {
    if (!imageRawDetached) {
      setImageRawJsonBuffer(imagePayloadText)
    }
  }, [imagePayloadText, imageRawDetached])

  useEffect(() => {
    if (!editImageRawDetached) {
      setEditImageRawJsonBuffer(editImagePayloadText)
    }
  }, [editImagePayloadText, editImageRawDetached])

  useEffect(() => {
    if (!videoRawDetached) {
      setVideoRawJsonBuffer(videoPayloadText)
    }
  }, [videoPayloadText, videoRawDetached])

  const currentRawJsonBuffer =
    composerMode === 'image'
      ? imageRawJsonBuffer
      : composerMode === 'edit_image'
        ? editImageRawJsonBuffer
        : videoRawJsonBuffer
  const currentRawDetached =
    composerMode === 'image' ? imageRawDetached : composerMode === 'edit_image' ? editImageRawDetached : videoRawDetached
  const currentPayloadPreview = safeParseJson(currentRawJsonBuffer)

  const jobsListQuery = useQuery({
    queryKey: ['jobs', activeTarget.baseUrl, activeTarget.prefix, attempt],
    queryFn: () => listJobs(activeTarget, { limit: 20 }),
    enabled: composerEnabled,
    refetchInterval: (query) => {
      const data = query.state.data as EdgeDitJobListResponse | undefined
      return data?.data.some((job) => isActiveJobStatus(job.status)) ? 2_500 : false
    },
  })

  useEffect(() => {
    if (!selectedJobId && jobsListQuery.data?.data[0]) {
      setSelectedJobId(jobsListQuery.data.data[0].id)
    }
  }, [jobsListQuery.data, selectedJobId])

  const selectedJobSummary = jobsListQuery.data?.data.find((job) => job.id === selectedJobId) ?? null

  const selectedJobQuery = useQuery({
    queryKey: ['job', activeTarget.baseUrl, activeTarget.prefix, attempt, selectedJobId],
    queryFn: () => getJob(activeTarget, selectedJobId!),
    enabled: composerEnabled && Boolean(selectedJobId),
    refetchInterval: (query) => {
      const data = query.state.data as EdgeDitJob | undefined
      return data && isActiveJobStatus(data.status) ? 1_000 : false
    },
  })

  const selectedJob = selectedJobQuery.data ?? selectedJobSummary

  const selectedResultQuery = useQuery({
    queryKey: ['job-result', activeTarget.baseUrl, activeTarget.prefix, attempt, selectedJobId],
    queryFn: () => getJobResult(activeTarget, selectedJobId!),
    enabled: composerEnabled && selectedJob?.status === 'succeeded',
    refetchInterval: (query) => {
      if (query.state.data) {
        return false
      }
      const error = query.state.error
      return error instanceof ApiClientError && error.code === 'job_not_ready' ? 1_000 : false
    },
    retry: false,
  })

  const decodedImagesState = useMemo(() => {
    try {
      return {
        error: null as string | null,
        items:
          selectedResultQuery.data && isImageGenerationResult(selectedResultQuery.data)
            ? decodeImageResult(selectedResultQuery.data)
            : [],
      }
    } catch (error) {
      return {
        error: describeApiFailure(error),
        items: [],
      }
    }
  }, [selectedResultQuery.data])

  const decodedVideoFramesState = useMemo(() => {
    try {
      return {
        error: null as string | null,
        items:
          selectedResultQuery.data && isVideoGenerationResult(selectedResultQuery.data)
            ? decodeVideoResult(selectedResultQuery.data)
            : [],
      }
    } catch (error) {
      return {
        error: describeApiFailure(error),
        items: [],
      }
    }
  }, [selectedResultQuery.data])

  const currentDecodedCount = selectedJob?.kind === 'video'
    ? decodedVideoFramesState.items.length
    : decodedImagesState.items.length

  useEffect(() => {
    setSelectedMediaIndex(0)
    setIsVideoPlaying(false)
  }, [selectedResultQuery.data?.id])

  useEffect(() => {
    if (selectedMediaIndex >= currentDecodedCount && currentDecodedCount > 0) {
      setSelectedMediaIndex(0)
    }
  }, [currentDecodedCount, selectedMediaIndex])

  useEffect(() => {
    if (!isVideoPlaying || decodedVideoFramesState.items.length < 2) {
      return
    }

    const intervalId = window.setInterval(() => {
      setSelectedMediaIndex((index) => {
        const next = index + 1
        if (next < decodedVideoFramesState.items.length) {
          return next
        }
        if (videoLoop) {
          return 0
        }
        setIsVideoPlaying(false)
        return index
      })
    }, Math.max(120, Math.round(1000 / videoFps)))

    return () => {
      window.clearInterval(intervalId)
    }
  }, [decodedVideoFramesState.items.length, isVideoPlaying, videoFps, videoLoop])

  useEffect(() => {
    const job = selectedJobQuery.data
    if (!job || isActiveJobStatus(job.status)) {
      return
    }
    const key = `${job.id}:${job.status}`
    if (loggedTerminalStatuses.current.has(key)) {
      return
    }
    loggedTerminalStatuses.current.add(key)
    appendActivityLog(setActionLogRows, {
      method: 'GET',
      path: `${activeTarget.prefix}/jobs/${job.id}`,
      requestId: job.request_id,
      status: `${job.status}${job.error ? ` · ${job.error}` : ''}`,
    })
  }, [activeTarget.prefix, selectedJobQuery.data])

  useEffect(() => {
    const result = selectedResultQuery.data
    if (!result || loggedResultRequestIds.current.has(result.request_id)) {
      return
    }
    loggedResultRequestIds.current.add(result.request_id)
    appendActivityLog(setActionLogRows, {
      method: 'GET',
      path: `${activeTarget.prefix}/jobs/${result.id}/result`,
      requestId: result.request_id,
      status: summarizeResult(result),
    })
  }, [activeTarget.prefix, selectedResultQuery.data])

  const sendBrowserNotification = useEffectEvent((title: string, message: string) => {
    maybeSendBrowserNotification(notificationPermission, title, message)
  })

  const emitRuntimeEventFeedback = useEffectEvent(
    (event: ManagedRuntimeStatusResponse['recent_events'][number]) => {
      if (event.type === 'backend_ready' || event.type === 'backend_recovered') {
        if (managedRuntimeRecommendedTarget && isSameConnectionTarget(activeTarget, managedRuntimeRecommendedTarget)) {
          // A managed model switch keeps the same base URL, so force a fresh
          // health/capabilities/job pass once the new backend is actually ready.
          refreshConnectionSession(managedRuntimeRecommendedTarget, { clearActivity: false })
        }
        toast.success(event.message)
        sendBrowserNotification('Managed backend ready', event.message)
        return
      }

      if (event.level === 'error') {
        toast.error(event.message)
        sendBrowserNotification('Managed backend issue', event.message)
        return
      }

      if (event.level === 'warning') {
        toast.warning(event.message)
        sendBrowserNotification('Managed backend warning', event.message)
      }
    },
  )

  useEffect(() => {
    const nextAvailability = managedRuntimeQuery.isSuccess
      ? 'online'
      : managedRuntimeQuery.isError
        ? 'offline'
        : 'unknown'
    const previousAvailability = runtimeAvailability.current

    if (previousAvailability === nextAvailability) {
      return
    }

    if (previousAvailability === 'online' && nextAvailability === 'offline') {
      toast.error('Local runtime manager is unreachable. Crash diagnostics and self-heal feedback are temporarily unavailable.')
      sendBrowserNotification(
        'Runtime manager offline',
        'Crash diagnostics and self-heal feedback are temporarily unavailable.',
      )
    } else if (previousAvailability === 'offline' && nextAvailability === 'online') {
      toast.success('Local runtime manager is reachable again.')
    }

    runtimeAvailability.current = nextAvailability
  }, [managedRuntimeQuery.isError, managedRuntimeQuery.isSuccess])

  useEffect(() => {
    const recentEvents = managedRuntime?.recent_events
    if (!recentEvents || recentEvents.length === 0) {
      return
    }

    if (!runtimeEventsPrimed.current) {
      for (const event of recentEvents) {
        seenRuntimeEventIds.current.add(event.id)
      }
      runtimeEventsPrimed.current = true
      return
    }

    for (const event of [...recentEvents].sort((left, right) => left.time_ms - right.time_ms)) {
      if (seenRuntimeEventIds.current.has(event.id)) {
        continue
      }
      seenRuntimeEventIds.current.add(event.id)
      emitRuntimeEventFeedback(event)
    }
  }, [managedRuntime?.recent_events])

  async function handleAcceptedJob(job: EdgeDitJob, path: string, label: string) {
    setSubmitErrorText(null)
    setSelectedJobId(job.id)
    appendActivityLog(setActionLogRows, {
      method: 'POST',
      path,
      requestId: job.request_id,
      status: `${job.status} · ${job.id}`,
    })
    toast.success(`${label} job queued: ${job.id}`)
    await queryClient.invalidateQueries({
      queryKey: ['jobs', activeTarget.baseUrl, activeTarget.prefix, attempt],
    })
    await queryClient.invalidateQueries({
      queryKey: ['job', activeTarget.baseUrl, activeTarget.prefix, attempt, job.id],
    })
  }

  async function refreshJobQueries(jobId?: string | null) {
    await queryClient.invalidateQueries({
      queryKey: ['jobs', activeTarget.baseUrl, activeTarget.prefix, attempt],
    })

    if (!jobId) {
      return
    }

    await queryClient.invalidateQueries({
      queryKey: ['job', activeTarget.baseUrl, activeTarget.prefix, attempt, jobId],
    })
    await queryClient.invalidateQueries({
      queryKey: ['job-result', activeTarget.baseUrl, activeTarget.prefix, attempt, jobId],
    })
  }

  function clearSelectedJobArtifacts(jobId: string) {
    if (selectedJobId === jobId) {
      setSelectedJobId(null)
      setIsVideoPlaying(false)
    }

    queryClient.removeQueries({
      queryKey: ['job', activeTarget.baseUrl, activeTarget.prefix, attempt, jobId],
    })
    queryClient.removeQueries({
      queryKey: ['job-result', activeTarget.baseUrl, activeTarget.prefix, attempt, jobId],
    })
  }

  const createImageMutation = useMutation({
    mutationFn: (payload: Record<string, unknown>) => createImageJob(activeTarget, payload),
    onError: (error) => {
      const message = describeApiFailure(error)
      setSubmitErrorText(message)
      appendActivityLog(setActionLogRows, {
        method: 'POST',
        path: `${activeTarget.prefix}/images/generations`,
        status: message,
      })
    },
    onSuccess: async (job) => {
      await handleAcceptedJob(job, `${activeTarget.prefix}/images/generations`, 'Image')
    },
  })

  const createVideoMutation = useMutation({
    mutationFn: (payload: Record<string, unknown>) => createVideoJob(activeTarget, payload),
    onError: (error) => {
      const message = describeApiFailure(error)
      setSubmitErrorText(message)
      appendActivityLog(setActionLogRows, {
        method: 'POST',
        path: `${activeTarget.prefix}/videos/generations`,
        status: message,
      })
    },
    onSuccess: async (job) => {
      await handleAcceptedJob(job, `${activeTarget.prefix}/videos/generations`, 'Video')
    },
  })

  const downloadVideoMutation = useMutation({
    mutationFn: ({ fps, jobId }: { fps: number; jobId: string }) => downloadVideo(activeTarget, jobId, fps),
    onError: (error) => {
      toast.error(`Video download failed: ${describeApiFailure(error)}`)
    },
    onSuccess: ({ blob, filename }) => {
      saveBlobDownload(blob, filename)
      toast.success(`Saved video: ${filename}`)
    },
  })

  const cancelJobMutation = useMutation({
    mutationFn: (jobId: string) => cancelJob(activeTarget, jobId),
    onError: (error, jobId) => {
      const message = describeApiFailure(error)
      appendActivityLog(setActionLogRows, {
        method: 'POST',
        path: `${activeTarget.prefix}/jobs/${jobId}/cancel`,
        status: message,
      })
      toast.error(`Cancel failed: ${message}`)
    },
    onSuccess: async (job) => {
      appendActivityLog(setActionLogRows, {
        method: 'POST',
        path: `${activeTarget.prefix}/jobs/${job.id}/cancel`,
        requestId: job.request_id,
        status: `${job.status} · ${job.id}`,
      })
      toast.success(job.status === 'cancelled' ? `Job cancelled: ${job.id}` : `Cancellation requested: ${job.id}`)
      await refreshJobQueries(job.id)
    },
  })

  const deleteJobMutation = useMutation({
    mutationFn: (jobId: string) => deleteJob(activeTarget, jobId),
    onError: (error, jobId) => {
      const message = describeApiFailure(error)
      appendActivityLog(setActionLogRows, {
        method: 'DELETE',
        path: `${activeTarget.prefix}/jobs/${jobId}`,
        status: message,
      })
      toast.error(`Delete failed: ${message}`)
    },
    onSuccess: async (deleted) => {
      clearSelectedJobArtifacts(deleted.id)
      appendActivityLog(setActionLogRows, {
        method: 'DELETE',
        path: `${activeTarget.prefix}/jobs/${deleted.id}`,
        requestId: deleted.request_id,
        status: `${deleted.status} removed`,
      })
      toast.success(`Removed job: ${deleted.id}`)
      await refreshJobQueries()
    },
  })

  const cleanupJobsMutation = useMutation({
    mutationFn: () => cleanupJobs(activeTarget),
    onError: (error) => {
      const message = describeApiFailure(error)
      appendActivityLog(setActionLogRows, {
        method: 'POST',
        path: `${activeTarget.prefix}/jobs/cleanup`,
        status: message,
      })
      toast.error(`Cleanup failed: ${message}`)
    },
    onSuccess: async (cleanup) => {
      if (selectedJobId && cleanup.removed_ids.includes(selectedJobId)) {
        clearSelectedJobArtifacts(selectedJobId)
      }
      appendActivityLog(setActionLogRows, {
        method: 'POST',
        path: `${activeTarget.prefix}/jobs/cleanup`,
        requestId: cleanup.request_id,
        status: `${cleanup.removed_count} expired removed`,
      })
      toast.success(
        cleanup.removed_count > 0
          ? `Removed ${cleanup.removed_count} expired job${cleanup.removed_count === 1 ? '' : 's'}`
          : 'No expired jobs to clean up',
      )
      await refreshJobQueries()
    },
  })

  function resetConsoleSession(nextTarget: ConnectionTarget) {
    refreshConnectionSession(nextTarget, { clearActivity: true })
  }

  function syncManagedRuntimeSnapshot(status: ManagedRuntimeStatusResponse) {
    queryClient.setQueryData(['managed-runtime-status'], status)
    resetConsoleSession(status.recommended_connection_target)
  }

  const startManagedRuntimeMutation = useMutation({
    mutationFn: (profileSlug: string) => startManagedRuntimeProfile(profileSlug),
    onError: (error) => {
      toast.error(`Start failed: ${describeApiFailure(error)}`)
    },
    onSuccess: (status, profileSlug) => {
      setSelectedManagedProfileSlug(status.backend.profile?.slug ?? profileSlug)
      syncManagedRuntimeSnapshot(status)
      const profile = status.profiles.find((item) => item.slug === (status.backend.profile?.slug ?? profileSlug))
      toast.success(`Managed profile starting: ${profile?.name ?? profileSlug}`)
    },
  })

  const restartManagedRuntimeMutation = useMutation({
    mutationFn: () => restartManagedRuntime(),
    onError: (error) => {
      toast.error(`Restart failed: ${describeApiFailure(error)}`)
    },
    onSuccess: (status) => {
      syncManagedRuntimeSnapshot(status)
      toast.success('Managed backend restart requested')
    },
  })

  const stopManagedRuntimeMutation = useMutation({
    mutationFn: () => stopManagedRuntime(),
    onError: (error) => {
      toast.error(`Stop failed: ${describeApiFailure(error)}`)
    },
    onSuccess: (status) => {
      syncManagedRuntimeSnapshot(status)
      toast.message('Managed backend stop requested')
    },
  })

  const createPending = composerMode === 'image' ? createImageMutation.isPending : createVideoMutation.isPending

  const capabilityRows = useMemo(() => buildCapabilityRows(capabilitiesQuery.data), [capabilitiesQuery.data])
  const connectionLogRows = useMemo(
    () =>
      buildConnectionLogRows(
        activeTarget,
        healthQuery.data,
        capabilitiesQuery.data,
        healthQuery.error,
        capabilitiesQuery.error,
      ),
    [activeTarget, capabilitiesQuery.data, capabilitiesQuery.error, healthQuery.data, healthQuery.error],
  )
  const activityRows = useMemo(
    () => [...actionLogRows, ...connectionLogRows].slice(0, 12),
    [actionLogRows, connectionLogRows],
  )

  const healthErrorText = healthQuery.error ? describeApiFailure(healthQuery.error) : null
  const capabilitiesErrorText = capabilitiesQuery.error ? describeApiFailure(capabilitiesQuery.error) : null
  const decodeErrorText = selectedJob?.kind === 'video' ? decodedVideoFramesState.error : decodedImagesState.error
  const resultErrorText =
    selectedResultQuery.error instanceof ApiClientError && selectedResultQuery.error.code === 'job_not_ready'
      ? null
      : selectedResultQuery.error
        ? describeApiFailure(selectedResultQuery.error)
        : decodeErrorText
  const resultWaitingForReadiness =
    selectedResultQuery.isPending ||
    (selectedResultQuery.error instanceof ApiClientError && selectedResultQuery.error.code === 'job_not_ready')
  const managedEditLogStepProgress = deriveManagedEditLogStepProgress(
    managedRuntime?.log_tail ?? [],
    selectedManagedProfile,
    managedRuntimeBackend?.profile ?? null,
    capabilitiesQuery.data,
  )
  const progressVm = deriveProgressVm(selectedJob, managedEditLogStepProgress)
  const managedRuntimeErrorText = managedRuntimeQuery.error ? describeApiFailure(managedRuntimeQuery.error) : null
  const managedRuntimeBusy =
    startManagedRuntimeMutation.isPending ||
    restartManagedRuntimeMutation.isPending ||
    stopManagedRuntimeMutation.isPending
  const canRestartManaged = Boolean(managedRuntimeBackend?.profile) && !managedRuntimeBusy
  const canStopManaged =
    Boolean(managedRuntimeBackend?.profile) &&
    managedRuntimeBackend?.status !== 'idle' &&
    managedRuntimeBackend?.status !== 'stopping' &&
    !managedRuntimeBusy
  const selectedManagedAlreadyRunning =
    managedRuntimeBackend?.status === 'running' &&
    managedRuntimeBackend.profile?.slug === selectedManagedProfile?.slug
  const canStartManaged = Boolean(selectedManagedProfile) && !managedRuntimeBusy && !selectedManagedAlreadyRunning
  const startManagedLabel =
    managedRuntimeBackend?.profile?.slug && managedRuntimeBackend.profile.slug !== selectedManagedProfile?.slug
      ? 'Switch model'
      : selectedManagedAlreadyRunning
        ? 'Running'
        : 'Start model'

  function handleConnect() {
    const nextTarget = {
      baseUrl: draftTarget.baseUrl.trim() || persistedTarget.baseUrl,
      prefix: draftTarget.prefix,
    }
    resetConsoleSession(nextTarget)
  }

  function handleImageDraftChange(field: keyof ImageDraft, value: string) {
    setImageDraft((current) => ({ ...current, [field]: value }))
    setImageDraftErrors((current) => ({ ...current, [field]: undefined }))
  }

  function handleEditImageDraftChange(field: keyof EditImageDraft, value: string) {
    setEditImageDraft((current) => ({ ...current, [field]: value }))
    setEditImageDraftErrors((current) => ({ ...current, [field]: undefined }))
  }

  function handleVideoDraftChange(field: keyof VideoDraft, value: string) {
    setVideoDraft((current) => ({ ...current, [field]: value }))
    setVideoDraftErrors((current) => ({ ...current, [field]: undefined }))
  }

  async function handleEditImageFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) {
      return
    }
    if (!file.type.startsWith('image/')) {
      setEditImageDraftErrors((current) => ({ ...current, inputImage: 'Please select a valid image file.' }))
      toast.error('Only image files can be used as edit inputs.')
      return
    }

    try {
      const inputImage = await readFileAsDataUrl(file)
      setEditImageDraft((current) => ({
        ...current,
        inputImage,
        inputImageName: file.name,
      }))
      setEditImageDraftErrors((current) => ({ ...current, inputImage: undefined }))
      setEditImageRawDetached(false)
      setSubmitErrorText(null)
      toast.success(`${editImageBinding.inputLabel} ready: ${file.name}`)
    } catch {
      setEditImageDraftErrors((current) => ({ ...current, inputImage: 'Failed to read the image, please try again.' }))
      toast.error('The selected image could not be read.')
    }
  }

  function handleClearEditImage() {
    setEditImageDraft((current) => ({
      ...current,
      inputImage: '',
      inputImageName: '',
    }))
    setEditImageDraftErrors((current) => ({ ...current, inputImage: undefined }))
    setEditImageRawDetached(false)
    setSubmitErrorText(null)
  }

  function handleJsonBufferChange(value: string) {
    if (composerMode === 'image') {
      setImageRawJsonBuffer(value)
      setImageRawDetached(true)
    } else if (composerMode === 'edit_image') {
      setEditImageRawJsonBuffer(value)
      setEditImageRawDetached(true)
    } else {
      setVideoRawJsonBuffer(value)
      setVideoRawDetached(true)
    }
    setSubmitErrorText(null)
  }

  function handleResetJsonBuffer() {
    if (composerMode === 'image') {
      setImageRawDetached(false)
      setImageRawJsonBuffer(imagePayloadText)
    } else if (composerMode === 'edit_image') {
      setEditImageRawDetached(false)
      setEditImageRawJsonBuffer(editImagePayloadText)
    } else {
      setVideoRawDetached(false)
      setVideoRawJsonBuffer(videoPayloadText)
    }
    setSubmitErrorText(null)
  }

  async function handleCopyJson() {
    try {
      await navigator.clipboard.writeText(currentRawJsonBuffer)
      toast.success('Payload JSON copied')
    } catch {
      toast.error('Clipboard is not available in this browser context')
    }
  }

  function handlePromptHotkey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault()
      void handleSubmitCurrentJob()
    }
  }

  async function handleEnableNotifications() {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      toast.error('This browser does not support system notifications.')
      setNotificationPermission('unsupported')
      return
    }

    const permission = await window.Notification.requestPermission()
    setNotificationPermission(permission)

    if (permission === 'granted') {
      toast.success('System notifications enabled for runtime recovery events.')
      maybeSendBrowserNotification(
        'granted',
        'Notifications enabled',
        'You will now receive local runtime alerts in this browser.',
      )
      return
    }

    if (permission === 'denied') {
      toast.warning('System notifications were blocked by the browser.')
      return
    }

    toast.message('System notifications remain optional.')
  }

  function handleUseManagedTarget() {
    if (!managedRuntimeRecommendedTarget) {
      return
    }

    resetConsoleSession(managedRuntimeRecommendedTarget)
    toast.success('Console target synced to the managed backend.')
  }

  function handleApplyManagedPreset(profile: ManagedRuntimeProfile | null) {
    if (!profile || !profile.request_example || Array.isArray(profile.request_example)) {
      toast.warning('This profile does not expose a verified request example yet.')
      return
    }

    const payload = profile.request_example as Record<string, unknown>
    if (profile.kind === 'video') {
      const nextDraft = buildVideoDraftFromPayload(payload)
      setVideoDraft(nextDraft)
      setVideoDraftErrors({})
      setVideoRawJsonBuffer(stringifyPayload(payload))
      setVideoRawDetached(!requestPayloadFitsDraft(payload, 'video'))
      setVideoFps(profile.slug === 'minimax-h3' ? 24 : 16)
      setComposerMode('video')
      toast.success('Verified video preset loaded into the video composer.')
      return
    }

    if (payloadHasEditImageInput(payload)) {
      const nextDraft = buildEditImageDraftFromPayload(payload)
      setEditImageDraft(nextDraft)
      setEditImageDraftErrors({})
      setEditImageRawJsonBuffer(stringifyPayload(payload))
      setEditImageRawDetached(!requestPayloadFitsDraft(payload, 'edit_image'))
      setComposerMode('edit_image')
      toast.success('Verified image-edit preset loaded into the edit composer.')
      return
    }

    const nextDraft = buildImageDraftFromPayload(payload)
    setImageDraft(nextDraft)
    setImageDraftErrors({})
    setImageRawJsonBuffer(stringifyPayload(payload))
    setImageRawDetached(!requestPayloadFitsDraft(payload, 'image'))
    setComposerMode('image')
    toast.success('Verified image preset loaded into the image composer.')
  }

  async function handleSubmitCurrentJob() {
    if (!composerEnabled || createPending) {
      return
    }

    if (composerMode === 'image') {
      if (!supportsImage) {
        return
      }
      if (!imageRawDetached) {
        const validationErrors = validateImageDraft(imageDraft)
        setImageDraftErrors(validationErrors)
        if (Object.keys(validationErrors).length > 0) {
          setSubmitErrorText('Please fix the field errors in the form first.')
          return
        }
      }
    } else if (composerMode === 'edit_image') {
      if (!supportsImage || !supportsEditImage) {
        return
      }
      if (!editImageRawDetached) {
        const validationErrors = validateEditImageDraft(editImageDraft)
        setEditImageDraftErrors(validationErrors)
        if (Object.keys(validationErrors).length > 0) {
          setSubmitErrorText('Please fix the field errors in the form first.')
          return
        }
      }
    } else {
      if (!supportsVideo) {
        return
      }
      if (!videoRawDetached) {
        const validationErrors = validateVideoDraft(videoDraft)
        setVideoDraftErrors(validationErrors)
        if (Object.keys(validationErrors).length > 0) {
          setSubmitErrorText('Please fix the field errors in the form first.')
          return
        }
      }
    }

    let payload: Record<string, unknown>
    try {
      const parsed = JSON.parse(currentRawJsonBuffer) as unknown
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        setSubmitErrorText('The request body must be a JSON object.')
        return
      }
      payload = parsed as Record<string, unknown>
    } catch {
      setSubmitErrorText('Invalid JSON format, please fix the syntax first.')
      return
    }

    setSubmitErrorText(null)
    if (composerMode === 'image' || composerMode === 'edit_image') {
      await createImageMutation.mutateAsync(payload)
    } else {
      await createVideoMutation.mutateAsync(payload)
    }
  }

  return (
    <main className="min-h-screen overflow-hidden bg-[var(--bg-app)] text-[var(--text-primary)]">
      <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_12%_10%,rgba(42,96,220,0.16),transparent_34%),radial-gradient(circle_at_76%_0%,rgba(36,128,116,0.12),transparent_30%),linear-gradient(135deg,rgba(255,255,255,0.55),rgba(232,235,230,0.18))]" />
      <div className="pointer-events-none fixed inset-0 -z-10 opacity-[0.35] [background-image:linear-gradient(rgba(30,45,65,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(30,45,65,0.05)_1px,transparent_1px)] [background-size:32px_32px]" />

      <section className="mx-auto flex min-h-screen w-full max-w-[1720px] flex-col px-5 py-5 lg:px-7">
        <ConsoleHeader
          activeTarget={activeTarget}
          capabilities={capabilitiesQuery.data}
          connectionStatus={connectionStatus}
          managedRuntimeIsOnline={managedRuntimeIsOnline}
          notificationPermission={notificationPermission}
        />

        <div className="grid flex-1 gap-4 pt-5 xl:grid-cols-[360px_minmax(580px,1fr)_420px]">
          <aside className="flex min-h-0 flex-col gap-4 overflow-y-auto pr-1">
            <ConnectionCard
              capabilitiesErrorText={capabilitiesErrorText}
              connectionStatus={connectionStatus}
              draftTarget={draftTarget}
              healthErrorText={healthErrorText}
              isBusy={connectionStatus === 'connecting'}
              managedRuntimeHint={managedRuntimeHint}
              onConnect={handleConnect}
              onDraftChange={setDraftTarget}
            />
            <LocalRuntimeCard
              busy={managedRuntimeBusy}
              canStart={canStartManaged}
              canRestart={canRestartManaged}
              canStop={canStopManaged}
              errorText={managedRuntimeErrorText}
              isOnline={managedRuntimeIsOnline}
              isUsingManagedTarget={isUsingManagedTarget}
              notificationPermission={notificationPermission}
              onApplyPreset={() => handleApplyManagedPreset(selectedManagedProfile)}
              onEnableNotifications={() => {
                void handleEnableNotifications()
              }}
              onRestart={() => {
                void restartManagedRuntimeMutation.mutateAsync()
              }}
              onSelectProfile={setSelectedManagedProfileSlug}
              onStart={() => {
                if (!selectedManagedProfile) {
                  return
                }
                void startManagedRuntimeMutation.mutateAsync(selectedManagedProfile.slug)
              }}
              onStop={() => {
                void stopManagedRuntimeMutation.mutateAsync()
              }}
              onUseManagedTarget={handleUseManagedTarget}
              profile={selectedManagedProfile}
              profiles={managedRuntimeProfiles}
              runtime={managedRuntime}
              startLabel={startManagedLabel}
            />
          </aside>

          <section className="flex min-h-0 flex-col gap-4 overflow-y-auto pr-1" ref={composerColumnRef}>
            <ComposerPanel
              composerEnabled={composerEnabled}
              composerMode={composerMode}
              createPending={createPending}
              editImageBinding={editImageBinding}
              editImageDraft={editImageDraft}
              editImageErrors={editImageDraftErrors}
              imageDraft={imageDraft}
              imageErrors={imageDraftErrors}
              onDraftModeChange={setComposerMode}
              onEditImageDraftChange={handleEditImageDraftChange}
              onEditImageFileChange={handleEditImageFileChange}
              onEditImageClear={handleClearEditImage}
              onImageDraftChange={handleImageDraftChange}
              onPromptHotkey={handlePromptHotkey}
              onSubmit={handleSubmitCurrentJob}
              onVideoDraftChange={handleVideoDraftChange}
              submitErrorText={submitErrorText}
              supportsEditImage={supportsEditImage}
              supportsImage={supportsImage}
              supportsVideo={supportsVideo}
              videoDraft={videoDraft}
              videoErrors={videoDraftErrors}
            />
            <ProgressPanel progressVm={progressVm} />
            <ResultPreview
              decodedImages={decodedImagesState.items}
              decodedVideoFrames={decodedVideoFramesState.items}
              downloadPending={downloadVideoMutation.isPending}
              isVideoPlaying={isVideoPlaying}
              job={selectedJob}
              onDownloadVideo={(jobId, fps) => {
                downloadVideoMutation.mutate({ fps, jobId })
              }}
              onFpsChange={setVideoFps}
              onLoopChange={setVideoLoop}
              onPlayChange={setIsVideoPlaying}
              result={selectedResultQuery.data}
              resultErrorText={resultErrorText}
              resultPending={resultWaitingForReadiness}
              selectedMediaIndex={selectedMediaIndex}
              setSelectedMediaIndex={setSelectedMediaIndex}
              videoFps={videoFps}
              videoLoop={videoLoop}
            />
          </section>

          <aside className="flex min-h-0 flex-col gap-4 overflow-y-auto pr-1">
            <JobsPanel
              cleanupPending={cleanupJobsMutation.isPending}
              deletePending={deleteJobMutation.isPending}
              isLoading={jobsListQuery.isPending}
              jobs={jobsListQuery.data?.data ?? []}
              onCancelSelected={() => {
                if (!selectedJobId) {
                  return
                }
                void cancelJobMutation.mutateAsync(selectedJobId)
              }}
              onCleanup={() => {
                void cleanupJobsMutation.mutateAsync()
              }}
              onDeleteSelected={() => {
                if (!selectedJobId) {
                  return
                }
                void deleteJobMutation.mutateAsync(selectedJobId)
              }}
              onRefresh={() => {
                void jobsListQuery.refetch()
                if (selectedJobId) {
                  void selectedJobQuery.refetch()
                }
              }}
              selectedJob={selectedJob}
              selectedJobId={selectedJobId}
              setSelectedJobId={setSelectedJobId}
              cancelPending={cancelJobMutation.isPending}
            />
            <PayloadPreviewPanel
              canSubmit={
                composerEnabled &&
                (composerMode === 'video'
                  ? supportsVideo
                  : composerMode === 'edit_image'
                    ? supportsImage && supportsEditImage
                    : supportsImage)
              }
              composerMode={composerMode}
              createPending={createPending}
              isRawDetached={currentRawDetached}
              onCopyJson={handleCopyJson}
              onRawJsonChange={handleJsonBufferChange}
              onResetJsonBuffer={handleResetJsonBuffer}
              onSubmit={handleSubmitCurrentJob}
              rawJsonBuffer={currentRawJsonBuffer}
            />
            <CapabilitiesCard
              capabilities={capabilitiesQuery.data}
              connectionStatus={connectionStatus}
              rows={capabilityRows}
            />
            <InspectorPanel
              capabilities={capabilitiesQuery.data}
              connectionStatus={connectionStatus}
              health={healthQuery.data}
              payloadPreview={currentPayloadPreview}
              result={selectedResultQuery.data}
              selectedJob={selectedJob}
            />
            <ActivityLog rows={activityRows.length > 0 ? activityRows : [...placeholderLogRows]} />
          </aside>
        </div>
      </section>
    </main>
  )
}

function ConsoleHeader({
  activeTarget,
  capabilities,
  connectionStatus,
  managedRuntimeIsOnline,
  notificationPermission,
}: {
  activeTarget: ConnectionTarget
  capabilities?: CapabilitiesResponse
  connectionStatus: ConnectionStatus
  managedRuntimeIsOnline: boolean
  notificationPermission: BrowserNotificationState
}) {
  return (
    <header className="rounded-[22px] border border-[var(--border-subtle)] bg-[var(--surface-shell)]/86 px-5 py-4 shadow-[var(--shadow-soft)] backdrop-blur-xl">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center gap-4">
          <div className="grid size-12 place-items-center rounded-2xl border border-[var(--accent-border)] bg-[var(--accent-soft)] text-[var(--accent)]">
            <MonitorUp className="size-6" />
          </div>
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-[var(--text-muted)]">
              edge-dit local console
            </p>
            <h1 className="mt-1 text-2xl font-semibold tracking-[-0.035em] text-[var(--text-primary)] sm:text-3xl">
              Python Server Console
            </h1>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <StatusPill tone={connectionTone(connectionStatus)} icon={<Server className="size-3.5" />}>
            {activeTarget.baseUrl}
          </StatusPill>
          <StatusPill tone="info" icon={<CircleDot className="size-3.5" />}>
            {activeTarget.prefix}
          </StatusPill>
          <StatusPill tone={connectionTone(connectionStatus)} icon={<RefreshCcw className="size-3.5" />}>
            {formatConnectionStatus(connectionStatus)}
          </StatusPill>
          <StatusPill tone={managedRuntimeIsOnline ? 'success' : 'warning'} icon={<Activity className="size-3.5" />}>
            runtime {managedRuntimeIsOnline ? 'online' : 'offline'}
          </StatusPill>
          <StatusPill tone={notificationPermission === 'granted' ? 'success' : 'warning'} icon={<Bell className="size-3.5" />}>
            {formatNotificationPermission(notificationPermission)}
          </StatusPill>
          {capabilities?.model ? <StatusPill tone="neutral">{capabilities.model}</StatusPill> : null}
        </div>
      </div>
    </header>
  )
}

function ConnectionCard({
  capabilitiesErrorText,
  connectionStatus,
  draftTarget,
  healthErrorText,
  isBusy,
  managedRuntimeHint,
  onConnect,
  onDraftChange,
}: {
  capabilitiesErrorText: string | null
  connectionStatus: ConnectionStatus
  draftTarget: ConnectionTarget
  healthErrorText: string | null
  isBusy: boolean
  managedRuntimeHint: string | null
  onConnect: () => void
  onDraftChange: (target: ConnectionTarget) => void
}) {
  return (
    <Panel title="Connection" eyebrow="M1 live" icon={<Server className="size-4" />}>
      <div className="space-y-3">
        <label className="block">
          <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
            Base URL
          </span>
          <input
            className="w-full rounded-2xl border border-[var(--border-strong)] bg-[var(--surface-card)] px-4 py-3 text-sm outline-none transition focus:border-[var(--accent)] focus:ring-4 focus:ring-[var(--accent-ring)]"
            onChange={(event) => onDraftChange({ ...draftTarget, baseUrl: event.target.value })}
            placeholder="http://127.0.0.1:8080"
            value={draftTarget.baseUrl}
          />
        </label>

        <label className="block">
          <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
            API prefix
          </span>
          <select
            className="w-full rounded-2xl border border-[var(--border-strong)] bg-[var(--surface-card)] py-3 pl-3 pr-4 text-sm outline-none transition focus:border-[var(--accent)] focus:ring-4 focus:ring-[var(--accent-ring)]"
            onChange={(event) =>
              onDraftChange({ ...draftTarget, prefix: event.target.value as ConnectionTarget['prefix'] })
            }
            value={draftTarget.prefix}
          >
            {API_PREFIXES.map((prefix) => (
              <option key={prefix} value={prefix}>
                {prefix}
              </option>
            ))}
          </select>
        </label>

        <div className="grid grid-cols-2 gap-2">
          <ButtonGhost icon={<Activity className="size-4" />} onClick={onConnect}>
            Refresh probes
          </ButtonGhost>
          <ButtonPrimary icon={<Play className="size-4" />} onClick={onConnect}>
            {isBusy ? 'Connecting' : 'Connect'}
          </ButtonPrimary>
        </div>

        <StatusBanner status={connectionStatus} />

        {healthErrorText ? <InlineError title="Health request failed" value={healthErrorText} /> : null}
        {capabilitiesErrorText ? (
          <InlineError title="Capabilities request failed" value={capabilitiesErrorText} />
        ) : null}
        {managedRuntimeHint ? <InlineInfo title="Managed runtime" value={managedRuntimeHint} /> : null}

        <p className="rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-muted)] px-3 py-2 text-xs leading-5 text-[var(--text-secondary)]">
          Default dev traffic will use the Vite proxy for canonical prefixes. Direct local targets remain supported by the connection form.
        </p>
      </div>
    </Panel>
  )
}

function LocalRuntimeCard({
  busy,
  canStart,
  canRestart,
  canStop,
  errorText,
  isOnline,
  isUsingManagedTarget,
  notificationPermission,
  onApplyPreset,
  onEnableNotifications,
  onRestart,
  onSelectProfile,
  onStart,
  onStop,
  onUseManagedTarget,
  profile,
  profiles,
  runtime,
  startLabel,
}: {
  busy: boolean
  canStart: boolean
  canRestart: boolean
  canStop: boolean
  errorText: string | null
  isOnline: boolean
  isUsingManagedTarget: boolean
  notificationPermission: BrowserNotificationState
  onApplyPreset: () => void
  onEnableNotifications: () => void
  onRestart: () => void
  onSelectProfile: (slug: string) => void
  onStart: () => void
  onStop: () => void
  onUseManagedTarget: () => void
  profile: ManagedRuntimeProfile | null
  profiles: ManagedRuntimeProfile[]
  runtime: ManagedRuntimeStatusResponse | null
  startLabel: string
}) {
  const backend = runtime?.backend ?? null
  const recentEvents = runtime?.recent_events.slice(-4).reverse() ?? []
  const logTail = runtime?.log_tail.slice(-3).reverse() ?? []
  const notes = profile?.notes ?? []
  const statusCopy = buildManagedRuntimeStatusCopy(backend, profile)
  const modelValue = profile ? `${profile.name} (${profile.kind})` : 'waiting for runtime manager'
  const loadProgress = deriveManagedModelLoadProgress(backend, runtime?.log_tail ?? [], profile)
  const targetValue = runtime
    ? `${runtime.recommended_connection_target.baseUrl}${runtime.recommended_connection_target.prefix}`
    : 'manager offline'

  return (
    <Panel title="Local Runtime" eyebrow="self-heal + model switch" icon={<RefreshCcw className="size-4" />}>
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <StatusPill tone={isOnline ? 'success' : 'warning'} icon={<Activity className="size-3.5" />}>
            manager {isOnline ? 'online' : 'offline'}
          </StatusPill>
          <StatusPill tone={managedBackendTone(backend?.status)} icon={<Server className="size-3.5" />}>
            {formatManagedBackendStatus(backend?.status)}
          </StatusPill>
          <StatusPill
            tone={notificationPermission === 'granted' ? 'success' : notificationPermission === 'unsupported' ? 'neutral' : 'warning'}
            icon={<Bell className="size-3.5" />}
          >
            {formatNotificationPermission(notificationPermission)}
          </StatusPill>
          {backend?.profile ? <StatusPill tone="neutral">{backend.profile.name}</StatusPill> : null}
        </div>

        <label className="block">
          <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
            Verified model
          </span>
          <select
            className="w-full rounded-2xl border border-[var(--border-strong)] bg-[var(--surface-card)] py-3 pl-3 pr-4 text-sm outline-none transition focus:border-[var(--accent)] focus:ring-4 focus:ring-[var(--accent-ring)]"
            onChange={(event) => onSelectProfile(event.target.value)}
            value={profile?.slug ?? ''}
          >
            {profiles.length === 0 ? (
              <option value="">Waiting for runtime manager…</option>
            ) : (
              profiles.map((item) => (
                <option key={item.slug} value={item.slug}>
                  {item.name} · {item.kind}
                </option>
              ))
            )}
          </select>
        </label>

        <div className="grid grid-cols-2 gap-2">
          <ButtonPrimary disabled={!canStart} icon={<Play className="size-4" />} onClick={onStart}>
            {busy ? 'Working' : startLabel}
          </ButtonPrimary>
          <ButtonGhost disabled={!canRestart} icon={<RefreshCcw className="size-4" />} onClick={onRestart}>
            Restart
          </ButtonGhost>
          <ButtonGhost disabled={!canStop} icon={<Pause className="size-4" />} onClick={onStop}>
            Stop
          </ButtonGhost>
          <ButtonGhost
            disabled={!runtime || isUsingManagedTarget}
            icon={<CircleDot className="size-4" />}
            onClick={onUseManagedTarget}
          >
            {isUsingManagedTarget ? 'Target synced' : 'Use target'}
          </ButtonGhost>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <ButtonGhost
            disabled={!profile?.request_example}
            icon={<Sparkles className="size-4" />}
            onClick={onApplyPreset}
          >
            Apply preset
          </ButtonGhost>
          <ButtonGhost
            disabled={notificationPermission === 'granted' || notificationPermission === 'unsupported'}
            icon={<Bell className="size-4" />}
            onClick={onEnableNotifications}
          >
            {notificationPermission === 'granted' ? 'Notifications on' : 'Enable alerts'}
          </ButtonGhost>
        </div>

        <div className="rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-muted)] px-3 py-3 text-sm leading-6 text-[var(--text-secondary)]">
          {statusCopy}
        </div>

        {loadProgress ? <ManagedModelLoadCard progress={loadProgress} /> : null}

        {errorText ? <InlineError title="Runtime manager request failed" value={errorText} /> : null}
        <div className="grid gap-2 sm:grid-cols-2">
          <FieldPreview label="Selected model" value={modelValue} />
          <FieldPreview label="Managed target" value={targetValue} />
          <FieldPreview
            label="Auto-restart"
            value={
              backend
                ? `${backend.restart_count_consecutive} / ${backend.auto_restart_limit} consecutive`
                : 'waiting for backend'
            }
          />
          <FieldPreview
            label="Last health"
            value={backend?.last_health ? formatHealthSummary(backend.last_health) : 'not probed yet'}
          />
          <FieldPreview label="PID" value={backend?.pid ? String(backend.pid) : 'not running'} />
          <FieldPreview label="Last exit" value={formatLastExit(backend?.last_exit ?? null)} />
        </div>

        {notes.length > 0 ? (
          <div className="space-y-2">
            {notes.map((note) => (
              <div
                className="rounded-xl border border-[var(--border-subtle)] bg-[var(--surface-card)] px-3 py-2 text-xs leading-5 text-[var(--text-secondary)]"
                key={note}
              >
                {note}
              </div>
            ))}
          </div>
        ) : null}

        <div className="space-y-2">
          {recentEvents.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-[var(--border-strong)] bg-[var(--surface-muted)] px-4 py-4 text-sm leading-6 text-[var(--text-secondary)]">
              Runtime manager event history will appear here once the local controller starts observing the backend.
            </div>
          ) : (
            recentEvents.map((event) => (
              <div
                className="rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-card)] px-3 py-3"
                key={event.id}
              >
                <div className="flex items-start justify-between gap-3">
                  <p className="text-sm font-semibold text-[var(--text-primary)]">{event.message}</p>
                  <StatusPill tone={eventTone(event.level)}>{event.level}</StatusPill>
                </div>
                <p className="mt-2 font-mono text-[11px] text-[var(--text-muted)]">
                  {formatTimestamp(event.time_ms)} · {event.type}
                </p>
              </div>
            ))
          )}
        </div>

        {logTail.length > 0 ? (
          <LogTailViewer entries={logTail} />
        ) : null}
      </div>
    </Panel>
  )
}

function CapabilitiesCard({
  capabilities,
  connectionStatus,
  rows,
}: {
  capabilities?: CapabilitiesResponse
  connectionStatus: ConnectionStatus
  rows: Array<{ label: string; tone: 'info' | 'success'; value: string }>
}) {
  return (
    <Panel title="Capabilities" eyebrow="server contract" icon={<Sparkles className="size-4" />}>
      {connectionStatus !== 'connected' || !capabilities ? (
        <div className="rounded-2xl border border-dashed border-[var(--border-strong)] bg-[var(--surface-muted)] px-4 py-4 text-sm leading-6 text-[var(--text-secondary)]">
          Connect successfully to unlock capability-driven UI. Composer buttons stay gated until `supports.image` and `supports.video` are known.
        </div>
      ) : (
        <div className="space-y-2">
          <div className="grid gap-2 sm:grid-cols-2">
            <FieldPreview label="Model" value={capabilities.model} />
            <FieldPreview label="Package" value={capabilities.package_version} />
            <FieldPreview label="Pipeline" value={capabilities.pipeline_name ?? 'n/a'} />
            <FieldPreview label="Version" value={capabilities.version_name ?? 'n/a'} />
          </div>

          {rows.map((row) => (
            <div
              className="rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-card)] px-3 py-2"
              key={row.label}
            >
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-medium text-[var(--text-primary)]">{row.label}</span>
                <StatusDot tone={row.tone} />
              </div>
              <p className="mt-1 font-mono text-[11px] text-[var(--text-muted)]">{row.value}</p>
            </div>
          ))}
        </div>
      )}
    </Panel>
  )
}

function ComposerPanel({
  composerEnabled,
  composerMode,
  createPending,
  editImageBinding,
  editImageDraft,
  editImageErrors,
  imageDraft,
  imageErrors,
  onDraftModeChange,
  onEditImageDraftChange,
  onEditImageFileChange,
  onEditImageClear,
  onImageDraftChange,
  onPromptHotkey,
  onSubmit,
  onVideoDraftChange,
  submitErrorText,
  supportsEditImage,
  supportsImage,
  supportsVideo,
  videoDraft,
  videoErrors,
}: {
  composerEnabled: boolean
  composerMode: ComposerMode
  createPending: boolean
  editImageBinding: ReturnType<typeof deriveEditImageBinding>
  editImageDraft: EditImageDraft
  editImageErrors: Partial<Record<keyof EditImageDraft, string>>
  imageDraft: ImageDraft
  imageErrors: Partial<Record<keyof ImageDraft, string>>
  onDraftModeChange: (mode: ComposerMode) => void
  onEditImageDraftChange: (field: keyof EditImageDraft, value: string) => void
  onEditImageFileChange: (event: ChangeEvent<HTMLInputElement>) => void
  onEditImageClear: () => void
  onImageDraftChange: (field: keyof ImageDraft, value: string) => void
  onPromptHotkey: (event: KeyboardEvent<HTMLTextAreaElement>) => void
  onSubmit: () => Promise<void>
  onVideoDraftChange: (field: keyof VideoDraft, value: string) => void
  submitErrorText: string | null
  supportsEditImage: boolean
  supportsImage: boolean
  supportsVideo: boolean
  videoDraft: VideoDraft
  videoErrors: Partial<Record<keyof VideoDraft, string>>
}) {
  const isImageMode = composerMode === 'image'
  const isEditImageMode = composerMode === 'edit_image'
  const isVideoMode = composerMode === 'video'
  const canSubmitCurrentMode =
    composerEnabled &&
    (isVideoMode ? supportsVideo : isEditImageMode ? supportsImage && supportsEditImage : supportsImage)
  const primaryLabel = isVideoMode
    ? createPending
      ? 'Submitting video job'
      : 'Create video job'
    : isEditImageMode
      ? createPending
        ? 'Submitting edit job'
        : 'Create edit job'
      : createPending
        ? 'Submitting image job'
        : 'Create image job'
  const presetCopy = isVideoMode
    ? 'video mode starts with `416 x 240`, `9` frames, and `20` steps.'
    : isEditImageMode
      ? `${editImageBinding.modeLabel} starts with a single input image plus a conservative \`256 x 256\` / \`1\` step smoke envelope.`
      : 'image mode starts with `256 x 256` and `1` step for a conservative local smoke envelope.'

  return (
    <Panel
      title="Generation Composer"
      eyebrow="image + edit + video"
      icon={<Layers3 className="size-4" />}
      className="min-h-[360px]"
    >
      <div className="space-y-4">
        <div className="flex rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-muted)] p-1">
          <button
            className={cn(
              'flex-1 rounded-xl px-3 py-2 text-sm font-semibold shadow-sm transition',
              isImageMode ? 'bg-[var(--surface-card)] text-[var(--text-primary)]' : 'text-[var(--text-muted)]',
            )}
            disabled={!supportsImage}
            onClick={() => onDraftModeChange('image')}
            type="button"
          >
            Image
          </button>
          <button
            className={cn(
              'flex-1 rounded-xl px-3 py-2 text-sm font-semibold shadow-sm transition',
              isEditImageMode ? 'bg-[var(--surface-card)] text-[var(--text-primary)]' : 'text-[var(--text-muted)]',
            )}
            disabled={!supportsEditImage}
            onClick={() => onDraftModeChange('edit_image')}
            type="button"
          >
            Edit Image
          </button>
          <button
            className={cn(
              'flex-1 rounded-xl px-3 py-2 text-sm font-semibold shadow-sm transition',
              isVideoMode ? 'bg-[var(--surface-card)] text-[var(--text-primary)]' : 'text-[var(--text-muted)]',
            )}
            disabled={!supportsVideo}
            onClick={() => onDraftModeChange('video')}
            type="button"
          >
            Video
          </button>
        </div>

        <div className="rounded-2xl border border-[var(--info-border)] bg-[var(--info-soft)] px-3 py-3 text-sm leading-6 text-[var(--text-secondary)]">
          <span className="font-semibold text-[var(--info)]">UI preset:</span> {presetCopy} These are console defaults, not
          backend guarantees. Use the Local Runtime panel to apply a verified model-specific request example.
        </div>

        {isImageMode ? (
          <ImageComposerFields
            draft={imageDraft}
            errors={imageErrors}
            onDraftChange={onImageDraftChange}
            onPromptHotkey={onPromptHotkey}
          />
        ) : isEditImageMode ? (
          <EditImageComposerFields
            binding={editImageBinding}
            draft={editImageDraft}
            errors={editImageErrors}
            onClearImage={onEditImageClear}
            onDraftChange={onEditImageDraftChange}
            onFileChange={onEditImageFileChange}
            onPromptHotkey={onPromptHotkey}
          />
        ) : (
          <VideoComposerFields
            draft={videoDraft}
            errors={videoErrors}
            onDraftChange={onVideoDraftChange}
            onPromptHotkey={onPromptHotkey}
          />
        )}

        <div className="flex flex-wrap gap-2">
          <ButtonPrimary
            disabled={!canSubmitCurrentMode || createPending}
            icon={isVideoMode ? <Video className="size-4" /> : <Image className="size-4" />}
            onClick={() => {
              void onSubmit()
            }}
          >
            {primaryLabel}
          </ButtonPrimary>
          <ButtonGhost disabled icon={<Braces className="size-4" />}>
            JSON editor stays mode-scoped
          </ButtonGhost>
        </div>

        {!composerEnabled ? (
          <p className="text-sm leading-6 text-[var(--text-secondary)]">
            Composer is gated until `health` and `capabilities` both succeed.
          </p>
        ) : null}

        {submitErrorText ? <InlineError title="Submit blocked" value={submitErrorText} /> : null}
      </div>
    </Panel>
  )
}

function ImageComposerFields({
  draft,
  errors,
  onDraftChange,
  onPromptHotkey,
}: {
  draft: ImageDraft
  errors: Partial<Record<keyof ImageDraft, string>>
  onDraftChange: (field: keyof ImageDraft, value: string) => void
  onPromptHotkey: (event: KeyboardEvent<HTMLTextAreaElement>) => void
}) {
  return (
    <>
      <TextAreaField
        error={errors.prompt}
        helper="Describe the target image. Use Ctrl/Cmd + Enter to submit."
        label="Prompt"
        onChange={(value) => onDraftChange('prompt', value)}
        onKeyDown={onPromptHotkey}
        placeholder="A clean product demo image, elegant lighting, precise interface details..."
        value={draft.prompt}
      />

      <TextAreaField
        helper="Optional negative prompt."
        label="Negative Prompt"
        onChange={(value) => onDraftChange('negativePrompt', value)}
        placeholder="blurry, distorted, low detail"
        rows={3}
        value={draft.negativePrompt}
      />

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <InputField error={errors.width} label="Width" onChange={(value) => onDraftChange('width', value)} value={draft.width} />
        <InputField error={errors.height} label="Height" onChange={(value) => onDraftChange('height', value)} value={draft.height} />
        <InputField error={errors.steps} label="Steps" onChange={(value) => onDraftChange('steps', value)} value={draft.steps} />
        <InputField
          error={errors.seed}
          helper="Blank means backend decides."
          label="Seed"
          onChange={(value) => onDraftChange('seed', value)}
          placeholder="random"
          value={draft.seed}
        />
      </div>

      <div className="rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-card)] p-3">
        <div className="mb-3">
          <p className="text-sm font-semibold text-[var(--text-primary)]">Advanced</p>
          <p className="text-xs leading-5 text-[var(--text-muted)]">
            Rich but lightweight first-pass image controls.
          </p>
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <InputField error={errors.guidance} label="Guidance" onChange={(value) => onDraftChange('guidance', value)} value={draft.guidance} />
          <InputField error={errors.batchCount} label="Batch Count" onChange={(value) => onDraftChange('batchCount', value)} value={draft.batchCount} />
          <InputField error={errors.cfgScale} label="CFG Scale" onChange={(value) => onDraftChange('cfgScale', value)} value={draft.cfgScale} />
          <InputField error={errors.imageCfgScale} label="Image CFG" onChange={(value) => onDraftChange('imageCfgScale', value)} value={draft.imageCfgScale} />
          <InputField error={errors.eta} label="Eta" onChange={(value) => onDraftChange('eta', value)} value={draft.eta} />
          <InputField error={errors.flowShift} label="Flow Shift" onChange={(value) => onDraftChange('flowShift', value)} value={draft.flowShift} />
          <InputField label="Sampler" onChange={(value) => onDraftChange('sampler', value)} placeholder="free input" value={draft.sampler} />
          <InputField label="Scheduler" onChange={(value) => onDraftChange('scheduler', value)} placeholder="free input" value={draft.scheduler} />
        </div>
      </div>
    </>
  )
}

function EditImageComposerFields({
  binding,
  draft,
  errors,
  onClearImage,
  onDraftChange,
  onFileChange,
  onPromptHotkey,
}: {
  binding: ReturnType<typeof deriveEditImageBinding>
  draft: EditImageDraft
  errors: Partial<Record<keyof EditImageDraft, string>>
  onClearImage: () => void
  onDraftChange: (field: keyof EditImageDraft, value: string) => void
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void
  onPromptHotkey: (event: KeyboardEvent<HTMLTextAreaElement>) => void
}) {
  const previewSrc = buildEditImagePreviewSrc(draft.inputImage)

  return (
    <>
      <TextAreaField
        error={errors.prompt}
        helper={`Describe the transformation. ${binding.shortHelper} Use Ctrl/Cmd + Enter to submit.`}
        label="Prompt"
        onChange={(value) => onDraftChange('prompt', value)}
        onKeyDown={onPromptHotkey}
        placeholder="Turn this desk photo into a polished editorial product shot..."
        value={draft.prompt}
      />

      <TextAreaField
        helper="Optional negative prompt."
        label="Negative Prompt"
        onChange={(value) => onDraftChange('negativePrompt', value)}
        placeholder="blurry, distorted, low detail"
        rows={3}
        value={draft.negativePrompt}
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_240px]">
        <div className="rounded-[24px] border border-[var(--border-strong)] bg-[linear-gradient(180deg,#ffffff,#f5f8f3)] p-4 shadow-[var(--shadow-soft)]">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-[var(--text-primary)]">{binding.inputLabel}</p>
              <p className="mt-1 text-xs leading-5 text-[var(--text-muted)]">{binding.helper}</p>
            </div>
            <span className="rounded-full border border-[var(--border-subtle)] bg-white/70 px-2 py-1 font-mono text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
              {binding.transport ?? 'unverified'}
            </span>
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-[minmax(0,1fr)_150px]">
            <label className="group flex min-h-[156px] cursor-pointer flex-col justify-between rounded-[22px] border border-dashed border-[var(--border-strong)] bg-[var(--surface-card)] p-4 transition hover:border-[var(--accent-border)] hover:bg-[var(--surface-hover)]">
              <input accept="image/*" className="sr-only" onChange={onFileChange} type="file" />
              <div>
                <p className="text-sm font-semibold text-[var(--text-primary)]">
                  {draft.inputImage ? 'Replace input image' : 'Choose input image'}
                </p>
                <p className="mt-2 text-sm leading-6 text-[var(--text-secondary)]">
                  PNG, JPEG, or any browser-readable image. The console will inline it into the request JSON.
                </p>
              </div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--accent)] group-hover:text-[var(--accent-hover)]">
                click to browse
              </p>
            </label>

            <div className="overflow-hidden rounded-[22px] border border-[var(--border-subtle)] bg-[linear-gradient(135deg,#eef2f4,#dfe8e6)]">
              {previewSrc ? (
                <img alt="Edit input preview" className="h-full w-full object-cover" src={previewSrc} />
              ) : (
                <div className="grid h-full min-h-[156px] place-items-center px-4 text-center text-sm leading-6 text-[var(--text-secondary)]">
                  No input image selected yet.
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <ButtonGhost disabled={!draft.inputImage} icon={<Trash2 className="size-4" />} onClick={onClearImage}>
              Clear image
            </ButtonGhost>
            {draft.inputImageName ? <FieldPreview label="Selected file" value={draft.inputImageName} /> : null}
          </div>
          {errors.inputImage ? <p className="mt-2 text-xs text-[var(--error)]">{errors.inputImage}</p> : null}
        </div>

        <div className="space-y-3">
          <div className="grid gap-3">
            <InputField error={errors.width} label="Width" onChange={(value) => onDraftChange('width', value)} value={draft.width} />
            <InputField error={errors.height} label="Height" onChange={(value) => onDraftChange('height', value)} value={draft.height} />
            <InputField error={errors.steps} label="Steps" onChange={(value) => onDraftChange('steps', value)} value={draft.steps} />
            <InputField
              error={errors.seed}
              helper="Blank means backend decides."
              label="Seed"
              onChange={(value) => onDraftChange('seed', value)}
              placeholder="random"
              value={draft.seed}
            />
          </div>

          <div className="rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-card)] p-3">
            <div className="mb-3">
              <p className="text-sm font-semibold text-[var(--text-primary)]">Advanced</p>
              <p className="text-xs leading-5 text-[var(--text-muted)]">
                Lightweight controls shared across the verified image-edit paths.
              </p>
            </div>
            <div className="grid gap-3">
              <InputField error={errors.guidance} label="Guidance" onChange={(value) => onDraftChange('guidance', value)} value={draft.guidance} />
              <InputField error={errors.cfgScale} label="CFG Scale" onChange={(value) => onDraftChange('cfgScale', value)} value={draft.cfgScale} />
              <InputField error={errors.flowShift} label="Flow Shift" onChange={(value) => onDraftChange('flowShift', value)} value={draft.flowShift} />
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

function VideoComposerFields({
  draft,
  errors,
  onDraftChange,
  onPromptHotkey,
}: {
  draft: VideoDraft
  errors: Partial<Record<keyof VideoDraft, string>>
  onDraftChange: (field: keyof VideoDraft, value: string) => void
  onPromptHotkey: (event: KeyboardEvent<HTMLTextAreaElement>) => void
}) {
  return (
    <>
      <TextAreaField
        error={errors.prompt}
        helper="Describe the target motion clip. Use Ctrl/Cmd + Enter to submit."
        label="Prompt"
        onChange={(value) => onDraftChange('prompt', value)}
        onKeyDown={onPromptHotkey}
        placeholder="A small robot walking through rain, cinematic lighting, smooth motion..."
        value={draft.prompt}
      />

      <TextAreaField
        helper="Optional negative prompt."
        label="Negative Prompt"
        onChange={(value) => onDraftChange('negativePrompt', value)}
        placeholder="flicker, distortion, messy motion"
        rows={3}
        value={draft.negativePrompt}
      />

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
        <InputField error={errors.width} label="Width" onChange={(value) => onDraftChange('width', value)} value={draft.width} />
        <InputField error={errors.height} label="Height" onChange={(value) => onDraftChange('height', value)} value={draft.height} />
        <InputField error={errors.frames} label="Frames" onChange={(value) => onDraftChange('frames', value)} value={draft.frames} />
        <InputField error={errors.steps} label="Steps" onChange={(value) => onDraftChange('steps', value)} value={draft.steps} />
        <InputField
          error={errors.seed}
          helper="Blank means backend decides."
          label="Seed"
          onChange={(value) => onDraftChange('seed', value)}
          placeholder="random"
          value={draft.seed}
        />
      </div>

      <div className="rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-card)] p-3">
        <div className="mb-3">
          <p className="text-sm font-semibold text-[var(--text-primary)]">Advanced</p>
          <p className="text-xs leading-5 text-[var(--text-muted)]">
            These fields shape the backend video request; the viewer remains a frontend frame carousel.
          </p>
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <InputField error={errors.guidance} label="Guidance" onChange={(value) => onDraftChange('guidance', value)} value={draft.guidance} />
          <InputField error={errors.cfgScale} label="CFG Scale" onChange={(value) => onDraftChange('cfgScale', value)} value={draft.cfgScale} />
          <InputField error={errors.eta} label="Eta" onChange={(value) => onDraftChange('eta', value)} value={draft.eta} />
          <InputField error={errors.flowShift} label="Flow Shift" onChange={(value) => onDraftChange('flowShift', value)} value={draft.flowShift} />
          <InputField label="Sampler" onChange={(value) => onDraftChange('sampler', value)} placeholder="free input" value={draft.sampler} />
          <InputField label="Scheduler" onChange={(value) => onDraftChange('scheduler', value)} placeholder="free input" value={draft.scheduler} />
        </div>
      </div>
    </>
  )
}

function PayloadPreviewPanel({
  canSubmit,
  composerMode,
  createPending,
  isRawDetached,
  onCopyJson,
  onRawJsonChange,
  onResetJsonBuffer,
  onSubmit,
  rawJsonBuffer,
}: {
  canSubmit: boolean
  composerMode: ComposerMode
  createPending: boolean
  isRawDetached: boolean
  onCopyJson: () => Promise<void>
  onRawJsonChange: (value: string) => void
  onResetJsonBuffer: () => void
  onSubmit: () => Promise<void>
  rawJsonBuffer: string
}) {
  return (
    <Panel
      title="Payload Preview"
      eyebrow="mode-scoped expert json"
      icon={<Braces className="size-4" />}
      className="flex h-[390px] flex-col"
    >
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="rounded-full border border-[var(--border-subtle)] bg-white/70 px-2 py-1 font-mono text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
          {formatComposerModeLabel(composerMode)}
        </span>
        <span className="rounded-full border border-[var(--border-subtle)] bg-white/70 px-2 py-1 font-mono text-[10px] uppercase tracking-[0.18em] text-[var(--text-muted)]">
          {isRawDetached ? 'detached' : 'synced'}
        </span>
      </div>

      <JsonCodeEditor
        ariaLabel="Payload preview JSON editor"
        className="min-h-0 flex-1"
        maxHeight={276}
        minHeight={276}
        onChange={onRawJsonChange}
        value={rawJsonBuffer}
      />

      <div className="mt-3 flex flex-col gap-2">
        <ButtonPrimary
          className="w-full justify-between px-4"
          disabled={!canSubmit || createPending}
          icon={<Play className="size-4" />}
          onClick={() => {
            void onSubmit()
          }}
        >
          Send JSON
        </ButtonPrimary>
        <ButtonGhost
          className="w-full justify-between px-4"
          disabled={!isRawDetached}
          icon={<RotateCcw className="size-4" />}
          onClick={onResetJsonBuffer}
        >
          Reset to form
        </ButtonGhost>
        <ButtonGhost className="w-full justify-between px-4" icon={<Copy className="size-4" />} onClick={() => void onCopyJson()}>
          Copy JSON
        </ButtonGhost>
      </div>
    </Panel>
  )
}

function ProgressPanel({ progressVm }: { progressVm: ReturnType<typeof deriveProgressVm> }) {
  return (
    <Panel title="Progress" eyebrow="truthful derived phase" icon={<TimerReset className="size-4" />}>
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-[var(--text-primary)]">{progressVm.phaseLabel}</p>
            <p className="mt-1 text-xs text-[var(--text-muted)]">{progressVm.description}</p>
          </div>
          <StatusPill tone={progressVm.tone}>{progressVm.stepsLabel}</StatusPill>
        </div>
        <div className="h-3 overflow-hidden rounded-full bg-[var(--surface-muted)]">
          <div
            className="h-full rounded-full bg-[linear-gradient(90deg,var(--accent),var(--accent-2))] shadow-[0_0_24px_rgba(42,96,220,0.28)] transition-[width]"
            style={{ width: `${progressVm.percent}%` }}
          />
        </div>
      </div>
    </Panel>
  )
}

function ResultPreview({
  decodedImages,
  decodedVideoFrames,
  downloadPending,
  isVideoPlaying,
  job,
  onDownloadVideo,
  onFpsChange,
  onLoopChange,
  onPlayChange,
  result,
  resultErrorText,
  resultPending,
  selectedMediaIndex,
  setSelectedMediaIndex,
  videoFps,
  videoLoop,
}: {
  decodedImages: Array<ReturnType<typeof decodeImageResult>[number]>
  decodedVideoFrames: Array<ReturnType<typeof decodeVideoResult>[number]>
  downloadPending: boolean
  isVideoPlaying: boolean
  job: EdgeDitJob | EdgeDitJobSummary | null | undefined
  onDownloadVideo: (jobId: string, fps: number) => void
  onFpsChange: (value: number) => void
  onLoopChange: (value: boolean) => void
  onPlayChange: (value: boolean) => void
  result?: EdgeDitGenerationResult
  resultErrorText: string | null
  resultPending: boolean
  selectedMediaIndex: number
  setSelectedMediaIndex: (index: number) => void
  videoFps: number
  videoLoop: boolean
}) {
  const isVideoJob = job?.kind === 'video'
  const selectedImage = decodedImages[selectedMediaIndex] ?? null
  const selectedFrame = decodedVideoFrames[selectedMediaIndex] ?? null
  const resultJsonHref = result ? jsonDownloadHref(result) : null

  return (
    <Panel
      title="Result Viewer"
      eyebrow={isVideoJob ? 'frame carousel + metadata' : 'image decode + metadata'}
      icon={isVideoJob ? <Video className="size-4" /> : <Image className="size-4" />}
      className="flex-1"
    >
      {!job ? (
        <ResultPlaceholder
          title="Select or create a job"
          body="The console watches the selected job, polls detail, and fetches result JSON once the backend reports success."
        />
      ) : resultErrorText ? (
        <ResultPlaceholder title="Result fetch failed" body={resultErrorText} tone="error" />
      ) : job.status === 'failed' || job.status === 'cancelled' ? (
        <ResultPlaceholder
          title={`Job ${job.status}`}
          body={job.error ?? 'The selected job reached a terminal state before producing a usable result.'}
          tone="warning"
        />
      ) : resultPending || job.status === 'queued' || job.status === 'running' || job.status === 'cancelling' ? (
        <ResultPlaceholder
          title={job.status === 'succeeded' ? 'Fetching result JSON' : `Job ${job.status}`}
          body={
            job.status === 'succeeded'
              ? 'The console is fetching `/jobs/{id}/result` and will retry if the backend still reports `job_not_ready`.'
              : 'Progress comes from the selected job detail endpoint. The viewer activates after a successful result fetch.'
          }
          tone="info"
        />
      ) : isVideoJob ? (
        selectedFrame && result && isVideoGenerationResult(result) ? (
          <div className="grid min-h-[320px] gap-4 lg:grid-cols-[1fr_240px]">
            <div className="space-y-3">
              <div className="relative overflow-hidden rounded-[24px] border border-[var(--border-strong)] bg-[linear-gradient(135deg,#eef2f4,#dfe8e6)]">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_32%_22%,rgba(42,96,220,0.16),transparent_28%),radial-gradient(circle_at_74%_66%,rgba(28,124,112,0.16),transparent_30%)]" />
                <div className="relative flex min-h-[320px] items-center justify-center p-6">
                  <img
                    alt={`Generated video frame ${selectedMediaIndex + 1}`}
                    className="max-h-[420px] max-w-full rounded-2xl border border-white/70 bg-white shadow-xl"
                    src={selectedFrame.src}
                  />
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                <ButtonGhost
                  disabled={selectedMediaIndex === 0}
                  icon={<SkipBack className="size-4" />}
                  onClick={() => setSelectedMediaIndex(Math.max(0, selectedMediaIndex - 1))}
                >
                  Prev
                </ButtonGhost>
                <ButtonPrimary
                  disabled={decodedVideoFrames.length < 2}
                  icon={isVideoPlaying ? <Pause className="size-4" /> : <Play className="size-4" />}
                  onClick={() => onPlayChange(!isVideoPlaying)}
                >
                  {isVideoPlaying ? 'Pause' : 'Play'}
                </ButtonPrimary>
                <ButtonGhost
                  disabled={selectedMediaIndex >= decodedVideoFrames.length - 1}
                  icon={<SkipForward className="size-4" />}
                  onClick={() =>
                    setSelectedMediaIndex(Math.min(decodedVideoFrames.length - 1, selectedMediaIndex + 1))
                  }
                >
                  Next
                </ButtonGhost>
                <ButtonGhost
                  icon={<RotateCcw className="size-4" />}
                  onClick={() => {
                    setSelectedMediaIndex(0)
                    onPlayChange(false)
                  }}
                >
                  Reset
                </ButtonGhost>
              </div>

              <div className="grid grid-cols-4 gap-2 sm:grid-cols-6">
                {decodedVideoFrames.map((frame, index) => (
                  <button
                    className={cn(
                      'overflow-hidden rounded-2xl border p-1 transition',
                      index === selectedMediaIndex
                        ? 'border-[var(--accent)] bg-[var(--accent-soft)]'
                        : 'border-[var(--border-subtle)] bg-[var(--surface-card)]',
                    )}
                    key={`${frame.metadata.index}-${frame.metadata.width}-${frame.metadata.height}`}
                    onClick={() => {
                      setSelectedMediaIndex(index)
                      onPlayChange(false)
                    }}
                    type="button"
                  >
                    <img alt={`Frame thumbnail ${index + 1}`} className="aspect-square w-full rounded-xl object-cover" src={frame.src} />
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-3">
              <FieldPreview label="Frame" value={`${selectedMediaIndex + 1} / ${decodedVideoFrames.length}`} />
              <FieldPreview label="Width" value={String(selectedFrame.metadata.width)} />
              <FieldPreview label="Height" value={String(selectedFrame.metadata.height)} />
              <FieldPreview label="Format" value={selectedFrame.metadata.format} />
              <FieldPreview label="Playback" value="Frontend frame carousel" />

              <label className="block">
                <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
                  FPS
                </span>
                <select
                  className="w-full rounded-2xl border border-[var(--border-strong)] bg-[var(--surface-card)] px-3 py-2 text-sm outline-none transition focus:border-[var(--accent)] focus:ring-4 focus:ring-[var(--accent-ring)]"
                  onChange={(event) => onFpsChange(Number(event.target.value))}
                  value={videoFps}
                >
                  {[2, 4, 6, 8, 12, 16, 24].map((fps) => (
                    <option key={fps} value={fps}>
                      {fps} fps
                    </option>
                  ))}
                </select>
              </label>

              <button
                className={cn(
                  'inline-flex w-full items-center justify-center gap-2 rounded-xl border px-3 py-2 text-sm font-semibold transition',
                  videoLoop
                    ? 'border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]'
                    : 'border-[var(--border-subtle)] bg-[var(--surface-card)] text-[var(--text-primary)]',
                )}
                onClick={() => onLoopChange(!videoLoop)}
                type="button"
              >
                <RotateCcw className="size-4" />
                Loop {videoLoop ? 'On' : 'Off'}
              </button>

              <ButtonPrimary
                disabled={downloadPending}
                icon={<Download className="size-4" />}
                onClick={() => onDownloadVideo(result.id, videoFps)}
              >
                {downloadPending ? 'Encoding MP4' : 'Save Video as MP4'}
              </ButtonPrimary>

              <a
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-[var(--accent)] px-3 py-2 text-sm font-semibold text-white shadow-[0_12px_28px_rgba(42,96,220,0.26)] transition hover:-translate-y-0.5 hover:bg-[var(--accent-hover)]"
                download={`${result.id}-frame-${selectedMediaIndex + 1}.png`}
                href={selectedFrame.src}
              >
                <Box className="size-4" />
                Download Current Frame
              </a>
              {resultJsonHref ? (
                <a
                  className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-[var(--border-subtle)] bg-[var(--surface-card)] px-3 py-2 text-sm font-semibold text-[var(--text-primary)] transition hover:-translate-y-0.5 hover:border-[var(--accent-border)] hover:bg-[var(--surface-hover)]"
                  download={`${result.id}-result.json`}
                  href={resultJsonHref}
                >
                  <Braces className="size-4" />
                  Download Result JSON
                </a>
              ) : null}
            </div>
          </div>
        ) : (
          <ResultPlaceholder
            title="No decoded frames yet"
            body="The backend returned a video result, but the console has not decoded a visible frame payload."
            tone="warning"
          />
        )
      ) : selectedImage && result && isImageGenerationResult(result) ? (
        <div className="grid min-h-[320px] gap-4 lg:grid-cols-[1fr_220px]">
          <div className="space-y-3">
            <div className="relative overflow-hidden rounded-[24px] border border-[var(--border-strong)] bg-[linear-gradient(135deg,#eef2f4,#dfe8e6)]">
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_32%_22%,rgba(42,96,220,0.16),transparent_28%),radial-gradient(circle_at_74%_66%,rgba(28,124,112,0.16),transparent_30%)]" />
              <div className="relative flex min-h-[320px] items-center justify-center p-6">
                <img
                  alt={`Generated result preview ${selectedMediaIndex + 1}`}
                  className="max-h-[420px] max-w-full rounded-2xl border border-white/70 bg-white shadow-xl"
                  src={selectedImage.src}
                />
              </div>
            </div>

            {decodedImages.length > 1 ? (
              <div className="grid grid-cols-4 gap-2 sm:grid-cols-6">
                {decodedImages.map((image, index) => (
                  <button
                    className={cn(
                      'overflow-hidden rounded-2xl border p-1 transition',
                      index === selectedMediaIndex
                        ? 'border-[var(--accent)] bg-[var(--accent-soft)]'
                        : 'border-[var(--border-subtle)] bg-[var(--surface-card)]',
                    )}
                    key={`${image.metadata.index}-${image.metadata.width}-${image.metadata.height}`}
                    onClick={() => setSelectedMediaIndex(index)}
                    type="button"
                  >
                    <img alt={`Thumbnail ${index + 1}`} className="aspect-square w-full rounded-xl object-cover" src={image.src} />
                  </button>
                ))}
              </div>
            ) : null}
          </div>

          <div className="space-y-3">
            <FieldPreview label="Image" value={`${selectedMediaIndex + 1} / ${decodedImages.length}`} />
            <FieldPreview label="Width" value={String(selectedImage.metadata.width)} />
            <FieldPreview label="Height" value={String(selectedImage.metadata.height)} />
            <FieldPreview label="Channels" value={String(selectedImage.metadata.channels)} />
            <FieldPreview label="Format" value={selectedImage.metadata.format} />
            <a
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-[var(--accent)] px-3 py-2 text-sm font-semibold text-white shadow-[0_12px_28px_rgba(42,96,220,0.26)] transition hover:-translate-y-0.5 hover:bg-[var(--accent-hover)]"
              download={`${result.id}-${selectedMediaIndex + 1}.png`}
              href={selectedImage.src}
            >
              <Box className="size-4" />
              Download PNG
            </a>
            {resultJsonHref ? (
              <a
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-[var(--border-subtle)] bg-[var(--surface-card)] px-3 py-2 text-sm font-semibold text-[var(--text-primary)] transition hover:-translate-y-0.5 hover:border-[var(--accent-border)] hover:bg-[var(--surface-hover)]"
                download={`${result.id}-result.json`}
                href={resultJsonHref}
              >
                <Braces className="size-4" />
                Download Result JSON
              </a>
            ) : null}
          </div>
        </div>
      ) : (
        <ResultPlaceholder
          title="No decoded image yet"
          body="The backend returned an image result, but the console has not decoded a visible PNG payload."
          tone="warning"
        />
      )}
    </Panel>
  )
}

function JobsPanel({
  cancelPending,
  cleanupPending,
  deletePending,
  isLoading,
  jobs,
  onCancelSelected,
  onCleanup,
  onDeleteSelected,
  onRefresh,
  selectedJob,
  selectedJobId,
  setSelectedJobId,
}: {
  cancelPending: boolean
  cleanupPending: boolean
  deletePending: boolean
  isLoading: boolean
  jobs: EdgeDitJobSummary[]
  onCancelSelected: () => void
  onCleanup: () => void
  onDeleteSelected: () => void
  onRefresh: () => void
  selectedJob: EdgeDitJob | EdgeDitJobSummary | null | undefined
  selectedJobId: string | null
  setSelectedJobId: (jobId: string) => void
}) {
  const canCancelSelected = Boolean(
    selectedJob && isActiveJobStatus(selectedJob.status) && !selectedJob.cancel_requested && !cancelPending,
  )
  const canDeleteSelected = Boolean(selectedJob && !isActiveJobStatus(selectedJob.status) && !deletePending)

  return (
    <Panel title="Task List" eyebrow="serial worker queue" icon={<ClipboardList className="size-4" />}>
      {jobs.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-[var(--border-strong)] bg-[var(--surface-muted)] px-4 py-4 text-sm leading-6 text-[var(--text-secondary)]">
          {isLoading
            ? 'Loading current jobs from `GET /jobs`...'
            : 'No jobs yet. Create an image or video job to start the next closed loop.'}
        </div>
      ) : (
        <div className="space-y-2">
          {jobs.map((job) => (
            <button
              className={cn(
                'w-full rounded-2xl border px-3 py-3 text-left transition hover:-translate-y-0.5 hover:shadow-[var(--shadow-soft)]',
                selectedJobId === job.id
                  ? 'border-[var(--accent)] bg-[var(--accent-soft)]'
                  : 'border-[var(--border-subtle)] bg-[var(--surface-card)] hover:border-[var(--accent-border)]',
              )}
              key={job.id}
              onClick={() => setSelectedJobId(job.id)}
              type="button"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="font-mono text-xs font-semibold text-[var(--text-primary)]">{job.id}</p>
                  <p className="mt-1 text-xs text-[var(--text-muted)]">
                    {job.kind} · {summarizeJobParameters(job)}
                  </p>
                </div>
                <StatusPill tone={jobTone(job.status)}>{job.status}</StatusPill>
              </div>
            </button>
          ))}
        </div>
      )}

      <div className="mt-3 grid grid-cols-3 gap-2">
        <ButtonGhost icon={<Activity className="size-4" />} onClick={onRefresh}>
          Refresh
        </ButtonGhost>
        <ButtonGhost disabled={!canCancelSelected} icon={<TimerReset className="size-4" />} onClick={onCancelSelected}>
          {selectedJob?.cancel_requested ? 'Cancel requested' : cancelPending ? 'Cancelling' : 'Cancel selected'}
        </ButtonGhost>
        <ButtonGhost disabled={!canDeleteSelected} icon={<Trash2 className="size-4" />} onClick={onDeleteSelected}>
          {deletePending ? 'Removing' : 'Delete selected'}
        </ButtonGhost>
      </div>

      <div className="mt-2">
        <ButtonGhost
          disabled={cleanupPending}
          icon={<RefreshCcw className="size-4" />}
          onClick={onCleanup}
        >
          {cleanupPending ? 'Cleaning up' : 'Cleanup expired'}
        </ButtonGhost>
      </div>
    </Panel>
  )
}

function sanitizeInspectorValue(value: unknown, key = ''): unknown {
  if (typeof value === 'string') {
    const normalizedKey = key.toLowerCase()
    if (
      normalizedKey.includes('b64') ||
      normalizedKey.includes('base64') ||
      value.startsWith('data:') ||
      value.length > 4096
    ) {
      return `<binary or long string omitted: ${value.length} characters>`
    }
    return value
  }
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeInspectorValue(item, key))
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([childKey, childValue]) => [
        childKey,
        sanitizeInspectorValue(childValue, childKey),
      ]),
    )
  }
  return value
}

function InspectorPanel({
  capabilities,
  connectionStatus,
  health,
  payloadPreview,
  result,
  selectedJob,
}: {
  capabilities?: CapabilitiesResponse
  connectionStatus: ConnectionStatus
  health?: { model: string; request_id: string; service: string; status: string }
  payloadPreview: unknown
  result?: EdgeDitGenerationResult
  selectedJob: EdgeDitJob | EdgeDitJobSummary | null | undefined
}) {
  return (
    <Panel title="JSON Inspector" eyebrow="request / response / errors" icon={<Braces className="size-4" />}>
      <JsonCodeEditor
        ariaLabel="JSON inspector"
        maxHeight={520}
        minHeight={320}
        readOnly
        value={JSON.stringify(
          sanitizeInspectorValue({
            connection_status: connectionStatus,
            health,
            capabilities: capabilities
              ? {
                  model: capabilities.model,
                  supports: capabilities.supports,
                  semantics: capabilities.semantics,
                }
              : null,
            payload_preview: payloadPreview,
            selected_job: selectedJob,
            result,
          }),
          null,
          2,
        )}
      />
    </Panel>
  )
}

function ActivityLog({
  rows,
}: {
  rows: Array<{ method: string; path: string; requestId?: string; status: string; time: string }>
}) {
  return (
    <Panel title="Activity Log" eyebrow="local session" icon={<Activity className="size-4" />}>
      <div className="space-y-2">
        {rows.map((entry) => (
          <div
            className="rounded-xl bg-[var(--surface-muted)] px-3 py-2"
            key={`${entry.method}-${entry.path}-${entry.requestId ?? entry.time}-${entry.status}`}
          >
            <div className="grid grid-cols-[64px_48px_1fr] gap-2">
              <span className="font-mono text-[11px] text-[var(--text-muted)]">{entry.time}</span>
              <span className="font-mono text-[11px] font-semibold text-[var(--accent)]">{entry.method}</span>
              <span className="truncate font-mono text-[11px] text-[var(--text-secondary)]">{entry.path}</span>
            </div>
            <div className="mt-2 flex items-center justify-between gap-3">
              <span className="text-xs text-[var(--text-secondary)]">{entry.status}</span>
              {entry.requestId ? (
                <span className="truncate font-mono text-[10px] uppercase tracking-[0.1em] text-[var(--text-muted)]">
                  {entry.requestId}
                </span>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  )
}

function Panel({
  children,
  className,
  eyebrow,
  icon,
  title,
}: {
  children: ReactNode
  className?: string
  eyebrow: string
  icon: ReactNode
  title: string
}) {
  return (
    <section
      className={cn(
        'rounded-[24px] border border-[var(--border-subtle)] bg-[var(--surface-shell)]/90 p-4 shadow-[var(--shadow-soft)] backdrop-blur-xl',
        className,
      )}
    >
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-[var(--text-muted)]">{eyebrow}</p>
          <h2 className="mt-1 flex items-center gap-2 text-base font-semibold tracking-[-0.02em] text-[var(--text-primary)]">
            {icon}
            {title}
          </h2>
        </div>
      </div>
      {children}
    </section>
  )
}

function InputField({
  error,
  helper,
  label,
  onChange,
  placeholder,
  value,
}: {
  error?: string
  helper?: string
  label: string
  onChange: (value: string) => void
  placeholder?: string
  value: string
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
        {label}
      </span>
      <input
        className={cn(
          'w-full rounded-2xl border bg-[var(--surface-card)] px-3 py-2 text-sm outline-none transition focus:ring-4 focus:ring-[var(--accent-ring)]',
          error
            ? 'border-[var(--error)] focus:border-[var(--error)]'
            : 'border-[var(--border-strong)] focus:border-[var(--accent)]',
        )}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        value={value}
      />
      {error ? (
        <p className="mt-1 text-xs text-[var(--error)]">{error}</p>
      ) : helper ? (
        <p className="mt-1 text-xs text-[var(--text-muted)]">{helper}</p>
      ) : null}
    </label>
  )
}

function TextAreaField({
  error,
  helper,
  label,
  onChange,
  onKeyDown,
  placeholder,
  rows = 5,
  value,
}: {
  error?: string
  helper?: string
  label: string
  onChange: (value: string) => void
  onKeyDown?: (event: KeyboardEvent<HTMLTextAreaElement>) => void
  placeholder?: string
  rows?: number
  value: string
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
        {label}
      </span>
      <textarea
        className={cn(
          'w-full resize-none rounded-2xl border bg-[var(--surface-card)] px-4 py-3 text-sm leading-6 outline-none transition focus:ring-4 focus:ring-[var(--accent-ring)]',
          error
            ? 'border-[var(--error)] focus:border-[var(--error)]'
            : 'border-[var(--border-strong)] focus:border-[var(--accent)]',
        )}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        rows={rows}
        value={value}
      />
      {error ? (
        <p className="mt-1 text-xs text-[var(--error)]">{error}</p>
      ) : helper ? (
        <p className="mt-1 text-xs text-[var(--text-muted)]">{helper}</p>
      ) : null}
    </label>
  )
}

function FieldPreview({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-card)] px-3 py-2">
      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">{label}</p>
      <p className="mt-1 truncate font-mono text-xs text-[var(--text-primary)]">{value}</p>
    </div>
  )
}

function ManagedModelLoadCard({
  progress,
}: {
  progress: ReturnType<typeof deriveManagedModelLoadProgress>
}) {
  if (!progress) {
    return null
  }

  return (
    <div className="relative overflow-hidden rounded-[26px] border border-[var(--info-border)] bg-[linear-gradient(135deg,rgba(42,96,220,0.08),rgba(255,255,255,0.96))] px-4 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.72)]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(42,96,220,0.16),transparent_32%),radial-gradient(circle_at_bottom_left,rgba(28,124,112,0.12),transparent_30%)]" />
      <div className="relative space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <StatusPill icon={<Box className="size-3.5" />} tone="info">
                Tensor loading
              </StatusPill>
              <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-[var(--text-muted)]">
                {progress.percentLabel}
              </span>
            </div>
            <p className="mt-2 text-sm font-semibold text-[var(--text-primary)]">{progress.modelLabel}</p>
            <p className="mt-1 text-sm leading-6 text-[var(--text-secondary)]">
              Native loader logs are reporting live tensor intake. The console does not invent an ETA.
            </p>
          </div>
          <div className="shrink-0 rounded-2xl border border-white/70 bg-white/70 px-3 py-2 text-right shadow-[0_10px_24px_rgba(42,96,220,0.08)]">
            <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-[var(--text-muted)]">Tensors</p>
            <p className="mt-1 font-mono text-sm font-semibold text-[var(--text-primary)]">{progress.tensorLabel}</p>
          </div>
        </div>

        <div className="space-y-2">
          <div
            aria-label="Model load progress"
            aria-valuemax={progress.total}
            aria-valuemin={0}
            aria-valuenow={progress.current}
            className="h-2.5 overflow-hidden rounded-full bg-white/80 shadow-[inset_0_1px_3px_rgba(31,42,56,0.08)]"
            role="progressbar"
          >
            <div
              className="h-full rounded-full bg-[linear-gradient(90deg,var(--accent),#3cb7a7)] transition-[width] duration-700 ease-out"
              style={{ width: `${progress.percent}%` }}
            />
          </div>
          <div className="flex items-center justify-between gap-3 font-mono text-[11px] text-[var(--text-muted)]">
            <span>{progress.current} loaded</span>
            <span>{progress.total} total</span>
          </div>
        </div>

        <div className="grid gap-2 sm:grid-cols-3">
          <FieldPreview label="Tensors" value={progress.tensorLabel} />
          <FieldPreview label="Memory" value={progress.memoryLabel} />
          <FieldPreview label="Elapsed" value={progress.elapsedLabel} />
        </div>
      </div>
    </div>
  )
}

function ResultPlaceholder({
  body,
  title,
  tone = 'info',
}: {
  body: string
  title: string
  tone?: 'error' | 'info' | 'warning'
}) {
  return (
    <div className="relative overflow-hidden rounded-[24px] border border-[var(--border-strong)] bg-[linear-gradient(135deg,#eef2f4,#dfe8e6)]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_32%_22%,rgba(42,96,220,0.18),transparent_28%),radial-gradient(circle_at_74%_66%,rgba(28,124,112,0.16),transparent_30%)]" />
      <div className="relative grid min-h-[320px] place-items-center p-8 text-center">
        <div>
          <div
            className={cn(
              'mx-auto grid size-16 place-items-center rounded-3xl border bg-white/55 shadow-xl backdrop-blur',
              tone === 'info' && 'border-white/70 text-[var(--accent)]',
              tone === 'warning' && 'border-[var(--warning-border)] text-[var(--warning)]',
              tone === 'error' && 'border-[var(--error-border)] text-[var(--error)]',
            )}
          >
            <Image className="size-7" />
          </div>
          <h2 className="mt-5 text-xl font-semibold tracking-[-0.03em] text-[var(--text-primary)]">{title}</h2>
          <p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-[var(--text-secondary)]">{body}</p>
        </div>
      </div>
    </div>
  )
}

function ButtonPrimary({
  children,
  className,
  disabled,
  icon,
  onClick,
}: {
  children: ReactNode
  className?: string
  disabled?: boolean
  icon?: ReactNode
  onClick?: () => void
}) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-xl bg-[var(--accent)] px-3 py-2 text-sm font-semibold text-white shadow-[0_12px_28px_rgba(42,96,220,0.26)] transition hover:-translate-y-0.5 hover:bg-[var(--accent-hover)] focus:outline-none focus:ring-4 focus:ring-[var(--accent-ring)] disabled:cursor-not-allowed disabled:opacity-55 disabled:hover:translate-y-0',
        className,
      )}
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      {icon}
      {children}
    </button>
  )
}

function ButtonGhost({
  children,
  className,
  disabled,
  icon,
  onClick,
}: {
  children: ReactNode
  className?: string
  disabled?: boolean
  icon?: ReactNode
  onClick?: () => void
}) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-xl border border-[var(--border-subtle)] bg-[var(--surface-card)] px-3 py-2 text-sm font-semibold text-[var(--text-primary)] transition hover:-translate-y-0.5 hover:border-[var(--accent-border)] hover:bg-[var(--surface-hover)] focus:outline-none focus:ring-4 focus:ring-[var(--accent-ring)] disabled:cursor-not-allowed disabled:opacity-55 disabled:hover:translate-y-0',
        className,
      )}
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      {icon}
      {children}
    </button>
  )
}

function StatusPill({
  children,
  icon,
  tone = 'neutral',
}: {
  children: ReactNode
  icon?: ReactNode
  tone?: 'error' | 'info' | 'neutral' | 'success' | 'warning'
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-semibold',
        tone === 'info' && 'border-[var(--info-border)] bg-[var(--info-soft)] text-[var(--info)]',
        tone === 'neutral' && 'border-[var(--border-subtle)] bg-[var(--surface-card)] text-[var(--text-secondary)]',
        tone === 'success' && 'border-[var(--success-border)] bg-[var(--success-soft)] text-[var(--success)]',
        tone === 'warning' && 'border-[var(--warning-border)] bg-[var(--warning-soft)] text-[var(--warning)]',
        tone === 'error' && 'border-[var(--error-border)] bg-[var(--error-soft)] text-[var(--error)]',
      )}
    >
      {icon}
      {children}
    </span>
  )
}

function StatusDot({ tone }: { tone: 'info' | 'success' }) {
  return (
    <span
      className={cn(
        'size-2.5 rounded-full shadow-[0_0_0_4px_var(--surface-muted)]',
        tone === 'info' && 'bg-[var(--info)]',
        tone === 'success' && 'bg-[var(--success)]',
      )}
    />
  )
}

function InlineError({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-2xl border border-[var(--error-border)] bg-[var(--error-soft)] px-3 py-3">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--error)]">{title}</p>
      <p className="mt-1 text-sm leading-6 text-[var(--text-primary)]">{value}</p>
    </div>
  )
}

function InlineInfo({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-2xl border border-[var(--info-border)] bg-[var(--info-soft)] px-3 py-3">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--info)]">{title}</p>
      <p className="mt-1 text-sm leading-6 text-[var(--text-primary)]">{value}</p>
    </div>
  )
}

function StatusBanner({ status }: { status: ConnectionStatus }) {
  const copy =
    status === 'connected'
      ? 'Health and capabilities both succeeded. Composer can now be driven by the live backend contract.'
      : status === 'connecting'
        ? 'Running health first, then capabilities. The UI stays conservative until both responses return.'
        : status === 'health_failed'
          ? 'The backend target could not answer health. Check the Python server process, base URL, or CORS/proxy path.'
          : status === 'capabilities_failed'
            ? 'Health succeeded but capabilities failed. The service is reachable, but capability-driven UI stays disabled.'
            : 'No connection attempt yet.'

  return (
    <div className="rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-muted)] px-3 py-3 text-sm leading-6 text-[var(--text-secondary)]">
      {copy}
    </div>
  )
}

function getBrowserNotificationPermission(): BrowserNotificationState {
  if (typeof window === 'undefined' || !('Notification' in window)) {
    return 'unsupported'
  }

  return window.Notification.permission
}

function maybeSendBrowserNotification(
  permission: BrowserNotificationState,
  title: string,
  message: string,
) {
  if (permission !== 'granted' || typeof window === 'undefined' || !('Notification' in window)) {
    return
  }

  try {
    new window.Notification(title, {
      body: message,
      tag: `${title}:${message}`,
    })
  } catch {
    // Ignore browser notification failures and keep in-app toasts alive.
  }
}

function formatNotificationPermission(permission: BrowserNotificationState) {
  switch (permission) {
    case 'granted':
      return 'notifications on'
    case 'denied':
      return 'notifications blocked'
    case 'unsupported':
      return 'notifications unavailable'
    default:
      return 'notifications optional'
  }
}

function managedBackendTone(status?: ManagedBackendStatus | null): 'error' | 'info' | 'neutral' | 'success' | 'warning' {
  if (!status || status === 'idle') {
    return 'neutral'
  }
  if (status === 'running') {
    return 'success'
  }
  if (status === 'crashed') {
    return 'error'
  }
  if (status === 'unhealthy' || status === 'restarting') {
    return 'warning'
  }
  return 'info'
}

function formatManagedBackendStatus(status?: ManagedBackendStatus | null) {
  switch (status) {
    case 'starting':
      return 'backend starting'
    case 'running':
      return 'backend running'
    case 'restarting':
      return 'backend restarting'
    case 'stopping':
      return 'backend stopping'
    case 'unhealthy':
      return 'backend unhealthy'
    case 'crashed':
      return 'backend crashed'
    case 'idle':
      return 'backend idle'
    default:
      return 'backend unknown'
  }
}

function buildManagedRuntimeHint(
  backend: ManagedRuntimeStatusResponse['backend'] | null,
  runtimeError: unknown,
) {
  if (runtimeError) {
    return 'The local runtime manager is not answering. Start `npm run runtime:manager` or `npm run dev:managed` to recover crash diagnostics and backend controls.'
  }

  if (!backend) {
    return null
  }

  if (backend.status === 'starting' || backend.status === 'restarting') {
    return `${backend.profile?.name ?? 'Managed backend'} is restarting. The console should recover automatically once health probes succeed.`
  }

  if (backend.status === 'unhealthy') {
    return 'Managed health probes are failing. The runtime manager will keep checking and can restart the backend if the failures continue.'
  }

  if (backend.status === 'crashed') {
    return 'Managed backend crashed and exhausted auto-restart. Open Local Runtime for the last exit reason and log tail.'
  }

  if (backend.status === 'idle' && backend.profile) {
    return `${backend.profile.name} is currently stopped. Start or restart it from Local Runtime.`
  }

  return null
}

function buildManagedRuntimeStatusCopy(
  backend: ManagedRuntimeStatusResponse['backend'] | null,
  profile: ManagedRuntimeProfile | null,
) {
  if (!backend) {
    return 'Runtime manager is online. Pick a verified model, start it, and the console will follow the same managed target automatically.'
  }

  if (backend.status === 'running') {
    return `${backend.profile?.name ?? profile?.name ?? 'Selected profile'} is healthy. If it exits unexpectedly, the manager will attempt an in-place recovery on the same 8080 backend slot.`
  }

  if (backend.status === 'starting') {
    return `${backend.profile?.name ?? profile?.name ?? 'Selected profile'} is booting. Health and capabilities will unlock once the Python server finishes loading the model.`
  }

  if (backend.status === 'restarting') {
    return `${backend.profile?.name ?? profile?.name ?? 'Selected profile'} is being restarted${backend.next_restart_at_ms ? ` in about ${formatCountdown(backend.next_restart_at_ms)}` : ''}. Keep the page open to watch recovery events.`
  }

  if (backend.status === 'unhealthy') {
    return `Health probes are failing for ${backend.profile?.name ?? profile?.name ?? 'the managed backend'}. If the streak keeps climbing, the manager will trigger a restart automatically.`
  }

  if (backend.status === 'crashed') {
    return `${backend.profile?.name ?? profile?.name ?? 'The managed backend'} is currently down. Review the exit reason below, then start it again or switch to a lighter verified profile.`
  }

  if (backend.profile) {
    return `${backend.profile.name} is selected but not currently serving. Start it again whenever you are ready.`
  }

  return 'No managed backend is active yet. Select a verified profile and start it from this panel.'
}

function formatHealthSummary(lastHealth: ManagedRuntimeStatusResponse['backend']['last_health']) {
  if (!lastHealth) {
    return 'not probed yet'
  }

  if (lastHealth.ok) {
    return `${lastHealth.response_ms} ms ok`
  }

  return `${lastHealth.status ?? 'network'} · ${lastHealth.error ?? 'probe failed'}`
}

function formatLastExit(lastExit: ManagedRuntimeStatusResponse['backend']['last_exit']) {
  if (!lastExit) {
    return 'no exit observed yet'
  }

  if (lastExit.intentional) {
    return `intentional · ${lastExit.reason}`
  }

  if (lastExit.signal) {
    return `signal ${lastExit.signal}`
  }

  return `code ${lastExit.code ?? 'unknown'}`
}

function formatCountdown(targetTimeMs: number) {
  const deltaMs = Math.max(0, targetTimeMs - Date.now())
  const seconds = Math.max(1, Math.ceil(deltaMs / 1000))
  return `${seconds}s`
}

function formatTimestamp(timeMs: number) {
  return new Date(timeMs).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function deriveManagedModelLoadProgress(
  backend: ManagedRuntimeStatusResponse['backend'] | null,
  logTail: ManagedRuntimeStatusResponse['log_tail'],
  profile: ManagedRuntimeProfile | null,
) {
  if (!backend || (backend.status !== 'starting' && backend.status !== 'restarting')) {
    return null
  }

  for (let index = logTail.length - 1; index >= 0; index -= 1) {
    const parsed = parseTensorLoadLine(logTail[index]?.line ?? '')
    if (!parsed) {
      continue
    }

    const percent = parsed.total > 0 ? Math.min(100, Math.max(0, (parsed.current / parsed.total) * 100)) : 0
    const modelLabel = backend.profile?.name ?? profile?.name ?? 'Managed backend'

    return {
      current: parsed.current,
      elapsedLabel: formatElapsedSeconds(parsed.elapsedSeconds),
      memoryLabel: formatTensorLoadMemory(parsed.memoryValue, parsed.memoryUnit),
      modelLabel,
      percent,
      percentLabel: `${Math.round(percent)}%`,
      tensorLabel: `${parsed.current} / ${parsed.total}`,
      total: parsed.total,
    }
  }

  return null
}

function parseTensorLoadLine(line: string) {
  const match =
    /loading tensors\s+(\d+)\/(\d+)\s+(\d+(?:\.\d+)?)(mb|mib|gb|gib)\s+(\d+(?:\.\d+)?)s/i.exec(line)

  if (!match) {
    return null
  }

  const current = Number.parseInt(match[1] ?? '', 10)
  const total = Number.parseInt(match[2] ?? '', 10)
  const memoryValue = Number.parseFloat(match[3] ?? '')
  const memoryUnit = (match[4] ?? 'MB').toUpperCase()
  const elapsedSeconds = Number.parseFloat(match[5] ?? '')

  if (
    !Number.isFinite(current) ||
    !Number.isFinite(total) ||
    !Number.isFinite(memoryValue) ||
    !Number.isFinite(elapsedSeconds) ||
    total <= 0
  ) {
    return null
  }

  return {
    current,
    elapsedSeconds,
    memoryUnit,
    memoryValue,
    total,
  }
}

function formatTensorLoadMemory(value: number, unit: string) {
  if (!Number.isFinite(value)) {
    return 'n/a'
  }

  if (unit === 'GB' || unit === 'GIB') {
    return `${value.toFixed(2)} ${unit}`
  }

  if (value >= 1024) {
    const nextUnit = unit === 'MIB' ? 'GiB' : 'GB'
    const divisor = unit === 'MIB' ? 1024 : 1000
    return `${(value / divisor).toFixed(2)} ${nextUnit}`
  }

  return `${value.toFixed(0)} ${unit}`
}

function formatElapsedSeconds(seconds: number) {
  if (!Number.isFinite(seconds)) {
    return 'n/a'
  }

  if (seconds >= 60) {
    return `${(seconds / 60).toFixed(1)} min`
  }

  return `${seconds.toFixed(1)} s`
}

function eventTone(level: ManagedRuntimeStatusResponse['recent_events'][number]['level']) {
  if (level === 'error') {
    return 'error' as const
  }
  if (level === 'warning') {
    return 'warning' as const
  }
  return 'info' as const
}

function isSameConnectionTarget(left: ConnectionTarget, right: ConnectionTarget) {
  return left.baseUrl === right.baseUrl && left.prefix === right.prefix
}

function requestPayloadFitsDraft(payload: Record<string, unknown>, kind: ComposerMode) {
  const knownKeys =
    kind === 'image'
      ? [
          'batch_count',
          'cfg_scale',
          'eta',
          'flow_shift',
          'guidance',
          'height',
          'image_cfg_scale',
          'negative_prompt',
          'prompt',
          'sampler',
          'scheduler',
          'seed',
          'steps',
          'width',
        ]
      : kind === 'edit_image'
        ? [
            'cfg_scale',
            'flow_shift',
            'guidance',
            'height',
            'init_image_b64',
            'negative_prompt',
            'prompt',
            'ref_images_b64',
            'seed',
            'steps',
            'width',
          ]
      : [
          'cfg_scale',
          'eta',
          'flow_shift',
          'frames',
          'guidance',
          'height',
          'negative_prompt',
          'prompt',
          'sampler',
          'scheduler',
          'seed',
          'steps',
          'width',
        ]

  if (!Object.keys(payload).every((key) => knownKeys.includes(key))) {
    return false
  }
  if (kind !== 'edit_image') {
    return true
  }

  if ('init_image_b64' in payload) {
    return typeof payload.init_image_b64 === 'string' && looksLikeInlineImagePayload(payload.init_image_b64)
  }
  if ('ref_images_b64' in payload) {
    return (
      Array.isArray(payload.ref_images_b64) &&
      payload.ref_images_b64.length === 1 &&
      typeof payload.ref_images_b64[0] === 'string' &&
      looksLikeInlineImagePayload(payload.ref_images_b64[0])
    )
  }
  return false
}

function payloadString(value: unknown) {
  if (typeof value === 'number' || typeof value === 'string') {
    return String(value)
  }
  return ''
}

function buildImageDraftFromPayload(payload: Record<string, unknown>): ImageDraft {
  return {
    ...DEFAULT_IMAGE_DRAFT,
    batchCount: payloadString(payload.batch_count),
    cfgScale: payloadString(payload.cfg_scale),
    eta: payloadString(payload.eta),
    flowShift: payloadString(payload.flow_shift),
    guidance: payloadString(payload.guidance),
    height: payloadString(payload.height) || DEFAULT_IMAGE_DRAFT.height,
    imageCfgScale: payloadString(payload.image_cfg_scale),
    negativePrompt: typeof payload.negative_prompt === 'string' ? payload.negative_prompt : '',
    prompt: typeof payload.prompt === 'string' ? payload.prompt : '',
    sampler: typeof payload.sampler === 'string' ? payload.sampler : '',
    scheduler: typeof payload.scheduler === 'string' ? payload.scheduler : '',
    seed: payloadString(payload.seed),
    steps: payloadString(payload.steps) || DEFAULT_IMAGE_DRAFT.steps,
    width: payloadString(payload.width) || DEFAULT_IMAGE_DRAFT.width,
  }
}

function buildVideoDraftFromPayload(payload: Record<string, unknown>): VideoDraft {
  return {
    ...DEFAULT_VIDEO_DRAFT,
    cfgScale: payloadString(payload.cfg_scale),
    eta: payloadString(payload.eta),
    flowShift: payloadString(payload.flow_shift),
    frames: payloadString(payload.frames) || DEFAULT_VIDEO_DRAFT.frames,
    guidance: payloadString(payload.guidance),
    height: payloadString(payload.height) || DEFAULT_VIDEO_DRAFT.height,
    negativePrompt: typeof payload.negative_prompt === 'string' ? payload.negative_prompt : '',
    prompt: typeof payload.prompt === 'string' ? payload.prompt : '',
    sampler: typeof payload.sampler === 'string' ? payload.sampler : '',
    scheduler: typeof payload.scheduler === 'string' ? payload.scheduler : '',
    seed: payloadString(payload.seed),
    steps: payloadString(payload.steps) || DEFAULT_VIDEO_DRAFT.steps,
    width: payloadString(payload.width) || DEFAULT_VIDEO_DRAFT.width,
  }
}

function payloadHasEditImageInput(payload: Record<string, unknown>) {
  return 'init_image_b64' in payload || 'ref_images_b64' in payload
}

function looksLikeInlineImagePayload(value: string) {
  return value.startsWith('data:image/') || /^[A-Za-z0-9+/=]+$/.test(value)
}

function formatComposerModeLabel(mode: ComposerMode) {
  switch (mode) {
    case 'edit_image':
      return 'edit image'
    case 'video':
      return 'video'
    default:
      return 'image'
  }
}

function buildEditImagePreviewSrc(value: string) {
  if (!value.trim()) {
    return null
  }
  if (value.startsWith('data:image/')) {
    return value
  }
  if (/^[A-Za-z0-9+/=]+$/.test(value)) {
    return `data:image/png;base64,${value}`
  }
  return null
}

function collectModelIdentityTokens(
  selectedProfile: ManagedRuntimeProfile | null,
  backendProfile: ManagedRuntimeProfile | null,
  capabilities?: CapabilitiesResponse,
) {
  return [
    ...(backendProfile?.slug ? [backendProfile.slug] : []),
    ...(backendProfile ? [] : selectedProfile?.slug ? [selectedProfile.slug] : []),
    capabilities?.pipeline_name,
    capabilities?.version_name,
    capabilities?.model,
  ]
    .filter((value): value is string => Boolean(value))
    .map((value) => value.toLowerCase())
}

function deriveEditImageBinding(
  selectedProfile: ManagedRuntimeProfile | null,
  backendProfile: ManagedRuntimeProfile | null,
  capabilities?: CapabilitiesResponse,
) {
  const tokens = collectModelIdentityTokens(selectedProfile, backendProfile, capabilities)

  if (tokens.some((value) => value.includes('flux-kontext'))) {
    return {
      helper: 'FLUX.1-Kontext-dev expects at least one reference image. The console sends your upload as `ref_images_b64[0]`.',
      inputLabel: 'Reference image',
      modeLabel: 'Kontext edit mode',
      shortHelper: 'A reference image is required for this model.',
      transport: 'ref_images_b64' as EditImageTransport,
    }
  }

  if (tokens.some((value) => value.includes('qwen-image-edit'))) {
    return {
      helper: 'Qwen-Image-Edit expects an init image. The console sends your upload as `init_image_b64`.',
      inputLabel: 'Input image',
      modeLabel: 'Image edit mode',
      shortHelper: 'An input image is required for this model.',
      transport: 'init_image_b64' as EditImageTransport,
    }
  }

  return {
    helper: 'The current model does not advertise a verified single-image edit contract in this console yet.',
    inputLabel: 'Input image',
    modeLabel: 'Unverified edit mode',
    shortHelper: 'Use a verified edit profile to unlock the upload flow.',
    transport: null as EditImageTransport,
  }
}

function deriveManagedEditLogStepProgress(
  logTail: ManagedRuntimeStatusResponse['log_tail'],
  selectedProfile: ManagedRuntimeProfile | null,
  backendProfile: ManagedRuntimeProfile | null,
  capabilities?: CapabilitiesResponse,
) {
  const tokens = collectModelIdentityTokens(selectedProfile, backendProfile, capabilities)
  const matcher = tokens.some((value) => value.includes('flux-kontext'))
    ? /flux-kontext\b.*?\bstep\s+(\d+)\/(\d+)/i
    : tokens.some((value) => value.includes('qwen-image-edit'))
      ? /qwen-image-edit\b.*?\bstep\s+(\d+)\/(\d+)/i
      : null

  if (!matcher) {
    return null
  }

  for (let index = logTail.length - 1; index >= 0; index -= 1) {
    const line = logTail[index]?.line ?? ''
    const match = matcher.exec(line)
    if (!match) {
      continue
    }

    const current = Number.parseInt(match[1] ?? '', 10)
    const total = Number.parseInt(match[2] ?? '', 10)

    if (Number.isFinite(current) && Number.isFinite(total) && total > 0) {
      return {
        current,
        total,
      }
    }
  }

  return null
}

function readFileAsDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error('File read failed'))
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result)
        return
      }
      reject(new Error('File read failed'))
    }
    reader.readAsDataURL(file)
  })
}

function connectionTone(status: ConnectionStatus): 'error' | 'info' | 'neutral' | 'success' {
  if (status === 'connected') {
    return 'success'
  }
  if (status === 'health_failed' || status === 'capabilities_failed') {
    return 'error'
  }
  if (status === 'connecting') {
    return 'info'
  }
  return 'neutral'
}

function formatConnectionStatus(status: ConnectionStatus) {
  switch (status) {
    case 'connected':
      return 'connected'
    case 'connecting':
      return 'connecting'
    case 'health_failed':
      return 'health failed'
    case 'capabilities_failed':
      return 'capabilities failed'
    default:
      return 'idle'
  }
}

function buildCapabilityRows(capabilities?: CapabilitiesResponse) {
  if (!capabilities) {
    return [
      { label: 'Image generation', value: 'waiting for capabilities', tone: 'info' as const },
      { label: 'Video generation', value: 'waiting for capabilities', tone: 'info' as const },
      { label: 'Progress', value: 'sampling_step_only', tone: 'info' as const },
      { label: 'Results', value: 'stored_in_memory', tone: 'info' as const },
    ]
  }

  return [
    {
      label: 'Image generation',
      value: capabilities.supports.image ? 'supported by loaded model' : 'not supported by loaded model',
      tone: capabilities.supports.image ? ('success' as const) : ('info' as const),
    },
    {
      label: 'Video generation',
      value: capabilities.supports.video ? 'supported by loaded model' : 'not supported by loaded model',
      tone: capabilities.supports.video ? ('success' as const) : ('info' as const),
    },
    {
      label: 'Progress',
      value: capabilities.semantics.progress,
      tone: 'info' as const,
    },
    {
      label: 'Results',
      value: capabilities.semantics.results,
      tone: 'info' as const,
    },
  ]
}

function buildConnectionLogRows(
  activeTarget: ConnectionTarget,
  health?: { request_id: string },
  capabilities?: { request_id: string },
  healthError?: unknown,
  capabilitiesError?: unknown,
) {
  const rows: Array<{ method: string; path: string; requestId?: string; status: string; time: string }> = []

  if (health) {
    rows.push({
      method: 'GET',
      path: `${activeTarget.prefix}/health`,
      requestId: health.request_id,
      status: '200 ok',
      time: 'live',
    })
  } else if (healthError) {
    rows.push({
      method: 'GET',
      path: `${activeTarget.prefix}/health`,
      status: describeApiFailure(healthError),
      time: 'live',
    })
  }

  if (capabilities) {
    rows.push({
      method: 'GET',
      path: `${activeTarget.prefix}/capabilities`,
      requestId: capabilities.request_id,
      status: '200 ok',
      time: 'live',
    })
  } else if (capabilitiesError) {
    rows.push({
      method: 'GET',
      path: `${activeTarget.prefix}/capabilities`,
      status: describeApiFailure(capabilitiesError),
      time: 'live',
    })
  }

  return rows.length > 0 ? rows : [...placeholderLogRows]
}

function deriveProgressVm(
  job: EdgeDitJob | EdgeDitJobSummary | null | undefined,
  managedEditLogStepProgress?: {
    current: number
    total: number
  } | null,
) {
  if (!job) {
    return {
      description: 'Select a job to inspect truthful step progress from `GET /jobs/{id}`.',
      percent: 0,
      phaseLabel: 'No selected job',
      stepsLabel: '0 / 0',
      tone: 'neutral' as const,
    }
  }

  if (!hasJobProgress(job)) {
    return {
      description:
        'Task list items come from `GET /jobs`, which does not include step counters. The console is waiting for `GET /jobs/{id}` detail to report progress.',
      percent: job.status === 'succeeded' ? 100 : 0,
      phaseLabel: summaryPhaseLabel(job.status),
      stepsLabel: job.status === 'succeeded' ? 'done' : 'pending detail',
      tone: jobTone(job.status),
    }
  }

  const total = job.progress.total_steps
  const current = job.progress.current_step
  const requestedSteps =
    typeof job.parameters.steps === 'number' && job.parameters.steps > 0 ? job.parameters.steps : 0
  const requestedBatchCount =
    job.kind === 'image' &&
    typeof job.parameters.batch_count === 'number' &&
    job.parameters.batch_count > 0
      ? job.parameters.batch_count
      : 1
  const requestedTotal = requestedSteps * requestedBatchCount
  const observedTotal = total > 0 ? total : managedEditLogStepProgress?.total ?? requestedTotal
  const observedCurrent = total > 0 ? current : managedEditLogStepProgress?.current ?? current
  const percent =
    observedTotal > 0
      ? Math.min(100, Math.max(0, (observedCurrent / observedTotal) * 100))
      : job.status === 'succeeded'
        ? 100
        : 0
  const samplingBudgetPublished = total > 0
  const hasManagedLogFallback = !samplingBudgetPublished && Boolean(managedEditLogStepProgress)
  const samplingStarted = observedCurrent > 0

  if (job.status === 'queued') {
    return {
      description: 'The backend accepted the request and queued it behind the single serial worker.',
      percent,
      phaseLabel: 'Queued',
      stepsLabel: `${observedCurrent} / ${observedTotal}`,
      tone: 'neutral' as const,
    }
  }
  if (job.status === 'running') {
    if (hasManagedLogFallback) {
      return {
        description:
          'Sampling progress is derived from managed backend step logs because this verified edit pipeline is not publishing `GET /jobs/{id}` counters yet.',
        percent,
        phaseLabel: 'Sampling',
        stepsLabel: `${observedCurrent} / ${observedTotal}`,
        tone: 'info' as const,
      }
    }
    if (!samplingStarted) {
      return {
        description:
          observedTotal > 0
            ? `The sampling budget is ${observedTotal} steps. The backend is preparing inputs or running the first sampling step.`
            : 'The backend has started the job and is preparing inputs; no sampling budget is available yet.',
        percent,
        phaseLabel: 'Preparing',
        stepsLabel: `${observedCurrent} / ${observedTotal}`,
        tone: 'info' as const,
      }
    }
    return {
      description: samplingBudgetPublished
        ? 'Sampling progress comes directly from the active backend job. The console does not invent an ETA.'
        : 'Sampling progress is being observed while the requested sampling budget supplies the total.',
      percent,
      phaseLabel: 'Sampling',
      stepsLabel: `${observedCurrent} / ${observedTotal}`,
      tone: 'info' as const,
    }
  }
  if (job.status === 'cancelling') {
    return {
      description: hasManagedLogFallback
        ? 'Cancellation is cooperative. Step progress is being inferred from managed backend logs for this edit pipeline.'
        : 'Cancellation is cooperative and resolves at step boundaries according to server semantics.',
      percent,
      phaseLabel: 'Cancelling',
      stepsLabel: `${observedCurrent} / ${observedTotal}`,
      tone: 'warning' as const,
    }
  }
  if (job.status === 'succeeded') {
    return {
      description: hasManagedLogFallback
        ? 'The backend finished successfully. Sampling steps were recovered from managed backend logs while result JSON is being fetched.'
        : 'The backend finished successfully. The console will fetch result JSON until it is ready to decode.',
      percent: 100,
      phaseLabel: 'Finalized',
      stepsLabel: observedTotal > 0 ? `${observedTotal} / ${observedTotal}` : 'done',
      tone: 'success' as const,
    }
  }

  return {
    description: job.error ?? 'The selected job ended before producing a usable result.',
    percent,
    phaseLabel: job.status === 'failed' ? 'Failed' : 'Cancelled',
    stepsLabel: `${observedCurrent} / ${observedTotal}`,
    tone: 'warning' as const,
  }
}

function appendActivityLog(
  setRows: Dispatch<SetStateAction<ActivityLogEntry[]>>,
  entry: Omit<ActivityLogEntry, 'id' | 'time'>,
) {
  setRows((rows) =>
    [
      {
        ...entry,
        id: activityId(),
        time: activityTime(),
      },
      ...rows,
    ].slice(0, 12),
  )
}

function activityId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `log_${Date.now()}_${Math.random().toString(16).slice(2)}`
}

function activityTime() {
  return new Date().toLocaleTimeString('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function summarizeJobParameters(job: EdgeDitJob | EdgeDitJobSummary) {
  const width = typeof job.parameters.width === 'number' ? job.parameters.width : 'n/a'
  const height = typeof job.parameters.height === 'number' ? job.parameters.height : 'n/a'
  const steps = typeof job.parameters.steps === 'number' ? job.parameters.steps : 'n/a'
  if (job.kind === 'video') {
    const frames = typeof job.parameters.frames === 'number' ? job.parameters.frames : 'n/a'
    return `${width} x ${height} · ${frames} frames · ${steps} steps`
  }
  return `${width} x ${height} · ${steps} steps`
}

function summarizeResult(result: EdgeDitGenerationResult) {
  if (isImageGenerationResult(result)) {
    return `200 ok · ${result.data.length} image${result.data.length === 1 ? '' : 's'}`
  }
  return `200 ok · ${result.frames.length} frame${result.frames.length === 1 ? '' : 's'}`
}

function jobTone(status: JobStatus): 'error' | 'info' | 'neutral' | 'success' | 'warning' {
  if (status === 'succeeded') {
    return 'success'
  }
  if (status === 'running') {
    return 'info'
  }
  if (status === 'cancelling') {
    return 'warning'
  }
  if (status === 'failed') {
    return 'error'
  }
  return 'neutral'
}

function summaryPhaseLabel(status: JobStatus) {
  switch (status) {
    case 'queued':
      return 'Queued'
    case 'running':
      return 'Loading detail'
    case 'cancelling':
      return 'Cancelling'
    case 'succeeded':
      return 'Finalized'
    case 'failed':
      return 'Failed'
    case 'cancelled':
      return 'Cancelled'
    default:
      return 'Selected job'
  }
}

function safeParseJson(raw: string) {
  try {
    return JSON.parse(raw)
  } catch {
    return {
      invalid_json_buffer: raw,
    }
  }
}

function jsonDownloadHref(payload: unknown) {
  return `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(payload, null, 2))}`
}

function saveBlobDownload(blob: Blob, filename: string) {
  const href = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = href
  anchor.download = filename
  anchor.click()
  setTimeout(() => URL.revokeObjectURL(href), 0)
}
