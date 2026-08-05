import { http, HttpResponse } from 'msw'

const prefixes = ['/ed/v2', '/edgedit/v2', '/edge-dit/v2']

const runtimeStatusPayload = {
  backend: {
    auto_restart_enabled: true,
    auto_restart_limit: 3,
    base_url: 'http://127.0.0.1:8080',
    capabilities: {
      model: 'edge-dit-model',
      supports: {
        image: true,
        video: true,
      },
    },
    health_failure_streak: 0,
    last_exit: null,
    last_health: {
      checked_at_ms: 1,
      error: null,
      ok: true,
      request_id: 'runtime-health',
      response_ms: 18,
      status: 200,
    },
    last_ready_at_ms: 1,
    next_restart_at_ms: null,
    pid: 424242,
    profile: {
      description: 'Stable local image smoke preset for FLUX.1-dev.',
      kind: 'image',
      model_env: 'EDGE_DIT_FLUX_MODEL_PATH',
      name: 'FLUX.1-dev',
      notes: ['Validated through the Python Server HTTP job API in this workspace.'],
      request_example: {
        height: 256,
        prompt: 'smoke test teapot',
        seed: 42,
        steps: 1,
        width: 256,
      },
      slug: 'flux-dev',
    },
    restart_count_consecutive: 0,
    restart_count_total: 0,
    started_at_ms: 1,
    status: 'running',
    uptime_ms: 5000,
  },
  log_tail: [
    {
      line: 'INFO backend ready',
      stream: 'stdout',
      time_ms: 1,
    },
  ],
  manager: {
    app_root: '/tmp/server-console',
    host: '127.0.0.1',
    port: 8090,
    started_at_ms: 1,
    status: 'online',
    version: 'local-dev-runtime-manager',
  },
  profiles: [
    {
      description: 'Stable local image smoke preset for FLUX.1-dev.',
      kind: 'image',
      model_env: 'EDGE_DIT_FLUX_MODEL_PATH',
      name: 'FLUX.1-dev',
      notes: ['Validated through the Python Server HTTP job API in this workspace.'],
      request_example: {
        height: 256,
        prompt: 'smoke test teapot',
        seed: 42,
        steps: 1,
        width: 256,
      },
      slug: 'flux-dev',
    },
    {
      description: 'Validated local video preset for Wan2.1-T2V-1.3B-Diffusers.',
      kind: 'video',
      model_env: 'EDGE_DIT_WAN_VIDEO_MODEL_PATH',
      name: 'Wan2.1-T2V-1.3B-Diffusers',
      notes: ['Switch to this profile when you want the console to expose video generation.'],
      request_example: {
        cfg_scale: 5,
        flow_shift: 5,
        frames: 9,
        height: 240,
        prompt: 'a small robot walking through rain',
        seed: 42,
        steps: 1,
        width: 416,
      },
      slug: 'wan-t2v',
    },
  ],
  recent_events: [
    {
      detail: {
        profile_slug: 'flux-dev',
      },
      id: 'runtime-event-ready',
      level: 'info',
      message: 'Managed backend is healthy and ready.',
      time_ms: 1,
      type: 'backend_ready',
    },
  ],
  recommended_connection_target: {
    baseUrl: 'http://127.0.0.1:8080',
    prefix: '/ed/v2',
  },
  request_id: 'runtime-status',
}

const apiHandlers = prefixes.flatMap((prefix) => [
  http.get(`${prefix}/health`, () =>
    HttpResponse.json({
      model: 'edge-dit-model',
      request_id: 'req-health',
      service: 'edge-dit-python-server',
      status: 'ok',
    }),
  ),
  http.get(`${prefix}/capabilities`, () =>
    HttpResponse.json({
      aliases: ['/edgedit/v2', '/edge-dit/v2'],
      defaults: {
        sampler: null,
        scheduler: null,
      },
      endpoints: [
        '/ed/v2/health',
        '/ed/v2/capabilities',
        '/ed/v2/images/generations',
        '/ed/v2/videos/generations',
        '/ed/v2/jobs',
        '/ed/v2/jobs/cleanup',
        '/ed/v2/jobs/{job_id}',
        '/ed/v2/jobs/{job_id}/cancel',
        '/ed/v2/jobs/{job_id}/result',
      ],
      model: 'edge-dit-model',
      package_version: '0.1.0',
      pipeline_name: 'demo-pipeline',
      request_id: 'req-capabilities',
      semantics: {
        cancellation: 'cooperative_step_boundary',
        job_ttl_ms: 3600000,
        progress: 'sampling_step_only',
        results: 'stored_in_memory',
      },
      service: 'edge-dit-python-server',
      supports: {
        image: true,
        video: true,
      },
      version_name: 'demo-version',
    }),
  ),
  http.get(`${prefix}/jobs`, () =>
    HttpResponse.json({
      data: [],
      has_more: false,
      object: 'list',
      request_id: 'req-jobs',
    }),
  ),
])

export const handlers = [
  ...apiHandlers,
  http.get('/runtime/v1/status', () => HttpResponse.json(runtimeStatusPayload)),
  http.post('/runtime/v1/backend/start', async () => HttpResponse.json(runtimeStatusPayload, { status: 202 })),
  http.post('/runtime/v1/backend/restart', async () => HttpResponse.json(runtimeStatusPayload, { status: 202 })),
  http.post('/runtime/v1/backend/stop', async () => HttpResponse.json(runtimeStatusPayload, { status: 202 })),
]
