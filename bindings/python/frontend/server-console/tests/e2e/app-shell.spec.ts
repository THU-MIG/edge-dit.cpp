import { expect, test } from '@playwright/test'

test('shows the console shell', async ({ page }) => {
  await page.route('**/runtime/v1/status', async (route) => {
    await route.fulfill({
      body: JSON.stringify({
        backend: {
          auto_restart_enabled: true,
          auto_restart_limit: 3,
          base_url: 'http://127.0.0.1:8080',
          capabilities: null,
          health_failure_streak: 0,
          last_exit: null,
          last_health: {
            checked_at_ms: 1,
            error: null,
            ok: true,
            request_id: 'runtime-health',
            response_ms: 20,
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
          uptime_ms: 1000,
        },
        log_tail: [],
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
        ],
        recent_events: [],
        recommended_connection_target: {
          baseUrl: 'http://127.0.0.1:8080',
          prefix: '/ed/v2',
        },
        request_id: 'runtime-status',
      }),
      contentType: 'application/json',
      status: 200,
    })
  })

  await page.route('**/ed/v2/health', async (route) => {
    await route.fulfill({
      body: JSON.stringify({
        model: 'edge-dit-model',
        request_id: 'e2e-health',
        service: 'edge-dit-python-server',
        status: 'ok',
      }),
      contentType: 'application/json',
      status: 200,
    })
  })

  await page.route('**/ed/v2/capabilities', async (route) => {
    await route.fulfill({
      body: JSON.stringify({
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
        request_id: 'e2e-capabilities',
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
      contentType: 'application/json',
      status: 200,
    })
  })

  await page.route('**/ed/v2/jobs*', async (route) => {
    await route.fulfill({
      body: JSON.stringify({
        data: [],
        has_more: false,
        object: 'list',
        request_id: 'e2e-jobs',
      }),
      contentType: 'application/json',
      status: 200,
    })
  })

  await page.goto('/')

  await expect(page.getByRole('heading', { name: 'Python Server Console' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Connection' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Local Runtime' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Generation Composer' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Task List' })).toBeVisible()
})
