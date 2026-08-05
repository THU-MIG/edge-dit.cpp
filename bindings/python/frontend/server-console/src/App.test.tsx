import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { delay, http, HttpResponse } from 'msw'

import { server } from '@/shared/test/msw/server'
import { App } from './App'

describe('App shell', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('renders the connected Python Server Console scaffold', async () => {
    render(<App />)

    expect(screen.getByRole('heading', { name: /python server console/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /connection/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /local runtime/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /generation composer/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /task list/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /json inspector/i })).toBeInTheDocument()

    expect(await screen.findAllByText('edge-dit-model')).not.toHaveLength(0)
    expect(await screen.findByText(/manager online/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /create image job/i })).toBeEnabled()
    expect(screen.getByRole('button', { name: /^Video$/i })).toBeEnabled()
  })

  it('renders managed tensor loading progress from runtime log tail', async () => {
    const fluxProfile = {
      description: 'Stable local image smoke preset for FLUX.1-dev.',
      kind: 'image' as const,
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
    }

    server.use(
      http.get('/runtime/v1/status', () =>
        HttpResponse.json({
          backend: {
            auto_restart_enabled: true,
            auto_restart_limit: 3,
            base_url: 'http://127.0.0.1:8080',
            capabilities: null,
            health_failure_streak: 0,
            last_exit: null,
            last_health: null,
            last_ready_at_ms: null,
            next_restart_at_ms: null,
            pid: 8181,
            profile: fluxProfile,
            restart_count_consecutive: 0,
            restart_count_total: 0,
            started_at_ms: 1,
            status: 'starting',
            uptime_ms: 49_000,
          },
          log_tail: [
            {
              line: 'INFO flux backend booting',
              stream: 'stdout' as const,
              time_ms: 1,
            },
            {
              line: 'loading tensors 435/2855 5245.17MB 48.60s',
              stream: 'stderr' as const,
              time_ms: 2,
            },
            {
              line: 'loading tensors 437/2855 5263.17MB 48.79s',
              stream: 'stderr' as const,
              time_ms: 3,
            },
            {
              line: 'loading tensors 439/2855 5281.18MB 48.99s',
              stream: 'stderr' as const,
              time_ms: 4,
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
          profiles: [fluxProfile],
          recent_events: [
            {
              detail: {
                profile_slug: 'flux-dev',
              },
              id: 'runtime-event-flux-spawn',
              level: 'info' as const,
              message: 'Starting managed backend for FLUX.',
              time_ms: 1,
              type: 'backend_spawn',
            },
          ],
          recommended_connection_target: {
            baseUrl: 'http://127.0.0.1:8080',
            prefix: '/ed/v2' as const,
          },
          request_id: 'runtime-status-starting-with-tensors',
        }),
      ),
    )

    render(<App />)

    expect(await screen.findByText('Tensor loading')).toBeInTheDocument()
    expect(screen.getByRole('progressbar', { name: /model load progress/i })).toBeInTheDocument()
    expect(screen.getByText('15%')).toBeInTheDocument()
    expect(screen.getAllByText('439 / 2855')).not.toHaveLength(0)
    expect(screen.getByText('5.28 GB')).toBeInTheDocument()
    expect(screen.getByText('49.0 s')).toBeInTheDocument()
  })

  it('derives edit-model sampling steps from managed runtime logs when server still reports 0/0', async () => {
    const qwenEditProfile = {
      description: 'Verified local edit preset for Qwen-Image-Edit.',
      kind: 'image' as const,
      model_env: 'EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH',
      name: 'Qwen-Image-Edit',
      notes: ['Validated through the Python Server HTTP job API in this workspace.'],
      request_example: {
        height: 256,
        init_image_b64: 'data:image/png;base64,AAAA',
        prompt: 'make it cinematic',
        steps: 20,
        width: 256,
      },
      slug: 'qwen-image-edit',
    }

    server.use(
      http.get('/runtime/v1/status', () =>
        HttpResponse.json({
          backend: {
            auto_restart_enabled: true,
            auto_restart_limit: 3,
            base_url: 'http://127.0.0.1:8080',
            capabilities: {
              model: 'Qwen-Image-Edit',
              supports: {
                image: true,
                video: false,
              },
            },
            health_failure_streak: 0,
            last_exit: null,
            last_health: {
              checked_at_ms: 2,
              error: null,
              ok: true,
              request_id: 'runtime-health-qwen-edit',
              response_ms: 11,
              status: 200,
            },
            last_ready_at_ms: 1,
            next_restart_at_ms: null,
            pid: 8484,
            profile: qwenEditProfile,
            restart_count_consecutive: 0,
            restart_count_total: 0,
            started_at_ms: 1,
            status: 'running',
            uptime_ms: 8_000,
          },
          log_tail: [
            {
              line: '/tmp/qwen.cpp:812 [info] qwen-image-edit step 1/20 sigma=1.000000 next=0.800000',
              stream: 'stderr' as const,
              time_ms: 2,
            },
            {
              line: '/tmp/qwen.cpp:812 [info] qwen-image-edit step 2/20 sigma=0.800000 next=0.600000',
              stream: 'stderr' as const,
              time_ms: 3,
            },
            {
              line: '/tmp/qwen.cpp:812 [info] qwen-image-edit step 3/20 sigma=0.600000 next=0.400000',
              stream: 'stderr' as const,
              time_ms: 4,
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
          profiles: [qwenEditProfile],
          recent_events: [
            {
              detail: {
                profile_slug: 'qwen-image-edit',
              },
              id: 'runtime-event-qwen-edit-ready',
              level: 'info' as const,
              message: 'Managed backend is healthy and ready.',
              time_ms: 1,
              type: 'backend_ready',
            },
          ],
          recommended_connection_target: {
            baseUrl: 'http://127.0.0.1:8080',
            prefix: '/ed/v2' as const,
          },
          request_id: 'runtime-status-qwen-edit-running',
        }),
      ),
      http.get('/ed/v2/jobs', () =>
        HttpResponse.json({
          data: [
            {
              cancel_requested: false,
              cancel_url: '/ed/v2/jobs/job-qwen-edit-001/cancel',
              created_ms: 1,
              error: null,
              expires_ms: null,
              finished_ms: null,
              id: 'job-qwen-edit-001',
              kind: 'image',
              model: 'Qwen-Image-Edit',
              object: 'edge_dit.job',
              parameters: {
                height: 256,
                prompt: 'make it cinematic',
                steps: 20,
                width: 256,
              },
              request_id: 'req-jobs-qwen-edit',
              result_url: '/ed/v2/jobs/job-qwen-edit-001/result',
              started_ms: 1,
              status: 'running',
              status_url: '/ed/v2/jobs/job-qwen-edit-001',
            },
          ],
          has_more: false,
          object: 'list',
          request_id: 'req-jobs-qwen-edit',
        }),
      ),
      http.get('/ed/v2/jobs/job-qwen-edit-001', () =>
        HttpResponse.json({
          cancel_requested: false,
          cancel_url: '/ed/v2/jobs/job-qwen-edit-001/cancel',
          created_ms: 1,
          error: null,
          expires_ms: null,
          finished_ms: null,
          id: 'job-qwen-edit-001',
          kind: 'image',
          model: 'Qwen-Image-Edit',
          object: 'edge_dit.job',
          parameters: {
            height: 256,
            prompt: 'make it cinematic',
            steps: 20,
            width: 256,
          },
          progress: {
            current_step: 0,
            total_steps: 0,
          },
          request_id: 'req-detail-qwen-edit',
          result_url: '/ed/v2/jobs/job-qwen-edit-001/result',
          started_ms: 1,
          status: 'running',
          status_url: '/ed/v2/jobs/job-qwen-edit-001',
        }),
      ),
    )

    render(<App />)

    expect(await screen.findByText('Sampling')).toBeInTheDocument()
    expect(screen.getByText('3 / 20')).toBeInTheDocument()
    expect(screen.getByText(/derived from managed backend step logs/i)).toBeInTheDocument()
  })

  it('shows preparing before sampling counters are published for a running job', async () => {
    server.use(
      http.get('/ed/v2/jobs', () =>
        HttpResponse.json({
          data: [
            {
              cancel_requested: false,
              cancel_url: '/ed/v2/jobs/job-preparing-001/cancel',
              created_ms: 1,
              error: null,
              expires_ms: null,
              finished_ms: null,
              id: 'job-preparing-001',
              kind: 'image',
              model: 'edge-dit-model',
              object: 'edge_dit.job',
              parameters: {
                height: 256,
                prompt: 'A preparing edit job',
                steps: 20,
                width: 256,
              },
              request_id: 'req-jobs-preparing',
              result_url: '/ed/v2/jobs/job-preparing-001/result',
              started_ms: 1,
              status: 'running',
              status_url: '/ed/v2/jobs/job-preparing-001',
            },
          ],
          has_more: false,
          object: 'list',
          request_id: 'req-jobs-preparing',
        }),
      ),
      http.get('/ed/v2/jobs/job-preparing-001', () =>
        HttpResponse.json({
          cancel_requested: false,
          cancel_url: '/ed/v2/jobs/job-preparing-001/cancel',
          created_ms: 1,
          error: null,
          expires_ms: null,
          finished_ms: null,
          id: 'job-preparing-001',
          kind: 'image',
          model: 'edge-dit-model',
          object: 'edge_dit.job',
          parameters: {
            height: 256,
            prompt: 'A preparing edit job',
            steps: 20,
            width: 256,
          },
          progress: {
            current_step: 0,
            total_steps: 0,
          },
          request_id: 'req-detail-preparing',
          result_url: '/ed/v2/jobs/job-preparing-001/result',
          started_ms: 1,
          status: 'running',
          status_url: '/ed/v2/jobs/job-preparing-001',
        }),
      ),
    )

    render(<App />)

    expect(await screen.findByText('Preparing')).toBeInTheDocument()
    expect(screen.getByText('0 / 0')).toBeInTheDocument()
    expect(
      screen.getByText(/has started the job, but Python Server has not published sampling-step counters yet/i),
    ).toBeInTheDocument()
    expect(screen.queryByText(/^Sampling$/i)).not.toBeInTheDocument()
  })

  it('creates an image job and renders the decoded result preview', async () => {
    const user = userEvent.setup()
    let detailCalls = 0

    server.use(
      http.get('/ed/v2/jobs', () =>
        HttpResponse.json({
          data: [
            {
              cancel_requested: false,
              cancel_url: '/ed/v2/jobs/job-image-001/cancel',
              created_ms: 1,
              error: null,
              expires_ms: 3600001,
              finished_ms: detailCalls > 0 ? 2 : null,
              id: 'job-image-001',
              kind: 'image',
              model: 'edge-dit-model',
              object: 'edge_dit.job',
              parameters: {
                height: 1024,
                prompt: 'Elegant console demo image',
                steps: 20,
                width: 1024,
              },
              request_id: 'req-jobs-image',
              result_url: '/ed/v2/jobs/job-image-001/result',
              started_ms: detailCalls > 0 ? 1 : null,
              status: detailCalls > 0 ? 'succeeded' : 'queued',
              status_url: '/ed/v2/jobs/job-image-001',
            },
          ],
          has_more: false,
          object: 'list',
          request_id: 'req-jobs-image',
        }),
      ),
      http.post('/ed/v2/images/generations', async () =>
        HttpResponse.json(
          {
            cancel_requested: false,
            cancel_url: '/ed/v2/jobs/job-image-001/cancel',
            created_ms: 1,
            error: null,
            expires_ms: null,
            finished_ms: null,
            id: 'job-image-001',
            kind: 'image',
            model: 'edge-dit-model',
            object: 'edge_dit.job',
            parameters: {
              height: 1024,
              prompt: 'Elegant console demo image',
              steps: 20,
              width: 1024,
            },
            progress: {
              current_step: 0,
              total_steps: 0,
            },
            request_id: 'req-create-image',
            result_url: '/ed/v2/jobs/job-image-001/result',
            started_ms: null,
            status: 'queued',
            status_url: '/ed/v2/jobs/job-image-001',
          },
          { status: 202 },
        ),
      ),
      http.get('/ed/v2/jobs/job-image-001', () => {
        detailCalls += 1

        if (detailCalls === 1) {
          return HttpResponse.json({
            cancel_requested: false,
            cancel_url: '/ed/v2/jobs/job-image-001/cancel',
            created_ms: 1,
            error: null,
            expires_ms: null,
            finished_ms: null,
            id: 'job-image-001',
            kind: 'image',
            model: 'edge-dit-model',
            object: 'edge_dit.job',
            parameters: {
              height: 1024,
              prompt: 'Elegant console demo image',
              steps: 20,
              width: 1024,
            },
            progress: {
              current_step: 1,
              total_steps: 2,
            },
            request_id: 'req-detail-running',
            result_url: '/ed/v2/jobs/job-image-001/result',
            started_ms: 1,
            status: 'running',
            status_url: '/ed/v2/jobs/job-image-001',
          })
        }

        return HttpResponse.json({
          cancel_requested: false,
          cancel_url: '/ed/v2/jobs/job-image-001/cancel',
          created_ms: 1,
          error: null,
          expires_ms: 3600001,
          finished_ms: 2,
          id: 'job-image-001',
          kind: 'image',
          model: 'edge-dit-model',
          object: 'edge_dit.job',
          parameters: {
            height: 1024,
            prompt: 'Elegant console demo image',
            steps: 20,
            width: 1024,
          },
          progress: {
            current_step: 2,
            total_steps: 2,
          },
          request_id: 'req-detail-succeeded',
          result_url: '/ed/v2/jobs/job-image-001/result',
          started_ms: 1,
          status: 'succeeded',
          status_url: '/ed/v2/jobs/job-image-001',
        })
      }),
      http.get('/ed/v2/jobs/job-image-001/result', () =>
        HttpResponse.json({
          completed_ms: 2,
          created_ms: 1,
          data: [
            {
              b64_png:
                'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFAAH/GMurNwAAAABJRU5ErkJggg==',
              metadata: {
                channels: 3,
                format: 'png',
                height: 1,
                index: 0,
                width: 1,
              },
            },
          ],
          id: 'job-image-001',
          model: 'edge-dit-model',
          object: 'edge_dit.image_generation',
          parameters: {
            height: 1024,
            prompt: 'Elegant console demo image',
            steps: 20,
            width: 1024,
          },
          request_id: 'req-result-image',
        }),
      ),
    )

    render(<App />)

    await user.type(
      await screen.findByPlaceholderText(/clean product demo image/i),
      'Elegant console demo image',
    )
    await user.click(await screen.findByRole('button', { name: /create image job/i }))

    expect(await screen.findByText('job-image-001')).toBeInTheDocument()
    expect(await screen.findByAltText(/generated result preview 1/i)).toBeInTheDocument()
  })

  it('creates a video job and renders the decoded frame carousel', async () => {
    const user = userEvent.setup()
    let detailCalls = 0

    server.use(
      http.get('/ed/v2/jobs', () =>
        HttpResponse.json({
          data: [
            {
              cancel_requested: false,
              cancel_url: '/ed/v2/jobs/job-video-001/cancel',
              created_ms: 11,
              error: null,
              expires_ms: 3600011,
              finished_ms: detailCalls > 1 ? 22 : null,
              id: 'job-video-001',
              kind: 'video',
              model: 'edge-dit-model',
              object: 'edge_dit.job',
              parameters: {
                frames: 3,
                height: 240,
                prompt: 'A cinematic robot walk cycle',
                steps: 20,
                width: 416,
              },
              request_id: 'req-jobs-video',
              result_url: '/ed/v2/jobs/job-video-001/result',
              started_ms: detailCalls > 0 ? 12 : null,
              status: detailCalls > 1 ? 'succeeded' : detailCalls > 0 ? 'running' : 'queued',
              status_url: '/ed/v2/jobs/job-video-001',
            },
          ],
          has_more: false,
          object: 'list',
          request_id: 'req-jobs-video',
        }),
      ),
      http.post('/ed/v2/videos/generations', async () =>
        HttpResponse.json(
          {
            cancel_requested: false,
            cancel_url: '/ed/v2/jobs/job-video-001/cancel',
            created_ms: 11,
            error: null,
            expires_ms: null,
            finished_ms: null,
            id: 'job-video-001',
            kind: 'video',
            model: 'edge-dit-model',
            object: 'edge_dit.job',
            parameters: {
              frames: 3,
              height: 240,
              prompt: 'A cinematic robot walk cycle',
              steps: 20,
              width: 416,
            },
            progress: {
              current_step: 0,
              total_steps: 0,
            },
            request_id: 'req-create-video',
            result_url: '/ed/v2/jobs/job-video-001/result',
            started_ms: null,
            status: 'queued',
            status_url: '/ed/v2/jobs/job-video-001',
          },
          { status: 202 },
        ),
      ),
      http.get('/ed/v2/jobs/job-video-001', () => {
        detailCalls += 1

        if (detailCalls === 1) {
          return HttpResponse.json({
            cancel_requested: false,
            cancel_url: '/ed/v2/jobs/job-video-001/cancel',
            created_ms: 11,
            error: null,
            expires_ms: null,
            finished_ms: null,
            id: 'job-video-001',
            kind: 'video',
            model: 'edge-dit-model',
            object: 'edge_dit.job',
            parameters: {
              frames: 3,
              height: 240,
              prompt: 'A cinematic robot walk cycle',
              steps: 20,
              width: 416,
            },
            progress: {
              current_step: 1,
              total_steps: 2,
            },
            request_id: 'req-detail-video-running',
            result_url: '/ed/v2/jobs/job-video-001/result',
            started_ms: 12,
            status: 'running',
            status_url: '/ed/v2/jobs/job-video-001',
          })
        }

        return HttpResponse.json({
          cancel_requested: false,
          cancel_url: '/ed/v2/jobs/job-video-001/cancel',
          created_ms: 11,
          error: null,
          expires_ms: 3600011,
          finished_ms: 22,
          id: 'job-video-001',
          kind: 'video',
          model: 'edge-dit-model',
          object: 'edge_dit.job',
          parameters: {
            frames: 3,
            height: 240,
            prompt: 'A cinematic robot walk cycle',
            steps: 20,
            width: 416,
          },
          progress: {
            current_step: 2,
            total_steps: 2,
          },
          request_id: 'req-detail-video-succeeded',
          result_url: '/ed/v2/jobs/job-video-001/result',
          started_ms: 12,
          status: 'succeeded',
          status_url: '/ed/v2/jobs/job-video-001',
        })
      }),
      http.get('/ed/v2/jobs/job-video-001/result', () =>
        HttpResponse.json({
          completed_ms: 22,
          created_ms: 11,
          frame_format: 'png',
          frames: [
            {
              b64_png:
                'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFAAH/GMurNwAAAABJRU5ErkJggg==',
              metadata: {
                channels: 3,
                format: 'png',
                height: 1,
                index: 0,
                width: 1,
              },
            },
            {
              b64_png:
                'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFAAH/GMurNwAAAABJRU5ErkJggg==',
              metadata: {
                channels: 3,
                format: 'png',
                height: 1,
                index: 1,
                width: 1,
              },
            },
            {
              b64_png:
                'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFAAH/GMurNwAAAABJRU5ErkJggg==',
              metadata: {
                channels: 3,
                format: 'png',
                height: 1,
                index: 2,
                width: 1,
              },
            },
          ],
          id: 'job-video-001',
          model: 'edge-dit-model',
          object: 'edge_dit.video_generation',
          parameters: {
            frames: 3,
            height: 240,
            prompt: 'A cinematic robot walk cycle',
            steps: 20,
            width: 416,
          },
          request_id: 'req-result-video',
        }),
      ),
    )

    render(<App />)

    await screen.findAllByText('edge-dit-model')
    await user.click(screen.getByRole('button', { name: /^Video$/i }))
    await user.type(
      screen.getByPlaceholderText(/a small robot walking through rain/i),
      'A cinematic robot walk cycle',
    )
    await user.click(screen.getByRole('button', { name: /create video job/i }))

    expect(await screen.findByText('job-video-001')).toBeInTheDocument()
    expect(await screen.findByAltText(/generated video frame 1/i)).toBeInTheDocument()
    expect(await screen.findByText(/^Frontend frame carousel$/i)).toBeInTheDocument()
  })

  it('deletes a selected terminal job and refreshes the task list', async () => {
    const user = userEvent.setup()
    let deleted = false

    server.use(
      http.get('/ed/v2/jobs', () =>
        HttpResponse.json({
          data: deleted
            ? []
            : [
                {
                  cancel_requested: false,
                  cancel_url: '/ed/v2/jobs/job-cancelled-001/cancel',
                  created_ms: 31,
                  error: 'generation cancelled before start',
                  expires_ms: 3600031,
                  finished_ms: 32,
                  id: 'job-cancelled-001',
                  kind: 'image',
                  model: 'edge-dit-model',
                  object: 'edge_dit.job',
                  parameters: {
                    height: 1024,
                    prompt: 'Remove me from the list',
                    steps: 20,
                    width: 1024,
                  },
                  request_id: 'req-jobs-delete',
                  result_url: '/ed/v2/jobs/job-cancelled-001/result',
                  started_ms: null,
                  status: 'cancelled',
                  status_url: '/ed/v2/jobs/job-cancelled-001',
                },
              ],
          has_more: false,
          object: 'list',
          request_id: 'req-jobs-delete',
        }),
      ),
      http.get('/ed/v2/jobs/job-cancelled-001', () =>
        HttpResponse.json({
          cancel_requested: false,
          cancel_url: '/ed/v2/jobs/job-cancelled-001/cancel',
          created_ms: 31,
          error: 'generation cancelled before start',
          expires_ms: 3600031,
          finished_ms: 32,
          id: 'job-cancelled-001',
          kind: 'image',
          model: 'edge-dit-model',
          object: 'edge_dit.job',
          parameters: {
            height: 1024,
            prompt: 'Remove me from the list',
            steps: 20,
            width: 1024,
          },
          progress: {
            current_step: 0,
            total_steps: 0,
          },
          request_id: 'req-detail-delete',
          result_url: '/ed/v2/jobs/job-cancelled-001/result',
          started_ms: null,
          status: 'cancelled',
          status_url: '/ed/v2/jobs/job-cancelled-001',
        }),
      ),
      http.delete('/ed/v2/jobs/job-cancelled-001', () => {
        deleted = true

        return HttpResponse.json({
          id: 'job-cancelled-001',
          kind: 'image',
          object: 'edge_dit.job_deleted',
          request_id: 'req-delete-job',
          status: 'cancelled',
        })
      }),
    )

    render(<App />)

    expect(await screen.findByText('job-cancelled-001')).toBeInTheDocument()
    await user.click(await screen.findByRole('button', { name: /delete selected/i }))

    expect(await screen.findByText(/no jobs yet\./i)).toBeInTheDocument()
    expect(screen.queryByText('job-cancelled-001')).not.toBeInTheDocument()
  })

  it('keeps the progress panel stable while only job summary data is available', async () => {
    server.use(
      http.get('/ed/v2/jobs', () =>
        HttpResponse.json({
          data: [
            {
              cancel_requested: false,
              cancel_url: '/ed/v2/jobs/job-summary-001/cancel',
              created_ms: 51,
              error: null,
              expires_ms: null,
              finished_ms: null,
              id: 'job-summary-001',
              kind: 'image',
              model: 'edge-dit-model',
              object: 'edge_dit.job',
              parameters: {
                height: 256,
                prompt: 'Summary first, detail later',
                steps: 1,
                width: 256,
              },
              request_id: 'req-jobs-summary',
              result_url: '/ed/v2/jobs/job-summary-001/result',
              started_ms: 52,
              status: 'running',
              status_url: '/ed/v2/jobs/job-summary-001',
            },
          ],
          has_more: false,
          object: 'list',
          request_id: 'req-jobs-summary',
        }),
      ),
      http.get('/ed/v2/jobs/job-summary-001', async () => {
        await delay(1_000)

        return HttpResponse.json({
          cancel_requested: false,
          cancel_url: '/ed/v2/jobs/job-summary-001/cancel',
          created_ms: 51,
          error: null,
          expires_ms: null,
          finished_ms: null,
          id: 'job-summary-001',
          kind: 'image',
          model: 'edge-dit-model',
          object: 'edge_dit.job',
          parameters: {
            height: 256,
            prompt: 'Summary first, detail later',
            steps: 1,
            width: 256,
          },
          progress: {
            current_step: 0,
            total_steps: 1,
          },
          request_id: 'req-detail-summary',
          result_url: '/ed/v2/jobs/job-summary-001/result',
          started_ms: 52,
          status: 'running',
          status_url: '/ed/v2/jobs/job-summary-001',
        })
      }),
    )

    render(<App />)

    expect(await screen.findByText('job-summary-001')).toBeInTheDocument()
    expect(screen.getByText('Loading detail')).toBeInTheDocument()
    expect(screen.getByText('pending detail')).toBeInTheDocument()
  })

  it('automatically refreshes probes after a managed model switch becomes ready', async () => {
    const user = userEvent.setup()
    let runtimePhase: 'initial' | 'starting' | 'ready' = 'initial'
    let healthCalls = 0
    let capabilitiesCalls = 0

    const fluxProfile = {
      description: 'Stable local image smoke preset for FLUX.1-dev.',
      kind: 'image' as const,
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
    }

    const wanProfile = {
      description: 'Validated local video preset for Wan2.1-T2V-1.3B-Diffusers.',
      kind: 'video' as const,
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
    }

    function runtimeStatusPayload() {
      if (runtimePhase === 'ready') {
        return {
          backend: {
            auto_restart_enabled: true,
            auto_restart_limit: 3,
            base_url: 'http://127.0.0.1:8080',
            capabilities: {
              model: 'wan-model',
              supports: {
                image: false,
                video: true,
              },
            },
            health_failure_streak: 0,
            last_exit: null,
            last_health: {
              checked_at_ms: 3,
              error: null,
              ok: true,
              request_id: 'runtime-health-wan',
              response_ms: 12,
              status: 200,
            },
            last_ready_at_ms: 3,
            next_restart_at_ms: null,
            pid: 9090,
            profile: wanProfile,
            restart_count_consecutive: 0,
            restart_count_total: 0,
            started_at_ms: 2,
            status: 'running',
            uptime_ms: 1_000,
          },
          log_tail: [
            {
              line: 'INFO wan backend ready',
              stream: 'stdout' as const,
              time_ms: 3,
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
          profiles: [fluxProfile, wanProfile],
          recent_events: [
            {
              detail: {
                profile_slug: 'flux-dev',
              },
              id: 'runtime-event-flux-ready',
              level: 'info' as const,
              message: 'Managed backend is healthy and ready.',
              time_ms: 1,
              type: 'backend_ready',
            },
            {
              detail: {
                profile_slug: 'wan-t2v',
              },
              id: 'runtime-event-wan-ready',
              level: 'info' as const,
              message: 'Managed backend switched to Wan and is healthy.',
              time_ms: 3,
              type: 'backend_ready',
            },
          ],
          recommended_connection_target: {
            baseUrl: 'http://127.0.0.1:8080',
            prefix: '/ed/v2' as const,
          },
          request_id: 'runtime-status-ready',
        }
      }

      if (runtimePhase === 'starting') {
        return {
          backend: {
            auto_restart_enabled: true,
            auto_restart_limit: 3,
            base_url: 'http://127.0.0.1:8080',
            capabilities: null,
            health_failure_streak: 0,
            last_exit: null,
            last_health: null,
            last_ready_at_ms: 1,
            next_restart_at_ms: null,
            pid: 9090,
            profile: wanProfile,
            restart_count_consecutive: 0,
            restart_count_total: 0,
            started_at_ms: 2,
            status: 'starting',
            uptime_ms: 200,
          },
          log_tail: [
            {
              line: 'INFO wan backend booting',
              stream: 'stdout' as const,
              time_ms: 2,
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
          profiles: [fluxProfile, wanProfile],
          recent_events: [
            {
              detail: {
                profile_slug: 'flux-dev',
              },
              id: 'runtime-event-flux-ready',
              level: 'info' as const,
              message: 'Managed backend is healthy and ready.',
              time_ms: 1,
              type: 'backend_ready',
            },
            {
              detail: {
                profile_slug: 'wan-t2v',
              },
              id: 'runtime-event-wan-spawn',
              level: 'info' as const,
              message: 'Starting managed backend for Wan.',
              time_ms: 2,
              type: 'backend_spawn',
            },
          ],
          recommended_connection_target: {
            baseUrl: 'http://127.0.0.1:8080',
            prefix: '/ed/v2' as const,
          },
          request_id: 'runtime-status-starting',
        }
      }

      return {
        backend: {
          auto_restart_enabled: true,
          auto_restart_limit: 3,
          base_url: 'http://127.0.0.1:8080',
          capabilities: {
            model: 'flux-model',
            supports: {
              image: true,
              video: false,
            },
          },
          health_failure_streak: 0,
          last_exit: null,
          last_health: {
            checked_at_ms: 1,
            error: null,
            ok: true,
            request_id: 'runtime-health-flux',
            response_ms: 10,
            status: 200,
          },
          last_ready_at_ms: 1,
          next_restart_at_ms: null,
          pid: 8081,
          profile: fluxProfile,
          restart_count_consecutive: 0,
          restart_count_total: 0,
          started_at_ms: 1,
          status: 'running',
          uptime_ms: 5_000,
        },
        log_tail: [
          {
            line: 'INFO flux backend ready',
            stream: 'stdout' as const,
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
        profiles: [fluxProfile, wanProfile],
        recent_events: [
          {
            detail: {
              profile_slug: 'flux-dev',
            },
            id: 'runtime-event-flux-ready',
            level: 'info' as const,
            message: 'Managed backend is healthy and ready.',
            time_ms: 1,
            type: 'backend_ready',
          },
        ],
        recommended_connection_target: {
          baseUrl: 'http://127.0.0.1:8080',
          prefix: '/ed/v2' as const,
        },
        request_id: 'runtime-status-flux',
      }
    }

    server.use(
      http.get('/runtime/v1/status', () => HttpResponse.json(runtimeStatusPayload())),
      http.post('/runtime/v1/backend/start', async () => {
        runtimePhase = 'starting'
        return HttpResponse.json(runtimeStatusPayload(), { status: 202 })
      }),
      http.get('/ed/v2/health', () => {
        healthCalls += 1

        if (runtimePhase === 'starting') {
          return HttpResponse.json(
            {
              code: 'backend_starting',
              message: 'backend is still starting',
              request_id: 'req-health-starting',
            },
            { status: 503 },
          )
        }

        return HttpResponse.json({
          model: runtimePhase === 'ready' ? 'wan-model' : 'flux-model',
          request_id: runtimePhase === 'ready' ? 'req-health-wan' : 'req-health-flux',
          service: 'edge-dit-python-server',
          status: 'ok',
        })
      }),
      http.get('/ed/v2/capabilities', () => {
        capabilitiesCalls += 1

        return HttpResponse.json({
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
          model: runtimePhase === 'ready' ? 'wan-model' : 'flux-model',
          package_version: '0.1.0',
          pipeline_name: runtimePhase === 'ready' ? 'wan' : 'flux',
          request_id: runtimePhase === 'ready' ? 'req-capabilities-wan' : 'req-capabilities-flux',
          semantics: {
            cancellation: 'cooperative_step_boundary',
            job_ttl_ms: 3600000,
            progress: 'sampling_step_only',
            results: 'stored_in_memory',
          },
          service: 'edge-dit-python-server',
          supports: {
            image: runtimePhase !== 'ready',
            video: runtimePhase === 'ready',
          },
          version_name: runtimePhase === 'ready' ? 'wan-version' : 'flux-version',
        })
      }),
    )

    render(<App />)

    expect(await screen.findAllByText('flux-model')).not.toHaveLength(0)
    expect(await screen.findByRole('button', { name: /create image job/i })).toBeEnabled()

    await user.selectOptions(screen.getByRole('combobox', { name: /verified model/i }), 'wan-t2v')
    await user.click(screen.getByRole('button', { name: /switch model/i }))

    runtimePhase = 'ready'

    expect(await screen.findAllByText('wan-model', {}, { timeout: 8_000 })).not.toHaveLength(0)
    expect(await screen.findByRole('button', { name: /create video job/i })).toBeEnabled()
    await waitFor(
      () => {
        expect(healthCalls).toBeGreaterThanOrEqual(2)
        expect(capabilitiesCalls).toBeGreaterThanOrEqual(2)
      },
      { timeout: 8_000 },
    )
  }, 12_000)
})
