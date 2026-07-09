# server_v2 Web Console

Local React console for `bindings/python/src/edge_dit/server_v2.py`.

This app follows the construction docs in:

```text
dev_docs/ecosystem-integration/server_v2-frontend/
```

For day-to-day machine setup, see:

```text
RUNTIME_CONFIGURATION.md
```

## Current Milestone

The local console now includes:

- Vite + React + TypeScript
- Tailwind CSS v4 design tokens
- React Query provider
- three-rail console shell for `health`, `capabilities`, jobs, progress, and results
- local runtime manager integration at `http://127.0.0.1:8090`
- managed `server_v2` backend slot at `http://127.0.0.1:8080`
- verified model switching for `flux-dev`, `sd3-medium`, `qwen-image`, `qwen-image-edit`, `flux-kontext`, and `wan-t2v`
- runtime event feedback with in-app toasts plus optional browser system notifications
- self-healing restart loop for unexpected managed-backend exits
- Vitest + Testing Library + MSW
- Playwright smoke test

## Local Development

Install dependencies:

```bash
npm install
```

Start only the frontend:

```bash
npm run dev
```

Start the full managed local stack:

```bash
npm run dev:managed
```

Start the same managed stack on all interfaces so another device can open it through your machine IP:

```bash
npm run dev:managed:network
```

This launches:

```text
frontend:         http://127.0.0.1:5173
runtime manager:  http://127.0.0.1:8090
managed backend:  http://127.0.0.1:8080
managed prefix:   /ed/v2
```

With `dev:managed:network`, the frontend, runtime manager, and managed backend all bind to `0.0.0.0`. Remote browsers can either stay on the Vite proxy at `:5173` or call the runtime/backend directly on `:8090` and `:8080`.

The default managed profile is `flux-dev`. Override it by forwarding runtime-manager args:

```bash
npm run dev:managed -- --auto-start-profile wan-t2v
```

Start only the runtime manager:

```bash
npm run runtime:manager -- --auto-start-profile sd3-medium
```

Run a verified profile directly without the manager:

```bash
bash ./scripts/profiles/flux-dev.sh --host 127.0.0.1 --port 8080
```

Managed development targets:

```text
frontend: http://127.0.0.1:5173
runtime:  http://127.0.0.1:8090/runtime/v1
backend:  http://127.0.0.1:8080/ed/v2
```

The Vite dev server proxies canonical `server_v2` prefixes to the local Python backend and proxies `/runtime/v1` to the local runtime manager.

## Verification

```bash
npm run build
npm test
npm run test:e2e
```

## Notes

- Startup environment for the managed backend is centralized in [scripts/runtime-env.sh](scripts/runtime-env.sh).
- The runtime manager will try to restart the managed backend on unexpected exits and will surface recent events plus log tail in the console UI.
- Browser system notifications are opt-in and can be enabled from the `Local Runtime` panel.
