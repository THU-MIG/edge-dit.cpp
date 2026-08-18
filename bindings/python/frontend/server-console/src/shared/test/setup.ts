import '@testing-library/jest-dom/vitest'

import { afterAll, afterEach, beforeAll } from 'vitest'

import { server } from './msw/server'

class MemoryStorage implements Storage {
  readonly #items = new Map<string, string>()

  get length() {
    return this.#items.size
  }

  clear() {
    this.#items.clear()
  }

  getItem(key: string) {
    return this.#items.get(String(key)) ?? null
  }

  key(index: number) {
    return [...this.#items.keys()][index] ?? null
  }

  removeItem(key: string) {
    this.#items.delete(String(key))
  }

  setItem(key: string, value: string) {
    this.#items.set(String(key), String(value))
  }
}

// Node 26 exposes an unusable experimental localStorage unless it receives a
// --localstorage-file flag. Vitest's jsdom window inherits that value, so use
// a deterministic in-memory implementation for browser persistence tests.
Object.defineProperty(window, 'localStorage', {
  configurable: true,
  value: new MemoryStorage(),
})

beforeAll(() => {
  server.listen({ onUnhandledRequest: 'error' })
})

afterEach(() => {
  server.resetHandlers()
})

afterAll(() => {
  server.close()
})
