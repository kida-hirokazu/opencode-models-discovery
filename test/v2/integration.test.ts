import { createServer, type Server } from "node:http"
import { describe, expect, it } from "vitest"
import { Model, Provider, type Plugin } from "@opencode/plugin"
import { discoverInventory } from "../../src/v2/discovery.js"
import { parseProviderDiscoveryOptions } from "../../src/v2/provider-config.js"
import plugin from "../../src/v2/index.js"
import type { RpcStatusOutput } from "../../src/v2/rpc.js"

async function startMockProvider(): Promise<{ server: Server; baseURL: string; requests: Array<{ path: string; authorization?: string; body?: string }> }> {
  const requests: Array<{ path: string; authorization?: string; body?: string }> = []
  const server = createServer((request, response) => {
    const chunks: Buffer[] = []
    request.on("data", (chunk: Buffer) => chunks.push(chunk))
    request.on("end", () => {
      const body = Buffer.concat(chunks).toString("utf8")
      requests.push({ path: request.url ?? "", authorization: request.headers.authorization, ...(body ? { body } : {}) })
      response.setHeader("content-type", "application/json")

      if (request.url === "/v1/models" && request.method === "GET") {
        response.end(JSON.stringify({ data: [{ id: "remote/model-alpha", name: "Remote Alpha" }] }))
        return
      }
      if (request.url === "/v1/chat/completions" && request.method === "POST") {
        response.end(JSON.stringify({ id: "completion-1", model: "remote/model-alpha", choices: [{ message: { content: "ok" } }] }))
        return
      }
      response.statusCode = 404
      response.end(JSON.stringify({ error: "not found" }))
    })
  })
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve))
  const address = server.address()
  if (!address || typeof address === "string") throw new Error("Mock provider did not bind to a TCP port")
  return { server, baseURL: `http://127.0.0.1:${address.port}/v1`, requests }
}

function createReplayHost(baseURL: string, sourceModels: readonly Model.Info[] = []) {
  const providerID = Provider.ID.make("local")
  const sourceProvider = {
    ...Provider.Info.empty(providerID),
    name: "Local", integrationID: providerID,
    package: "@opencode/ai/providers/openai-compatible",
    settings: { baseURL, modelsDiscovery: { enabled: true } },
  }
  const transforms: Parameters<Plugin.Context["provider"]["transform"]>[0][] = []
  let published = new Map(sourceModels.map((model) => [String(model.id), model]))
  const storage = new Map<string, unknown>()
  const handlers: Record<string, (input?: unknown) => Promise<unknown>> = {}
  const replay = () => {
    // The V2 contract replays transforms on fresh source state. Previous output
    // from THIS plugin is not an explicit source on the next reload.
    const provider = structuredClone(sourceProvider)
    const models = new Map(sourceModels.map((model) => [String(model.id), structuredClone(model)]))
    for (const transform of transforms) {
      transform({
        get: () => ({ provider, models }),
        update: (_id, update) => update(provider as never),
        models: {
          set: (_id, next) => {
            models.clear()
            for (const model of next) models.set(String(model.id), Model.Info.make(structuredClone(model)))
          },
        },
      } as Parameters<typeof transform>[0])
    }
    published = models
  }
  const ctx = {
    app: { version: "2.2.5" }, options: {},
    provider: {
      list: async () => [sourceProvider],
      transform: async (callback: typeof transforms[number]) => { transforms.push(callback); replay() },
      reload: async () => { replay() },
    },
    integration: {
      transform: async () => {}, reload: async () => {},
      connection: { active: async () => undefined },
    },
    tool: { transform: async () => {} },
    event: { subscribe: () => ({ async *[Symbol.asyncIterator]() {} }) },
    storage: {
      get: async (key: string) => storage.get(key),
      set: async (key: string, value: unknown) => { storage.set(key, structuredClone(value)) },
      remove: async (key: string) => { storage.delete(key) },
      scan: async ({ prefix }: { prefix: string }) => ({
        entries: [...storage].filter(([key]) => key.startsWith(prefix)).map(([key, value]) => ({ key, value })),
      }),
    },
    rpc: { register: async (_definition: unknown, registered: typeof handlers) => { Object.assign(handlers, registered) } },
  }
  return { ctx, handlers, model: () => published.get("remote/model-alpha"), storage }
}

describe("V2 local OpenAI-compatible integration", () => {
  it("updates this plugin's previously injected model on replay and restores it after override deletion", async () => {
    const mock = await startMockProvider()
    const host = createReplayHost(mock.baseURL)
    let cleanup: (() => void) | void = undefined
    try {
      cleanup = await plugin.setup(host.ctx as never)
      const before = host.model()!
      expect(before.variants).toEqual([])
      const variants = [{ id: "max", settings: { reasoningEffort: "max", providerMode: "careful" }, body: { thinking: { enabled: true } } }]
      await host.handlers.overrideSet({ providerID: "local", modelID: before.id, patch: {
        reasoning: true, attachment: true, variants, compatibility: { reasoningField: "reasoning" },
      } })

      const updated = host.model()!
      expect(updated.variants).toEqual(variants)
      expect(updated.compatibility).toEqual({ reasoningField: "reasoning" })
      // Validate actual host-shaped output, not only the plugin's status strings.
      expect(updated).not.toHaveProperty("reasoning")
      expect(updated).not.toHaveProperty("attachment")
      expect(before.variants).toEqual([])
      expect(before.compatibility).toBeUndefined()

      await host.handlers.refresh({ force: true })
      expect(host.model()).toEqual(updated)
      await host.handlers.overrideDelete({ providerID: "local", modelID: before.id })
      expect(host.model()).toEqual(before)
      expect(updated.variants).toEqual(variants)
      expect(mock.requests.filter((request) => request.path === "/v1/models")).toHaveLength(4)
    } finally {
      cleanup?.()
      await new Promise<void>((resolve, reject) => mock.server.close((error) => error ? reject(error) : resolve()))
    }
  })

  it.each(["explicit configuration", "an earlier plugin"])("preserves an existing source model from %s rather than replacing it with discovery overrides", async (source) => {
    const mock = await startMockProvider()
    // Both are immutable source definitions visible before this transform runs.
    const explicit = Model.Info.make({
      ...Model.Info.default(Provider.ID.make("local"), Model.ID.make("remote/model-alpha")),
      name: source,
      variants: [{ id: "high", settings: { reasoningEffort: "high", providerMode: "keep" }, body: { thinking: { enabled: true } }, headers: { "x-mode": "keep" } }],
      compatibility: { reasoningField: "reasoning_text", requireReasoning: false },
    })
    const snapshot = structuredClone(explicit)
    const host = createReplayHost(mock.baseURL, [explicit])
    let cleanup: (() => void) | void = undefined
    try {
      cleanup = await plugin.setup(host.ctx as never)
      expect(host.model()).toEqual(snapshot)
      await host.handlers.overrideSet({ providerID: "local", modelID: explicit.id, patch: {
        name: "Discovery override", variants: [{ id: "max", settings: { reasoningEffort: "max" } }],
        compatibility: { reasoningField: "reasoning" },
      } })

      expect(host.storage.has("overrides:local:remote/model-alpha")).toBe(true)
      const status = await host.handlers.status({ details: true }) as RpcStatusOutput
      expect(status.providers[0]?.models[0]?.name).toBe("Discovery override")
      expect(host.model()).toEqual(snapshot)
      await host.handlers.refresh({ force: true })
      expect(host.model()).toEqual(snapshot)
      await host.handlers.overrideDelete({ providerID: "local", modelID: explicit.id })
      expect(host.model()).toEqual(snapshot)
      expect(explicit).toEqual(snapshot)
    } finally {
      cleanup?.()
      await new Promise<void>((resolve, reject) => mock.server.close((error) => error ? reject(error) : resolve()))
    }
  })

  it("drops plugin-only and unsupported fields with the installed V2 runtime model schema", () => {
    const variants = [{ id: "high", settings: { reasoningEffort: "high", providerMode: "keep" }, body: { thinking: { enabled: true } }, headers: { "x-mode": "keep" } }]
    const model = Model.Info.make({
      ...Model.Info.default(Provider.ID.make("local"), Model.ID.make("model")),
      reasoning: true, attachment: true, options: { providerMode: "ignored" },
      variants, settings: { providerMode: "keep" },
      compatibility: { reasoningField: "reasoning_content" },
    } as Model.Info)
    expect(model).not.toHaveProperty("reasoning")
    expect(model).not.toHaveProperty("attachment")
    expect(model).not.toHaveProperty("options")
    expect(model.variants).toEqual(variants)
    expect(model.settings).toEqual({ providerMode: "keep" })
    expect(model.compatibility).toEqual({ reasoningField: "reasoning_content" })
  })

  it("discovers models and sends a runtime request with the expected credential and model ID", async () => {
    const mock = await startMockProvider()
    try {
      const provider = {
        id: "local",
        package: "@opencode/ai/providers/openai-compatible",
        apiKey: "example-integration-key",
        settings: { baseURL: mock.baseURL },
      }
      const discovery = new Map([["local", parseProviderDiscoveryOptions({ enabled: true })!]])
      const inventory = await discoverInventory([provider], discovery, fetch)
      const model = inventory.get("local")?.get("remote/model-alpha")

      expect(model).toMatchObject({ id: "remote/model-alpha", modelID: "remote/model-alpha" })
      expect(mock.requests[0]).toMatchObject({ path: "/v1/models", authorization: "Bearer example-integration-key" })

      const response = await fetch(`${mock.baseURL}/chat/completions`, {
        method: "POST",
        headers: { authorization: "Bearer example-integration-key", "content-type": "application/json" },
        body: JSON.stringify({ model: model?.modelID, messages: [{ role: "user", content: "hello" }] }),
      })
      expect(response.ok).toBe(true)
      expect(await response.json()).toMatchObject({ model: "remote/model-alpha" })
      expect(mock.requests[1]).toMatchObject({
        path: "/v1/chat/completions",
        authorization: "Bearer example-integration-key",
        body: JSON.stringify({ model: "remote/model-alpha", messages: [{ role: "user", content: "hello" }] }),
      })
    } finally {
      await new Promise<void>((resolve, reject) => mock.server.close((error) => error ? reject(error) : resolve()))
    }
  })

  it("uses a fresh cache entry and bypasses it for force refresh", async () => {
    const mock = await startMockProvider()
    const cache = new Map<string, unknown>()
    const storage = {
      get: async (key: string) => cache.get(key),
      set: async (key: string, value: unknown) => { cache.set(key, value) },
    }
    const provider = {
      id: "local",
      package: "@opencode/ai/providers/openai-compatible",
      settings: { baseURL: mock.baseURL },
    }
    const discovery = new Map([["local", parseProviderDiscoveryOptions({ enabled: true, cache: { enabled: true } })!]])

    try {
      await discoverInventory([provider], discovery, fetch, storage)
      await discoverInventory([provider], discovery, fetch, storage)
      await discoverInventory([provider], discovery, fetch, storage, { force: true })
      expect(mock.requests.filter((request) => request.path === "/v1/models")).toHaveLength(2)
    } finally {
      await new Promise<void>((resolve, reject) => mock.server.close((error) => error ? reject(error) : resolve()))
    }
  })
})
