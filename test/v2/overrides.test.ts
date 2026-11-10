import { describe, expect, it, vi } from "vitest"
import {
  applyModelOverride,
  deleteModelOverride,
  listModelOverrides,
  overrideStorageKey,
  setModelOverride,
} from "../../src/v2/overrides.js"

function storage() {
  const values = new Map<string, unknown>()
  return {
    values,
    get: vi.fn(async (key: string) => values.get(key)),
    set: vi.fn(async (key: string, value: unknown) => { values.set(key, value) }),
    remove: vi.fn(async (key: string) => { values.delete(key) }),
    scan: vi.fn(async ({ prefix }: { prefix: string }) => ({
      entries: [...values.entries()]
        .filter(([key]) => key.startsWith(prefix))
        .map(([key, value]) => ({ key, value })),
    })),
  }
}

describe("V2 model overrides", () => {
  it("stores and lists overrides in the provider/model namespace", async () => {
    const backend = storage()
    const saved = await setModelOverride(backend, {
      providerID: "provider-a",
      modelID: "model-1",
      patch: { name: "Custom model", limit: { context: 65536 } },
    })

    expect(backend.values.has("overrides:provider-a:model-1")).toBe(true)
    expect(overrideStorageKey("provider-a", "model-1")).toBe("overrides:provider-a:model-1")
    expect(await listModelOverrides(backend, "provider-a")).toEqual([saved])
    expect(await listModelOverrides(backend, "provider-b")).toEqual([])
  })

  it("rejects unsupported or sensitive patch fields", async () => {
    const backend = storage()
    await expect(setModelOverride(backend, {
      providerID: "provider-a",
      modelID: "model-1",
      patch: { apiKey: "secret" },
    })).rejects.toThrow("not allowed")
    await expect(setModelOverride(backend, {
      providerID: "provider-a",
      modelID: "model-1",
      patch: { unknown: true },
    })).rejects.toThrow("Unsupported override field")
    await expect(setModelOverride(backend, {
      providerID: "provider-a",
      modelID: "model-1",
      patch: { variants: [{ headers: { authorization: "secret" } }] },
    })).rejects.toThrow("not allowed")
  })

  it("deep-merges an enabled override without changing the API model ID", () => {
    const model = {
      id: "model-1",
      modelID: "remote-model",
      name: "Discovered",
      limit: { context: 32768, output: 4096 },
      capabilities: { tools: false, input: ["text"], output: ["text"] },
    }
    const result = applyModelOverride(model, {
      providerID: "provider-a",
      modelID: "model-1",
      patch: { name: "Custom", limit: { context: 65536 }, capabilities: { tools: true } },
      enabled: true,
      updatedAt: new Date().toISOString(),
    })

    expect(result).toEqual({
      ...model,
      name: "Custom",
      limit: { context: 65536, output: 4096 },
      capabilities: { tools: true, input: ["text"], output: ["text"] },
    })
  })

  it("stores explicit reasoning variants and a reasoning field without generating defaults", () => {
    const patch = {
      reasoning: true,
      variants: [
        { id: "low", settings: { reasoningEffort: "low" } },
        { id: "high", settings: { reasoningEffort: "high" } },
        { id: "max", settings: { reasoningEffort: "max" } },
      ],
      compatibility: { reasoningField: "reasoning" },
    }
    const result = applyModelOverride({
      id: "model-1",
      modelID: "model-1",
      name: "Discovered",
      limit: { context: 131072, output: 8192 },
      capabilities: { tools: true, input: ["text"], output: ["text"] },
    }, {
      providerID: "provider-a",
      modelID: "model-1",
      patch,
      enabled: true,
      updatedAt: new Date().toISOString(),
    })

    expect(result.reasoning).toBe(true)
    expect(result.variants).toEqual(patch.variants)
    expect(result.compatibility).toEqual({ reasoningField: "reasoning" })
  })

  it("does not derive reasoning variants from a reasoning flag alone", () => {
    const result = applyModelOverride({
      id: "model-1",
      modelID: "model-1",
      name: "Discovered",
      limit: { context: 131072, output: 8192 },
      capabilities: { tools: true, input: ["text"], output: ["text"] },
    }, {
      providerID: "provider-a",
      modelID: "model-1",
      patch: { reasoning: true },
      enabled: true,
      updatedAt: new Date().toISOString(),
    })

    expect(result.reasoning).toBe(true)
    expect(result.variants).toBeUndefined()
    expect(result.compatibility).toBeUndefined()
  })

  it("replaces discovered variants with an explicit override list", () => {
    const result = applyModelOverride({
      id: "model-1",
      modelID: "model-1",
      name: "Discovered",
      limit: { context: 131072, output: 8192 },
      capabilities: { tools: true, input: ["text"], output: ["text"] },
      variants: [{ id: "medium", settings: { reasoningEffort: "medium" } }],
    }, {
      providerID: "provider-a",
      modelID: "model-1",
      patch: { variants: [{ id: "max", settings: { reasoningEffort: "max" } }] },
      enabled: true,
      updatedAt: new Date().toISOString(),
    })

    expect(result.variants).toEqual([{ id: "max", settings: { reasoningEffort: "max" } }])
  })

  it("keeps discovered variants when an override does not set them", () => {
    const discoveredVariants = [{ id: "low", settings: { reasoningEffort: "low" } }]
    const result = applyModelOverride({
      id: "model-1",
      modelID: "model-1",
      name: "Discovered",
      limit: { context: 131072, output: 8192 },
      capabilities: { tools: true, input: ["text"], output: ["text"] },
      variants: discoveredVariants,
    }, {
      providerID: "provider-a",
      modelID: "model-1",
      patch: { name: "Renamed" },
      enabled: true,
      updatedAt: new Date().toISOString(),
    })

    expect(result.variants).toBe(discoveredVariants)
  })

  it("accepts max and reasoning fields as override patch fields", async () => {
    const backend = storage()
    await expect(setModelOverride(backend, {
      providerID: "provider-a",
      modelID: "model-1",
      patch: {
        reasoning: true,
        variants: [{ id: "max", settings: { reasoningEffort: "max" } }],
        compatibility: { reasoningField: "reasoning" },
      },
    })).resolves.toMatchObject({
      patch: {
        variants: [{ id: "max", settings: { reasoningEffort: "max" } }],
        compatibility: { reasoningField: "reasoning" },
      },
    })
  })

  it("deletes an existing override and reports whether it existed", async () => {
    const backend = storage()
    await setModelOverride(backend, { providerID: "provider-a", modelID: "model-1", patch: { reasoning: true } })
    await expect(deleteModelOverride(backend, "provider-a", "model-1")).resolves.toBe(true)
    await expect(deleteModelOverride(backend, "provider-a", "model-1")).resolves.toBe(false)
  })
})
