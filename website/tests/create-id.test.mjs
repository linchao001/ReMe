import assert from "node:assert/strict";
import test from "node:test";

import { createId } from "../app/create-id.ts";

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

test("createId returns a UUID v4 string", () => {
  const id = createId();
  assert.match(id, UUID_RE);
});

test("createId falls back when crypto.randomUUID is unavailable", () => {
  const original = globalThis.crypto?.randomUUID;
  if (globalThis.crypto) {
    Object.defineProperty(globalThis.crypto, "randomUUID", {
      configurable: true,
      value: undefined,
    });
  }

  try {
    const id = createId();
    assert.match(id, UUID_RE);
  } finally {
    if (globalThis.crypto && original) {
      Object.defineProperty(globalThis.crypto, "randomUUID", {
        configurable: true,
        value: original,
      });
    }
  }
});

test("createId generates distinct values", () => {
  const ids = new Set(Array.from({ length: 20 }, () => createId()));
  assert.equal(ids.size, 20);
});
