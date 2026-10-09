import assert from "node:assert/strict";
import test from "node:test";
import {
  filterPathsBySource,
  filterWorkspacePaths,
  knowledgeTabDirectory,
  parseWorkspaceExtensions,
  sharedKnowledgeMounted,
  workspaceFileListing,
} from "../app/workspace-files.ts";

test("workspace filter hides dot paths and keeps configured file types", () => {
  const paths = [
    ".DS_Store",
    ".hidden/note.md",
    "daily/.draft.md",
    "daily/note.md",
    "digest/summary.TXT",
    "resource/data.json",
  ];

  assert.deepEqual(filterWorkspacePaths(paths, parseWorkspaceExtensions()), [
    "daily/note.md",
    "digest/summary.TXT",
  ]);
  assert.deepEqual(
    filterWorkspacePaths(paths, parseWorkspaceExtensions("json")),
    ["resource/data.json"],
  );
});

test("workspace sources expose journal and knowledge files without an archive source", () => {
  const paths = [
    "daily/2026-08-05.md",
    "digest/wiki/topic.md",
    "notes/idea.md",
  ];
  const config = { daily_dir: "daily", digest_dir: "digest" };

  assert.deepEqual(filterPathsBySource(paths, "workspace", config), paths);
  assert.deepEqual(filterPathsBySource(paths, "daily", config), [
    "daily/2026-08-05.md",
  ]);
  assert.deepEqual(filterPathsBySource(paths, "digest", config), [
    "digest/wiki/topic.md",
  ]);
});

test("mounted shared knowledge bases use the knowledge tab and leave the workspace tab", () => {
  const paths = [
    "daily/2026-08-05.md",
    "digest/wiki/topic.md",
    "knowledge/business/wiki/topic.md",
    "resource/input.txt",
  ];
  const config = {
    daily_dir: "daily",
    digest_dir: "digest",
    knowledge_dir: "knowledge",
    knowledge_base_id: "zhb",
  };

  assert.equal(sharedKnowledgeMounted(config), true);
  assert.equal(knowledgeTabDirectory(config), "knowledge");
  assert.deepEqual(filterPathsBySource(paths, "workspace", config), [
    "daily/2026-08-05.md",
    "digest/wiki/topic.md",
    "resource/input.txt",
  ]);
  assert.deepEqual(filterPathsBySource(paths, "digest", config), [
    "knowledge/business/wiki/topic.md",
  ]);
});

test("workspace listing reports when the service result reaches its limit", () => {
  assert.deepEqual(workspaceFileListing(["a.md", "b.md"], 2), {
    paths: ["a.md", "b.md"],
    limited: true,
  });
  assert.deepEqual(workspaceFileListing(["a.md"], 2), {
    paths: ["a.md"],
    limited: false,
  });
});
