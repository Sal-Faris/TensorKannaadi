import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(new Request("http://localhost/", { headers: { accept: "text/html" } }), {
    ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
  }, { waitUntil() {}, passThroughOnException() {} });
}

test("server-renders the Kannaadi workbench", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Kannaadi/);
  assert.match(html, /Mechanistic interpretability workbench/);
  assert.match(html, /Model explorer/);
  assert.match(html, /Inspector/);
  assert.match(html, /blocks\.0\.attn\.head\.0/);
  assert.match(html, /aria-label="Select blocks\.3\.attn\.head\.0"/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/);
});
