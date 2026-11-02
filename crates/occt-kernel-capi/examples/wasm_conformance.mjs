// node wasm_conformance.mjs <occt_capi.mjs> <wasm-bindgen --target nodejs output .js>
// A trap poisons the Rust instance, so each check gets its own process-fresh module pair.
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const [modulePath, rustPath, only] = process.argv.slice(2);
const require = createRequire(import.meta.url);
const { default: createOcctCapi } = await import(pathToFileURL(resolve(modulePath)));
globalThis.occtCapi = await createOcctCapi();
const rust = require(resolve(rustPath));

if (only !== undefined) {
  rust.run_check(Number(only));
  process.exit(0);
}

const count = rust.check_count();
let passed = 0;
const failed = [];
try {
  // Fast path: run in one instance until something traps, then isolate the rest.
  for (let i = 0; i < count; ++i) {
    try {
      rust.run_check(i);
      passed += 1;
    } catch (e) {
      failed.push([rust.check_name(i), String(e).slice(0, 300)]);
      for (let j = i + 1; j < count; ++j) {
        const r = spawnSync(process.execPath, [fileURLToPath(import.meta.url), modulePath, rustPath, String(j)], { encoding: "utf8" });
        if (r.status === 0) passed += 1;
        else failed.push([rust.check_name(j), (r.stderr || "").split("\n").filter((l) => /panicked|Error|assert/.test(l)).slice(0, 3).join(" | ").slice(0, 300)]);
      }
      break;
    }
  }
} finally {
  console.log(`conformance on wasm: ${passed} passed, ${failed.length} failed of ${count}`);
  for (const [name, why] of failed) console.log(`FAILED ${name}: ${why}`);
}
process.exit(failed.length ? 1 : 0);
