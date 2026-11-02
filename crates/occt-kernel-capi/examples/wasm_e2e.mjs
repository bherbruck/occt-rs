// node wasm_e2e.mjs <occt_capi.mjs> <wasm-bindgen --target nodejs output .js>
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [modulePath, rustPath] = process.argv.slice(2).map((p) => resolve(p));
const rust = createRequire(import.meta.url)(rustPath);

const before = rust.before_load();
console.log("before load:", before);
if (!before.startsWith("Err(Unavailable")) throw new Error("expected Unavailable");

const { default: createOcctCapi } = await import(pathToFileURL(modulePath));
globalThis.occtCapi = await createOcctCapi();

console.log("after load:", rust.before_load());
console.log(rust.report());
