// node smoke.mjs <occt_capi.mjs>: checks the wasm module through its raw exports.
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const { default: createOcctCapi } = await import(pathToFileURL(resolve(process.argv[2])));
const m = await createOcctCapi();

const text = (ptr, len) => new TextDecoder().decode(m.HEAPU8.slice(ptr, ptr + len));
const lastError = () => text(m._occt_last_error_ptr(), m._occt_last_error_len());
const shape = (h) => {
  assert.notEqual(h, 0, lastError());
  return h;
};
const take = (h) => {
  assert.notEqual(h, 0, lastError());
  const ptr = m._occt_buffer_ptr(h) >>> 0;
  const bytes = m.HEAPU8.slice(ptr, ptr + (m._occt_buffer_len(h) >>> 0));
  m._occt_buffer_free(h);
  return bytes;
};

assert.equal(m._occt_abi_version(), 3);
const version = text(m._occt_library_version_ptr(), m._occt_library_version_len());
assert.ok(version.startsWith("8.0.1"), version);

const box = shape(m._occt_make_box(0, 0, 0, 10, 10, 2));
const cylinder = shape(m._occt_make_cylinder(5, 5, -1, 0, 0, 1, 2, 4));
const part = shape(m._occt_cut(box, cylinder));
const expected = 200 - Math.PI * 8;
const volume = m._occt_volume(part);
assert.ok(Math.abs(volume - expected) < 1e-6, `volume ${volume}`);
assert.equal(m._occt_shape_kind(part), 2);

const mesh = take(m._occt_mesh(part, 0.01, 0.35, 0));
const header = new DataView(mesh.buffer);
const vertices = header.getUint32(0, true);
const triangles = header.getUint32(4, true);
assert.equal(mesh.length, 8 + vertices * 48 + triangles * 12);
assert.ok(triangles > 12);

const step = take(m._occt_step_write(part));
assert.equal(new TextDecoder().decode(step.slice(0, 12)), "ISO-10303-21");
const ptr = m._occt_alloc(step.length) >>> 0;
m.HEAPU8.set(step, ptr);
const back = shape(m._occt_step_read(ptr, step.length));
m._occt_dealloc(ptr);
const backVolume = m._occt_volume(back);
assert.ok(Math.abs(backVolume - expected) < 1e-6, `STEP volume ${backVolume}`);

// C++ exceptions become errors, not aborts.
assert.equal(m._occt_make_box(0, 0, 0, 0, 0, 0), 0);
const zeroBox = lastError();
assert.ok(zeroBox.length > 0);
assert.ok(Number.isNaN(m._occt_volume(0xfffffff)));
const badHandle = lastError();
assert.ok(badHandle.includes("invalid shape handle"), badHandle);
assert.ok(m._occt_shape_kind(box) === 2, "the module still works after errors");

// Memory growth must keep HEAPU8 usable.
const before = m.HEAPU8.buffer.byteLength;
const big = m._occt_alloc(before * 2) >>> 0;
assert.notEqual(big, 0);
assert.ok(m.HEAPU8.buffer.byteLength > before, "memory did not grow");
m.HEAPU8[big + before] = 7;
assert.equal(m.HEAPU8[big + before], 7);
m._occt_dealloc(big);

for (const h of [box, cylinder, part, back]) m._occt_shape_free(h);

console.log(
  `ok: OCCT ${version}, cut volume ${volume.toFixed(6)}, ${triangles} triangles, ` +
    `${step.length} STEP bytes, STEP volume ${backVolume.toFixed(6)}, ` +
    `zero box: "${zeroBox}", memory ${before} -> ${m.HEAPU8.buffer.byteLength} bytes`,
);
