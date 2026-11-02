//! Writing meshes to the file formats 3D printers and other tools read.
//!
//! All of these are methods on [`Mesh`], take anything that implements [`Write`], and
//! buffer internally, so passing a [`File`](std::fs::File) directly is fine. Each writer
//! checks the mesh first: an index out of range or a coordinate that isn't finite is an
//! [`ErrorKind::InvalidData`] error before anything is written.
//!
//! ```
//! use occt_math::{Point3, Vec3};
//! use occt_mesh::Mesh;
//!
//! # fn main() -> std::io::Result<()> {
//! let triangle = Mesh {
//!     positions: vec![
//!         Point3::new(0.0, 0.0, 0.0),
//!         Point3::new(1.0, 0.0, 0.0),
//!         Point3::new(0.0, 1.0, 0.0),
//!     ],
//!     normals: vec![Vec3::new(0.0, 0.0, 1.0); 3],
//!     indices: vec![[0, 1, 2]],
//! };
//! let mut stl = Vec::new();
//! triangle.write_stl_binary(&mut stl)?;
//! assert_eq!(stl.len(), 84 + 50);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::{self, BufWriter, ErrorKind, Write};

use occt_math::{Point3, Vec3};

use crate::Mesh;

/// Goes in a binary STL's 80-byte header. It must not begin with `solid`, which is how
/// readers recognise the text form.
const STL_HEADER: &[u8] = b"binary STL written by occt-mesh";

/// Positions rounded to this many parts per model unit are one vertex in a 3MF file. It
/// matches the coincidence distance of modelling kernels, about 1e-7, to the nearest
/// decade above, since vertices that a kernel holds as shared may differ by that much.
const WELD_PER_UNIT: f64 = 1e6;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

/// A triangle's unit normal from its winding; zero if it has no area.
fn facet_normal([a, b, c]: [Point3; 3]) -> Vec3 {
    let normal = (b - a).cross(c - a);
    let length = normal.length();
    if length > 0.0 {
        normal / length
    } else {
        Vec3::ZERO
    }
}

fn little_endian_f32s(out: &mut Vec<u8>, [x, y, z]: [f64; 3]) {
    for value in [x, y, z] {
        // The conversion to single precision is what the format holds.
        #[allow(clippy::cast_possible_truncation)]
        out.extend_from_slice(&(value as f32).to_le_bytes());
    }
}

impl Mesh {
    fn check(&self) -> io::Result<()> {
        if !self.positions.iter().all(|p| p.is_finite()) {
            return Err(invalid("the mesh has a position that isn't finite"));
        }
        let count = self.positions.len();
        if self.indices.iter().flatten().any(|&i| i as usize >= count) {
            return Err(invalid("the mesh has a triangle index out of range"));
        }
        Ok(())
    }

    /// Writes a binary STL: single precision and compact, so the usual choice.
    ///
    /// Facet normals come from the winding, as STL requires of them, not from
    /// [`normals`](Mesh::normals).
    ///
    /// # Errors
    /// [`ErrorKind::InvalidData`] if the mesh is malformed, a coordinate doesn't fit in
    /// single precision or it has more triangles than the format can count. Otherwise
    /// whatever the writer reports.
    ///
    /// # Examples
    /// ```
    /// # use occt_math::Point3;
    /// # use occt_mesh::Mesh;
    /// # fn main() -> std::io::Result<()> {
    /// let mesh = Mesh::default();
    /// let mut out = Vec::new();
    /// mesh.write_stl_binary(&mut out)?;
    /// assert_eq!(out.len(), 84, "a header and a zero count");
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_stl_binary(&self, writer: impl Write) -> io::Result<()> {
        self.check()?;
        let count = u32::try_from(self.indices.len())
            .map_err(|_| invalid("a binary STL holds fewer triangles than this mesh has"))?;
        let representable = |p: &Point3| {
            #[allow(clippy::cast_possible_truncation)]
            let fits = |v: f64| (v as f32).is_finite();
            fits(p.x) && fits(p.y) && fits(p.z)
        };
        if !self.positions.iter().all(representable) {
            return Err(invalid("a coordinate doesn't fit in single precision"));
        }

        let mut out = BufWriter::new(writer);
        let mut header = [0; 80];
        header[..STL_HEADER.len()].copy_from_slice(STL_HEADER);
        out.write_all(&header)?;
        out.write_all(&count.to_le_bytes())?;
        let mut record = Vec::with_capacity(50);
        for triangle in self.triangles() {
            record.clear();
            let n = facet_normal(triangle);
            little_endian_f32s(&mut record, [n.x, n.y, n.z]);
            for p in triangle {
                little_endian_f32s(&mut record, [p.x, p.y, p.z]);
            }
            record.extend_from_slice(&[0, 0]); // The attribute count, always zero.
            out.write_all(&record)?;
        }
        out.flush()
    }

    /// Writes a text STL. Larger than the binary form, but readable, and it keeps double
    /// precision.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidData`] if the mesh is malformed. Otherwise whatever the writer
    /// reports.
    ///
    /// # Examples
    /// ```
    /// # use occt_mesh::Mesh;
    /// # fn main() -> std::io::Result<()> {
    /// let mut out = Vec::new();
    /// Mesh::default().write_stl_ascii(&mut out)?;
    /// assert_eq!(String::from_utf8_lossy(&out), "solid mesh\nendsolid mesh\n");
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_stl_ascii(&self, writer: impl Write) -> io::Result<()> {
        self.check()?;
        let mut out = BufWriter::new(writer);
        writeln!(out, "solid mesh")?;
        for triangle in self.triangles() {
            let n = facet_normal(triangle);
            writeln!(out, "facet normal {} {} {}", n.x, n.y, n.z)?;
            writeln!(out, "  outer loop")?;
            for p in triangle {
                writeln!(out, "    vertex {} {} {}", p.x, p.y, p.z)?;
            }
            writeln!(out, "  endloop")?;
            writeln!(out, "endfacet")?;
        }
        writeln!(out, "endsolid mesh")?;
        out.flush()
    }

    /// Writes a Wavefront OBJ, with normals when the mesh has one per position.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidData`] if the mesh is malformed. Otherwise whatever the writer
    /// reports.
    ///
    /// # Examples
    /// ```
    /// # use occt_math::{Point3, Vec3};
    /// # use occt_mesh::Mesh;
    /// # fn main() -> std::io::Result<()> {
    /// let mesh = Mesh {
    ///     positions: vec![Point3::ORIGIN; 3],
    ///     normals: vec![Vec3::new(0.0, 0.0, 1.0); 3],
    ///     indices: vec![[0, 1, 2]],
    /// };
    /// let mut out = Vec::new();
    /// mesh.write_obj(&mut out)?;
    /// assert!(String::from_utf8_lossy(&out).ends_with("f 1//1 2//2 3//3\n"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_obj(&self, writer: impl Write) -> io::Result<()> {
        self.check()?;
        let mut out = BufWriter::new(writer);
        for p in &self.positions {
            writeln!(out, "v {} {} {}", p.x, p.y, p.z)?;
        }
        let with_normals = self.normals.len() == self.positions.len();
        if with_normals {
            if !self.normals.iter().all(|n| n.is_finite()) {
                return Err(invalid("the mesh has a normal that isn't finite"));
            }
            for n in &self.normals {
                writeln!(out, "vn {} {} {}", n.x, n.y, n.z)?;
            }
        }
        for &[a, b, c] in &self.indices {
            // The format counts from one, and past u32 only if the mesh could.
            let (a, b, c) = (u64::from(a) + 1, u64::from(b) + 1, u64::from(c) + 1);
            if with_normals {
                writeln!(out, "f {a}//{a} {b}//{b} {c}//{c}")?;
            } else {
                writeln!(out, "f {a} {b} {c}")?;
            }
        }
        out.flush()
    }

    /// Writes a 3MF package, in millimetres: a zip archive of XML, the format slicers
    /// prefer because it can say a mesh is one closed part.
    ///
    /// The format wants each corner shared between the triangles that meet there, so
    /// vertices that coincide are merged, and triangles this collapses are left out. The
    /// archive's entries are stored rather than compressed, which the format allows; to
    /// shrink one, recompress it with any zip tool.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidData`] if the mesh is malformed or would make an archive over
    /// the 4 GiB this writer can describe. Otherwise whatever the writer reports.
    ///
    /// # Examples
    /// ```
    /// # use occt_math::{Point3, Vec3};
    /// # use occt_mesh::Mesh;
    /// # fn main() -> std::io::Result<()> {
    /// let mut out = Vec::new();
    /// Mesh::default().write_3mf(&mut out)?;
    /// assert_eq!(&out[..2], b"PK");
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_3mf(&self, writer: impl Write) -> io::Result<()> {
        self.check()?;
        let model = self.model_xml();
        write_stored_zip(
            writer,
            &[
                ("[Content_Types].xml", CONTENT_TYPES.as_bytes()),
                ("_rels/.rels", RELATIONSHIPS.as_bytes()),
                ("3D/3dmodel.model", model.as_bytes()),
            ],
        )
    }

    fn model_xml(&self) -> String {
        let key = |p: Point3| {
            // Saturating is harmless: coordinates that large aren't being told apart anyway.
            #[allow(clippy::cast_possible_truncation)]
            [p.x, p.y, p.z].map(|v| (v * WELD_PER_UNIT).round() as i64)
        };
        let mut welded: HashMap<[i64; 3], usize> = HashMap::new();
        let mut vertices = String::new();
        let mut map = Vec::with_capacity(self.positions.len());
        for &p in &self.positions {
            let next = welded.len();
            let index = *welded.entry(key(p)).or_insert(next);
            if index == next {
                // Writing to a String can't fail.
                let _ = writeln!(
                    vertices,
                    "<vertex x=\"{}\" y=\"{}\" z=\"{}\"/>",
                    p.x, p.y, p.z
                );
            }
            map.push(index);
        }
        let mut triangles = String::new();
        for t in &self.indices {
            let [a, b, c] = t.map(|i| map[i as usize]);
            if a != b && b != c && a != c {
                let _ = writeln!(triangles, "<triangle v1=\"{a}\" v2=\"{b}\" v3=\"{c}\"/>");
            }
        }
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <model unit=\"millimeter\" xml:lang=\"en-US\" \
             xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n\
             <resources>\n<object id=\"1\" type=\"model\">\n<mesh>\n\
             <vertices>\n{vertices}</vertices>\n<triangles>\n{triangles}</triangles>\n\
             </mesh>\n</object>\n</resources>\n\
             <build>\n<item objectid=\"1\"/>\n</build>\n</model>\n"
        )
    }
}

const CONTENT_TYPES: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\n\
<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\n\
<Default Extension=\"model\" ContentType=\"application/vnd.ms-package.3dmanufacturing-3dmodel+xml\"/>\n\
</Types>\n";

const RELATIONSHIPS: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n\
<Relationship Target=\"/3D/3dmodel.model\" Id=\"rel0\" \
Type=\"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel\"/>\n\
</Relationships>\n";

const CRC_TABLE: [u32; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        #[allow(clippy::cast_possible_truncation)]
        let mut value = i as u32;
        let mut bit = 0;
        while bit < 8 {
            value = if value & 1 == 1 {
                0xEDB8_8320 ^ (value >> 1)
            } else {
                value >> 1
            };
            bit += 1;
        }
        table[i] = value;
        i += 1;
    }
    table
};

fn crc32(data: &[u8]) -> u32 {
    !data.iter().fold(u32::MAX, |crc, &byte| {
        CRC_TABLE[usize::from(crc as u8 ^ byte)] ^ (crc >> 8)
    })
}

/// A zip archive of uncompressed entries. A few fixed fields is all that needs, so a zip
/// library would be a dependency for nothing.
fn write_stored_zip(writer: impl Write, entries: &[(&str, &[u8])]) -> io::Result<()> {
    const VERSION: u16 = 20;
    const DOS_DATE: u16 = 0x0021; // 1980-01-01, the earliest there is.
    let too_large = || invalid("the archive is over the 4 GiB a plain zip can describe");

    let mut out = BufWriter::new(writer);
    let mut directory = Vec::new();
    let mut offset = 0u32;
    for &(name, data) in entries {
        let crc = crc32(data);
        let size = u32::try_from(data.len()).map_err(|_| too_large())?;
        let name_len = u16::try_from(name.len()).map_err(|_| too_large())?;
        let mut common = Vec::with_capacity(26);
        for field in [VERSION, 0, 0, 0, DOS_DATE] {
            common.extend_from_slice(&field.to_le_bytes());
        }
        common.extend_from_slice(&crc.to_le_bytes());
        common.extend_from_slice(&size.to_le_bytes());
        common.extend_from_slice(&size.to_le_bytes());
        common.extend_from_slice(&name_len.to_le_bytes());
        common.extend_from_slice(&0u16.to_le_bytes());

        out.write_all(b"PK\x03\x04")?;
        out.write_all(&common)?;
        out.write_all(name.as_bytes())?;
        out.write_all(data)?;

        directory.extend_from_slice(b"PK\x01\x02");
        directory.extend_from_slice(&VERSION.to_le_bytes());
        directory.extend_from_slice(&common);
        directory.extend_from_slice(&[0; 10]); // Comment, disk, and file attributes.
        directory.extend_from_slice(&offset.to_le_bytes());
        directory.extend_from_slice(name.as_bytes());

        offset = u32::try_from(30 + name.len() + data.len())
            .ok()
            .and_then(|written| offset.checked_add(written))
            .ok_or_else(too_large)?;
    }
    let count = u16::try_from(entries.len()).map_err(|_| too_large())?;
    let directory_len = u32::try_from(directory.len()).map_err(|_| too_large())?;
    offset.checked_add(directory_len).ok_or_else(too_large)?;
    out.write_all(&directory)?;
    out.write_all(b"PK\x05\x06")?;
    out.write_all(&[0; 4])?; // This disk and the directory's.
    out.write_all(&count.to_le_bytes())?;
    out.write_all(&count.to_le_bytes())?;
    out.write_all(&directory_len.to_le_bytes())?;
    out.write_all(&offset.to_le_bytes())?;
    out.write_all(&[0; 2])?; // No comment.
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetrahedron() -> Mesh {
        Mesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            ],
            normals: vec![Vec3::new(0.0, 0.0, 1.0); 4],
            indices: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        }
    }

    fn u16_at(bytes: &[u8], at: usize) -> u16 {
        u16::from_le_bytes([bytes[at], bytes[at + 1]])
    }

    fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }

    fn f32_at(bytes: &[u8], at: usize) -> f32 {
        f32::from_bits(u32_at(bytes, at))
    }

    /// Stops accepting after a few bytes, to prove errors come back.
    struct Full(usize);

    impl Write for Full {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            if self.0 < data.len() {
                return Err(io::Error::new(ErrorKind::StorageFull, "full"));
            }
            self.0 -= data.len();
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    type Writer = fn(&Mesh, &mut dyn Write) -> io::Result<()>;

    fn every_format() -> [(&'static str, Writer); 4] {
        [
            ("binary STL", |m, w| m.write_stl_binary(w)),
            ("text STL", |m, w| m.write_stl_ascii(w)),
            ("OBJ", |m, w| m.write_obj(w)),
            ("3MF", |m, w| m.write_3mf(w)),
        ]
    }

    #[test]
    fn binary_stl_reads_back_by_hand() {
        let mesh = tetrahedron();
        let mut out = Vec::new();
        mesh.write_stl_binary(&mut out).unwrap();

        assert_eq!(out.len(), 84 + 4 * 50);
        assert!(!out.starts_with(b"solid"), "that marks the text form");
        assert_eq!(u32_at(&out, 80), 4);

        let mut volume = 0.0;
        for (i, t) in mesh.triangles().enumerate() {
            let record = &out[84 + i * 50..84 + (i + 1) * 50];
            let n = facet_normal(t);
            let read = |at: usize| {
                Vec3::new(
                    f64::from(f32_at(record, at)),
                    f64::from(f32_at(record, at + 4)),
                    f64::from(f32_at(record, at + 8)),
                )
            };
            let normal = read(0);
            assert!((normal - n).length() < 1e-6, "{normal:?} vs {n:?}");
            let corners: Vec<Vec3> = (0..3).map(|c| read(12 + c * 12)).collect();
            for (corner, want) in corners.iter().zip(t) {
                assert!((*corner - (want - Point3::ORIGIN)).length() < 1e-6);
            }
            assert_eq!(u16_at(record, 48), 0);
            volume += corners[0].dot(corners[1].cross(corners[2])) / 6.0;
        }
        assert!((volume - 1.0 / 6.0).abs() < 1e-6, "volume {volume}");
    }

    #[test]
    fn facet_normals_face_out_and_ignore_stored_normals() {
        let mesh = tetrahedron();
        let mut out = Vec::new();
        mesh.write_stl_binary(&mut out).unwrap();
        // The first triangle lies in z = 0 and winds to face down, whatever `normals` says.
        assert_eq!(
            [f32_at(&out, 84), f32_at(&out, 88), f32_at(&out, 92)],
            [0.0, 0.0, -1.0]
        );
    }

    #[test]
    fn text_stl_has_every_facet() {
        let mut out = Vec::new();
        tetrahedron().write_stl_ascii(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("solid mesh\n"));
        assert!(text.ends_with("endsolid mesh\n"));
        assert_eq!(text.matches("facet normal").count(), 4);
        assert_eq!(text.matches("vertex ").count(), 12);
        assert_eq!(text.matches("endfacet").count(), 4);
        assert!(text.contains("facet normal 0 0 -1\n"));
        // Parses back to the original coordinates.
        let vertices: Vec<f64> = text
            .lines()
            .filter_map(|l| l.trim().strip_prefix("vertex "))
            .flat_map(|l| l.split(' ').map(|v| v.parse::<f64>().unwrap()))
            .collect();
        assert_eq!(vertices.len(), 36);
        assert_eq!(&vertices[3..6], &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn obj_counts_from_one() {
        let mut out = Vec::new();
        tetrahedron().write_obj(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(text.lines().filter(|l| l.starts_with("v ")).count(), 4);
        assert_eq!(text.lines().filter(|l| l.starts_with("vn ")).count(), 4);
        assert_eq!(text.lines().filter(|l| l.starts_with("f ")).count(), 4);
        assert!(text.contains("\nf 1//1 3//3 2//2\n"));
        assert!(text.contains("\nv 1 0 0\n"));
    }

    #[test]
    fn obj_without_a_normal_per_position_omits_them() {
        let mut mesh = tetrahedron();
        mesh.normals.clear();
        let mut out = Vec::new();
        mesh.write_obj(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("vn "));
        assert!(text.contains("\nf 1 3 2\n"));
    }

    #[test]
    fn malformed_meshes_write_nothing() {
        let mut bad_index = tetrahedron();
        bad_index.indices[1][2] = 4;
        let mut bad_position = tetrahedron();
        bad_position.positions[2].y = f64::NAN;
        for mesh in [bad_index, bad_position] {
            for (name, write) in every_format() {
                let mut out = Vec::new();
                let e = write(&mesh, &mut out).unwrap_err();
                assert_eq!(e.kind(), ErrorKind::InvalidData, "{name}");
                assert!(out.is_empty(), "{name} wrote {} bytes", out.len());
            }
        }

        let mut huge = tetrahedron();
        huge.positions[0].x = 1e300;
        let e = huge.write_stl_binary(&mut Vec::new()).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData);
        huge.write_stl_ascii(&mut Vec::new()).unwrap();
    }

    #[test]
    fn write_errors_come_back() {
        for (name, write) in every_format() {
            let e = write(&tetrahedron(), &mut Full(3)).unwrap_err();
            assert_eq!(e.kind(), ErrorKind::StorageFull, "{name}");
        }
    }

    #[test]
    fn crc_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    /// Walks the archive the way a reader does: from the end, through the directory.
    fn read_zip(bytes: &[u8]) -> Vec<(String, Vec<u8>)> {
        let end = bytes.len() - 22;
        assert_eq!(&bytes[end..end + 4], b"PK\x05\x06");
        let count = usize::from(u16_at(bytes, end + 10));
        assert_eq!(u16_at(bytes, end + 8), u16_at(bytes, end + 10));
        let directory_size = u32_at(bytes, end + 12) as usize;
        let mut at = u32_at(bytes, end + 16) as usize;
        assert_eq!(at + directory_size, end);

        let mut out = Vec::new();
        for _ in 0..count {
            assert_eq!(&bytes[at..at + 4], b"PK\x01\x02");
            assert_eq!(u16_at(bytes, at + 10), 0, "stored");
            let crc = u32_at(bytes, at + 16);
            let size = u32_at(bytes, at + 24) as usize;
            assert_eq!(u32_at(bytes, at + 20) as usize, size);
            let name_len = usize::from(u16_at(bytes, at + 28));
            let local = u32_at(bytes, at + 42) as usize;
            let name = String::from_utf8(bytes[at + 46..at + 46 + name_len].to_vec()).unwrap();

            assert_eq!(&bytes[local..local + 4], b"PK\x03\x04");
            assert_eq!(usize::from(u16_at(bytes, local + 26)), name_len);
            assert_eq!(u32_at(bytes, local + 14), crc);
            let start = local + 30 + name_len;
            assert_eq!(&bytes[local + 30..start], name.as_bytes());
            let data = bytes[start..start + size].to_vec();
            assert_eq!(crc32(&data), crc, "{name}");
            out.push((name, data));
            at += 46 + name_len;
        }
        assert_eq!(at, end);
        out
    }

    #[test]
    fn three_mf_is_a_readable_package() {
        let mut out = Vec::new();
        tetrahedron().write_3mf(&mut out).unwrap();
        let entries = read_zip(&out);
        let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            ["[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"]
        );
        let model = std::str::from_utf8(&entries[2].1).unwrap();
        assert_eq!(model.matches("<vertex ").count(), 4);
        assert_eq!(model.matches("<triangle ").count(), 4);
        assert!(model.contains("unit=\"millimeter\""));
        assert!(model.contains("<triangle v1=\"0\" v2=\"2\" v3=\"1\"/>"));
    }

    #[test]
    fn three_mf_merges_coincident_corners() {
        // The tetrahedron with each triangle owning its corners, as a mesh from faces does,
        // and one sliver that merging collapses.
        let source = tetrahedron();
        let mut mesh = Mesh::default();
        for (i, t) in source.triangles().enumerate() {
            let base = u32::try_from(i * 3).unwrap();
            mesh.positions.extend(t);
            mesh.normals.extend([Vec3::ZERO; 3]);
            mesh.indices.push([base, base + 1, base + 2]);
        }
        let near = source.positions[0];
        let at = u32::try_from(mesh.positions.len()).unwrap();
        mesh.positions.extend([
            near,
            Point3::new(near.x + 1e-9, near.y, near.z),
            source.positions[1],
        ]);
        mesh.normals.extend([Vec3::ZERO; 3]);
        mesh.indices.push([at, at + 1, at + 2]);

        let mut out = Vec::new();
        mesh.write_3mf(&mut out).unwrap();
        let entries = read_zip(&out);
        let model = std::str::from_utf8(&entries[2].1).unwrap();
        assert_eq!(model.matches("<vertex ").count(), 4);
        assert_eq!(model.matches("<triangle ").count(), 4);
    }
}
