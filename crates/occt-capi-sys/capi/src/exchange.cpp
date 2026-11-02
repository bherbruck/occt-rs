// Entry points of the exchange domain. SPDX-License-Identifier: MIT OR Apache-2.0
//
// Nothing here reads a resource file or an environment variable. Documents are plain
// TDocStd_Document objects with no application, and every option is set in code. The one
// thing that touches the file system is reading IGES, because OCCT's IGES parser only
// opens files by name: see TempFile.
#include "internal.hpp"

#include <BRepBuilderAPI_MakeSolid.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepCheck_Shell.hxx>
#include <BRepLib.hxx>
#include <BRep_Builder.hxx>
#include <BinTools.hxx>
#include <DESTEP_Parameters.hxx>
#include <IGESControl_Reader.hxx>
#include <IGESData_GlobalSection.hxx>
#include <IGESData_IGESModel.hxx>
#include <IGESControl_Writer.hxx>
#include <Interface_InterfaceModel.hxx>
#include <NCollection_DataMap.hxx>
#include <OSD_OpenFile.hxx>
#include <Quantity_Color.hxx>
#include <STEPCAFControl_Reader.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <STEPConstruct_UnitContext.hxx>
#include <StepData_Factors.hxx>
#include <StepRepr_GlobalUnitAssignedContext.hxx>
#include <StepGeom_GeomRepContextAndGlobUnitAssCtxAndGlobUncertaintyAssCtx.hxx>
#include <StepGeom_GeometricRepresentationContextAndGlobalUnitAssignedContext.hxx>
#include <TCollection_AsciiString.hxx>
#include <TCollection_ExtendedString.hxx>
#include <TCollection_HAsciiString.hxx>
#include <TDF_Label.hxx>
#include <TDF_Tool.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Solid.hxx>
#include <UnitsMethods_LengthUnit.hxx>
#include <XCAFDoc_ColorTool.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <XSControl_WorkSession.hxx>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <system_error>

using namespace occt_capi;

namespace {

// A STEP file can't place a top-level product, or use one product at the top level twice.
// A document that does gets one extra assembly holding its roots, with this product name.
// Reading removes it again, so such a document still round trips exactly.
constexpr const char* kDocumentRoot = "occt-document-root";

// The deepest product structure accepted, so that recursion can't exhaust the stack.
constexpr int kMaxDepth = 256;

// OCCT's readers report some failures straight to std::cout rather than through a
// messenger. Runs under the kernel lock, which is the only thing that writes there.
class NullBuffer : public std::streambuf {
 protected:
  int overflow(int c) override { return traits_type::not_eof(c); }
};

class SilentCout {
 public:
  SilentCout() : saved_(std::cout.rdbuf(&null_)) {}
  ~SilentCout() {
    std::cout.rdbuf(saved_);
    std::cout.clear();
  }
  SilentCout(const SilentCout&) = delete;
  SilentCout& operator=(const SilentCout&) = delete;

 private:
  NullBuffer null_;
  std::streambuf* saved_;
};

std::string to_string_input(const uint8_t* data, size_t len) {
  return len == 0 ? std::string() : std::string(reinterpret_cast<const char*>(data), len);
}

// Frees handles already issued for an output that then fails, as no caller will ever
// learn of them.
class IssuedShapes {
 public:
  ~IssuedShapes() {
    if (committed_) return;
    for (const uint32_t h : handles_) g_shapes.erase(h);
  }
  uint32_t put(const TopoDS_Shape& shape) {
    handles_.reserve(handles_.size() + 1);
    const uint32_t h = put_shape(shape);
    handles_.push_back(h);
    return h;
  }
  void commit() { committed_ = true; }

 private:
  std::vector<uint32_t> handles_;
  bool committed_ = false;
};

// --- native saves --------------------------------------------------------------------

constexpr uint8_t kNativeMagic[8] = {'O', 'C', 'C', 'T', 'S', 'A', 'V', '1'};
constexpr size_t kNativeHeader = sizeof(kNativeMagic) + 16;

uint64_t fnv1a(const void* data, size_t len) {
  uint64_t hash = 0xcbf29ce484222325ull;
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < len; ++i) {
    hash ^= bytes[i];
    hash *= 0x100000001b3ull;
  }
  return hash;
}

void put_le64(Bytes& out, uint64_t value) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
}

uint64_t get_le64(const uint8_t* bytes) {
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
  return value;
}

// --- documents, common -------------------------------------------------------------

occ::handle<TDocStd_Document> new_document() {
  // No application: creating or using one would bring in format plugins and resources.
  occ::handle<TDocStd_Document> doc = new TDocStd_Document("MDTV-XCAF");
  XCAFDoc_DocumentTool::Set(doc->Main(), false);
  // Always millimetres, whatever xstep.cascade.unit may have been set to.
  XCAFDoc_DocumentTool::SetLengthUnit(doc, 1.0, UnitsMethods_LengthUnit_Millimeter);
  return doc;
}

void set_name(const TDF_Label& label, const std::string& name) {
  if (!name.empty()) {
    TDataStd_Name::Set(label, TCollection_ExtendedString(name.c_str(), /*isMultiByte=*/true));
  }
}

std::string get_name(const TDF_Label& label) {
  occ::handle<TDataStd_Name> attribute;
  if (!label.FindAttribute(TDataStd_Name::GetID(), attribute)) return std::string();
  return TCollection_AsciiString(attribute->Get()).ToCString();
}

// --- documents, reading --------------------------------------------------------------

// Instance and product names that some program made up rather than a person.
bool is_generated_name(const std::string& name) {
  if (name.empty() || name.rfind("=>[", 0) == 0) return true;
  if (name.rfind("Open CASCADE STEP translator", 0) == 0) return true;
  if (name.rfind("NAUO", 0) == 0 && name.size() > 4) {
    return name.find_first_not_of("0123456789", 4) == std::string::npos;
  }
  return false;
}

bool label_color(const TDF_Label& label, Quantity_Color& color) {
  return XCAFDoc_ColorTool::GetColor(label, XCAFDoc_ColorSurf, color) ||
         XCAFDoc_ColorTool::GetColor(label, XCAFDoc_ColorGen, color) ||
         XCAFDoc_ColorTool::GetColor(label, XCAFDoc_ColorCurv, color);
}

// One colour covering the whole part, where the file styled its solid or all its faces
// rather than the part. Faces with colours of their own have nowhere to go in a node.
bool uniform_subshape_color(const TDF_Label& part, const TopoDS_Shape& shape,
                            Quantity_Color& color) {
  NCollection_Sequence<TDF_Label> subs;
  if (!XCAFDoc_ShapeTool::GetSubShapes(part, subs) || subs.IsEmpty()) return false;
  ShapeMap faces, covered;
  TopExp::MapShapes(shape, TopAbs_FACE, faces);
  if (faces.IsEmpty()) return false;
  bool found = false;
  for (NCollection_Sequence<TDF_Label>::Iterator it(subs); it.More(); it.Next()) {
    Quantity_Color here;
    if (!label_color(it.Value(), here)) continue;
    if (found && !(here == color)) return false;
    color = here;
    found = true;
    TopExp::MapShapes(XCAFDoc_ShapeTool::GetShape(it.Value()), TopAbs_FACE, covered);
  }
  return found && covered.Extent() >= faces.Extent();
}

void append_string(Bytes& out, const std::string& text) {
  append<uint32_t>(out, static_cast<uint32_t>(text.size()));
  out.insert(out.end(), text.begin(), text.end());
}

// One node, as `decode_node` in occt-kernel-capi reads it:
//   u32 name length, then UTF-8 (length 0 means no name)
//   u32 has colour, then 3 f64 (sRGB, as the file holds them) if so
//   12 f64 placement by rows
//   u32 0 and a shape handle for a part, or u32 1, u32 count and nodes for an assembly
void append_node(Bytes& out, IssuedShapes& issued, const TDF_Label& label, int depth) {
  if (depth > kMaxDepth) throw std::runtime_error("the product structure is too deep");

  TDF_Label product = label;
  TopLoc_Location placement;
  if (XCAFDoc_ShapeTool::IsReference(label)) {
    if (!XCAFDoc_ShapeTool::GetReferredShape(label, product)) {
      throw std::runtime_error("an instance refers to no product");
    }
    placement = XCAFDoc_ShapeTool::GetLocation(label);
  }

  std::string name = get_name(label);
  if (is_generated_name(name)) name = get_name(product);
  if (is_generated_name(name)) name.clear();
  append_string(out, name);

  const bool assembly = XCAFDoc_ShapeTool::IsAssembly(product);
  TopoDS_Shape shape;
  if (!assembly) {
    shape = XCAFDoc_ShapeTool::GetShape(product);
    if (shape.IsNull()) throw std::runtime_error("a part has no shape");
    // A node's part is shared and carries no placement of its own.
    placement = placement * shape.Location();
    shape.Location(TopLoc_Location());
  }

  Quantity_Color color;
  const bool has_color =
      label_color(label, color) || label_color(product, color) ||
      (!assembly && uniform_subshape_color(product, shape, color));
  append<uint32_t>(out, has_color ? 1 : 0);
  if (has_color) {
    double r, g, b;
    color.Values(r, g, b, Quantity_TOC_sRGB);
    append(out, r);
    append(out, g);
    append(out, b);
  }

  const gp_Trsf trsf = placement.Transformation();
  for (int row = 1; row <= 3; ++row) {
    for (int column = 1; column <= 4; ++column) append(out, trsf.Value(row, column));
  }

  if (assembly) {
    NCollection_Sequence<TDF_Label> children;
    XCAFDoc_ShapeTool::GetComponents(product, children, false);
    append<uint32_t>(out, 1);
    append<uint32_t>(out, static_cast<uint32_t>(children.Length()));
    for (NCollection_Sequence<TDF_Label>::Iterator it(children); it.More(); it.Next()) {
      append_node(out, issued, it.Value(), depth + 1);
    }
  } else {
    append<uint32_t>(out, 0);
    append<uint32_t>(out, issued.put(shape));
  }
}

// How many of the file's length units fit a metre, or NaN if it declares none. Each
// representation may have its own unit, in which case the first is reported; geometry
// converts correctly either way.
double file_units_per_metre(const occ::handle<Interface_InterfaceModel>& model) {
  StepData_Factors factors;
  factors.SetCascadeUnit(1.0);
  for (int i = 1; i <= model->NbEntities(); ++i) {
    const occ::handle<Standard_Transient> entity = model->Value(i);
    occ::handle<StepRepr_GlobalUnitAssignedContext> units;
    if (const auto a =
            occ::down_cast<StepGeom_GeomRepContextAndGlobUnitAssCtxAndGlobUncertaintyAssCtx>(
                entity)) {
      units = a->GlobalUnitAssignedContext();
    } else if (const auto b =
                   occ::down_cast<StepGeom_GeometricRepresentationContextAndGlobalUnitAssignedContext>(
                       entity)) {
      units = b->GlobalUnitAssignedContext();
    } else if (const auto c = occ::down_cast<StepRepr_GlobalUnitAssignedContext>(entity)) {
      units = c;
    }
    if (units.IsNull()) continue;
    STEPConstruct_UnitContext context;
    context.ComputeFactors(units, factors);
    // The factor is millimetres per file unit.
    if (!context.LengthDone()) continue;
    if (context.LengthFactor() > 0.0) return 1000.0 / context.LengthFactor();
  }
  return kNaN;
}

// --- documents, writing --------------------------------------------------------------

struct NodeColor {
  bool set = false;
  double r = 0.0, g = 0.0, b = 0.0;
  bool operator==(const NodeColor& o) const {
    return set == o.set && r == o.r && g == o.g && b == o.b;
  }
  Quantity_Color get() const { return Quantity_Color(r, g, b, Quantity_TOC_sRGB); }
};

struct InNode {
  std::string name;
  NodeColor color;
  double rows[12] = {};
  TopoDS_Shape shape;  // Null for an assembly.
  std::vector<InNode> children;
  int part = -1;
  bool identity = true;
};

// A part as the document holds it: one shape, however many nodes use it.
struct PartEntry {
  TopoDS_Shape shape;
  std::string name;
  NodeColor color;
  bool uniform = true;
  int uses = 0;
  TDF_Label label;
};

using PartIndex = NCollection_DataMap<TopoDS_Shape, int, TopTools_ShapeMapHasher>;

void check_rigid(const double* m) {
  const auto dot = [m](int a, int b) {
    return m[a * 4] * m[b * 4] + m[a * 4 + 1] * m[b * 4 + 1] + m[a * 4 + 2] * m[b * 4 + 2];
  };
  constexpr double tolerance = 1e-9;
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      if (std::abs(dot(a, b) - (a == b ? 1.0 : 0.0)) > tolerance) {
        throw std::invalid_argument("a placement must be a rigid transform");
      }
    }
  }
  const double det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
                     m[2] * (m[4] * m[9] - m[5] * m[8]);
  if (det < 0.0) throw std::invalid_argument("a placement must not mirror");
  for (int i = 0; i < 12; ++i) {
    if (!std::isfinite(m[i])) throw std::invalid_argument("a placement must be finite");
  }
}

bool rows_are_identity(const double* m) {
  constexpr double identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  for (int i = 0; i < 12; ++i) {
    if (std::abs(m[i] - identity[i]) > 1e-12) return false;
  }
  return true;
}

gp_Trsf trsf_from(const double* m) {
  gp_Trsf t;
  t.SetValues(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11]);
  return t;
}

TopLoc_Location location_of(const InNode& node) {
  return node.identity ? TopLoc_Location() : TopLoc_Location(trsf_from(node.rows));
}

// The layout of `encode_node` in occt-kernel-capi, the same as append_node's. Colours
// inherit downward here, because STEP can't colour an assembly.
InNode read_node(Reader& in, const NodeColor& inherited, int depth) {
  if (depth > kMaxDepth) throw std::invalid_argument("the product structure is too deep");
  InNode node;
  const uint32_t name_len = in.u32();
  if (name_len > in.remaining()) throw std::invalid_argument("truncated input");
  node.name.reserve(name_len);
  for (uint32_t i = 0; i < name_len; ++i) node.name.push_back(in.get<char>());
  node.color = inherited;
  if (in.u32() != 0) {
    node.color.set = true;
    node.color.r = in.f64();
    node.color.g = in.f64();
    node.color.b = in.f64();
  }
  for (double& v : node.rows) v = in.f64();
  check_rigid(node.rows);

  if (in.u32() == 0) {
    node.shape = in.shape();
    // Normalise: an instance is a placement of a part with no location of its own, so
    // that placed copies of one shape are one part in the file.
    if (!node.shape.Location().IsIdentity()) {
      const gp_Trsf own = node.shape.Location().Transformation();
      gp_Trsf all = trsf_from(node.rows);
      all.Multiply(own);
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 4; ++column) {
          node.rows[row * 4 + column] = all.Value(row + 1, column + 1);
        }
      }
      node.shape.Location(TopLoc_Location());
    }
  } else {
    const uint32_t count = in.u32();
    if (count == 0) throw std::invalid_argument("an assembly needs at least one child");
    if (count > in.remaining()) throw std::invalid_argument("truncated input");
    node.children.reserve(count);
    for (uint32_t i = 0; i < count; ++i) node.children.push_back(read_node(in, node.color, depth + 1));
  }
  node.identity = rows_are_identity(node.rows);
  return node;
}

void collect_parts(InNode& node, PartIndex& index, std::vector<PartEntry>& parts) {
  if (node.shape.IsNull()) {
    for (InNode& child : node.children) collect_parts(child, index, parts);
    return;
  }
  const int* found = index.Seek(node.shape);
  if (found == nullptr) {
    node.part = static_cast<int>(parts.size());
    index.Bind(node.shape, node.part);
    PartEntry part;
    part.shape = node.shape;
    part.name = node.name;
    part.color = node.color;
    part.uses = 1;
    parts.push_back(std::move(part));
    return;
  }
  node.part = *found;
  PartEntry& part = parts[static_cast<size_t>(node.part)];
  ++part.uses;
  if (part.name.empty()) part.name = node.name;
  if (!(part.color == node.color)) part.uniform = false;
}

// Builds the product of an assembly and everything under it. Identical assemblies below
// the top level become one product, as they were in a file that used one twice.
class AssemblyBuilder {
 public:
  AssemblyBuilder(const occ::handle<XCAFDoc_ShapeTool>& shapes,
                  const occ::handle<XCAFDoc_ColorTool>& colors, std::vector<PartEntry>& parts)
      : shapes_(shapes), colors_(colors), parts_(parts) {}

  TDF_Label product(const InNode& node, bool share) {
    std::string key;
    std::vector<TDF_Label> child_products;
    child_products.reserve(node.children.size());
    for (const InNode& child : node.children) {
      const TDF_Label product_label = child.shape.IsNull()
                                          ? product(child, true)
                                          : parts_[static_cast<size_t>(child.part)].label;
      child_products.push_back(product_label);
      if (share) append_key(key, child, product_label);
    }
    if (share) {
      key.append(node.name);
      const auto found = shared_.find(key);
      if (found != shared_.end()) return found->second;
    }
    const TDF_Label label = shapes_->NewShape();
    set_name(label, node.name);
    for (size_t i = 0; i < node.children.size(); ++i) add_component(label, node.children[i], child_products[i]);
    if (share) shared_.emplace(std::move(key), label);
    return label;
  }

 private:
  void add_component(const TDF_Label& assembly, const InNode& child, const TDF_Label& product) {
    const TDF_Label component = shapes_->AddComponent(assembly, product, location_of(child));
    if (component.IsNull()) throw std::runtime_error("an instance could not be added");
    set_name(component, child.name);
    const bool part_colored =
        !child.shape.IsNull() && parts_[static_cast<size_t>(child.part)].uniform;
    if (child.color.set && !part_colored && !child.shape.IsNull()) {
      colors_->SetColor(component, child.color.get(), XCAFDoc_ColorGen);
    }
  }

  static void append_key(std::string& key, const InNode& child, const TDF_Label& product) {
    TCollection_AsciiString entry;
    TDF_Tool::Entry(product, entry);
    key.append(entry.ToCString()).push_back('\0');
    key.append(child.name).push_back('\0');
    key.append(reinterpret_cast<const char*>(child.rows), sizeof(child.rows));
    key.append(reinterpret_cast<const char*>(&child.color.r), sizeof(double) * 3);
    key.push_back(child.color.set ? '\1' : '\0');
  }

  occ::handle<XCAFDoc_ShapeTool> shapes_;
  occ::handle<XCAFDoc_ColorTool> colors_;
  std::vector<PartEntry>& parts_;
  std::map<std::string, TDF_Label> shared_;
};

int schema_of(int32_t schema) {
  switch (schema) {
    case 0: return DESTEP_Parameters::WriteMode_StepSchema_AP203;
    case 1: return DESTEP_Parameters::WriteMode_StepSchema_AP214IS;
    case 2: return DESTEP_Parameters::WriteMode_StepSchema_AP242DIS;
    default: throw std::invalid_argument("unknown STEP schema");
  }
}

// The unit named by a units-per-metre value. NaN means millimetres.
UnitsMethods_LengthUnit unit_of(double per_metre) {
  if (std::isnan(per_metre)) return UnitsMethods_LengthUnit_Millimeter;
  const struct {
    double millimetres;
    UnitsMethods_LengthUnit unit;
  } known[] = {
      {1.0, UnitsMethods_LengthUnit_Millimeter}, {1000.0, UnitsMethods_LengthUnit_Meter},
      {10.0, UnitsMethods_LengthUnit_Centimeter}, {25.4, UnitsMethods_LengthUnit_Inch},
      {304.8, UnitsMethods_LengthUnit_Foot},      {1e6, UnitsMethods_LengthUnit_Kilometer},
      {1609344.0, UnitsMethods_LengthUnit_Mile},  {0.0254, UnitsMethods_LengthUnit_Mil},
      {0.001, UnitsMethods_LengthUnit_Micron},    {0.0000254, UnitsMethods_LengthUnit_Microinch},
  };
  for (const auto& k : known) {
    if (std::abs(1000.0 / per_metre / k.millimetres - 1.0) < 1e-6) return k.unit;
  }
  throw std::invalid_argument("STEP can only be written in a common length unit");
}

// --- IGES ---------------------------------------------------------------------------

// OCCT's IGES parser takes a file name, with no stream form (IGESFile_Read opens it itself,
// and DEIGES_Provider's stream overload fails in the absence of a work library that reads
// streams), so reading goes through a private temporary file. On wasm this is the module's
// in-memory file system. The name is random and the file is created exclusively, so nothing
// else can be reading or replacing it.
class TempFile {
 public:
  TempFile(const uint8_t* data, size_t len) {
    std::random_device random;
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    for (int attempt = 0; attempt < 16 && file_ == nullptr; ++attempt) {
      std::ostringstream name;
      name << "occt-" << std::hex << random() << random() << random() << ".igs";
      path_ = (dir / name.str()).u8string();
      file_ = OSD_OpenFile(path_.c_str(), "wbx");
    }
    if (file_ == nullptr) throw std::runtime_error("no temporary file could be created");
    const size_t written = len == 0 ? 0 : std::fwrite(data, 1, len, file_);
    const bool closed = std::fclose(file_) == 0;
    file_ = nullptr;
    if (written != len || !closed) {
      remove();
      throw std::runtime_error("the temporary file could not be written");
    }
  }
  ~TempFile() { remove(); }
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  const char* path() const { return path_.c_str(); }

 private:
  void remove() {
    std::error_code ignored;
    std::filesystem::remove(std::filesystem::u8path(path_), ignored);
  }
  std::string path_;
  FILE* file_ = nullptr;
};

// A closed shell as a solid with its material inside. The shell itself otherwise.
TopoDS_Shape solid_from(const TopoDS_Shell& shell) {
  BRepCheck_Shell check(shell);
  if (check.Closed() != BRepCheck_NoError) return shell;
  BRepBuilderAPI_MakeSolid maker(shell);
  if (!maker.IsDone()) return shell;
  TopoDS_Solid solid = maker.Solid();
  if (!BRepLib::OrientClosedSolid(solid)) return shell;
  return solid;
}

// IGES usually stores a solid as loose faces. Sews whatever isn't already a solid, and
// turns closed results into solids. Curves and points are kept as they are.
TopoDS_Shape assemble_iges(const TopoDS_Shape& raw) {
  // Coincidence for imported faces: wider than the 1e-7 of a healthy model, as files keep
  // about six significant digits, and far below any feature a part would have.
  constexpr double kSewingTolerance = 1e-3;

  std::vector<TopoDS_Shape> kept;
  BRepBuilderAPI_Sewing sewing(kSewingTolerance);
  int loose = 0;
  for (TopExp_Explorer it(raw, TopAbs_SOLID); it.More(); it.Next()) kept.push_back(it.Current());
  for (TopExp_Explorer it(raw, TopAbs_SHELL, TopAbs_SOLID); it.More(); it.Next()) {
    sewing.Add(it.Current());
    ++loose;
  }
  for (TopExp_Explorer it(raw, TopAbs_FACE, TopAbs_SHELL); it.More(); it.Next()) {
    sewing.Add(it.Current());
    ++loose;
  }
  for (TopExp_Explorer it(raw, TopAbs_WIRE, TopAbs_FACE); it.More(); it.Next()) kept.push_back(it.Current());
  for (TopExp_Explorer it(raw, TopAbs_EDGE, TopAbs_WIRE); it.More(); it.Next()) kept.push_back(it.Current());
  for (TopExp_Explorer it(raw, TopAbs_VERTEX, TopAbs_EDGE); it.More(); it.Next()) kept.push_back(it.Current());

  if (loose > 0) {
    sewing.Perform();
    const TopoDS_Shape sewn = sewing.SewedShape();
    if (sewn.IsNull()) throw std::runtime_error("the IGES faces could not be sewn");
    for (TopExp_Explorer it(sewn, TopAbs_SHELL); it.More(); it.Next()) {
      kept.push_back(solid_from(TopoDS::Shell(it.Current())));
    }
    // Sewing leaves a lone face bare, even one that closes on itself like a sphere.
    for (TopExp_Explorer it(sewn, TopAbs_FACE, TopAbs_SHELL); it.More(); it.Next()) {
      TopoDS_Shell shell;
      BRep_Builder builder;
      builder.MakeShell(shell);
      builder.Add(shell, it.Current());
      const TopoDS_Shape solid = solid_from(shell);
      kept.push_back(solid.ShapeType() == TopAbs_SOLID ? solid : it.Current());
    }
  }

  if (kept.empty()) throw std::runtime_error("the IGES data has no shapes");
  if (kept.size() == 1) return kept.front();
  TopoDS_Compound all;
  BRep_Builder builder;
  builder.MakeCompound(all);
  for (const TopoDS_Shape& shape : kept) builder.Add(all, shape);
  return all;
}

}  // namespace

// --- native -------------------------------------------------------------------------

// Saves are OCCT's binary format, which stores each double as it is; its text format rounds
// vertices and parameters to 15 digits. A short frame goes first, because that format's
// reader trusts its input and can crash on damaged data: 8 bytes of magic, a u64 payload
// length and a u64 FNV-1a hash of the payload, little-endian. So truncation or damage is
// an ordinary error, and only a save forged on purpose can reach the parser.
API uint32_t occt_native_write(uint32_t h, int32_t with_triangulation) {
  return guard_handle([&]() -> uint32_t {
    SilentCout silent;
    const TopoDS_Shape shape = get_shape(h);
    std::ostringstream stream(std::ios::binary);
    const bool triangles = with_triangulation != 0;
    BinTools::Write(shape, stream, triangles, triangles, BinTools_FormatVersion_CURRENT);
    if (!stream.good()) throw std::runtime_error("the shape could not be written");
    const std::string payload = stream.str();

    Bytes out;
    out.reserve(kNativeHeader + payload.size());
    out.insert(out.end(), kNativeMagic, kNativeMagic + sizeof(kNativeMagic));
    put_le64(out, payload.size());
    put_le64(out, fnv1a(payload.data(), payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    return put_buffer(std::move(out));
  });
}

API uint32_t occt_native_read(const uint8_t* data, size_t len) {
  return guard_handle([&]() -> uint32_t {
    SilentCout silent;
    if (len < kNativeHeader || std::memcmp(data, kNativeMagic, sizeof(kNativeMagic)) != 0) {
      throw std::invalid_argument("the data is not a saved shape");
    }
    const uint8_t* payload = data + kNativeHeader;
    const size_t size = len - kNativeHeader;
    if (get_le64(data + sizeof(kNativeMagic)) != size ||
        get_le64(data + sizeof(kNativeMagic) + 8) != fnv1a(payload, size)) {
      throw std::invalid_argument("the saved shape is truncated or damaged");
    }
    std::istringstream stream(std::string(reinterpret_cast<const char*>(payload), size),
                              std::ios::binary);
    TopoDS_Shape shape;
    BinTools::Read(shape, stream);
    if (stream.fail()) throw std::runtime_error("the saved shape could not be read");
    if (shape.IsNull()) throw std::runtime_error("the data holds no shape");
    return put_shape(shape);
  });
}

// --- STEP documents ---------------------------------------------------------------------

// Buffer: f64 file units per metre (NaN if not declared), u32 root count, roots as written
// by append_node.
API uint32_t occt_step_document_read(const uint8_t* data, size_t len, size_t cancel) {
  return guard_handle([&]() -> uint32_t {
    silence_messages();
    SilentCout silent;
    FlagProgress progress(cancel);
    if (progress.UserBreak()) throw std::runtime_error("cancelled");

    std::istringstream stream(to_string_input(data, len));
    STEPCAFControl_Reader reader;
    reader.SetColorMode(true);
    reader.SetNameMode(true);
    reader.SetLayerMode(false);
    reader.SetPropsMode(false);
    reader.SetSHUOMode(false);
    reader.SetGDTMode(false);
    reader.SetMatMode(false);
    reader.SetViewMode(false);
    reader.SetMetaMode(false);
    reader.SetProductMetaMode(false);
    if (reader.ReadStream("input.step", stream) != IFSelect_RetDone) {
      throw std::runtime_error("the data is not readable as STEP");
    }
    if (progress.UserBreak()) throw std::runtime_error("cancelled");

    occ::handle<TDocStd_Document> doc = new_document();
    const bool transferred = reader.Transfer(doc, progress.Start());
    if (progress.UserBreak()) throw std::runtime_error("cancelled");
    if (!transferred) throw std::runtime_error("the STEP data could not be translated");

    occ::handle<XCAFDoc_ShapeTool> shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
    NCollection_Sequence<TDF_Label> roots;
    shapes->GetFreeShapes(roots);
    if (roots.IsEmpty()) throw std::runtime_error("the STEP data has no shapes");

    // Without the extra assembly a document may have needed.
    if (roots.Length() == 1 && XCAFDoc_ShapeTool::IsAssembly(roots.First()) &&
        get_name(roots.First()) == kDocumentRoot) {
      NCollection_Sequence<TDF_Label> children;
      XCAFDoc_ShapeTool::GetComponents(roots.First(), children, false);
      roots = children;
    }

    IssuedShapes issued;
    Bytes out;
    append(out, file_units_per_metre(reader.Reader().WS()->Model()));
    append<uint32_t>(out, static_cast<uint32_t>(roots.Length()));
    for (NCollection_Sequence<TDF_Label>::Iterator it(roots); it.More(); it.Next()) {
      append_node(out, issued, it.Value(), 0);
    }
    const uint32_t buffer = put_buffer(std::move(out));
    issued.commit();
    return buffer;
  });
}

// `nodes` holds a u32 root count and roots as read_node reads them. `schema` is 0 for
// AP203, 1 for AP214 and 2 for AP242; `units` is units per metre, or NaN for millimetres.
// Schema and unit go in through the writer's parameters, not OCCT's global statics, so
// nothing needs restoring afterward.
API uint32_t occt_step_document_write(const uint8_t* nodes, size_t len, int32_t schema, double units, size_t cancel) {
  return guard_handle([&]() -> uint32_t {
    silence_messages();
    SilentCout silent;
    FlagProgress progress(cancel);
    if (progress.UserBreak()) throw std::runtime_error("cancelled");

    Reader in(nodes, len);
    const uint32_t count = in.u32();
    if (count == 0) throw std::invalid_argument("a document needs at least one node");
    if (count > in.remaining()) throw std::invalid_argument("truncated input");
    std::vector<InNode> roots;
    roots.reserve(count);
    for (uint32_t i = 0; i < count; ++i) roots.push_back(read_node(in, NodeColor{}, 1));

    PartIndex index;
    std::vector<PartEntry> parts;
    for (InNode& root : roots) collect_parts(root, index, parts);

    occ::handle<TDocStd_Document> doc = new_document();
    occ::handle<XCAFDoc_ShapeTool> shapes = XCAFDoc_DocumentTool::ShapeTool(doc->Main());
    occ::handle<XCAFDoc_ColorTool> colors = XCAFDoc_DocumentTool::ColorTool(doc->Main());

    for (PartEntry& part : parts) {
      part.label = shapes->NewShape();
      shapes->SetShape(part.label, part.shape);
      set_name(part.label, part.name);
      if (part.uniform && part.color.set) {
        colors->SetColor(part.label, part.color.get(), XCAFDoc_ColorGen);
      }
    }

    // Plain roots are what a file holds natively. Anything else takes the extra assembly.
    bool wrap = false;
    for (const InNode& root : roots) {
      if (!root.identity) wrap = true;
      if (!root.shape.IsNull() && parts[static_cast<size_t>(root.part)].uses > 1) wrap = true;
    }
    AssemblyBuilder builder(shapes, colors, parts);
    if (wrap) {
      InNode outer;
      outer.name = kDocumentRoot;
      outer.children = std::move(roots);
      builder.product(outer, false);
    } else {
      for (const InNode& root : roots) {
        if (root.shape.IsNull()) builder.product(root, false);
      }
    }
    shapes->UpdateAssemblies();

    STEPCAFControl_Writer writer;
    writer.SetColorMode(true);
    writer.SetNameMode(true);
    writer.SetLayerMode(false);
    writer.SetPropsMode(false);
    writer.SetSHUOMode(false);
    writer.SetDimTolMode(false);
    writer.SetMaterialMode(false);

    DESTEP_Parameters parameters;
    parameters.InitFromStatic();
    parameters.WriteSchema = static_cast<DESTEP_Parameters::WriteMode_StepSchema>(schema_of(schema));
    parameters.WriteUnit = unit_of(units);
    const bool transferred =
        writer.Transfer(doc, parameters, STEPControl_AsIs, nullptr, progress.Start());
    if (progress.UserBreak()) throw std::runtime_error("cancelled");
    if (!transferred) throw std::runtime_error("the document could not be translated to STEP");

    std::ostringstream stream;
    if (writer.WriteStream(stream) != IFSelect_RetDone) {
      throw std::runtime_error("STEP write failed");
    }
    const std::string text = stream.str();
    return put_buffer(Bytes(text.begin(), text.end()));
  });
}

// --- IGES ---------------------------------------------------------------------------

API uint32_t occt_iges_read(const uint8_t* data, size_t len) {
  return guard_handle([&]() -> uint32_t {
    silence_messages();
    SilentCout silent;
    if (len == 0) throw std::invalid_argument("the data is empty");
    TempFile file(data, len);
    IGESControl_Reader reader;
    reader.SetReadVisible(false);
    if (reader.ReadFile(file.path()) != IFSelect_RetDone) {
      throw std::runtime_error("the data is not readable as IGES");
    }
    if (reader.TransferRoots() == 0) throw std::runtime_error("the IGES data has no shapes");
    return put_shape(assemble_iges(reader.OneShape()));
  });
}

API uint32_t occt_iges_write(const void* handles, size_t count) {
  return guard_handle([&]() -> uint32_t {
    silence_messages();
    SilentCout silent;
    if (count == 0) throw std::invalid_argument("no shapes given");
    const std::vector<TopoDS_Shape> shapes = read_shapes(handles, count);
    IGESControl_Writer writer("MM", 0);
    for (const TopoDS_Shape& shape : shapes) {
      if (!writer.AddShape(shape)) throw std::runtime_error("a shape could not be translated to IGES");
    }
    // OCCT fills this in with the login name of whoever runs the program.
    const occ::handle<IGESData_IGESModel> model = writer.Model();
    IGESData_GlobalSection section = model->GlobalSection();
    section.SetAuthorName(new TCollection_HAsciiString(""));
    model->SetGlobalSection(section);

    std::ostringstream stream;
    if (!writer.Write(stream)) throw std::runtime_error("IGES write failed");
    if (writer.Model()->NbEntities() == 0) throw std::runtime_error("the shapes hold nothing IGES can store");
    const std::string text = stream.str();
    return put_buffer(Bytes(text.begin(), text.end()));
  });
}
