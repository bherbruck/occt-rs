// Entry points of the features domain. SPDX-License-Identifier: MIT OR Apache-2.0
//
// Sweeps with history, lofts and local edits of solids. OCCT often reports success for a
// result that makes no sense, so each operation here checks what it can of the outcome.
#include "internal.hpp"

#include <BRepAlgoAPI_Defeaturing.hxx>
#include <BRepBuilderAPI_GTransform.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_TransitionMode.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepOffsetAPI_DraftAngle.hxx>
#include <BRepOffsetAPI_MakeOffsetShape.hxx>
#include <BRepOffsetAPI_MakePipeShell.hxx>
#include <BRepOffsetAPI_MakeThickSolid.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRepOffset_MakeOffset.hxx>
#include <BRepTools.hxx>
#include <BRepTools_History.hxx>
#include <BRep_Builder.hxx>
#include <GeomAbs_JoinType.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <TopoDS_Compound.hxx>
#include <gp_GTrsf.hxx>
#include <gp_Pln.hxx>

#include <algorithm>
#include <memory>

using namespace occt_capi;

namespace {

// The join tolerance of offsets: tighter than what callers measure, loose enough for
// surfaces meeting at the kernel's usual 1e-7 coincidence distance.
constexpr double kOffsetTolerance = 1e-6;
// How far a matrix may be from orthogonal and still count as rigid or similar. Products of
// a few rotations stay well inside this, and a shape at 1000 units moves by under 1e-6.
constexpr double kRigidTolerance = 1e-9;
// How far off a measured offset distance may be, relative to it. Offsets of curved faces can
// be fitted, conventionally to about 1e-4 of model size.
constexpr double kDistanceSlack = 1e-3;
// Results below this fraction of the input's volume have collapsed.
constexpr double kVolumeFloor = 1e-12;

double volume_of(const TopoDS_Shape& shape) {
  GProp_GProps props;
  BRepGProp::VolumeProperties(shape, props, /*OnlyClosed=*/true);
  return props.Mass();
}

void require_present(const TopoDS_Shape& shape, const char* what) {
  if (shape.IsNull()) throw std::runtime_error(std::string(what) + " produced nothing");
}

void require_valid(const TopoDS_Shape& shape, const char* what) {
  require_present(shape, what);
  if (!BRepCheck_Analyzer(shape).IsValid()) {
    throw std::runtime_error(std::string(what) + " produced an invalid shape");
  }
}

// A closed shell with its material outside reads as a negative volume.
TopoDS_Shape oriented_outward(const TopoDS_Shape& solid) {
  return volume_of(solid) < 0.0 ? solid.Reversed() : solid;
}

TopoDS_Shape only_solid(const TopoDS_Shape& raw, const char* what) {
  require_present(raw, what);
  const TopoDS_Shape shape = unwrap_single(raw);
  if (shape.ShapeType() != TopAbs_SOLID) {
    throw std::runtime_error(std::string(what) + " did not produce one solid");
  }
  return shape;
}

// Past a local radius of curvature an offset surface turns inside out, and the algorithm
// returns a valid shape that isn't the offset. A true offset is exactly the distance away.
void require_distance(const TopoDS_Shape& a, const TopoDS_Shape& b, double distance,
                      const char* message) {
  BRepExtrema_DistShapeShape measure(a, b);
  if (!measure.IsDone()) throw std::runtime_error(message);
  const double slack = distance * kDistanceSlack + 1e-6;
  if (std::abs(measure.Value() - distance) > slack) throw std::runtime_error(message);
}

// The faces of a shape as a compound, which distance measures treat as surfaces: a solid
// would count as filled, putting anything inside it at distance zero.
TopoDS_Compound faces_of(const TopoDS_Shape& shape) {
  ShapeMap faces;
  TopExp::MapShapes(shape, TopAbs_FACE, faces);
  BRep_Builder builder;
  TopoDS_Compound out;
  builder.MakeCompound(out);
  for (int i = 1; i <= faces.Extent(); ++i) builder.Add(out, faces(i));
  return out;
}

// A shape with one child is that child.
TopoDS_Shape unwrap_single_child(const TopoDS_Shape& shape) {
  TopoDS_Iterator it(shape);
  if (!it.More()) return shape;
  const TopoDS_Shape only = it.Value();
  it.Next();
  return it.More() ? shape : only;
}

const char* kind_name(const TopoDS_Shape& shape) {
  switch (shape.ShapeType()) {
    case TopAbs_FACE: return "face";
    case TopAbs_WIRE: return "wire";
    case TopAbs_EDGE: return "edge";
    case TopAbs_VERTEX: return "vertex";
    case TopAbs_SOLID: return "solid";
    case TopAbs_SHELL: return "shell";
    default: return "group of shapes";
  }
}

void require_solid(const TopoDS_Shape& shape, const char* what) {
  if (shape.ShapeType() != TopAbs_SOLID) {
    throw std::invalid_argument(std::string(what) + " needs a solid, not a " + kind_name(shape));
  }
}

// The distinct faces named by `handles`, all of them checked to belong to `solid`: the
// algorithms below ignore or crash on a foreign face.
std::vector<TopoDS_Face> own_faces(const TopoDS_Shape& solid, const void* handles, size_t count) {
  ShapeMap own;
  TopExp::MapShapes(solid, TopAbs_FACE, own);
  ShapeMap seen;
  std::vector<TopoDS_Face> faces;
  for (const TopoDS_Shape& shape : read_shapes(handles, count)) {
    if (shape.ShapeType() != TopAbs_FACE) {
      throw std::invalid_argument(std::string("expected a face, got a ") + kind_name(shape));
    }
    const int index = own.FindIndex(shape);
    if (index == 0) throw std::invalid_argument("the face is not part of the solid");
    const int before = seen.Extent();
    seen.Add(shape);
    if (seen.Extent() != before) faces.push_back(TopoDS::Face(own(index)));
  }
  return faces;
}

// A wire to sweep or loft: a wire itself, an edge as a one-edge wire, or a face's outline.
TopoDS_Wire as_wire(const TopoDS_Shape& shape, const char* what) {
  switch (shape.ShapeType()) {
    case TopAbs_WIRE:
      return TopoDS::Wire(shape);
    case TopAbs_EDGE: {
      BRepBuilderAPI_MakeWire wire(TopoDS::Edge(shape));
      if (!wire.IsDone()) throw std::runtime_error(std::string(what) + ": bad edge");
      return wire.Wire();
    }
    case TopAbs_FACE: {
      const TopoDS_Wire outer = BRepTools::OuterWire(TopoDS::Face(shape));
      if (outer.IsNull()) throw std::invalid_argument(std::string(what) + ": face has no outline");
      return outer;
    }
    default:
      throw std::invalid_argument(std::string(what) + " needs a wire, edge or face, not a " +
                                  kind_name(shape));
  }
}

// BRepTools_History has the three calls modeled_buffer wants, under other names.
class RecordedHistory {
 public:
  explicit RecordedHistory(const occ::handle<BRepTools_History>& history) : history_(history) {}
  const TopTools_ListOfShape& Modified(const TopoDS_Shape& s) { return history_->Modified(s); }
  const TopTools_ListOfShape& Generated(const TopoDS_Shape& s) { return history_->Generated(s); }
  bool IsDeleted(const TopoDS_Shape& s) { return history_->IsRemoved(s); }

 private:
  occ::handle<BRepTools_History> history_;
};

// Algorithms also list shapes they made along the way that didn't reach the result, or the
// result as a whole. History only makes sense for faces, edges and vertices that are in it.
template <typename Maker>
class ResultHistory {
 public:
  ResultHistory(Maker& maker, const TopoDS_Shape& result) : maker_(maker) {
    for (const TopAbs_ShapeEnum kind : {TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX}) {
      TopExp::MapShapes(result, kind, result_);
    }
  }
  const TopTools_ListOfShape& Modified(const TopoDS_Shape& s) { return kept(maker_.Modified(s)); }
  const TopTools_ListOfShape& Generated(const TopoDS_Shape& s) { return kept(maker_.Generated(s)); }
  bool IsDeleted(const TopoDS_Shape& s) { return maker_.IsDeleted(s) && !result_.Contains(s); }

 private:
  const TopTools_ListOfShape& kept(const TopTools_ListOfShape& shapes) {
    out_.Clear();
    for (TopTools_ListOfShape::Iterator it(shapes); it.More(); it.Next()) {
      if (result_.Contains(it.Value())) out_.Append(it.Value());
    }
    return out_;
  }

  Maker& maker_;
  ShapeMap result_;
  TopTools_ListOfShape out_;
};

template <typename Maker>
uint32_t modeled_handle(Maker& maker, const TopoDS_Shape& result,
                        const std::vector<TopoDS_Shape>& inputs) {
  ResultHistory<Maker> history(maker, result);
  return put_buffer(modeled_buffer(history, result, inputs));
}

// A full turn marks its input edges as unused, though each made a face, so the algorithm
// reports them as deleted and generating nothing. This asks the sweep itself as well.
class RevolveHistory {
 public:
  RevolveHistory(BRepPrimAPI_MakeRevol& maker, const TopoDS_Shape& result) : maker_(maker) {
    for (const TopAbs_ShapeEnum kind : {TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX}) {
      TopExp::MapShapes(result, kind, result_);
    }
  }
  const TopTools_ListOfShape& Modified(const TopoDS_Shape&) { return none_; }
  const TopTools_ListOfShape& Generated(const TopoDS_Shape& s) {
    out_ = maker_.Generated(s);
    if (out_.IsEmpty()) {
      // Shape() only fills the sweep's cache, but it isn't const and Revol() is.
      const TopoDS_Shape swept = const_cast<BRepSweep_Revol&>(maker_.Revol()).Shape(s);
      if (!swept.IsNull() && result_.Contains(swept)) out_.Append(swept);
    }
    return out_;
  }
  bool IsDeleted(const TopoDS_Shape& s) { return !result_.Contains(s) && Generated(s).IsEmpty(); }

 private:
  BRepPrimAPI_MakeRevol& maker_;
  ShapeMap result_;
  TopTools_ListOfShape none_;
  TopTools_ListOfShape out_;
};

// The draft algorithm files tilted faces under Generated and leaves edges out. A tilted face
// is the same face changed, so everything is reported as modified, keeping only what the
// result really holds: the algorithm can substitute shapes after the fact.
class DraftHistory {
 public:
  DraftHistory(BRepOffsetAPI_DraftAngle& maker, const TopoDS_Shape& result) : maker_(maker) {
    for (const TopAbs_ShapeEnum kind : {TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX}) {
      TopExp::MapShapes(result, kind, result_);
    }
  }
  const TopTools_ListOfShape& Modified(const TopoDS_Shape& s) {
    out_.Clear();
    if (s.ShapeType() == TopAbs_FACE) {
      keep(maker_.Generated(s), s);
      keep(maker_.Modified(s), s);
    } else {
      TopTools_ListOfShape only;
      only.Append(maker_.ModifiedShape(s));
      keep(only, s);
    }
    return out_;
  }
  const TopTools_ListOfShape& Generated(const TopoDS_Shape&) { return none_; }
  bool IsDeleted(const TopoDS_Shape& s) { return Modified(s).IsEmpty() && !result_.Contains(s); }

 private:
  void keep(const TopTools_ListOfShape& shapes, const TopoDS_Shape& input) {
    for (TopTools_ListOfShape::Iterator it(shapes); it.More(); it.Next()) {
      if (!it.Value().IsSame(input) && result_.Contains(it.Value())) out_.Append(it.Value());
    }
  }

  BRepOffsetAPI_DraftAngle& maker_;
  ShapeMap result_;
  TopTools_ListOfShape none_;
  TopTools_ListOfShape out_;
};

// A solid with a closed cavity: the outer solid's shells, then the inner solid's turned
// inside out. OCCT's thick solid has nowhere to put walls when no face is open, and returns
// just the offset.
TopoDS_Shape with_cavity(const TopoDS_Shape& outer, const TopoDS_Shape& inner) {
  BRep_Builder builder;
  TopoDS_Solid solid;
  builder.MakeSolid(solid);
  for (TopoDS_Iterator it(outer); it.More(); it.Next()) builder.Add(solid, it.Value());
  for (TopoDS_Iterator it(inner); it.More(); it.Next()) builder.Add(solid, it.Value().Reversed());
  return solid;
}

using Pipe = BRepOffsetAPI_MakePipeShell;

std::unique_ptr<Pipe> build_pipe(const TopoDS_Wire& section, const TopoDS_Wire& path,
                                 int32_t orientation, bool has_binormal, const gp_Dir& binormal,
                                 bool solid) {
  auto pipe = std::make_unique<Pipe>(path);
  if (has_binormal) {
    pipe->SetMode(binormal);
  } else if (orientation == 0) {
    pipe->SetMode(/*IsFrenet=*/false);
  } else if (orientation == 1) {
    pipe->SetMode(/*IsFrenet=*/true);
  } else if (orientation == 2) {
    pipe->SetMode(gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0), gp_Dir(1.0, 0.0, 0.0)));
  } else {
    throw std::invalid_argument("invalid sweep orientation");
  }
  // Mitres a path's sharp corners, where the default stretches the profile; a smooth path
  // has no corners for this to change.
  pipe->SetTransitionMode(BRepBuilderAPI_RightCorner);
  pipe->Add(section);
  if (!pipe->IsReady()) throw std::runtime_error("the sweep's profile could not be used");
  pipe->Build();
  if (!pipe->IsDone()) throw std::runtime_error("the profile could not follow the path");
  if (solid && !pipe->MakeSolid()) throw std::runtime_error("the sweep's ends could not be closed");
  return pipe;
}

// Where a sweep's input subshapes went: through each wire's sweep, then the cut of the
// holes if there was one. A face stands for the end caps.
class SweepHistory {
 public:
  SweepHistory(const std::vector<std::unique_ptr<Pipe>>& pipes, BRepAlgoAPI_Cut* cut,
               const TopoDS_Shape& face)
      : pipes_(pipes), cut_(cut), face_(face) {}
  const TopTools_ListOfShape& Modified(const TopoDS_Shape&) { return none_; }
  const TopTools_ListOfShape& Generated(const TopoDS_Shape& s) {
    out_.Clear();
    for (const auto& pipe : pipes_) {
      for (TopTools_ListOfShape::Iterator it(pipe->Generated(s)); it.More(); it.Next()) {
        push(it.Value());
      }
    }
    if (!face_.IsNull() && s.IsSame(face_)) {
      push(pipes_.front()->FirstShape());
      push(pipes_.front()->LastShape());
    }
    return out_;
  }
  bool IsDeleted(const TopoDS_Shape&) { return false; }

 private:
  void push(const TopoDS_Shape& s) {
    if (s.IsNull()) return;
    if (cut_ == nullptr) {
      out_.Append(s);
    } else if (!cut_->IsDeleted(s)) {
      const TopTools_ListOfShape& modified = cut_->Modified(s);
      if (modified.IsEmpty()) {
        out_.Append(s);
      } else {
        for (TopTools_ListOfShape::Iterator it(modified); it.More(); it.Next()) out_.Append(it.Value());
      }
    }
  }

  const std::vector<std::unique_ptr<Pipe>>& pipes_;
  BRepAlgoAPI_Cut* cut_;
  TopoDS_Shape face_;
  TopTools_ListOfShape none_;
  TopTools_ListOfShape out_;
};

// --- transforms ------------------------------------------------------------------------

struct Matrix {
  double m[3][4];
  double det() const {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  }
  // Whether the linear part is a rotation or reflection times `*scale`.
  bool is_similarity(double* scale) const {
    const double d = det();
    const double s = std::cbrt(std::abs(d));
    if (!(s > 0.0)) return false;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        double dot = 0.0;
        for (int k = 0; k < 3; ++k) dot += m[k][i] * m[k][j];
        if (std::abs(dot - (i == j ? s * s : 0.0)) > kRigidTolerance * s * s) return false;
      }
    }
    *scale = s;
    return true;
  }
  gp_Trsf trsf() const {
    gp_Trsf t;
    t.SetValues(m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0],
                m[2][1], m[2][2], m[2][3]);
    return t;
  }
  gp_GTrsf gtrsf() const {
    gp_GTrsf t;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) t.SetValue(r + 1, c + 1, m[r][c]);
    }
    return t;
  }
};

Matrix read_matrix(const void* data, size_t len) {
  Reader reader(data, len);
  Matrix out;
  for (auto& row : out.m) {
    for (double& cell : row) {
      cell = reader.f64();
      if (!std::isfinite(cell)) throw std::invalid_argument("the transform is not finite");
    }
  }
  return out;
}

}  // namespace

// --- sweeps ----------------------------------------------------------------------------

// A modeled_buffer.
API uint32_t occt_extrude_modeled(uint32_t h, double dx, double dy, double dz) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    BRepPrimAPI_MakePrism prism(shape, gp_Vec(dx, dy, dz));
    if (!prism.IsDone()) throw std::runtime_error("extrude failed");
    return modeled_handle(prism, prism.Shape(), {shape});
  });
}

// A modeled_buffer.
API uint32_t occt_revolve_modeled(uint32_t h, double px, double py, double pz, double dx, double dy,
                                  double dz, double angle) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    const double full = 6.283185307179586476925286766559;
    const gp_Ax1 axis(gp_Pnt(px, py, pz), gp_Dir(dx, dy, dz));
    BRepPrimAPI_MakeRevol revol(shape, axis, std::abs(angle - full) < 1e-9 ? full : angle);
    if (!revol.IsDone()) throw std::runtime_error("revolve failed");
    RevolveHistory history(revol, revol.Shape());
    return modeled_handle(history, revol.Shape(), {shape});
  });
}

// A prism with its sides tilted: positive `taper` removes material toward the far end.
API uint32_t occt_extrude_tapered(uint32_t h, double dx, double dy, double dz, double taper) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    if (shape.ShapeType() != TopAbs_FACE) throw std::invalid_argument("expected a face");
    const TopoDS_Face face = TopoDS::Face(shape);
    BRepAdaptor_Surface surface(face);
    if (surface.GetType() != GeomAbs_Plane) throw std::invalid_argument("the face is not planar");
    const gp_Pln plane = surface.Plane();
    const gp_Vec by(dx, dy, dz);
    const double height = std::abs(by.Dot(gp_Vec(plane.Axis().Direction())));
    if (height <= by.Magnitude() * 1e-6) {
      throw std::invalid_argument("the extrusion lies in the face's plane");
    }

    GProp_GProps area;
    BRepGProp::SurfaceProperties(face, area);
    const double straight = area.Mass() * height;

    BRepPrimAPI_MakePrism prism(face, by);
    if (!prism.IsDone()) throw std::runtime_error("extrude failed");
    if (taper == 0.0) return put_shape(prism.Shape());

    BRepOffsetAPI_DraftAngle draft(prism.Shape());
    ShapeMap edges;
    TopExp::MapShapes(face, TopAbs_EDGE, edges);
    const gp_Dir pull(by);
    for (int i = 1; i <= edges.Extent(); ++i) {
      for (TopTools_ListOfShape::Iterator it(prism.Generated(edges(i))); it.More(); it.Next()) {
        if (it.Value().ShapeType() != TopAbs_FACE) continue;
        draft.Add(TopoDS::Face(it.Value()), pull, taper, plane);
        if (!draft.AddDone()) {
          throw std::runtime_error("this profile's sides cannot be tapered: they must be planes, "
                                   "cylinders or cones");
        }
      }
    }
    draft.Build();
    if (!draft.IsDone()) throw std::runtime_error("the taper could not be built");
    const TopoDS_Shape result = only_solid(draft.Shape(), "taper");
    require_valid(result, "taper");
    const double volume = volume_of(result);
    if (!(volume > straight * kVolumeFloor) || (taper > 0.0) != (volume < straight)) {
      throw std::runtime_error("the taper is too steep for this height");
    }
    return put_shape(result);
  });
}

// orientation: 0 corrected, 1 Frenet, 2 fixed. A modeled_buffer.
API uint32_t occt_sweep(uint32_t profile, uint32_t path, int32_t orientation, int32_t has_binormal,
                        double bx, double by, double bz, int32_t solid) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape input = get_shape(profile);
    const TopoDS_Shape spine_input = get_shape(path);
    if (spine_input.ShapeType() != TopAbs_WIRE && spine_input.ShapeType() != TopAbs_EDGE) {
      throw std::invalid_argument(std::string("a sweep's path must be a wire or edge, not a ") +
                                  kind_name(spine_input));
    }
    const TopoDS_Wire spine = as_wire(spine_input, "sweep");
    const bool is_face = input.ShapeType() == TopAbs_FACE;
    const bool make_solid = is_face || solid != 0;
    const TopoDS_Wire outer = as_wire(input, "sweep");
    if (make_solid && !BRep_Tool::IsClosed(outer)) {
      throw std::invalid_argument("only a closed profile can sweep into a solid");
    }
    const gp_Dir binormal = has_binormal != 0 ? gp_Dir(bx, by, bz) : gp_Dir(0.0, 0.0, 1.0);

    std::vector<std::unique_ptr<Pipe>> pipes;
    pipes.push_back(
        build_pipe(outer, spine, orientation, has_binormal != 0, binormal, make_solid));

    // A face's holes run along the path too, as tunnels through the outline's solid.
    TopoDS_Compound holes;
    BRep_Builder builder;
    builder.MakeCompound(holes);
    bool has_holes = false;
    if (is_face) {
      for (TopoDS_Iterator it(input); it.More(); it.Next()) {
        if (it.Value().ShapeType() != TopAbs_WIRE || it.Value().IsSame(outer)) continue;
        pipes.push_back(build_pipe(TopoDS::Wire(it.Value()), spine, orientation,
                                   has_binormal != 0, binormal, true));
        builder.Add(holes, pipes.back()->Shape());
        has_holes = true;
      }
    }

    std::unique_ptr<BRepAlgoAPI_Cut> cut;
    TopoDS_Shape result = pipes.front()->Shape();
    if (has_holes) {
      cut = std::make_unique<BRepAlgoAPI_Cut>(result, holes);
      if (cut->HasErrors() || !cut->IsDone()) throw std::runtime_error("the holes could not be cut");
      require_present(cut->Shape(), "sweep");
      result = unwrap_single(cut->Shape());
    }
    require_valid(result, "sweep");
    if (make_solid && !(volume_of(result) > 0.0)) {
      throw std::runtime_error("the sweep has no volume");
    }
    SweepHistory history(pipes, cut.get(), is_face ? input : TopoDS_Shape());
    return modeled_handle(history, result, {input, spine_input});
  });
}

// `sections` is count uint32 handles of wires, edges or faces, with vertices allowed at the ends.
API uint32_t occt_loft(const void* sections, size_t count, int32_t solid, int32_t ruled,
                       double tolerance) {
  return guard_handle([&]() -> uint32_t {
    if (count < 2) throw std::invalid_argument("a loft needs at least two sections");
    const std::vector<TopoDS_Shape> shapes = read_shapes(sections, count);
    BRepOffsetAPI_ThruSections loft(solid != 0, ruled != 0, tolerance);
    size_t wires = 0;
    for (size_t i = 0; i < count; ++i) {
      if (shapes[i].ShapeType() == TopAbs_VERTEX) {
        if (i != 0 && i + 1 != count) {
          throw std::invalid_argument("a point can only be the first or last section of a loft");
        }
        loft.AddVertex(TopoDS::Vertex(shapes[i]));
      } else {
        const TopoDS_Wire wire = as_wire(shapes[i], "loft");
        if (solid != 0 && !BRep_Tool::IsClosed(wire)) {
          throw std::invalid_argument("only closed sections can loft into a solid");
        }
        loft.AddWire(wire);
        ++wires;
      }
    }
    if (wires == 0) throw std::invalid_argument("a loft needs a section that is not a point");
    loft.CheckCompatibility(true);
    loft.Build();
    if (!loft.IsDone()) throw std::runtime_error("the sections could not be joined");
    TopoDS_Shape result = loft.Shape();
    require_present(result, "loft");
    require_valid(result, "loft");
    if (solid != 0) {
      result = oriented_outward(only_solid(result, "loft"));
      if (!(volume_of(result) > 0.0)) throw std::runtime_error("the loft has no volume");
    }
    return put_shape(result);
  });
}

// --- features --------------------------------------------------------------------------

// `faces` is count uint32 handles; no faces makes a closed cavity. A modeled_buffer.
API uint32_t occt_shell(uint32_t h, const void* faces, size_t count, double thickness) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape solid = get_shape(h);
    require_solid(solid, "shell");
    TopTools_ListOfShape closing;
    for (const TopoDS_Face& face : own_faces(solid, faces, count)) closing.Append(face);
    const double before = volume_of(solid);
    if (closing.IsEmpty()) {
      BRepOffsetAPI_MakeOffsetShape offset;
      offset.PerformByJoin(solid, thickness, kOffsetTolerance, BRepOffset_Skin,
                           /*Intersection=*/false, /*SelfInter=*/false, GeomAbs_Intersection);
      if (!offset.IsDone() || offset.Shape().IsNull()) {
        throw std::runtime_error("the walls are too thick for this solid");
      }
      const TopoDS_Shape other = only_solid(offset.Shape(), "shell");
      require_valid(other, "shell");
      require_distance(faces_of(solid), faces_of(other), std::abs(thickness),
                       "the walls are too thick for this solid");
      const double inner = thickness < 0.0 ? volume_of(other) : before;
      const double outer = thickness < 0.0 ? before : volume_of(other);
      if (!(inner > before * kVolumeFloor) || !(outer > inner)) {
        throw std::runtime_error("the walls are too thick for this solid");
      }
      const TopoDS_Shape result =
          thickness < 0.0 ? with_cavity(solid, other) : with_cavity(other, solid);
      require_valid(result, "shell");
      return modeled_handle(offset, result, {solid});
    }
    BRepOffsetAPI_MakeThickSolid maker;
    maker.MakeThickSolidByJoin(solid, closing, thickness, kOffsetTolerance, BRepOffset_Skin,
                               /*Intersection=*/false, /*SelfInter=*/false, GeomAbs_Intersection);
    if (!maker.IsDone()) throw std::runtime_error("the walls could not be offset");
    const TopoDS_Shape result = only_solid(maker.Shape(), "shell");
    require_valid(result, "shell");
    // Each remaining face has a new wall the thickness away from it.
    ShapeMap kept;
    TopExp::MapShapes(solid, TopAbs_FACE, kept);
    BRep_Builder builder;
    TopoDS_Compound originals, walls;
    builder.MakeCompound(originals);
    builder.MakeCompound(walls);
    for (int i = 1; i <= kept.Extent(); ++i) {
      const TopTools_ListOfShape& made = maker.Generated(kept(i));
      if (made.IsEmpty()) continue;
      builder.Add(originals, kept(i));
      for (TopTools_ListOfShape::Iterator it(made); it.More(); it.Next()) {
        if (it.Value().ShapeType() == TopAbs_FACE) builder.Add(walls, it.Value());
      }
    }
    require_distance(originals, walls, std::abs(thickness),
                     "the walls are too thick for this solid");
    const double after = volume_of(result);
    if (!(after > before * kVolumeFloor) || (thickness < 0.0 && !(after < before))) {
      throw std::runtime_error("the walls are too thick for this solid");
    }
    return modeled_handle(maker, result, {solid});
  });
}

API uint32_t occt_offset(uint32_t h, double distance) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    BRepOffsetAPI_MakeOffsetShape maker;
    maker.PerformByJoin(shape, distance, kOffsetTolerance, BRepOffset_Skin,
                        /*Intersection=*/false, /*SelfInter=*/false, GeomAbs_Intersection);
    if (!maker.IsDone()) throw std::runtime_error("the faces could not be offset");
    if (maker.Shape().IsNull()) throw std::runtime_error("the offset is too large for this shape");
    TopoDS_Shape result = unwrap_single(maker.Shape());
    require_valid(result, "offset");
    require_distance(faces_of(shape), faces_of(result), std::abs(distance),
                     "the offset is too large for this shape");
    // A face comes back wrapped in a shell.
    if (shape.ShapeType() == TopAbs_FACE) result = unwrap_single_child(result);
    const double before = volume_of(shape);
    if (before != 0.0) {
      const double after = volume_of(result);
      if (!(after > before * kVolumeFloor) || (distance > 0.0) != (after > before)) {
        throw std::runtime_error("the offset is too large for this shape");
      }
    }
    return put_shape(result);
  });
}

API uint32_t occt_thicken(uint32_t h, double thickness) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape sheet = get_shape(h);
    if (sheet.ShapeType() != TopAbs_FACE && sheet.ShapeType() != TopAbs_SHELL) {
      throw std::invalid_argument(std::string("only a face or shell can be thickened, not a ") +
                                  kind_name(sheet));
    }
    BRepOffset_MakeOffset maker;
    maker.Initialize(sheet, thickness, kOffsetTolerance, BRepOffset_Skin, /*Intersection=*/false,
                     /*SelfInter=*/false, GeomAbs_Intersection, /*Thickening=*/true);
    maker.MakeOffsetShape();
    if (!maker.IsDone()) throw std::runtime_error("the sheet could not be thickened");
    const TopoDS_Shape result = oriented_outward(only_solid(maker.Shape(), "thicken"));
    require_valid(result, "thicken");
    BRep_Builder builder;
    TopoDS_Compound made;
    builder.MakeCompound(made);
    ShapeMap faces;
    TopExp::MapShapes(sheet, TopAbs_FACE, faces);
    for (int i = 1; i <= faces.Extent(); ++i) {
      const TopTools_ListOfShape& from = maker.Generated(faces(i));
      for (TopTools_ListOfShape::Iterator it(from); it.More(); it.Next()) {
        if (it.Value().ShapeType() == TopAbs_FACE) builder.Add(made, it.Value());
      }
    }
    if (TopoDS_Iterator(made).More()) {
      require_distance(faces_of(sheet), made, std::abs(thickness),
                       "the thickness is too large for this sheet's curvature");
    }
    if (!(volume_of(result) > 0.0)) throw std::runtime_error("the thickened sheet has no volume");
    return put_shape(result);
  });
}

// `faces` is count uint32 handles. A modeled_buffer.
API uint32_t occt_draft(uint32_t h, const void* faces, size_t count, double pull_x, double pull_y,
                        double pull_z, double px, double py, double pz, double nx, double ny,
                        double nz, double angle) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape solid = get_shape(h);
    require_solid(solid, "draft");
    const std::vector<TopoDS_Face> list = own_faces(solid, faces, count);
    if (list.empty()) throw std::invalid_argument("no faces given");
    const gp_Dir pull(pull_x, pull_y, pull_z);
    const gp_Pln neutral(gp_Pnt(px, py, pz), gp_Dir(nx, ny, nz));
    BRepOffsetAPI_DraftAngle maker(solid);
    for (const TopoDS_Face& face : list) {
      maker.Add(face, pull, angle, neutral);
      if (!maker.AddDone()) {
        throw std::runtime_error("this face cannot be drafted: it must be a plane, cylinder or "
                                 "cone that the pull direction does not cross");
      }
    }
    maker.Build();
    if (!maker.IsDone()) throw std::runtime_error("the draft could not be built");
    const TopoDS_Shape result = only_solid(maker.Shape(), "draft");
    require_valid(result, "draft");
    if (!(volume_of(result) > 0.0)) throw std::runtime_error("the draft left no volume");
    DraftHistory history(maker, result);
    return modeled_handle(history, result, {solid});
  });
}

// `faces` is count uint32 handles. A modeled_buffer.
API uint32_t occt_remove_faces(uint32_t h, const void* faces, size_t count) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape solid = get_shape(h);
    require_solid(solid, "remove_faces");
    const std::vector<TopoDS_Face> list = own_faces(solid, faces, count);
    if (list.empty()) throw std::invalid_argument("no faces given");
    BRepAlgoAPI_Defeaturing maker;
    maker.SetShape(solid);
    for (const TopoDS_Face& face : list) maker.AddFaceToRemove(face);
    maker.SetRunParallel(false);
    maker.SetToFillHistory(true);
    maker.Build();
    if (maker.HasErrors() || !maker.IsDone()) {
      std::ostringstream message;
      maker.DumpErrors(message);
      throw std::runtime_error("the faces could not be removed: " + message.str());
    }
    const TopoDS_Shape result = only_solid(maker.Shape(), "remove_faces");
    // The algorithm only warns when it gives up on a face, and returns it unchanged.
    ShapeMap left;
    TopExp::MapShapes(result, TopAbs_FACE, left);
    for (const TopoDS_Face& face : list) {
      if (left.Contains(face)) {
        throw std::runtime_error("a face could not be removed: its neighbours cannot close the gap");
      }
    }
    require_valid(result, "remove_faces");
    if (!(volume_of(result) > 0.0)) throw std::runtime_error("removing the faces left no volume");
    return modeled_handle(maker, result, {solid});
  });
}

// A modeled_buffer.
API uint32_t occt_unify(uint32_t h) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    ShapeUpgrade_UnifySameDomain maker(shape, /*UnifyEdges=*/true, /*UnifyFaces=*/true,
                                       /*ConcatBSplines=*/false);
    maker.Build();
    const TopoDS_Shape result = maker.Shape();
    require_present(result, "unify");
    RecordedHistory history(maker.History());
    return modeled_handle(history, result, {shape});
  });
}

API uint32_t occt_heal(uint32_t h, double tolerance) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    ShapeFix_Shape fix(shape);
    fix.SetPrecision(tolerance);
    fix.SetMinTolerance(std::min(tolerance, 1e-7));
    fix.SetMaxTolerance(std::max(tolerance, 1e-3));
    fix.Perform();
    TopoDS_Shape result = fix.Shape();
    require_present(result, "heal");
    if (result.ShapeType() == TopAbs_SOLID) result = oriented_outward(result);
    return put_shape(result);
  });
}

// `rows` is 12 doubles, the top three rows of the matrix.
API uint32_t occt_transformed(uint32_t h, const void* rows, size_t len) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    const Matrix matrix = read_matrix(rows, len);
    double scale = 0.0;
    if (matrix.is_similarity(&scale)) {
      BRepBuilderAPI_Transform transform(shape, matrix.trsf(), /*Copy=*/true);
      if (!transform.IsDone()) throw std::runtime_error("transform failed");
      return put_shape(transform.Shape());
    }
    if (std::abs(matrix.det()) <= 1e-12) throw std::invalid_argument("the transform is singular");
    BRepBuilderAPI_GTransform transform(shape, matrix.gtrsf(), /*Copy=*/true);
    if (!transform.IsDone()) throw std::runtime_error("transform failed");
    TopoDS_Shape result = transform.Shape();
    require_present(result, "transform");
    if (result.ShapeType() == TopAbs_SOLID) result = oriented_outward(result);
    return put_shape(result);
  });
}

// `rows` is 12 doubles, the top three rows of the matrix.
API uint32_t occt_placed(uint32_t h, const void* rows, size_t len) {
  return guard_handle([&]() -> uint32_t {
    const Matrix matrix = read_matrix(rows, len);
    double scale = 0.0;
    if (!matrix.is_similarity(&scale) || std::abs(scale - 1.0) > kRigidTolerance ||
        matrix.det() < 0.0) {
      throw std::invalid_argument("a placement must be a rotation and translation");
    }
    gp_Trsf trsf = matrix.trsf();
    // Locations refuse any scale at all, and rounding leaves one in the last bits.
    trsf.SetScaleFactor(1.0);
    return put_shape(get_shape(h).Moved(TopLoc_Location(trsf)));
  });
}

API uint32_t occt_reversed(uint32_t h) {
  return guard_handle([&]() -> uint32_t { return put_shape(get_shape(h).Reversed()); });
}

// 1 when both are the same topology wherever each is placed, 0 if not, -1 on error.
API int32_t occt_shape_shares_geometry(uint32_t a, uint32_t b) {
  return guard<int32_t>(-1, [&]() { return get_shape(a).IsPartner(get_shape(b)) ? 1 : 0; });
}
