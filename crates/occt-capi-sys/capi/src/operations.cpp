// Entry points of the operations domain. SPDX-License-Identifier: MIT OR Apache-2.0
#include "internal.hpp"

#include <BOPAlgo_Alerts.hxx>
#include <BOPAlgo_Operation.hxx>
#include <BRepAlgoAPI_BooleanOperation.hxx>
#include <BRepAlgoAPI_Section.hxx>
#include <BRepAlgoAPI_Splitter.hxx>
#include <NCollection_Array1.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Vertex.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt2d.hxx>

#include <algorithm>

using namespace occt_capi;

namespace {

constexpr uint32_t kUnify = 1;
constexpr uint32_t kParallel = 2;

constexpr uint32_t kOpFuse = 0;
constexpr uint32_t kOpCut = 1;
constexpr uint32_t kOpCommon = 2;

constexpr uint32_t kLawConstant = 0;
constexpr uint32_t kLawLinear = 1;
constexpr uint32_t kLawPoints = 2;

constexpr uint32_t kChamferEqual = 0;
constexpr uint32_t kChamferTwoDistances = 1;
constexpr uint32_t kChamferDistanceAngle = 2;

// Between two given radii the fillet law is a spline with flat ends, which is not a
// straight line. Collinear samples hold it to one: the flat ends' effect shrinks by about
// a factor of four per sample, so it stays within the first and last interval, where the
// radius is off by at most about a fifth of a percent of the law's whole change.
constexpr int kLinearLawIntervals = 16;

using ShapeList = NCollection_List<TopoDS_Shape>;

ShapeList read_list(Reader& in) {
  ShapeList out;
  const uint32_t n = in.u32();
  for (uint32_t i = 0; i < n; ++i) out.Append(in.shape());
  return out;
}

std::vector<TopoDS_Shape> to_vector(const ShapeList& list) {
  std::vector<TopoDS_Shape> out;
  for (ShapeList::Iterator it(list); it.More(); it.Next()) out.push_back(it.Value());
  return out;
}

double volume_of(const TopoDS_Shape& shape) {
  GProp_GProps props;
  BRepGProp::VolumeProperties(shape, props);
  return props.Mass();
}

// Whether a shape is nothing but solids, so that its volume means what it says.
bool only_solids(const TopoDS_Shape& shape) {
  return TopExp_Explorer(shape, TopAbs_SOLID).More() &&
         !TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SOLID).More() &&
         !TopExp_Explorer(shape, TopAbs_EDGE, TopAbs_FACE).More();
}

bool all_only_solids(const ShapeList& list) {
  for (ShapeList::Iterator it(list); it.More(); it.Next()) {
    if (!only_solids(it.Value())) return false;
  }
  return true;
}

struct VolumeBounds {
  double sum = 0.0;
  double largest = 0.0;
};

VolumeBounds bounds_of(const ShapeList& list) {
  VolumeBounds out;
  for (ShapeList::Iterator it(list); it.More(); it.Next()) {
    const double v = volume_of(it.Value());
    out.sum += v;
    out.largest = std::max(out.largest, v);
  }
  return out;
}

// How far a result's volume may stray from what set arithmetic allows: a part in a
// thousand, plus the layer that a fuzzy value lets geometry shift by.
double volume_slack(double scale, double fuzzy) {
  return 1e-3 * scale + 12.0 * std::max(fuzzy, 0.0) * std::cbrt(scale * scale);
}

// Set arithmetic bounds the result of a boolean between solids. A kernel that quietly
// loses or inverts a piece breaks those bounds.
void check_boolean_volume(uint32_t op, const ShapeList& arguments, const ShapeList& tools,
                          const TopoDS_Shape& result, double fuzzy) {
  if (!all_only_solids(arguments) || !all_only_solids(tools)) return;
  if (TopExp_Explorer(result, TopAbs_FACE, TopAbs_SOLID).More()) return;
  const VolumeBounds a = bounds_of(arguments);
  const VolumeBounds t = bounds_of(tools);
  const double v = volume_of(result);
  const double slack = volume_slack(std::max(a.sum, t.sum), fuzzy);
  double low = 0.0;
  double high = 0.0;
  switch (op) {
    case kOpFuse:
      low = std::max(a.largest, t.largest);
      high = a.sum + t.sum;
      break;
    case kOpCut:
      high = a.sum;
      break;
    default:
      high = std::min(a.sum, t.sum);
      break;
  }
  if (v < low - slack || v > high + slack) {
    std::ostringstream msg;
    msg << "the result's volume " << v << " is impossible for these inputs (expected between "
        << low << " and " << high << "); a fuzzy value may help";
    throw std::runtime_error(msg.str());
  }
}

// Splitting redistributes volume without creating or losing any.
void check_split_volume(const ShapeList& arguments, const TopoDS_Shape& result, double fuzzy) {
  if (!all_only_solids(arguments)) return;
  if (TopExp_Explorer(result, TopAbs_FACE, TopAbs_SOLID).More()) return;
  const VolumeBounds a = bounds_of(arguments);
  const double v = volume_of(result);
  const double slack = volume_slack(a.sum, fuzzy);
  if (v < a.largest - slack || v > a.sum + slack) {
    std::ostringstream msg;
    msg << "the pieces' volume " << v << " differs from the inputs' (" << a.sum
        << "); a fuzzy value may help";
    throw std::runtime_error(msg.str());
  }
}

// A valid input and an invalid output means the operation quietly went wrong.
void check_valid_result(const ShapeList& arguments, const ShapeList& tools,
                        const TopoDS_Shape& result, const char* what) {
  if (BRepCheck_Analyzer(result).IsValid()) return;
  for (const ShapeList* list : {&arguments, &tools}) {
    for (ShapeList::Iterator it(*list); it.More(); it.Next()) {
      if (!BRepCheck_Analyzer(it.Value()).IsValid()) return;
    }
  }
  throw std::runtime_error(std::string(what) +
                           " produced an invalid shape from valid inputs; a fuzzy value may help");
}

template <typename Algo>
void configure(Algo& algo, const ShapeList& arguments, const ShapeList& tools, double fuzzy,
               uint32_t flags) {
  algo.SetArguments(arguments);
  algo.SetTools(tools);
  if (fuzzy > 0.0) algo.SetFuzzyValue(fuzzy);
  algo.SetRunParallel((flags & kParallel) != 0);
  // Operations never change their inputs, not even by widening a tolerance.
  algo.SetNonDestructive(true);
  algo.SetToFillHistory(true);
}

// The reasons callers can act on, in words that don't depend on this kernel.
template <typename Algo>
const char* explain_failure(Algo& algo) {
  if (algo.HasError(STANDARD_TYPE(BOPAlgo_AlertBOPNotAllowed))) {
    return "this operation can't combine shapes of these kinds";
  }
  if (algo.HasError(STANDARD_TYPE(BOPAlgo_AlertTooFewArguments))) {
    return "it needs more shapes";
  }
  if (algo.HasError(STANDARD_TYPE(BOPAlgo_AlertSolidBuilderFailed)) ||
      algo.HasError(STANDARD_TYPE(BOPAlgo_AlertBuilderFailed))) {
    return "the result couldn't be assembled; a fuzzy value may help";
  }
  return "the shapes couldn't be intersected; a fuzzy value may help";
}

template <typename Algo>
void run(Algo& algo, size_t cancel, const char* what) {
  FlagProgress progress(cancel);
  if (progress.UserBreak()) throw std::runtime_error("cancelled");
  algo.Build(progress.Start());
  // Checked first: being stopped surfaces as an arbitrary error from whichever stage noticed.
  if (progress.UserBreak()) throw std::runtime_error("cancelled");
  if (algo.HasErrors() || !algo.IsDone()) {
    throw std::runtime_error(std::string(what) + " failed: " + explain_failure(algo));
  }
}

// In a cut, the tool's surface becomes the wall of the hole. As set operations see it that
// is a piece of the tool surviving, and it is reported as modified; to the application it
// is new surface that the tool made, so a tool's surviving pieces also count as generated.
class CutHistory {
 public:
  CutHistory(BRepAlgoAPI_BooleanOperation& algo, const ShapeList& tools, const TopoDS_Shape& result)
      : algo_(algo) {
    for (ShapeList::Iterator it(tools); it.More(); it.Next()) {
      for (const TopAbs_ShapeEnum kind : {TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX}) {
        TopExp::MapShapes(it.Value(), kind, tools_);
      }
    }
    for (const TopAbs_ShapeEnum kind : {TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX}) {
      TopExp::MapShapes(result, kind, result_);
    }
  }
  const ShapeList& Modified(const TopoDS_Shape& s) { return algo_.Modified(s); }
  bool IsDeleted(const TopoDS_Shape& s) { return algo_.IsDeleted(s); }
  const ShapeList& Generated(const TopoDS_Shape& s) {
    if (!tools_.Contains(s)) return algo_.Generated(s);
    merged_.Clear();
    for (ShapeList::Iterator it(algo_.Generated(s)); it.More(); it.Next()) merged_.Append(it.Value());
    const ShapeList& modified = algo_.Modified(s);
    for (ShapeList::Iterator it(modified); it.More(); it.Next()) merged_.Append(it.Value());
    // A tool face the cut didn't divide, like a ball inside a box, is its own wall.
    if (modified.IsEmpty() && result_.Contains(s)) merged_.Append(s);
    return merged_;
  }

 private:
  BRepAlgoAPI_BooleanOperation& algo_;
  ShapeMap tools_;
  ShapeMap result_;
  ShapeList merged_;
};

// The blend makers answer IsDeleted only for faces: for an edge or vertex it says "deleted"
// even when the result still has it. Those are answered from the result instead.
template <typename Maker>
class BlendHistory {
 public:
  BlendHistory(Maker& maker, const TopoDS_Shape& result) : maker_(maker) {
    for (const TopAbs_ShapeEnum kind : {TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX}) {
      TopExp::MapShapes(result, kind, result_);
    }
  }
  const ShapeList& Modified(const TopoDS_Shape& s) { return maker_.Modified(s); }
  const ShapeList& Generated(const TopoDS_Shape& s) { return maker_.Generated(s); }
  bool IsDeleted(const TopoDS_Shape& s) {
    return !result_.Contains(s) && maker_.Modified(s).IsEmpty();
  }

 private:
  Maker& maker_;
  ShapeMap result_;
};

TopoDS_Edge read_own_edge(Reader& in, const ShapeMap& own_edges) {
  const TopoDS_Shape shape = in.shape();
  if (shape.ShapeType() != TopAbs_EDGE) throw std::invalid_argument("expected an edge");
  // The makers silently ignore a foreign edge.
  if (!own_edges.Contains(shape)) throw std::invalid_argument("the edge is not part of the shape");
  return TopoDS::Edge(shape);
}

// The vertex the contract calls an edge's first: the first of its listed vertices.
TopoDS_Shape first_listed_vertex(const TopoDS_Edge& edge) {
  ShapeMap vertices;
  TopExp::MapShapes(edge, TopAbs_VERTEX, vertices);
  if (vertices.IsEmpty()) throw std::invalid_argument("the edge has no vertices");
  return vertices(1);
}

bool edge_is_closed(const TopoDS_Edge& edge) {
  ShapeMap vertices;
  TopExp::MapShapes(edge, TopAbs_VERTEX, vertices);
  return vertices.Extent() < 2;
}

using RadiusPoints = std::vector<std::pair<double, double>>;

// The radius samples for one edge, as (fraction, radius) from its first vertex.
RadiusPoints read_radius_law(Reader& in, const TopoDS_Edge& edge) {
  RadiusPoints points;
  const uint32_t kind = in.u32();
  if (kind == kLawConstant) {
    const double r = in.f64();
    points = {{0.0, r}, {1.0, r}};
  } else if (kind == kLawLinear) {
    const double start = in.f64();
    const double end = in.f64();
    for (int i = 0; i <= kLinearLawIntervals; ++i) {
      const double t = double(i) / kLinearLawIntervals;
      points.emplace_back(t, start + t * (end - start));
    }
  } else if (kind == kLawPoints) {
    const uint32_t n = in.u32();
    for (uint32_t i = 0; i < n; ++i) {
      const double u = in.f64();
      const double r = in.f64();
      points.emplace_back(u, r);
    }
    if (points.size() < 2) throw std::invalid_argument("a radius law needs at least two points");
    // A law holds its nearest radius out to the ends of the edge.
    const std::pair<double, double> first = points.front();
    const std::pair<double, double> last = points.back();
    if (first.first > 0.0) points.insert(points.begin(), std::make_pair(0.0, first.second));
    if (last.first < 1.0) points.push_back(std::make_pair(1.0, last.second));
  } else {
    throw std::invalid_argument("unknown radius law");
  }
  for (const auto& p : points) {
    if (!(p.second > 0.0)) throw std::invalid_argument("a fillet radius must be positive");
  }
  if (edge_is_closed(edge) && std::abs(points.front().second - points.back().second) > 1e-9) {
    throw std::invalid_argument("on a closed edge the radius must end where it starts");
  }
  return points;
}

// Gives one edge of a contour its radius law, in the direction the contour runs.
void set_radius_law(BRepFilletAPI_MakeFillet& maker, const TopoDS_Edge& edge, RadiusPoints points) {
  const int contour = maker.Contour(edge);
  if (contour == 0) {
    throw std::invalid_argument("the edge can't be rounded: it doesn't join two faces");
  }
  int index = 0;
  for (int i = 1; i <= maker.NbEdges(contour); ++i) {
    if (maker.Edge(contour, i).IsSame(edge)) index = i;
  }
  if (index == 0) throw std::runtime_error("the edge isn't on its own contour");
  const TopoDS_Edge& along = maker.Edge(contour, index);
  if (!edge_is_closed(edge) &&
      !TopExp::FirstVertex(along, /*CumOri=*/true).IsSame(first_listed_vertex(edge))) {
    std::reverse(points.begin(), points.end());
    for (auto& p : points) p.first = 1.0 - p.first;
  }
  if (points.size() == 2 && points[0].second == points[1].second) {
    maker.SetRadius(points[0].second, contour, index);
    return;
  }
  NCollection_Array1<gp_Pnt2d> law(1, int(points.size()));
  for (size_t i = 0; i < points.size(); ++i) {
    law(int(i) + 1) = gp_Pnt2d(points[i].first, points[i].second);
  }
  maker.SetRadius(law, contour, index);
}

// Finishes a blend: builds it, checks for silent failure and encodes the history.
template <typename Maker>
uint32_t finish_blend(Maker& maker, const TopoDS_Shape& shape, const std::vector<TopoDS_Edge>& edges,
                      const char* what) {
  maker.Build();
  if (!maker.IsDone()) {
    throw std::runtime_error(std::string(what) +
                             " failed: the sizes may be too large for the shape, or impossible at this edge");
  }
  const TopoDS_Shape result = unwrap_single(maker.Shape());
  if (result.IsNull()) throw std::runtime_error(std::string(what) + " produced nothing");
  for (const TopoDS_Edge& edge : edges) {
    if (maker.Generated(edge).IsEmpty()) {
      throw std::runtime_error(std::string(what) + " did not blend one of its edges");
    }
  }
  if (!BRepCheck_Analyzer(result).IsValid() && BRepCheck_Analyzer(shape).IsValid()) {
    throw std::runtime_error(std::string(what) +
                             " produced an invalid shape: the sizes may be too large for it");
  }
  BlendHistory<Maker> history(maker, result);
  return put_buffer(modeled_buffer(history, result, {shape}));
}

}  // namespace

// --- booleans -----------------------------------------------------------------

// `input` is u32 n, n handles of arguments, u32 m, m handles of tools. `fuzzy` <= 0 means
// the default. `flags` is kUnify | kParallel. `cancel` is the address of a flag byte or 0.
// Returns a buffer in the modeled layout.
API uint32_t occt_boolean_general(uint32_t op, const void* input, size_t len, double fuzzy,
                                  uint32_t flags, size_t cancel) {
  return guard_handle([&]() -> uint32_t {
    Reader in(input, len);
    const ShapeList arguments = read_list(in);
    const ShapeList tools = read_list(in);
    if (arguments.IsEmpty() || tools.IsEmpty()) {
      throw std::invalid_argument("a boolean needs arguments and tools");
    }
    BOPAlgo_Operation operation = BOPAlgo_UNKNOWN;
    switch (op) {
      case kOpFuse: operation = BOPAlgo_FUSE; break;
      case kOpCut: operation = BOPAlgo_CUT; break;
      case kOpCommon: operation = BOPAlgo_COMMON; break;
      default: throw std::invalid_argument("unknown boolean operation");
    }
    BRepAlgoAPI_BooleanOperation algo;
    algo.SetOperation(operation);
    configure(algo, arguments, tools, fuzzy, flags);
    run(algo, cancel, "boolean");
    if ((flags & kUnify) != 0) algo.SimplifyResult();
    if (algo.HasErrors()) throw std::runtime_error("merging the result's faces failed");
    const TopoDS_Shape result = unwrap_single(algo.Shape());
    check_boolean_volume(op, arguments, tools, result, fuzzy);
    check_valid_result(arguments, tools, result, "boolean");

    std::vector<TopoDS_Shape> inputs = to_vector(arguments);
    const std::vector<TopoDS_Shape> tool_inputs = to_vector(tools);
    inputs.insert(inputs.end(), tool_inputs.begin(), tool_inputs.end());
    if (op == kOpCut) {
      CutHistory history(algo, tools, result);
      return put_buffer(modeled_buffer(history, result, inputs));
    }
    return put_buffer(modeled_buffer(algo, result, inputs));
  });
}

// `input` is as for occt_boolean_general; the result is always a compound of the pieces.
API uint32_t occt_split(const void* input, size_t len, double fuzzy, uint32_t flags, size_t cancel) {
  return guard_handle([&]() -> uint32_t {
    Reader in(input, len);
    const ShapeList arguments = read_list(in);
    const ShapeList tools = read_list(in);
    if (arguments.IsEmpty() || tools.IsEmpty()) {
      throw std::invalid_argument("splitting needs arguments and tools");
    }
    BRepAlgoAPI_Splitter algo;
    configure(algo, arguments, tools, fuzzy, flags);
    run(algo, cancel, "split");
    if ((flags & kUnify) != 0) algo.SimplifyResult();
    if (algo.HasErrors()) throw std::runtime_error("merging the result's faces failed");
    const TopoDS_Shape result = algo.Shape();
    check_split_volume(arguments, result, fuzzy);
    check_valid_result(arguments, tools, result, "split");

    std::vector<TopoDS_Shape> inputs = to_vector(arguments);
    const std::vector<TopoDS_Shape> tool_inputs = to_vector(tools);
    inputs.insert(inputs.end(), tool_inputs.begin(), tool_inputs.end());
    return put_buffer(modeled_buffer(algo, result, inputs));
  });
}

// --- sections -----------------------------------------------------------------

API uint32_t occt_section(uint32_t a, uint32_t b) {
  return guard_handle([&]() -> uint32_t {
    BRepAlgoAPI_Section algo(get_shape(a), get_shape(b), /*PerformNow=*/false);
    algo.SetNonDestructive(true);
    run(algo, 0, "section");
    return put_shape(unwrap_single(algo.Shape()));
  });
}

API uint32_t occt_section_plane(uint32_t h, double ox, double oy, double oz, double nx, double ny,
                                double nz) {
  return guard_handle([&]() -> uint32_t {
    const gp_Pln plane(gp_Pnt(ox, oy, oz), gp_Dir(nx, ny, nz));
    BRepAlgoAPI_Section algo(get_shape(h), plane, /*PerformNow=*/false);
    algo.SetNonDestructive(true);
    run(algo, 0, "section");
    return put_shape(unwrap_single(algo.Shape()));
  });
}

// --- blends -------------------------------------------------------------------

// `input` is u32 count, then per edge: u32 edge handle, then its radius law:
//   u32 0, f64 radius | u32 1, f64 start, f64 end | u32 2, u32 n, n * (f64 fraction, f64 radius)
// Returns a buffer in the modeled layout.
API uint32_t occt_fillet_edges(uint32_t h, const void* input, size_t len) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    ShapeMap own_edges;
    TopExp::MapShapes(shape, TopAbs_EDGE, own_edges);
    Reader in(input, len);
    const uint32_t count = in.u32();
    if (count == 0) throw std::invalid_argument("no edges given");
    BRepFilletAPI_MakeFillet maker(shape);
    std::vector<TopoDS_Edge> edges;
    for (uint32_t i = 0; i < count; ++i) {
      const TopoDS_Edge edge = read_own_edge(in, own_edges);
      RadiusPoints points = read_radius_law(in, edge);
      // Adding an edge takes in every edge tangent to it, so it may be there already.
      maker.Add(edge);
      set_radius_law(maker, edge, std::move(points));
      edges.push_back(edge);
    }
    return finish_blend(maker, shape, edges, "fillet");
  });
}

// `input` is u32 count, then per edge: u32 edge handle, u32 reference face handle or 0,
// u32 kind (0 equal, 1 two distances, 2 distance and angle), f64 a, f64 b.
// Returns a buffer in the modeled layout.
API uint32_t occt_chamfer_edges(uint32_t h, const void* input, size_t len) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    ShapeMap own_edges, own_faces;
    TopExp::MapShapes(shape, TopAbs_EDGE, own_edges);
    TopExp::MapShapes(shape, TopAbs_FACE, own_faces);
    Reader in(input, len);
    const uint32_t count = in.u32();
    if (count == 0) throw std::invalid_argument("no edges given");
    BRepFilletAPI_MakeChamfer maker(shape);
    std::vector<TopoDS_Edge> edges;
    struct Size {
      int contour;
      uint32_t kind;
      double a, b;
    };
    std::vector<Size> sizes;
    for (uint32_t i = 0; i < count; ++i) {
      const TopoDS_Edge edge = read_own_edge(in, own_edges);
      const uint32_t face_handle = in.u32();
      const uint32_t kind = in.u32();
      const double a = in.f64();
      const double b = in.f64();
      TopoDS_Face face;
      if (face_handle != 0) {
        const TopoDS_Shape f = get_shape(face_handle);
        if (f.ShapeType() != TopAbs_FACE) throw std::invalid_argument("expected a face");
        if (!own_faces.Contains(f)) throw std::invalid_argument("the face is not part of the shape");
        ShapeMap face_edges;
        TopExp::MapShapes(f, TopAbs_EDGE, face_edges);
        if (!face_edges.Contains(edge)) throw std::invalid_argument("the face is not next to the edge");
        face = TopoDS::Face(f);
      } else if (kind != kChamferEqual) {
        throw std::invalid_argument("an asymmetric chamfer needs a reference face");
      }
      // Adding an edge takes in every edge tangent to it, and that chain has one size.
      const int existing = maker.Contour(edge);
      if (existing != 0) {
        const auto same = std::find_if(sizes.begin(), sizes.end(), [&](const Size& s) {
          return s.contour == existing && s.kind == kind && s.a == a && s.b == b;
        });
        if (same == sizes.end()) {
          throw std::invalid_argument("edges joined tangentially must have the same chamfer size");
        }
        edges.push_back(edge);
        continue;
      }
      switch (kind) {
        case kChamferEqual: maker.Add(a, edge); break;
        case kChamferTwoDistances: maker.Add(a, b, edge, face); break;
        case kChamferDistanceAngle: maker.AddDA(a, b, edge, face); break;
        default: throw std::invalid_argument("unknown chamfer kind");
      }
      const int contour = maker.Contour(edge);
      if (contour == 0) {
        throw std::invalid_argument("the edge can't be bevelled: it doesn't join two faces");
      }
      sizes.push_back({contour, kind, a, b});
      edges.push_back(edge);
    }
    return finish_blend(maker, shape, edges, "chamfer");
  });
}
