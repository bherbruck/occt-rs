// Entry points of the inspection domain. SPDX-License-Identifier: MIT OR Apache-2.0
#include "internal.hpp"

#include <BRepAdaptor_Curve2d.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepLib.hxx>
#include <BRepTools.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <GCPnts_TangentialDeflection.hxx>
#include <GProp_PrincipalProps.hxx>
#include <HLRAlgo_Projector.hxx>
#include <HLRBRep_Algo.hxx>
#include <HLRBRep_HLRToShape.hxx>
#include <IMeshData_Status.hxx>
#include <IMeshTools_Parameters.hxx>
#include <IntCurvesFace_Intersector.hxx>
#include <NCollection_IndexedDataMap.hxx>
#include <NCollection_List.hxx>
#include <Poly_Polygon3D.hxx>
#include <Poly_PolygonOnTriangulation.hxx>
#include <Precision.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Lin.hxx>
#include <gp_Mat.hxx>

#include <algorithm>
#include <cstddef>

using namespace occt_capi;

namespace {

using AncestorMap =
    NCollection_IndexedDataMap<TopoDS_Shape, NCollection_List<TopoDS_Shape>,
                               TopTools_ShapeMapHasher>;

// Below this angle in radians two faces count as tangent. Blends fit their surfaces, and
// their boundaries are only tangent to about this; an intended crease is far larger.
constexpr double kTangentAngle = 1e-3;

// How far apart hits on one face must be to count as two crossings, rather than one
// reported twice because it landed on the surface's seam.
constexpr double kSameHit = 1e-7;

// The largest dimension that has something to measure.
GProp_GProps measure(const TopoDS_Shape& shape) {
  GProp_GProps props;
  BRepGProp::VolumeProperties(shape, props, /*OnlyClosed=*/true);
  if (std::abs(props.Mass()) > 0.0) return props;
  props = GProp_GProps();
  BRepGProp::SurfaceProperties(shape, props);
  if (std::abs(props.Mass()) > 0.0) return props;
  props = GProp_GProps();
  BRepGProp::LinearProperties(shape, props, /*SkipShared=*/true);
  return props;
}

// A vertex of `other` strictly inside a solid of a group. The distance algorithm looks
// for this itself, but only when the shape it gets is directly a solid.
bool vertex_inside_group(const TopoDS_Shape& group, const TopoDS_Shape& other, gp_Pnt& found) {
  if (group.ShapeType() == TopAbs_SOLID || group.ShapeType() == TopAbs_COMPSOLID) return false;
  ShapeMap vertices;
  TopExp::MapShapes(other, TopAbs_VERTEX, vertices);
  for (TopExp_Explorer it(group, TopAbs_SOLID); it.More(); it.Next()) {
    BRepClass3d_SolidClassifier classifier(it.Current());
    for (int i = 1; i <= vertices.Extent(); ++i) {
      const gp_Pnt point = BRep_Tool::Pnt(TopoDS::Vertex(vertices(i)));
      classifier.Perform(point, Precision::Confusion());
      if (classifier.State() == TopAbs_IN) {
        found = point;
        return true;
      }
    }
  }
  return false;
}

struct Hit {
  uint32_t face;
  double distance;
  gp_Pnt point;
  double u;
  double v;
};

// Orders a wire's edges by following them, end to start. A greedy walk gets stuck on a
// seam, where a loop at a vertex must be taken before leaving it, so this is Hierholzer's
// algorithm for a trail through every edge. Empty if there is no such trail.
std::vector<TopoDS_Edge> connected_edges(const std::vector<TopoDS_Edge>& edges) {
  ShapeMap vertices;
  std::vector<int> start(edges.size()), end(edges.size());
  for (size_t i = 0; i < edges.size(); ++i) {
    TopoDS_Vertex first, last;
    TopExp::Vertices(edges[i], first, last, /*CumOri=*/true);
    if (first.IsNull() || last.IsNull()) return {};
    start[i] = vertices.Add(first) - 1;
    end[i] = vertices.Add(last) - 1;
  }
  const size_t count = static_cast<size_t>(vertices.Extent());
  std::vector<std::vector<int>> outgoing(count);
  std::vector<int> balance(count, 0);
  for (size_t i = 0; i < edges.size(); ++i) {
    outgoing[static_cast<size_t>(start[i])].push_back(static_cast<int>(i));
    ++balance[static_cast<size_t>(start[i])];
    --balance[static_cast<size_t>(end[i])];
  }
  // An open wire has to begin where more edges leave than arrive.
  int origin = start.front();
  for (size_t v = 0; v < count; ++v) {
    if (balance[v] == 1) origin = static_cast<int>(v);
  }

  std::vector<size_t> next(count, 0);
  std::vector<std::pair<int, int>> stack{{origin, -1}};
  std::vector<int> trail;
  while (!stack.empty()) {
    const auto [vertex, arrived_by] = stack.back();
    const auto v = static_cast<size_t>(vertex);
    if (next[v] < outgoing[v].size()) {
      const int edge = outgoing[v][next[v]++];
      stack.emplace_back(end[static_cast<size_t>(edge)], edge);
    } else {
      stack.pop_back();
      if (arrived_by >= 0) trail.push_back(arrived_by);
    }
  }
  std::reverse(trail.begin(), trail.end());
  if (trail.size() != edges.size()) return {};
  for (size_t i = 1; i < trail.size(); ++i) {
    if (end[static_cast<size_t>(trail[i - 1])] != start[static_cast<size_t>(trail[i])]) return {};
  }
  std::vector<TopoDS_Edge> ordered;
  ordered.reserve(edges.size());
  for (const int i : trail) ordered.push_back(edges[static_cast<size_t>(i)]);
  return ordered;
}

// The outward normal of a face at a parameter of an edge on it, and the edge's direction
// as the face uses it. False where the surface is singular.
bool frame_on_face(const TopoDS_Face& face, const TopoDS_Edge& edge, double t, gp_Dir& normal,
                   gp_Dir& tangent) {
  TopoDS_Edge used = edge;
  for (TopExp_Explorer it(face, TopAbs_EDGE); it.More(); it.Next()) {
    if (it.Current().IsSame(edge)) {
      used = TopoDS::Edge(it.Current());
      break;
    }
  }
  const BRepAdaptor_Curve2d on_surface(used, face);
  const gp_Pnt2d uv = on_surface.Value(t);
  const BRepAdaptor_Surface surface(face);
  gp_Pnt position;
  gp_Vec du, dv;
  surface.D1(uv.X(), uv.Y(), position, du, dv);
  gp_Vec n = du.Crossed(dv);
  if (n.Magnitude() <= Precision::Confusion() * du.Magnitude()) return false;
  if (face.Orientation() == TopAbs_REVERSED) n.Reverse();
  normal = gp_Dir(n);

  const BRepAdaptor_Curve curve(used);
  gp_Vec d;
  curve.D1(t, position, d);
  if (d.Magnitude() <= Precision::Confusion()) return false;
  if (used.Orientation() == TopAbs_REVERSED) d.Reverse();
  tangent = gp_Dir(d);
  return true;
}

// Whether every face has a triangulation with parameters, and every edge a polygon on the
// triangulation of each of its faces.
bool fully_meshed(const TopoDS_Shape& shape, const AncestorMap& ancestors) {
  for (TopExp_Explorer it(shape, TopAbs_FACE); it.More(); it.Next()) {
    TopLoc_Location location;
    const occ::handle<Poly_Triangulation> tri =
        BRep_Tool::Triangulation(TopoDS::Face(it.Current()), location);
    if (tri.IsNull() || !tri->HasUVNodes()) return false;
  }
  for (int e = 1; e <= ancestors.Extent(); ++e) {
    const TopoDS_Edge edge = TopoDS::Edge(ancestors.FindKey(e));
    if (BRep_Tool::Degenerated(edge)) continue;
    for (NCollection_List<TopoDS_Shape>::Iterator it(ancestors(e)); it.More(); it.Next()) {
      TopLoc_Location location;
      const occ::handle<Poly_Triangulation> tri =
          BRep_Tool::Triangulation(TopoDS::Face(it.Value()), location);
      if (tri.IsNull() || BRep_Tool::PolygonOnTriangulation(edge, tri, location).IsNull()) {
        return false;
      }
    }
  }
  return true;
}

void append_mesh(Bytes& out, const TopoDS_Face& face) {
  TopLoc_Location location;
  const occ::handle<Poly_Triangulation> tri = BRep_Tool::Triangulation(face, location);
  if (tri.IsNull() || tri->NbTriangles() == 0) {
    throw std::runtime_error("a face has no triangulation");
  }
  if (!tri->HasUVNodes()) throw std::runtime_error("a face's triangulation has no parameters");
  BRepLib_ToolTriangulatedShape::ComputeNormals(face, tri);

  const gp_Trsf trsf = location.Transformation();
  const bool reversed = face.Orientation() == TopAbs_REVERSED;
  append<uint32_t>(out, static_cast<uint32_t>(tri->NbNodes()));
  append<uint32_t>(out, static_cast<uint32_t>(tri->NbTriangles()));
  for (int i = 1; i <= tri->NbNodes(); ++i) append_point(out, tri->Node(i).Transformed(trsf));
  for (int i = 1; i <= tri->NbNodes(); ++i) {
    gp_Dir n = tri->Normal(i).Transformed(trsf);
    if (reversed) n.Reverse();
    append(out, n.X());
    append(out, n.Y());
    append(out, n.Z());
  }
  for (int i = 1; i <= tri->NbNodes(); ++i) {
    const gp_Pnt2d uv = tri->UVNode(i);
    append(out, uv.X());
    append(out, uv.Y());
  }
  for (int i = 1; i <= tri->NbTriangles(); ++i) {
    int a, b, c;
    tri->Triangle(i).Get(a, b, c);
    if (reversed) std::swap(b, c);
    append<uint32_t>(out, static_cast<uint32_t>(a - 1));
    append<uint32_t>(out, static_cast<uint32_t>(b - 1));
    append<uint32_t>(out, static_cast<uint32_t>(c - 1));
  }
}

// An edge's points from the triangulation of a face it bounds, so that they are mesh
// vertices, or else from its own polygon when it bounds no face.
void append_polyline(Bytes& out, const TopoDS_Edge& edge,
                     const NCollection_List<TopoDS_Shape>& faces) {
  for (NCollection_List<TopoDS_Shape>::Iterator it(faces); it.More(); it.Next()) {
    TopLoc_Location location;
    const occ::handle<Poly_Triangulation> tri =
        BRep_Tool::Triangulation(TopoDS::Face(it.Value()), location);
    if (tri.IsNull()) continue;
    const occ::handle<Poly_PolygonOnTriangulation> polygon =
        BRep_Tool::PolygonOnTriangulation(edge, tri, location);
    if (polygon.IsNull()) continue;
    const gp_Trsf trsf = location.Transformation();
    append<uint32_t>(out, static_cast<uint32_t>(polygon->NbNodes()));
    for (int i = 1; i <= polygon->NbNodes(); ++i) {
      append_point(out, tri->Node(polygon->Node(i)).Transformed(trsf));
    }
    return;
  }

  TopLoc_Location location;
  const occ::handle<Poly_Polygon3D> polygon = BRep_Tool::Polygon3D(edge, location);
  if (!polygon.IsNull()) {
    const gp_Trsf trsf = location.Transformation();
    const NCollection_Array1<gp_Pnt>& nodes = polygon->Nodes();
    append<uint32_t>(out, static_cast<uint32_t>(nodes.Length()));
    for (int i = nodes.Lower(); i <= nodes.Upper(); ++i) {
      append_point(out, nodes(i).Transformed(trsf));
    }
    return;
  }

  // A degenerate edge is one point, and the mesher doesn't always store a polygon for it.
  TopoDS_Vertex first, last;
  TopExp::Vertices(edge, first, last);
  if (!BRep_Tool::Degenerated(edge) || first.IsNull() || last.IsNull()) {
    throw std::runtime_error("an edge has no polyline");
  }
  append<uint32_t>(out, 2);
  append_point(out, BRep_Tool::Pnt(first));
  append_point(out, BRep_Tool::Pnt(last));
}

// Layout of one kind of drawing line. Each edge of `lines` becomes:
//   u32 kind, u32 visible, u32 point count, then count * 2 f64
void append_drawing_lines(Bytes& out, uint32_t& count, TopoDS_Shape lines, uint32_t kind,
                          bool visible, double deflection) {
  if (lines.IsNull()) return;
  // The drawing's edges exist only as curves on the projection plane until this.
  BRepLib::BuildCurves3d(lines);
  for (TopExp_Explorer it(lines, TopAbs_EDGE); it.More(); it.Next()) {
    const TopoDS_Edge edge = TopoDS::Edge(it.Current());
    const BRepAdaptor_Curve curve(edge);
    append<uint32_t>(out, kind);
    append<uint32_t>(out, visible ? 1u : 0u);
    if (curve.GetType() == GeomAbs_Line) {
      const gp_Pnt a = curve.Value(curve.FirstParameter());
      const gp_Pnt b = curve.Value(curve.LastParameter());
      append<uint32_t>(out, 2);
      for (const gp_Pnt& p : {a, b}) {
        append(out, p.X());
        append(out, p.Y());
      }
    } else {
      // The angle only keeps coarse deflections from flattening a curve to a chord.
      const GCPnts_TangentialDeflection points(curve, /*Angular=*/0.5, deflection,
                                               /*MinimumOfPoints=*/2);
      if (points.NbPoints() < 2) throw std::runtime_error("a drawing line has no points");
      append<uint32_t>(out, static_cast<uint32_t>(points.NbPoints()));
      for (int i = 1; i <= points.NbPoints(); ++i) {
        const gp_Pnt p = points.Value(i);
        append(out, p.X());
        append(out, p.Y());
      }
    }
    ++count;
  }
}

}  // namespace

// Buffer of 25 doubles: mass, centre (3), inertia by rows (9), principal moments (3),
// principal axes (9).
API uint32_t occt_mass_properties(uint32_t h) {
  return guard_handle([&]() -> uint32_t {
    TopoDS_Shape shape = get_shape(h).Oriented(TopAbs_FORWARD);
    GProp_GProps props = measure(shape);
    if (props.Mass() < 0.0) props = measure(shape.Reversed());  // An inside-out shell.
    if (!(props.Mass() > 0.0) || !std::isfinite(props.Mass())) {
      throw std::runtime_error("the shape has no volume, area or length to measure");
    }
    Bytes out;
    append(out, props.Mass());
    append_point(out, props.CentreOfMass());
    const gp_Mat inertia = props.MatrixOfInertia();
    for (int row = 1; row <= 3; ++row) {
      for (int column = 1; column <= 3; ++column) append(out, inertia.Value(row, column));
    }
    const GProp_PrincipalProps principal = props.PrincipalProperties();
    double moments[3];
    principal.Moments(moments[0], moments[1], moments[2]);
    for (const double moment : moments) append(out, moment);
    for (const gp_Vec& axis : {principal.FirstAxisOfInertia(), principal.SecondAxisOfInertia(),
                               principal.ThirdAxisOfInertia()}) {
      append(out, axis.X());
      append(out, axis.Y());
      append(out, axis.Z());
    }
    return put_buffer(std::move(out));
  });
}

// Buffer of 7 doubles: distance, point on a, point on b.
API uint32_t occt_distance(uint32_t a, uint32_t b) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape first = get_shape(a), second = get_shape(b);
    gp_Pnt inner;
    double value = 0.0;
    gp_Pnt on_a, on_b;
    if (vertex_inside_group(first, second, inner) || vertex_inside_group(second, first, inner)) {
      on_a = on_b = inner;
    } else {
      BRepExtrema_DistShapeShape distance(first, second);
      if (!distance.IsDone() || distance.NbSolution() < 1) {
        throw std::runtime_error("no closest points were found");
      }
      value = distance.Value();
      on_a = distance.PointOnShape1(1);
      on_b = distance.PointOnShape2(1);
      if (distance.InnerSolution()) {
        value = 0.0;
      } else if (std::abs(on_a.Distance(on_b) - value) > 1e-6 * (1.0 + value)) {
        throw std::runtime_error("the closest points disagree with the distance");
      }
    }
    Bytes out;
    append(out, value);
    append_point(out, on_a);
    append_point(out, on_b);
    return put_buffer(std::move(out));
  });
}

// 0 inside, 1 on the boundary, 2 outside, -1 on error.
API int32_t occt_locate_point(uint32_t h, double x, double y, double z, double tolerance) {
  return guard<int32_t>(-1, [&]() -> int32_t {
    const TopoDS_Shape shape = get_shape(h);
    if (shape.ShapeType() != TopAbs_SOLID) throw std::invalid_argument("the shape is not a solid");
    // A reversed solid still occupies the same space.
    BRepClass3d_SolidClassifier classifier(shape.Oriented(TopAbs_FORWARD));
    classifier.Perform(gp_Pnt(x, y, z), tolerance);
    switch (classifier.State()) {
      case TopAbs_IN:
        return 0;
      case TopAbs_ON:
        return 1;
      case TopAbs_OUT:
        return 2;
      default:
        throw std::runtime_error("the point could not be classified");
    }
  });
}

// `max_distance` below zero means no limit. Buffer: u32 hit count, then per hit:
//   u32 face handle, f64 distance, 3 f64 point, 2 f64 surface parameters
API uint32_t occt_ray_cast(uint32_t h, double ox, double oy, double oz, double dx, double dy,
                           double dz, double max_distance) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    const gp_Pnt origin(ox, oy, oz);
    const gp_Lin line(origin, gp_Dir(dx, dy, dz));
    std::vector<Hit> hits;
    Bnd_Box box;
    BRepBndLib::Add(shape, box);
    if (!box.IsVoid()) {
      // Intersectors work better on a segment than on an infinite line, and a ray can't
      // meet the shape past the far side of its box.
      const double reach = origin.Distance(box.CornerMin()) + origin.Distance(box.CornerMax()) +
                           box.CornerMin().Distance(box.CornerMax());
      const double limit = max_distance < 0.0 ? reach : std::min(max_distance, reach);
      ShapeMap faces;
      TopExp::MapShapes(shape, TopAbs_FACE, faces);
      for (int f = 1; f <= faces.Extent(); ++f) {
        IntCurvesFace_Intersector intersector(TopoDS::Face(faces(f)), Precision::Confusion());
        intersector.Perform(line, 0.0, limit);
        if (!intersector.IsDone()) throw std::runtime_error("intersecting a face failed");
        const size_t first_of_face = hits.size();
        for (int i = 1; i <= intersector.NbPnt(); ++i) {
          const double w = intersector.WParameter(i);
          const bool again = std::any_of(
              hits.begin() + static_cast<std::ptrdiff_t>(first_of_face), hits.end(),
              [&](const Hit& hit) { return std::abs(hit.distance - w) <= kSameHit; });
          if (!again) {
            hits.push_back(Hit{static_cast<uint32_t>(f), w, intersector.Pnt(i),
                               intersector.UParameter(i), intersector.VParameter(i)});
          }
        }
      }
      std::stable_sort(hits.begin(), hits.end(),
                       [](const Hit& a, const Hit& b) { return a.distance < b.distance; });
      Bytes out;
      append<uint32_t>(out, static_cast<uint32_t>(hits.size()));
      for (const Hit& hit : hits) {
        append<uint32_t>(out, put_shape(faces(static_cast<int>(hit.face))));
        append(out, hit.distance);
        append_point(out, hit.point);
        append(out, hit.u);
        append(out, hit.v);
      }
      return put_buffer(std::move(out));
    }
    Bytes out;
    append<uint32_t>(out, 0);
    return put_buffer(std::move(out));
  });
}

// Buffer of uint32 handles.
API uint32_t occt_ancestors(uint32_t root, uint32_t sub, int32_t kind) {
  return guard_handle([&]() -> uint32_t {
    if (kind < TopAbs_COMPOUND || kind > TopAbs_VERTEX) {
      throw std::invalid_argument("invalid shape kind");
    }
    const TopoDS_Shape shape = get_shape(root), part = get_shape(sub);
    ShapeMap own;
    TopExp::MapShapes(shape, part.ShapeType(), own);
    if (!own.Contains(part)) throw std::invalid_argument("the subshape is not part of the shape");
    AncestorMap ancestors;
    TopExp::MapShapesAndUniqueAncestors(shape, part.ShapeType(),
                                        static_cast<TopAbs_ShapeEnum>(kind), ancestors);
    Bytes out;
    if (const NCollection_List<TopoDS_Shape>* found = ancestors.Seek(part)) {
      for (NCollection_List<TopoDS_Shape>::Iterator it(*found); it.More(); it.Next()) {
        append<uint32_t>(out, put_shape(it.Value()));
      }
    }
    return put_buffer(std::move(out));
  });
}

// 0 convex, 1 concave, 2 tangent, 3 open, -1 on error.
API int32_t occt_edge_convexity(uint32_t root, uint32_t e) {
  return guard<int32_t>(-1, [&]() -> int32_t {
    const TopoDS_Shape shape = get_shape(root);
    const TopoDS_Edge edge = TopoDS::Edge(get_shape(e));
    AncestorMap ancestors;
    TopExp::MapShapesAndUniqueAncestors(shape, TopAbs_EDGE, TopAbs_FACE, ancestors);
    const NCollection_List<TopoDS_Shape>* found = ancestors.Seek(edge);
    if (found == nullptr) throw std::invalid_argument("the edge is not part of the shape");
    if (found->Size() > 2) throw std::runtime_error("the edge borders more than two faces");
    if (found->Size() < 2 || BRep_Tool::Degenerated(edge)) return 3;
    const TopoDS_Face f1 = TopoDS::Face(found->First());
    const TopoDS_Face f2 = TopoDS::Face(found->Last());

    // Several places along the edge, since faces can meet differently along it. Interior
    // ones only: the ends can be where a surface is singular.
    const BRepAdaptor_Curve curve(edge);
    const double first = curve.FirstParameter(), last = curve.LastParameter();
    int convex = 0, concave = 0, tangent = 0;
    for (const double fraction : {0.1, 0.3, 0.5, 0.7, 0.9}) {
      const double t = first + fraction * (last - first);
      gp_Dir n1, n2, along, ignored;
      if (!frame_on_face(f1, edge, t, n1, along) || !frame_on_face(f2, edge, t, n2, ignored)) {
        continue;
      }
      // With the edge running as the first face uses it, the normals turn around it by
      // this signed angle, positive where the material angle is under 180 degrees.
      const double sine = n1.XYZ().Crossed(n2.XYZ()).Dot(along.XYZ());
      const double cosine = n1.Dot(n2);
      if (cosine < 0.0 && std::abs(sine) < kTangentAngle) {
        throw std::runtime_error("the faces fold back on each other at the edge");
      }
      const double angle = std::atan2(sine, cosine);
      if (std::abs(angle) < kTangentAngle) {
        ++tangent;
      } else if (angle > 0.0) {
        ++convex;
      } else {
        ++concave;
      }
    }
    if (convex + concave + tangent == 0) {
      throw std::runtime_error("the faces have no normal along the edge");
    }
    if (convex > 0 && concave > 0) {
      throw std::runtime_error("the edge is convex in places and concave in others");
    }
    if (convex > 0) return 0;
    if (concave > 0) return 1;
    return 2;
  });
}

// Buffer of uint32 handles, the outer wire first.
API uint32_t occt_face_wires(uint32_t h) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Face face = TopoDS::Face(get_shape(h));
    const TopoDS_Wire outer = BRepTools::OuterWire(face);
    Bytes out;
    if (!outer.IsNull()) append<uint32_t>(out, put_shape(outer));
    for (TopExp_Explorer it(face, TopAbs_WIRE); it.More(); it.Next()) {
      if (!it.Current().IsSame(outer)) append<uint32_t>(out, put_shape(it.Current()));
    }
    return put_buffer(std::move(out));
  });
}

// Buffer of uint32 handles in connection order. An edge a wire uses twice, as a seam is
// by the wire around a cylinder's side, appears twice.
API uint32_t occt_wire_edges(uint32_t h) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Wire wire = TopoDS::Wire(get_shape(h));
    std::vector<TopoDS_Edge> edges;
    for (TopoDS_Iterator it(wire); it.More(); it.Next()) {
      if (it.Value().ShapeType() == TopAbs_EDGE) edges.push_back(TopoDS::Edge(it.Value()));
    }
    Bytes out;
    if (edges.empty()) return put_buffer(std::move(out));
    std::vector<TopoDS_Edge> ordered = connected_edges(edges);
    if (ordered.empty()) {
      // Edges whose stored orientations disagree have no directed trail, but the
      // explorer also accepts them joined either way round.
      for (BRepTools_WireExplorer it(wire); it.More(); it.Next()) ordered.push_back(it.Current());
      if (ordered.size() != edges.size()) {
        throw std::runtime_error("the wire's edges don't connect into one chain");
      }
    }
    for (const TopoDS_Edge& edge : ordered) append<uint32_t>(out, put_shape(edge));
    return put_buffer(std::move(out));
  });
}

// `cancel` is the address of a flag byte, or 0. Buffer:
//   u32 face count, then per face, in subshape order:
//     u32 vertices, u32 triangles, vertices * 3 f64 positions, vertices * 3 f64 normals,
//     vertices * 2 f64 surface parameters, triangles * 3 u32 indices
//   u32 edge count, then per edge, in subshape order:
//     u32 points, points * 3 f64
API uint32_t occt_tessellate(uint32_t h, double linear_deflection, double angular_deflection,
                             int32_t relative, size_t cancel) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    const occ::handle<FlagProgress> progress = new FlagProgress(cancel);
    IMeshTools_Parameters parameters;
    parameters.Deflection = linear_deflection;
    parameters.Angle = angular_deflection;
    parameters.Relative = relative != 0;
    parameters.InParallel = false;
    AncestorMap ancestors;
    TopExp::MapShapesAndUniqueAncestors(shape, TopAbs_EDGE, TopAbs_FACE, ancestors);

    const auto run_mesher = [&]() {
      const BRepMesh_IncrementalMesh mesher(shape, parameters, progress->Start());
      // Checked however the mesher ended, so a flag set just as it finished still gets the
      // answer the caller asked for. A stopped run leaves faces with triangles but edges
      // without polygons, which the next run would take as done, so it's undone here.
      if (progress->UserBreak()) {
        BRepTools::Clean(shape);
        throw std::runtime_error("cancelled");
      }
      if (!mesher.IsDone()) throw std::runtime_error("meshing failed");
      if ((mesher.GetStatusFlags() & IMeshData_Failure) != 0) {
        throw std::runtime_error("meshing failed on a face");
      }
    };
    run_mesher();
    // The mesher reuses a triangulation that's fine by its own standards, which needn't
    // have what this answer needs, if it was made another way.
    if (!fully_meshed(shape, ancestors)) {
      BRepTools::Clean(shape);
      run_mesher();
    }

    Bytes out;
    ShapeMap faces;
    TopExp::MapShapes(shape, TopAbs_FACE, faces);
    append<uint32_t>(out, static_cast<uint32_t>(faces.Extent()));
    for (int i = 1; i <= faces.Extent(); ++i) append_mesh(out, TopoDS::Face(faces(i)));

    ShapeMap edges;
    TopExp::MapShapes(shape, TopAbs_EDGE, edges);
    append<uint32_t>(out, static_cast<uint32_t>(edges.Extent()));
    const NCollection_List<TopoDS_Shape> no_faces;
    for (int i = 1; i <= edges.Extent(); ++i) {
      const NCollection_List<TopoDS_Shape>* found = ancestors.Seek(edges(i));
      append_polyline(out, TopoDS::Edge(edges(i)), found != nullptr ? *found : no_faces);
    }
    return put_buffer(std::move(out));
  });
}

// `view` is frame origin, frame z, frame x (9 f64), then perspective distance, or below
// zero for parallel, then deflection. Buffer: u32 line count, then the lines as
// append_drawing_lines writes them. Kinds: 0 sharp, 1 smooth, 2 outline.
API uint32_t occt_hidden_lines(uint32_t h, const void* view, size_t len) {
  return guard_handle([&]() -> uint32_t {
    silence_messages();
    Reader reader(view, len);
    const gp_Pnt origin = reader.point();
    const gp_Dir z = reader.dir();
    const gp_Dir x = reader.dir();
    const double perspective = reader.f64();
    const double deflection = reader.f64();
    const gp_Ax2 frame(origin, z, x);
    const HLRAlgo_Projector projector =
        perspective < 0.0 ? HLRAlgo_Projector(frame) : HLRAlgo_Projector(frame, perspective);

    // Which edges are smooth is a stored attribute that most builders leave unset. It
    // is worked out on a copy, as this is a query and the input stays as it was.
    BRepBuilderAPI_Copy copy(get_shape(h), /*copyGeom=*/false, /*copyMesh=*/false);
    if (!copy.IsDone()) throw std::runtime_error("copying the shape failed");
    const TopoDS_Shape shape = copy.Shape();
    BRepLib::EncodeRegularity(shape, kTangentAngle);

    const occ::handle<HLRBRep_Algo> algo = new HLRBRep_Algo();
    algo->Add(shape);
    algo->Projector(projector);
    algo->Update();
    algo->Hide();
    HLRBRep_HLRToShape lines(algo);

    // The extractor's "sewn" lines are left out: those are seams, which exist only because
    // of how a closed surface is parameterised, and aren't features of the part.
    Bytes body;
    uint32_t count = 0;
    append_drawing_lines(body, count, lines.VCompound(), 0, true, deflection);
    append_drawing_lines(body, count, lines.HCompound(), 0, false, deflection);
    append_drawing_lines(body, count, lines.Rg1LineVCompound(), 1, true, deflection);
    append_drawing_lines(body, count, lines.Rg1LineHCompound(), 1, false, deflection);
    append_drawing_lines(body, count, lines.OutLineVCompound(), 2, true, deflection);
    append_drawing_lines(body, count, lines.OutLineHCompound(), 2, false, deflection);

    Bytes out;
    append<uint32_t>(out, count);
    out.insert(out.end(), body.begin(), body.end());
    return put_buffer(std::move(out));
  });
}
