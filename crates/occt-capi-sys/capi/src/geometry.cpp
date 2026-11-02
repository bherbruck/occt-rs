// Entry points of the geometry domain: building topology from geometry, and reading exact
// geometry back. SPDX-License-Identifier: MIT OR Apache-2.0
#include "internal.hpp"

#include <BRepAdaptor_Curve.hxx>
#include <BRepBuilderAPI_FindPlane.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepCheck_Shell.hxx>
#include <BRepCheck_Status.hxx>
#include <BRepLProp_SLProps.hxx>
#include <BRepLib.hxx>
#include <BRepOffsetAPI_MakeFilling.hxx>
#include <BRepOffsetAPI_MakeOffset.hxx>
#include <BRepPrimAPI_MakeTorus.hxx>
#include <BRepTools.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <BRep_Builder.hxx>
#include <Geom2d_Line.hxx>
#include <GeomAPI_Interpolate.hxx>
#include <GeomAPI_PointsToBSpline.hxx>
#include <GeomAPI_ProjectPointOnCurve.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <GeomConvert.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_BezierCurve.hxx>
#include <Geom_BezierSurface.hxx>
#include <Geom_Circle.hxx>
#include <Geom_ConicalSurface.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_Ellipse.hxx>
#include <Geom_Line.hxx>
#include <Geom_Plane.hxx>
#include <Geom_RectangularTrimmedSurface.hxx>
#include <Geom_SphericalSurface.hxx>
#include <Geom_ToroidalSurface.hxx>
#include <Geom_TrimmedCurve.hxx>
#include <NCollection_Array1.hxx>
#include <NCollection_Array2.hxx>
#include <NCollection_HArray1.hxx>
#include <Precision.hxx>
#include <ShapeAnalysis_Surface.hxx>
#include <ShapeFix_Face.hxx>
#include <ShapeFix_Shell.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Solid.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Ax2.hxx>
#include <gp_Ax3.hxx>
#include <gp_Circ.hxx>
#include <gp_Dir2d.hxx>
#include <gp_Elips.hxx>
#include <gp_Pnt2d.hxx>

#include <algorithm>
#include <limits>

using namespace occt_capi;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// How far a wire may stray from the surface a face is built on. Wires commonly come from
// approximation, and fitting tolerances are around 1e-4 of model size; beyond this the
// face's boundary is not on its surface in any useful sense.
constexpr double kOnSurface = 1e-4;

// The byte formats, decoded and encoded in occt-kernel-capi/src/geometry.rs.
//
// Frame: origin, z, x, 9 doubles.
// Curve: u32 tag, then
//   0 other: nothing
//   1 line: origin, direction
//   2 circle: frame, radius
//   3 ellipse: frame, major, minor
//   4 spline: u32 degree, u32 flags (1 periodic, 2 rational), u32 knots, u32 poles,
//             knots as doubles, multiplicities as u32, poles, weights if rational
// Surface: u32 tag, then
//   0 other: nothing
//   1 plane: frame
//   2 cylinder: frame, radius
//   3 cone: frame, radius, half angle
//   4 sphere: frame, radius
//   5 torus: frame, major, minor
//   6 spline: u32 degree u, degree v, flags (1 periodic u, 2 periodic v, 4 rational),
//             knots u, knots v, poles u, poles v, then knots u, multiplicities u,
//             knots v, multiplicities v, poles with U varying slowest, weights if rational

// --- decoding ------------------------------------------------------------------------

gp_Ax3 read_frame(Reader& r) {
  const gp_Pnt origin = r.point();
  const gp_Dir z = r.dir();
  const gp_Dir x = r.dir();
  return gp_Ax3(origin, z, x);
}

// Bounds a count from the input by the bytes that are left, so that a corrupt count can't
// ask for a huge allocation.
int read_count(Reader& r, size_t item_size) {
  const uint32_t n = r.u32();
  if (static_cast<uint64_t>(n) * item_size > r.remaining()) {
    throw std::invalid_argument("truncated input");
  }
  return static_cast<int>(n);
}

NCollection_Array1<double> read_doubles(Reader& r, int n) {
  NCollection_Array1<double> out(1, std::max(n, 1));
  for (int i = 1; i <= n; ++i) out(i) = r.f64();
  return out;
}

NCollection_Array1<int> read_ints(Reader& r, int n) {
  NCollection_Array1<int> out(1, std::max(n, 1));
  for (int i = 1; i <= n; ++i) out(i) = static_cast<int>(r.u32());
  return out;
}

occ::handle<Geom_Curve> read_curve(Reader& r) {
  switch (r.u32()) {
    case 1: {
      const gp_Pnt origin = r.point();
      const gp_Dir direction = r.dir();
      return new Geom_Line(origin, direction);
    }
    case 2: {
      const gp_Ax3 frame = read_frame(r);
      return new Geom_Circle(frame.Ax2(), r.f64());
    }
    case 3: {
      const gp_Ax3 frame = read_frame(r);
      const double major = r.f64();
      return new Geom_Ellipse(frame.Ax2(), major, r.f64());
    }
    case 4: {
      const int degree = static_cast<int>(r.u32());
      const uint32_t flags = r.u32();
      const int knots_n = read_count(r, sizeof(double) + sizeof(uint32_t));
      const int poles_n = read_count(r, 3 * sizeof(double));
      if (knots_n < 2 || poles_n < 2) throw std::invalid_argument("too few knots or poles");
      const NCollection_Array1<double> knots = read_doubles(r, knots_n);
      const NCollection_Array1<int> mults = read_ints(r, knots_n);
      NCollection_Array1<gp_Pnt> poles(1, poles_n);
      for (int i = 1; i <= poles_n; ++i) poles(i) = r.point();
      const bool periodic = (flags & 1) != 0;
      if ((flags & 2) != 0) {
        const NCollection_Array1<double> weights = read_doubles(r, poles_n);
        return new Geom_BSplineCurve(poles, weights, knots, mults, degree, periodic);
      }
      return new Geom_BSplineCurve(poles, knots, mults, degree, periodic);
    }
    default:
      throw std::invalid_argument("this curve can't be built: it has no description");
  }
}

occ::handle<Geom_Surface> read_surface(Reader& r) {
  switch (r.u32()) {
    case 1:
      return new Geom_Plane(read_frame(r));
    case 2: {
      const gp_Ax3 frame = read_frame(r);
      return new Geom_CylindricalSurface(frame, r.f64());
    }
    case 3: {
      const gp_Ax3 frame = read_frame(r);
      const double radius = r.f64();
      return new Geom_ConicalSurface(frame, r.f64(), radius);
    }
    case 4: {
      const gp_Ax3 frame = read_frame(r);
      return new Geom_SphericalSurface(frame, r.f64());
    }
    case 5: {
      const gp_Ax3 frame = read_frame(r);
      const double major = r.f64();
      return new Geom_ToroidalSurface(frame, major, r.f64());
    }
    case 6: {
      const int degree_u = static_cast<int>(r.u32());
      const int degree_v = static_cast<int>(r.u32());
      const uint32_t flags = r.u32();
      const int knots_u_n = read_count(r, sizeof(double) + sizeof(uint32_t));
      const int knots_v_n = read_count(r, sizeof(double) + sizeof(uint32_t));
      const uint32_t poles_u_n = r.u32();
      const uint32_t poles_v_n = r.u32();
      if (static_cast<uint64_t>(poles_u_n) * poles_v_n * 3 * sizeof(double) > r.remaining()) {
        throw std::invalid_argument("truncated input");
      }
      if (knots_u_n < 2 || knots_v_n < 2 || poles_u_n < 2 || poles_v_n < 2) {
        throw std::invalid_argument("too few knots or poles");
      }
      const NCollection_Array1<double> knots_u = read_doubles(r, knots_u_n);
      const NCollection_Array1<int> mults_u = read_ints(r, knots_u_n);
      const NCollection_Array1<double> knots_v = read_doubles(r, knots_v_n);
      const NCollection_Array1<int> mults_v = read_ints(r, knots_v_n);
      const int nu = static_cast<int>(poles_u_n), nv = static_cast<int>(poles_v_n);
      NCollection_Array2<gp_Pnt> poles(1, nu, 1, nv);
      for (int i = 1; i <= nu; ++i) {
        for (int j = 1; j <= nv; ++j) poles(i, j) = r.point();
      }
      const bool periodic_u = (flags & 1) != 0, periodic_v = (flags & 2) != 0;
      if ((flags & 4) != 0) {
        NCollection_Array2<double> weights(1, nu, 1, nv);
        for (int i = 1; i <= nu; ++i) {
          for (int j = 1; j <= nv; ++j) weights(i, j) = r.f64();
        }
        return new Geom_BSplineSurface(poles, weights, knots_u, knots_v, mults_u, mults_v, degree_u,
                                       degree_v, periodic_u, periodic_v);
      }
      return new Geom_BSplineSurface(poles, knots_u, knots_v, mults_u, mults_v, degree_u, degree_v,
                                     periodic_u, periodic_v);
    }
    default:
      throw std::invalid_argument("this surface can't be built: it has no description");
  }
}

// --- encoding ------------------------------------------------------------------------

void append_dir(Bytes& out, const gp_Dir& d) {
  append(out, d.X());
  append(out, d.Y());
  append(out, d.Z());
}

void append_frame(Bytes& out, const gp_Pnt& origin, const gp_Dir& z, const gp_Dir& x) {
  append_point(out, origin);
  append_dir(out, z);
  append_dir(out, x);
}

void append_frame(Bytes& out, const gp_Ax2& frame) {
  append_frame(out, frame.Location(), frame.Direction(), frame.XDirection());
}

template <typename T>
void append_array(Bytes& out, const NCollection_Array1<T>& values) {
  for (int i = values.Lower(); i <= values.Upper(); ++i) append(out, values(i));
}

void append_ints(Bytes& out, const NCollection_Array1<int>& values) {
  for (int i = values.Lower(); i <= values.Upper(); ++i) {
    append<uint32_t>(out, static_cast<uint32_t>(values(i)));
  }
}

occ::handle<Geom_Curve> basis_curve(occ::handle<Geom_Curve> curve) {
  while (const auto trimmed = occ::down_cast<Geom_TrimmedCurve>(curve)) {
    curve = trimmed->BasisCurve();
  }
  return curve;
}

occ::handle<Geom_Surface> basis_surface(occ::handle<Geom_Surface> surface) {
  while (const auto trimmed = occ::down_cast<Geom_RectangularTrimmedSurface>(surface)) {
    surface = trimmed->BasisSurface();
  }
  return surface;
}

void append_bspline(Bytes& out, const occ::handle<Geom_BSplineCurve>& s) {
  append<uint32_t>(out, 4);
  append<uint32_t>(out, static_cast<uint32_t>(s->Degree()));
  append<uint32_t>(out, (s->IsPeriodic() ? 1u : 0u) | (s->IsRational() ? 2u : 0u));
  append<uint32_t>(out, static_cast<uint32_t>(s->NbKnots()));
  append<uint32_t>(out, static_cast<uint32_t>(s->NbPoles()));
  append_array(out, s->Knots());
  append_ints(out, s->Multiplicities());
  const NCollection_Array1<gp_Pnt>& poles = s->Poles();
  for (int i = poles.Lower(); i <= poles.Upper(); ++i) append_point(out, poles(i));
  if (s->IsRational()) {
    for (int i = 1; i <= s->NbPoles(); ++i) append(out, s->Weight(i));
  }
}

// Writes a curve that is not trimmed; see basis_curve.
void append_curve(Bytes& out, const occ::handle<Geom_Curve>& curve) {
  if (const auto line = occ::down_cast<Geom_Line>(curve)) {
    const gp_Lin lin = line->Lin();
    append<uint32_t>(out, 1);
    append_point(out, lin.Location());
    append_dir(out, lin.Direction());
  } else if (const auto circle = occ::down_cast<Geom_Circle>(curve)) {
    append<uint32_t>(out, 2);
    append_frame(out, circle->Position());
    append(out, circle->Radius());
  } else if (const auto ellipse = occ::down_cast<Geom_Ellipse>(curve)) {
    append<uint32_t>(out, 3);
    append_frame(out, ellipse->Position());
    append(out, ellipse->MajorRadius());
    append(out, ellipse->MinorRadius());
  } else if (const auto spline = occ::down_cast<Geom_BSplineCurve>(curve)) {
    append_bspline(out, spline);
  } else if (occ::down_cast<Geom_BezierCurve>(curve)) {
    append_bspline(out, GeomConvert::CurveToBSplineCurve(curve));
  } else {
    append<uint32_t>(out, 0);
  }
}

void append_bspline(Bytes& out, const occ::handle<Geom_BSplineSurface>& s) {
  const bool rational = s->IsURational() || s->IsVRational();
  append<uint32_t>(out, 6);
  append<uint32_t>(out, static_cast<uint32_t>(s->UDegree()));
  append<uint32_t>(out, static_cast<uint32_t>(s->VDegree()));
  append<uint32_t>(out, (s->IsUPeriodic() ? 1u : 0u) | (s->IsVPeriodic() ? 2u : 0u) |
                            (rational ? 4u : 0u));
  append<uint32_t>(out, static_cast<uint32_t>(s->NbUKnots()));
  append<uint32_t>(out, static_cast<uint32_t>(s->NbVKnots()));
  append<uint32_t>(out, static_cast<uint32_t>(s->NbUPoles()));
  append<uint32_t>(out, static_cast<uint32_t>(s->NbVPoles()));
  append_array(out, s->UKnots());
  append_ints(out, s->UMultiplicities());
  append_array(out, s->VKnots());
  append_ints(out, s->VMultiplicities());
  for (int i = 1; i <= s->NbUPoles(); ++i) {
    for (int j = 1; j <= s->NbVPoles(); ++j) append_point(out, s->Pole(i, j));
  }
  if (rational) {
    for (int i = 1; i <= s->NbUPoles(); ++i) {
      for (int j = 1; j <= s->NbVPoles(); ++j) append(out, s->Weight(i, j));
    }
  }
}

// An analytic surface with a left-handed placement, as mirroring produces, has no
// description as a right-handed frame with the same parameters. With Y negated it does,
// and U runs the other way. Reading code presents such a surface in the flipped form
// throughout, so that what it reports is self-consistent. A plane needs no flip: its
// parameters are unaffected if the frame's Z is taken as X x Y.
bool flips_u(const occ::handle<Geom_Surface>& surface) {
  if (occ::down_cast<Geom_Plane>(surface)) return false;
  const auto elementary = occ::down_cast<Geom_ElementarySurface>(surface);
  return !elementary.IsNull() && !elementary->Position().Direct();
}

// A plane keeps X and Y and takes the Z they imply. A surface with its U flipped keeps Z and
// X, which imply the negated Y it is described with.
void append_frame(Bytes& out, const gp_Ax3& frame) {
  const gp_Dir x = frame.XDirection();
  append_frame(out, frame.Location(), frame.Direction(), x);
}

void append_plane_frame(Bytes& out, const gp_Ax3& frame) {
  const gp_Dir x = frame.XDirection();
  append_frame(out, frame.Location(), x.Crossed(frame.YDirection()), x);
}

// Writes a surface that is not trimmed; see basis_surface.
void append_surface(Bytes& out, const occ::handle<Geom_Surface>& surface) {
  if (const auto plane = occ::down_cast<Geom_Plane>(surface)) {
    append<uint32_t>(out, 1);
    append_plane_frame(out, plane->Position());
  } else if (const auto cylinder = occ::down_cast<Geom_CylindricalSurface>(surface)) {
    append<uint32_t>(out, 2);
    append_frame(out, cylinder->Position());
    append(out, cylinder->Radius());
  } else if (const auto cone = occ::down_cast<Geom_ConicalSurface>(surface)) {
    append<uint32_t>(out, 3);
    append_frame(out, cone->Position());
    append(out, cone->RefRadius());
    append(out, cone->SemiAngle());
  } else if (const auto sphere = occ::down_cast<Geom_SphericalSurface>(surface)) {
    append<uint32_t>(out, 4);
    append_frame(out, sphere->Position());
    append(out, sphere->Radius());
  } else if (const auto torus = occ::down_cast<Geom_ToroidalSurface>(surface)) {
    append<uint32_t>(out, 5);
    append_frame(out, torus->Position());
    append(out, torus->MajorRadius());
    append(out, torus->MinorRadius());
  } else if (const auto spline = occ::down_cast<Geom_BSplineSurface>(surface)) {
    append_bspline(out, spline);
  } else if (occ::down_cast<Geom_BezierSurface>(surface)) {
    append_bspline(out, GeomConvert::SurfaceToBSplineSurface(surface));
  } else {
    append<uint32_t>(out, 0);
  }
}

// --- checks ----------------------------------------------------------------------------

double shape_area(const TopoDS_Shape& shape) {
  GProp_GProps props;
  BRepGProp::SurfaceProperties(shape, props);
  return props.Mass();
}

// The curve parameters at n evenly spaced places along an edge, ends included.
template <typename F>
void sample_edge(const TopoDS_Edge& edge, int n, F&& visit) {
  if (BRep_Tool::Degenerated(edge)) return;
  const BRepAdaptor_Curve curve(edge);
  const double first = curve.FirstParameter(), last = curve.LastParameter();
  for (int i = 0; i <= n; ++i) visit(curve.Value(first + (last - first) * i / n));
}

template <typename F>
void sample_edges(const TopoDS_Shape& shape, int n, F&& visit) {
  ShapeMap edges;
  TopExp::MapShapes(shape, TopAbs_EDGE, edges);
  for (int i = 1; i <= edges.Extent(); ++i) sample_edge(TopoDS::Edge(edges(i)), n, visit);
}

void check_wire_on_surface(const TopoDS_Shape& wire, const occ::handle<Geom_Surface>& surface) {
  sample_edges(wire, 8, [&](const gp_Pnt& p) {
    GeomAPI_ProjectPointOnSurf projection(p, surface);
    if (projection.NbPoints() == 0 || projection.LowerDistance() > kOnSurface) {
      throw std::invalid_argument("a wire does not lie on the surface");
    }
  });
}

// The wire with what its edges lack on the surface added, turned to run as an outer boundary.
// The orientation is settled alone: on a closed surface the fix can't tell a hole from a
// second region, and leaves wires as they come.
TopoDS_Wire oriented_on_surface(const occ::handle<Geom_Surface>& surface, const TopoDS_Wire& wire) {
  check_wire_on_surface(wire, surface);
  BRep_Builder builder;
  TopoDS_Face face;
  builder.MakeFace(face, surface, Precision::Confusion());
  builder.Add(face, wire);
  ShapeFix_Face fix(face);
  fix.FixAddNaturalBoundMode() = 0;
  fix.Perform();
  const TopoDS_Shape result = fix.Result();
  if (result.IsNull() || result.ShapeType() != TopAbs_FACE) {
    throw std::runtime_error("a boundary does not make a single face");
  }
  const TopoDS_Wire oriented = BRepTools::OuterWire(TopoDS::Face(result));
  if (oriented.IsNull()) throw std::runtime_error("a boundary does not make a face");
  return oriented;
}

TopoDS_Wire closed_wire(const TopoDS_Shape& shape, const char* what) {
  if (shape.ShapeType() != TopAbs_WIRE) {
    throw std::invalid_argument(std::string(what) + " is not a wire");
  }
  if (!BRep_Tool::IsClosed(shape)) {
    throw std::invalid_argument(std::string(what) + " is not closed");
  }
  return TopoDS::Wire(shape);
}

// A wire with its edges chained in order, or an error if the edges aren't one chain.
TopoDS_Wire chained_wire(const std::vector<TopoDS_Shape>& edges) {
  if (edges.empty()) throw std::invalid_argument("a wire needs at least one edge");
  ShapeMap distinct;
  NCollection_List<TopoDS_Shape> list;
  for (const TopoDS_Shape& edge : edges) {
    if (edge.ShapeType() != TopAbs_EDGE) throw std::invalid_argument("a wire takes only edges");
    const int before = distinct.Extent();
    distinct.Add(edge);
    if (distinct.Extent() == before) throw std::invalid_argument("an edge was given twice");
    list.Append(edge);
  }
  BRepBuilderAPI_MakeWire maker;
  maker.Add(list);
  if (!maker.IsDone() || maker.Error() != BRepBuilderAPI_WireDone) {
    throw std::runtime_error("the edges do not connect into one wire");
  }
  const TopoDS_Wire wire = maker.Wire();

  // A branch leaves more than two edge ends at a vertex, which no single chain has.
  NCollection_IndexedDataMap<TopoDS_Shape, int, TopTools_ShapeMapHasher> ends;
  for (TopExp_Explorer it(wire, TopAbs_VERTEX); it.More(); it.Next()) {
    if (int* n = ends.ChangeSeek(it.Current())) {
      if (++*n > 2) throw std::runtime_error("the edges branch instead of forming one wire");
    } else {
      ends.Add(it.Current(), 1);
    }
  }

  BRep_Builder builder;
  TopoDS_Wire ordered;
  builder.MakeWire(ordered);
  size_t count = 0;
  for (BRepTools_WireExplorer it(wire); it.More(); it.Next()) {
    builder.Add(ordered, it.Current());
    ++count;
  }
  if (count != edges.size()) {
    throw std::runtime_error("the edges do not connect into one wire");
  }
  ordered.Closed(BRep_Tool::IsClosed(ordered));
  return ordered;
}

void require_valid(const TopoDS_Shape& shape, const char* what) {
  if (!BRepCheck_Analyzer(shape).IsValid()) {
    throw std::runtime_error(std::string(what) + " came out invalid");
  }
}

const char* edge_error(BRepBuilderAPI_EdgeError error) {
  switch (error) {
    case BRepBuilderAPI_PointProjectionFailed:
      return "a point is not on the curve";
    case BRepBuilderAPI_ParameterOutOfRange:
      return "the range is outside the curve's parameters";
    case BRepBuilderAPI_DifferentPointsOnClosedCurve:
      return "the ends differ on a closed curve";
    case BRepBuilderAPI_PointWithInfiniteParameter:
      return "the range is infinite";
    case BRepBuilderAPI_DifferentsPointAndParameter:
      return "a point does not match its parameter";
    case BRepBuilderAPI_LineThroughIdenticPoints:
      return "the range is empty";
    default:
      return "the edge could not be built";
  }
}

// A span wider than one period would make the builders wrap it around without a word.
void check_span(double span, bool periodic, double period, const char* what) {
  if (!(span > 0.0)) throw std::invalid_argument(std::string(what) + " is empty or backwards");
  if (periodic && span > period * (1.0 + 1e-12)) {
    throw std::invalid_argument(std::string(what) + " is longer than one period");
  }
}

// Moves a periodic parameter into the period that starts at `low`.
double into_period(double value, double low, double period) {
  const double slack = 1e-9 * period;
  return value - std::floor((value - low + slack) / period) * period;
}

// Where a face's surface is evaluated from: its surface as written in the format, and the
// parameter flip that goes with that.
struct FaceSurface {
  occ::handle<Geom_Surface> surface;
  bool flip_u;
  explicit FaceSurface(const TopoDS_Face& face) {
    const occ::handle<Geom_Surface> raw = BRep_Tool::Surface(face);
    if (raw.IsNull()) throw std::runtime_error("the face has no surface");
    surface = basis_surface(raw);
    flip_u = flips_u(surface);
  }
};

// A curve built from points checked for being far enough apart, as a buffer of the format.
uint32_t curve_buffer(const occ::handle<Geom_Curve>& curve) {
  Bytes out;
  append_curve(out, basis_curve(curve));
  return put_buffer(std::move(out));
}

NCollection_Array1<gp_Pnt> read_points(const void* points, size_t count) {
  NCollection_Array1<gp_Pnt> out(1, static_cast<int>(std::max<size_t>(count, 1)));
  for (size_t i = 0; i < count; ++i) {
    const double x = read_unaligned<double>(points, 3 * i);
    const double y = read_unaligned<double>(points, 3 * i + 1);
    const double z = read_unaligned<double>(points, 3 * i + 2);
    out(static_cast<int>(i) + 1) = gp_Pnt(x, y, z);
  }
  return out;
}

double distance_to_curve(const gp_Pnt& point, const occ::handle<Geom_Curve>& curve) {
  double best = std::min(point.Distance(curve->Value(curve->FirstParameter())),
                         point.Distance(curve->Value(curve->LastParameter())));
  GeomAPI_ProjectPointOnCurve projection(point, curve);
  if (projection.NbPoints() > 0) best = std::min(best, projection.LowerDistance());
  return best;
}

}  // namespace

// --- construction ------------------------------------------------------------------

API uint32_t occt_make_vertex(double x, double y, double z) {
  return guard_handle([&]() -> uint32_t {
    BRepBuilderAPI_MakeVertex maker(gp_Pnt(x, y, z));
    return put_shape(maker.Shape());
  });
}

// Input: a curve, then u32 has_range and two doubles.
API uint32_t occt_make_edge(const void* data, size_t len) {
  return guard_handle([&]() -> uint32_t {
    Reader r(data, len);
    const occ::handle<Geom_Curve> curve = read_curve(r);
    const bool has_range = r.u32() != 0;
    const double first = r.f64(), last = r.f64();
    double from = first, to = last;
    if (has_range) {
      check_span(last - first, curve->IsPeriodic(), curve->IsPeriodic() ? curve->Period() : 0.0,
                 "the edge's range");
    } else {
      from = curve->FirstParameter();
      to = curve->LastParameter();
      if (Precision::IsInfinite(from) || Precision::IsInfinite(to)) {
        throw std::invalid_argument("an unbounded curve needs a range");
      }
    }
    BRepBuilderAPI_MakeEdge maker(curve, from, to);
    if (!maker.IsDone()) throw std::runtime_error(edge_error(maker.Error()));
    return put_shape(maker.Edge());
  });
}

API uint32_t occt_make_wire(const void* edges, size_t count) {
  return guard_handle(
      [&]() -> uint32_t { return put_shape(chained_wire(read_shapes(edges, count))); });
}

API uint32_t occt_make_planar_face(uint32_t outer, const void* holes, size_t count) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Wire outer_wire = closed_wire(get_shape(outer), "the outer boundary");
    BRepBuilderAPI_MakeFace base(outer_wire, /*OnlyPlane=*/true);
    if (!base.IsDone()) throw std::runtime_error("the outer boundary does not lie in a plane");
    const TopoDS_Face base_face = base.Face();
    const occ::handle<Geom_Surface> plane = BRep_Tool::Surface(base_face);

    double expected = shape_area(base_face);
    BRepBuilderAPI_MakeFace maker(base_face);
    for (const TopoDS_Shape& shape : read_shapes(holes, count)) {
      const TopoDS_Wire hole = closed_wire(shape, "a hole");
      sample_edges(hole, 8, [&](const gp_Pnt& p) {
        GeomAPI_ProjectPointOnSurf projection(p, plane);
        if (projection.NbPoints() == 0 || projection.LowerDistance() > Precision::Confusion() * 10) {
          throw std::invalid_argument("a hole does not lie in the outer boundary's plane");
        }
      });
      // Building a face on the wire alone orients it as an outer boundary, and a hole runs
      // the other way.
      BRepBuilderAPI_MakeFace alone(plane, hole, /*Inside=*/true);
      if (!alone.IsDone()) throw std::runtime_error("a hole could not be used");
      expected -= shape_area(alone.Face());
      const TopoDS_Wire oriented = BRepTools::OuterWire(alone.Face());
      maker.Add(TopoDS::Wire(oriented.Reversed()));
    }
    if (count != 0 && !maker.IsDone()) throw std::runtime_error("the face could not be built");
    const TopoDS_Face face = count != 0 ? maker.Face() : base_face;
    require_valid(face, "the planar face");
    // The validity check allows a hole that encloses nothing of the face in some cases.
    const double area = shape_area(face);
    if (!(area > 0.0) || std::abs(area - expected) > 1e-6 * std::max(expected, 1.0)) {
      throw std::invalid_argument("the holes are not inside the outer boundary");
    }
    return put_shape(face);
  });
}

// Input: a surface; u32 has_bounds and four doubles; u32 outer handle or 0; u32 hole count
// and the hole handles.
API uint32_t occt_make_surface_face(const void* data, size_t len) {
  return guard_handle([&]() -> uint32_t {
    Reader r(data, len);
    const occ::handle<Geom_Surface> surface = read_surface(r);
    const bool has_bounds = r.u32() != 0;
    const double u1 = r.f64(), u2 = r.f64(), v1 = r.f64(), v2 = r.f64();
    const uint32_t outer = r.u32();
    const int holes_n = read_count(r, sizeof(uint32_t));
    std::vector<TopoDS_Shape> holes;
    for (int i = 0; i < holes_n; ++i) holes.push_back(r.shape());

    if (outer == 0) {
      if (!holes.empty()) throw std::invalid_argument("holes need an outer boundary");
      double a = u1, b = u2, c = v1, d = v2;
      if (has_bounds) {
        check_span(u2 - u1, surface->IsUPeriodic(), surface->IsUPeriodic() ? surface->UPeriod() : 0,
                   "the U range");
        check_span(v2 - v1, surface->IsVPeriodic(), surface->IsVPeriodic() ? surface->VPeriod() : 0,
                   "the V range");
      } else {
        surface->Bounds(a, b, c, d);
        if (Precision::IsInfinite(a) || Precision::IsInfinite(b) || Precision::IsInfinite(c) ||
            Precision::IsInfinite(d)) {
          throw std::invalid_argument("an unbounded surface needs bounds or a boundary");
        }
      }
      BRepBuilderAPI_MakeFace maker(surface, a, b, c, d, Precision::Confusion());
      if (!maker.IsDone()) throw std::runtime_error("the parameter range is outside the surface");
      const TopoDS_Face face = maker.Face();
      require_valid(face, "the face");
      return put_shape(face);
    }

    BRep_Builder builder;
    TopoDS_Face face;
    builder.MakeFace(face, surface, Precision::Confusion());
    builder.Add(face, oriented_on_surface(surface,
                                          closed_wire(get_shape(outer), "the outer boundary")));
    for (const TopoDS_Shape& shape : holes) {
      const TopoDS_Wire hole = oriented_on_surface(surface, closed_wire(shape, "a hole"));
      builder.Add(face, TopoDS::Wire(hole.Reversed()));
    }
    require_valid(face, "the face");
    if (!(shape_area(face) > 0.0)) throw std::runtime_error("the face has no area");
    return put_shape(face);
  });
}

// Input: u32 edge count and handles; a handle per edge for the face it must be tangent to,
// or 0; u32 point count and points.
API uint32_t occt_fill_boundary(const void* data, size_t len) {
  return guard_handle([&]() -> uint32_t {
    Reader r(data, len);
    const int edges_n = read_count(r, sizeof(uint32_t));
    std::vector<TopoDS_Shape> edges;
    for (int i = 0; i < edges_n; ++i) edges.push_back(r.shape());
    std::vector<TopoDS_Face> supports;
    for (int i = 0; i < edges_n; ++i) {
      const uint32_t support = r.u32();
      if (support == 0) {
        supports.emplace_back();
        continue;
      }
      const TopoDS_Shape face = get_shape(support);
      if (face.ShapeType() != TopAbs_FACE) {
        throw std::invalid_argument("an edge can only be tangent to a face");
      }
      double first, last;
      if (BRep_Tool::CurveOnSurface(TopoDS::Edge(edges[i]), TopoDS::Face(face), first, last)
              .IsNull()) {
        throw std::invalid_argument("a tangent edge has to be an edge of its face");
      }
      supports.push_back(TopoDS::Face(face));
    }
    const int points_n = read_count(r, 3 * sizeof(double));
    std::vector<gp_Pnt> points;
    for (int i = 0; i < points_n; ++i) points.push_back(r.point());

    const TopoDS_Wire loop = chained_wire(edges);
    if (!BRep_Tool::IsClosed(loop)) throw std::invalid_argument("the boundary is not closed");

    BRepOffsetAPI_MakeFilling maker;
    // Curvature isn't offered: the plate solver misses it badly, see occt_blend_faces.
    // Chaining can rebuild an edge to merge its vertices, so a rebuilt one is found by where
    // it runs. The tangent constraint needs the original, which has the curve on its face.
    const auto midpoint = [](const TopoDS_Shape& edge) {
      const BRepAdaptor_Curve curve(TopoDS::Edge(edge));
      return curve.Value(0.5 * (curve.FirstParameter() + curve.LastParameter()));
    };
    for (BRepTools_WireExplorer it(loop); it.More(); it.Next()) {
      const TopoDS_Edge& edge = it.Current();
      const gp_Pnt middle = midpoint(edge);
      const auto given = std::find_if(edges.begin(), edges.end(), [&](const TopoDS_Shape& e) {
        return e.IsSame(edge) || midpoint(e).Distance(middle) < 1e-6;
      });
      if (given == edges.end()) throw std::runtime_error("the boundary lost an edge");
      const TopoDS_Face& support = supports[static_cast<size_t>(given - edges.begin())];
      if (support.IsNull()) {
        maker.Add(edge, GeomAbs_C0);
      } else {
        maker.Add(TopoDS::Edge(*given), support, GeomAbs_G1);
      }
    }
    for (const gp_Pnt& p : points) maker.Add(p);
    maker.Build();
    if (!maker.IsDone()) throw std::runtime_error("no surface fills this boundary");
    const TopoDS_Shape result = maker.Shape();
    if (result.IsNull() || result.ShapeType() != TopAbs_FACE) {
      throw std::runtime_error("filling did not produce a face");
    }
    require_valid(result, "the filled face");
    if (!(shape_area(result) > 0.0)) throw std::runtime_error("the filled face has no area");
    // The solver leaves a wrong answer unreported when a side edge can't meet the tangency:
    // 38 degrees in testing, with success returned.
    for (size_t i = 0; i < edges.size(); ++i) {
      if (supports[i].IsNull()) continue;
      const Continuity c =
          measure_continuity(TopoDS::Edge(edges[i]), TopoDS::Face(result), supports[i]);
      if (c.angle > 1.0 * M_PI / 180.0) {
        throw std::runtime_error(
            "the filled face could not be made tangent; the edges beside a tangent edge must "
            "leave it within the face's tangent plane");
      }
    }
    // The solver minimises rather than solves exactly. Far from a point means it gave up
    // on that point, which it reports only through this kind of check.
    const occ::handle<Geom_Surface> surface = BRep_Tool::Surface(TopoDS::Face(result));
    const Bnd_Box box = [&] {
      Bnd_Box b;
      BRepBndLib::Add(result, b);
      return b;
    }();
    const double allowed = std::max(1e-3, 1e-3 * std::sqrt(box.SquareExtent()));
    for (const gp_Pnt& p : points) {
      GeomAPI_ProjectPointOnSurf projection(p, surface);
      if (projection.NbPoints() == 0 || projection.LowerDistance() > allowed) {
        throw std::runtime_error("the filled face misses a point it had to pass through");
      }
    }
    return put_shape(result);
  });
}

API uint32_t occt_sew(const void* shapes, size_t count, double tolerance) {
  return guard_handle([&]() -> uint32_t {
    if (count == 0) throw std::invalid_argument("nothing to sew");
    BRepBuilderAPI_Sewing sewing(tolerance);
    for (const TopoDS_Shape& shape : read_shapes(shapes, count)) {
      if (!TopExp_Explorer(shape, TopAbs_FACE).More()) {
        throw std::invalid_argument("only faces and shells can be sewn");
      }
      sewing.Add(shape);
    }
    sewing.Perform();
    const TopoDS_Shape result = sewing.SewedShape();
    if (result.IsNull() || !TopExp_Explorer(result, TopAbs_FACE).More()) {
      throw std::runtime_error("sewing produced nothing");
    }
    return put_shape(unwrap_single(result));
  });
}

API uint32_t occt_make_solid(uint32_t h) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    if (shape.ShapeType() != TopAbs_SHELL) throw std::invalid_argument("a solid needs a shell");
    TopoDS_Shell shell = TopoDS::Shell(shape);
    if (!BRep_Tool::IsClosed(shell)) throw std::invalid_argument("the shell is not closed");
    shell.Closed(true);

    BRepCheck_Shell check(shell);
    if (check.Orientation() != BRepCheck_NoError) {
      // Faces disagree about which side is out; this also leaves them on the same side.
      ShapeFix_Shell fix(shell);
      fix.FixFaceOrientation(shell);
      const TopoDS_Shell fixed = fix.Shell();
      if (fixed.IsNull()) throw std::runtime_error("the shell's faces can't be oriented");
      shell = fixed;
      shell.Closed(true);
    }

    const auto build = [](const TopoDS_Shell& s) {
      BRep_Builder builder;
      TopoDS_Solid solid;
      builder.MakeSolid(solid);
      builder.Add(solid, s);
      return solid;
    };
    const auto volume = [](const TopoDS_Shape& s) {
      GProp_GProps props;
      BRepGProp::VolumeProperties(s, props);
      return props.Mass();
    };
    TopoDS_Solid solid = build(shell);
    if (volume(solid) < 0.0) solid = build(TopoDS::Shell(shell.Reversed()));
    if (!(volume(solid) > 0.0)) throw std::runtime_error("the shell encloses no volume");
    require_valid(solid, "the solid");
    return put_shape(solid);
  });
}

API uint32_t occt_make_compound(const void* shapes, size_t count) {
  return guard_handle([&]() -> uint32_t {
    BRep_Builder builder;
    TopoDS_Compound compound;
    builder.MakeCompound(compound);
    for (const TopoDS_Shape& shape : read_shapes(shapes, count)) builder.Add(compound, shape);
    return put_shape(compound);
  });
}

// Input: a frame, then the two radii.
API uint32_t occt_make_torus(const void* data, size_t len) {
  return guard_handle([&]() -> uint32_t {
    Reader r(data, len);
    const gp_Ax3 frame = read_frame(r);
    const double major = r.f64(), minor = r.f64();
    if (!(minor > 0.0) || !(major > minor)) {
      throw std::invalid_argument("a torus needs a major radius above its minor radius above 0");
    }
    BRepPrimAPI_MakeTorus maker(frame.Ax2(), major, minor);
    maker.Build();
    if (!maker.IsDone()) throw std::runtime_error("the torus could not be built");
    return put_shape(maker.Shape());
  });
}

API uint32_t occt_interpolate_curve(const void* points, size_t count, int32_t closed) {
  return guard_handle([&]() -> uint32_t {
    if (count < (closed != 0 ? 3u : 2u)) throw std::invalid_argument("too few points");
    const auto array = new NCollection_HArray1<gp_Pnt>(read_points(points, count));
    GeomAPI_Interpolate maker(array, closed != 0, Precision::Confusion());
    maker.Perform();
    if (!maker.IsDone()) throw std::runtime_error("the points could not be interpolated");
    return curve_buffer(maker.Curve());
  });
}

API uint32_t occt_fit_curve(const void* points, size_t count, double tolerance) {
  return guard_handle([&]() -> uint32_t {
    if (count < 2) throw std::invalid_argument("too few points");
    const NCollection_Array1<gp_Pnt> array = read_points(points, count);
    // The approximation's tolerance is a target, not a promise, so check, and ask for more
    // if it fell short. The deviation is checked at the points, which is all it's fit to.
    for (const double target : {tolerance, tolerance / 4.0, tolerance / 16.0}) {
      GeomAPI_PointsToBSpline maker(array, 3, 8, GeomAbs_C2, target);
      if (!maker.IsDone()) continue;
      const occ::handle<Geom_BSplineCurve> curve = maker.Curve();
      double worst = 0.0;
      for (int i = array.Lower(); i <= array.Upper(); ++i) {
        worst = std::max(worst, distance_to_curve(array(i), curve));
      }
      if (worst <= tolerance) return curve_buffer(curve);
    }
    throw std::runtime_error("no curve fits the points within the tolerance");
  });
}

API uint32_t occt_make_helix(double px, double py, double pz, double dx, double dy, double dz,
                             double radius, double pitch, double height, int32_t left_handed) {
  return guard_handle([&]() -> uint32_t {
    if (!(radius > 0.0) || !(pitch > 0.0) || !(height > 0.0)) {
      throw std::invalid_argument("a helix needs a positive radius, pitch and height");
    }
    if (height / pitch > 1e5) throw std::invalid_argument("a helix this long has too many turns");
    const gp_Dir z(dx, dy, dz);
    // X is chosen as the contract's Frame::from_normal does, so that the start is predictable.
    const gp_Vec hint = std::abs(z.X()) < 0.9 ? gp_Vec(1, 0, 0) : gp_Vec(0, 1, 0);
    const gp_Dir x(hint - gp_Vec(z) * hint.Dot(gp_Vec(z)));
    const occ::handle<Geom_Surface> cylinder =
        new Geom_CylindricalSurface(gp_Ax3(gp_Pnt(px, py, pz), z, x), radius);

    // One edge per turn: a long single spline approximates worse, and sweeps along it get
    // less robust with its length. The per-turn 3D curves meet tangentially.
    const double sign = left_handed != 0 ? -1.0 : 1.0;
    const gp_Dir2d direction(sign * kTwoPi, pitch);
    const double per_turn = std::hypot(kTwoPi, pitch);
    const double turns = height / pitch;
    const int edges = std::max(1, static_cast<int>(std::ceil(turns - 1e-9)));

    const auto point_at = [&](double turn) {
      return cylinder->Value(sign * kTwoPi * turn, pitch * turn);
    };
    BRep_Builder builder;
    TopoDS_Wire wire;
    builder.MakeWire(wire);
    TopoDS_Vertex start;
    builder.MakeVertex(start, point_at(0.0), Precision::Confusion());
    for (int i = 0; i < edges; ++i) {
      const double from = i, to = std::min(i + 1.0, turns);
      TopoDS_Vertex end;
      builder.MakeVertex(end, point_at(to), Precision::Confusion());
      const occ::handle<Geom2d_Line> line =
          new Geom2d_Line(gp_Pnt2d(sign * kTwoPi * from, pitch * from), direction);
      BRepBuilderAPI_MakeEdge maker(line, cylinder, start, end, 0.0, per_turn * (to - from));
      if (!maker.IsDone()) throw std::runtime_error(edge_error(maker.Error()));
      const TopoDS_Edge edge = maker.Edge();
      if (!BRepLib::BuildCurve3d(edge, 1e-7, GeomAbs_C2, 14, 100)) {
        throw std::runtime_error("the helix could not be approximated");
      }
      builder.Add(wire, edge);
      start = end;
    }
    wire.Closed(false);
    return put_shape(wire);
  });
}

API uint32_t occt_offset_wire(uint32_t h, double distance) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Shape shape = get_shape(h);
    if (shape.ShapeType() != TopAbs_WIRE) throw std::invalid_argument("only a wire is offset");
    const TopoDS_Wire wire = TopoDS::Wire(shape);
    const bool closed = BRep_Tool::IsClosed(wire);
    if (!BRepBuilderAPI_FindPlane(wire).Found()) {
      throw std::invalid_argument("the wire does not define a plane");
    }
    TopoDS_Shape result;
    double before = 0.0;
    if (closed) {
      BRepBuilderAPI_MakeFace face(wire, /*OnlyPlane=*/true);
      if (!face.IsDone()) throw std::runtime_error("the wire does not bound a planar face");
      before = shape_area(face.Face());
      BRepOffsetAPI_MakeOffset maker(face.Face(), GeomAbs_Arc);
      maker.Perform(distance);
      if (!maker.IsDone()) throw std::runtime_error("the wire could not be offset this far");
      result = maker.Shape();
    } else {
      BRepOffsetAPI_MakeOffset maker(wire, GeomAbs_Arc, /*IsOpenResult=*/true);
      maker.Perform(distance);
      if (!maker.IsDone()) throw std::runtime_error("the wire could not be offset this far");
      result = maker.Shape();
    }
    if (result.IsNull() || !TopExp_Explorer(result, TopAbs_EDGE).More()) {
      throw std::runtime_error("nothing is left of the wire at this offset");
    }
    if (closed) {
      double after = 0.0;
      for (TopExp_Explorer it(result, TopAbs_WIRE); it.More(); it.Next()) {
        BRepBuilderAPI_MakeFace face(TopoDS::Wire(it.Current()), /*OnlyPlane=*/true);
        if (!face.IsDone()) throw std::runtime_error("the offset wire is not a closed boundary");
        after += shape_area(face.Face());
      }
      if ((distance > 0.0) != (after > before)) {
        throw std::runtime_error("the offset went to the wrong side");
      }
    }
    return put_shape(unwrap_single(result));
  });
}

// --- reading -------------------------------------------------------------------------

// Output: a curve, two doubles for the range, u32 reversed.
API uint32_t occt_edge_geometry(uint32_t h) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Edge edge = TopoDS::Edge(get_shape(h));
    double first = 0.0, last = 0.0;
    const occ::handle<Geom_Curve> curve = BRep_Tool::Curve(edge, first, last);
    Bytes out;
    if (curve.IsNull()) {
      // Degenerate, or defined only on a surface.
      BRep_Tool::Range(edge, first, last);
      append<uint32_t>(out, 0);
    } else {
      append_curve(out, basis_curve(curve));
    }
    append(out, first);
    append(out, last);
    append<uint32_t>(out, edge.Orientation() == TopAbs_REVERSED ? 1 : 0);
    return put_buffer(std::move(out));
  });
}

// Output: point, derivative, curvature, 7 doubles. Curvature is NaN where the first
// derivative vanishes.
API uint32_t occt_edge_point(uint32_t h, double t) {
  return guard_handle([&]() -> uint32_t {
    const BRepAdaptor_Curve curve(TopoDS::Edge(get_shape(h)));
    const double first = curve.FirstParameter(), last = curve.LastParameter();
    const double slack = 1e-9 * std::max(1.0, last - first);
    // A periodic curve's edge may store its range a period away from where it was asked for.
    if (curve.IsPeriodic()) t = into_period(t, first, curve.Period());
    if (t < first - slack || t > last + slack) {
      throw std::invalid_argument("the parameter is outside the edge's range");
    }
    const double at = std::min(std::max(t, first), last);
    gp_Pnt point;
    gp_Vec d1, d2;
    curve.D2(at, point, d1, d2);
    const double speed = d1.Magnitude();
    const double curvature = speed > Precision::Confusion() * 1e-3
                                 ? d1.Crossed(d2).Magnitude() / (speed * speed * speed)
                                 : kNaN;
    Bytes out;
    append_point(out, point);
    append(out, d1.X());
    append(out, d1.Y());
    append(out, d1.Z());
    append(out, curvature);
    return put_buffer(std::move(out));
  });
}

// Output: a surface, u1, u2, v1, v2, u32 reversed. See flips_u.
API uint32_t occt_face_geometry(uint32_t h) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Face face = TopoDS::Face(get_shape(h));
    const FaceSurface at(face);
    double u1 = 0.0, u2 = 0.0, v1 = 0.0, v2 = 0.0;
    BRepTools::UVBounds(face, u1, u2, v1, v2);
    bool reversed = face.Orientation() == TopAbs_REVERSED;
    Bytes out;
    append_surface(out, at.surface);
    if (at.flip_u) {
      const double low = -u2, high = -u1;
      u1 = low;
      u2 = high;
      // The natural normal of the description is the opposite of the kernel's.
      reversed = !reversed;
    }
    append(out, u1);
    append(out, u2);
    append(out, v1);
    append(out, v2);
    append<uint32_t>(out, reversed ? 1 : 0);
    return put_buffer(std::move(out));
  });
}

// Output: u32 has_normal, point, normal, du, dv, larger and smaller curvature. Normal
// follows the face's orientation, and curvature is relative to it. Curvatures are NaN and
// has_normal 0 where the surface is singular.
API uint32_t occt_face_point(uint32_t h, double u, double v) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Face face = TopoDS::Face(get_shape(h));
    const FaceSurface at(face);
    const double raw_u = at.flip_u ? -u : u;

    double u1 = 0.0, u2 = 0.0, v1 = 0.0, v2 = 0.0;
    at.surface->Bounds(u1, u2, v1, v2);
    const double slack = 1e-9;
    const bool in_u = at.surface->IsUPeriodic() || (raw_u >= u1 - slack && raw_u <= u2 + slack);
    const bool in_v = at.surface->IsVPeriodic() || (v >= v1 - slack && v <= v2 + slack);
    if (!in_u || !in_v) throw std::invalid_argument("the parameters are outside the surface");

    const BRepAdaptor_Surface adaptor(face, /*Restriction=*/false);
    BRepLProp_SLProps props(adaptor, raw_u, v, 2, 1e-9);
    const bool reversed = face.Orientation() == TopAbs_REVERSED;
    const gp_Vec du = at.flip_u ? gp_Vec(props.D1U()).Reversed() : gp_Vec(props.D1U());
    const gp_Vec dv = props.D1V();

    // The normal is the limit of nearby ones where it can be, but a collapsed parameter line
    // is singular whatever its limit does.
    const double scale = std::max(du.SquareMagnitude(), dv.SquareMagnitude());
    const bool has_normal = props.IsNormalDefined() && du.Crossed(dv).Magnitude() > 1e-12 * scale;

    Bytes out;
    append<uint32_t>(out, has_normal ? 1 : 0);
    append_point(out, props.Value());
    gp_Vec normal(0.0, 0.0, 0.0);
    if (has_normal) normal = reversed ? gp_Vec(props.Normal()).Reversed() : gp_Vec(props.Normal());
    append(out, normal.X());
    append(out, normal.Y());
    append(out, normal.Z());
    append(out, du.X());
    append(out, du.Y());
    append(out, du.Z());
    append(out, dv.X());
    append(out, dv.Y());
    append(out, dv.Z());
    double k1 = kNaN, k2 = kNaN;
    if (has_normal && props.IsCurvatureDefined()) {
      k1 = reversed ? -props.MinCurvature() : props.MaxCurvature();
      k2 = reversed ? -props.MaxCurvature() : props.MinCurvature();
    }
    append(out, k1);
    append(out, k2);
    return put_buffer(std::move(out));
  });
}

// Output: u, v, point, 5 doubles. Periodic parameters fall in the face's range where they can.
API uint32_t occt_project_onto_face(uint32_t h, double x, double y, double z) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Face face = TopoDS::Face(get_shape(h));
    const FaceSurface at(face);
    const gp_Pnt target(x, y, z);

    double u = 0.0, v = 0.0, best = std::numeric_limits<double>::infinity();
    GeomAPI_ProjectPointOnSurf projection(target, at.surface);
    if (projection.NbPoints() > 0) {
      projection.LowerDistanceParameters(u, v);
      best = projection.LowerDistance();
    }
    // Extrema are stationary points, which misses a bounded surface's boundary and the
    // centre of a sphere. This searches the whole surface instead.
    const occ::handle<ShapeAnalysis_Surface> search = new ShapeAnalysis_Surface(at.surface);
    const gp_Pnt2d uv = search->ValueOfUV(target, Precision::Confusion());
    const double gap = target.Distance(at.surface->Value(uv.X(), uv.Y()));
    if (gap < best) {
      best = gap;
      u = uv.X();
      v = uv.Y();
    }
    if (!std::isfinite(best)) throw std::runtime_error("no closest point could be found");

    double u1 = 0.0, u2 = 0.0, v1 = 0.0, v2 = 0.0;
    BRepTools::UVBounds(face, u1, u2, v1, v2);
    if (at.surface->IsUPeriodic()) u = into_period(u, u1, at.surface->UPeriod());
    if (at.surface->IsVPeriodic()) v = into_period(v, v1, at.surface->VPeriod());

    Bytes out;
    append(out, at.flip_u ? -u : u);
    append(out, v);
    append_point(out, at.surface->Value(u, v));
    return put_buffer(std::move(out));
  });
}

// Output: t, point, 4 doubles.
API uint32_t occt_project_onto_edge(uint32_t h, double x, double y, double z) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Edge edge = TopoDS::Edge(get_shape(h));
    double first = 0.0, last = 0.0;
    const occ::handle<Geom_Curve> curve = BRep_Tool::Curve(edge, first, last);
    if (curve.IsNull()) throw std::invalid_argument("the edge has no curve to project onto");
    const gp_Pnt target(x, y, z);

    // Extrema are stationary points, and the closest point of an edge can be an end.
    double t = first, best = target.Distance(curve->Value(first));
    const double at_last = target.Distance(curve->Value(last));
    if (at_last < best) {
      best = at_last;
      t = last;
    }
    GeomAPI_ProjectPointOnCurve projection(target, curve, first, last);
    if (projection.NbPoints() > 0 && projection.LowerDistance() < best) {
      t = projection.LowerDistanceParameter();
    }
    Bytes out;
    append(out, t);
    append_point(out, curve->Value(t));
    return put_buffer(std::move(out));
  });
}

// 1 closed, 0 open, -1 error.
API int32_t occt_is_closed(uint32_t h) {
  return guard<int32_t>(-1, [&]() {
    const TopoDS_Shape shape = get_shape(h);
    const TopAbs_ShapeEnum kind = shape.ShapeType();
    if (kind != TopAbs_EDGE && kind != TopAbs_WIRE && kind != TopAbs_SHELL) {
      throw std::invalid_argument("only an edge, wire or shell can be closed or open");
    }
    return BRep_Tool::IsClosed(shape) ? 1 : 0;
  });
}

// The largest tolerance on the shape, 0 if it holds no geometry, NaN on error.
API double occt_tolerance(uint32_t h) {
  return guard<double>(kNaN, [&]() {
    const TopoDS_Shape shape = get_shape(h);
    double largest = 0.0;
    for (TopExp_Explorer it(shape, TopAbs_VERTEX); it.More(); it.Next()) {
      largest = std::max(largest, BRep_Tool::Tolerance(TopoDS::Vertex(it.Current())));
    }
    for (TopExp_Explorer it(shape, TopAbs_EDGE); it.More(); it.Next()) {
      largest = std::max(largest, BRep_Tool::Tolerance(TopoDS::Edge(it.Current())));
    }
    for (TopExp_Explorer it(shape, TopAbs_FACE); it.More(); it.Next()) {
      largest = std::max(largest, BRep_Tool::Tolerance(TopoDS::Face(it.Current())));
    }
    return largest;
  });
}
