// Surface work across shared edges: measuring continuity, curvature-matched blends and
// extending faces. SPDX-License-Identifier: MIT OR Apache-2.0
#include <BRepAdaptor_Curve.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepLib.hxx>
#include <BRep_Tool.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>
#include <GeomAPI_Interpolate.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_Surface.hxx>
#include <NCollection_Array1.hxx>
#include <NCollection_Array2.hxx>
#include <NCollection_HArray1.hxx>
#include <Precision.hxx>
#include <TopoDS.hxx>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "internal.hpp"

using namespace occt_capi;

namespace {

constexpr int kMeasureSamples = 21;
constexpr int kBlendSpans = 32;
constexpr double kDegree = M_PI / 180.0;

// Position, derivatives and normal of a surface at a point of an edge on it.
struct Local {
  gp_Pnt point;
  double gap = 0.0, u = 0.0, v = 0.0;
  gp_Vec su, sv, suu, suv, svv, normal;
};

bool local_at(const occ::handle<Geom_Surface>& surface, const gp_Pnt& p, Local& out) {
  GeomAPI_ProjectPointOnSurf projection(p, surface);
  if (projection.NbPoints() == 0) return false;
  projection.LowerDistanceParameters(out.u, out.v);
  out.gap = projection.LowerDistance();
  surface->D2(out.u, out.v, out.point, out.su, out.sv, out.suu, out.svv, out.suv);
  const gp_Vec n = out.su.Crossed(out.sv);
  if (n.Magnitude() < 1e-12) return false;
  out.normal = n.Normalized();
  return true;
}

// The surface parameter step that moves along a tangent-plane vector.
void to_parameters(const Local& l, const gp_Vec& d, double& du, double& dv) {
  const double e = l.su.Dot(l.su), f = l.su.Dot(l.sv), g = l.sv.Dot(l.sv);
  const double a = d.Dot(l.su), b = d.Dot(l.sv), det = e * g - f * f;
  du = (g * a - f * b) / det;
  dv = (e * b - f * a) / det;
}

// Normal curvature in a tangent direction, signed against l.normal.
double normal_curvature(const Local& l, const gp_Vec& d) {
  double du, dv;
  to_parameters(l, d, du, dv);
  const double second = du * du * l.suu.Dot(l.normal) + 2.0 * du * dv * l.suv.Dot(l.normal) +
                        dv * dv * l.svv.Dot(l.normal);
  const gp_Vec along = l.su * du + l.sv * dv;
  return second / along.Dot(along);
}

// The unit vector in the tangent plane, across the edge.
gp_Vec across(const Local& l, const gp_Vec& tangent) {
  const gp_Vec w = l.normal.Crossed(tangent);
  if (w.Magnitude() < 1e-9) throw std::runtime_error("the edge leaves the surface's tangent plane");
  return w.Normalized();
}

// +1 if `w` points into the face, -1 if away, 0 if both sides are the face, as across a
// seam. An edge with the face on neither side doesn't border it.
int side_of(BRepTopAdaptor_FClass2d& classifier, const Local& l, const gp_Vec& w, double step) {
  double du, dv;
  to_parameters(l, w, du, dv);
  const auto in = [&](double sign) {
    return classifier.Perform(gp_Pnt2d(l.u + sign * step * du, l.v + sign * step * dv)) ==
           TopAbs_IN;
  };
  const bool plus = in(1.0), minus = in(-1.0);
  if (plus && minus) return 0;
  if (plus) return 1;
  if (minus) return -1;
  throw std::invalid_argument("the edge does not border the face");
}

TopoDS_Edge as_edge(const TopoDS_Shape& shape) {
  if (shape.ShapeType() != TopAbs_EDGE) throw std::invalid_argument("expected an edge");
  return TopoDS::Edge(shape);
}

TopoDS_Face as_face(const TopoDS_Shape& shape) {
  if (shape.ShapeType() != TopAbs_FACE) throw std::invalid_argument("expected a face");
  return TopoDS::Face(shape);
}

double curve_length(const BRepAdaptor_Curve& curve) {
  return GCPnts_AbscissaPoint::Length(curve);
}

// An edge on its face, walked from 0 to 1 in the direction the blend wants.
struct Rail {
  BRepAdaptor_Curve curve;
  occ::handle<Geom_Surface> surface;
  TopoDS_Face face;
  double first, last;
  bool flipped = false;

  Rail(const TopoDS_Edge& edge, const TopoDS_Face& f)
      : curve(edge),
        surface(BRep_Tool::Surface(f)),
        face(f),
        first(curve.FirstParameter()),
        last(curve.LastParameter()) {}
  double at(double s) const {
    const double t = flipped ? 1.0 - s : s;
    return first + (last - first) * t;
  }
};

// The direction leaving the face across the edge at each sample, with the normal it pairs
// with. The side is read at the middle, where no corner confuses it, and carried along.
struct Leaving {
  gp_Pnt point;
  gp_Vec direction;
  gp_Vec normal;
  double curvature;
};

std::vector<Leaving> leaving(const Rail& rail, double step) {
  std::vector<Local> locals(kBlendSpans + 1);
  std::vector<gp_Vec> across_edge(kBlendSpans + 1);
  std::vector<gp_Pnt> points(kBlendSpans + 1);
  const double extent = curve_length(rail.curve);
  for (int i = 0; i <= kBlendSpans; ++i) {
    gp_Pnt p;
    gp_Vec t;
    rail.curve.D1(rail.at(double(i) / kBlendSpans), p, t);
    if (t.Magnitude() < 1e-12) throw std::invalid_argument("the edge has a singular point");
    points[i] = p;
    if (!local_at(rail.surface, p, locals[i])) {
      throw std::invalid_argument("the face's surface is singular along the edge");
    }
    if (locals[i].gap > std::max(1e-4, 1e-4 * extent)) {
      throw std::invalid_argument("the edge does not lie on the face");
    }
    across_edge[i] = across(locals[i], t.Normalized());
  }
  BRepTopAdaptor_FClass2d classifier(rail.face, 1e-6);
  const int mid = kBlendSpans / 2;
  const int side = side_of(classifier, locals[mid], across_edge[mid], step);
  if (side == 0) throw std::invalid_argument("the face lies on both sides of the edge");
  // The direction leaves the face, so it points away from the side the face is on.
  std::vector<Leaving> out(kBlendSpans + 1);
  gp_Vec previous = across_edge[mid] * double(-side);
  const auto fill = [&](int i) {
    gp_Vec d = across_edge[i];
    if (d.Dot(previous) < 0.0) d.Reverse();
    previous = d;
    out[i] = {points[i], d, locals[i].normal, normal_curvature(locals[i], d)};
  };
  for (int i = mid; i <= kBlendSpans; ++i) fill(i);
  previous = out[mid].direction;
  for (int i = mid - 1; i >= 0; --i) fill(i);
  return out;
}

occ::handle<Geom_BSplineCurve> interpolated(const std::vector<gp_Pnt>& row) {
  const int n = static_cast<int>(row.size());
  occ::handle<NCollection_HArray1<gp_Pnt>> points = new NCollection_HArray1<gp_Pnt>(1, n);
  occ::handle<NCollection_HArray1<double>> params = new NCollection_HArray1<double>(1, n);
  for (int i = 1; i <= n; ++i) {
    points->SetValue(i, row[i - 1]);
    params->SetValue(i, double(i - 1) / (n - 1));
  }
  GeomAPI_Interpolate maker(points, params, false, Precision::Confusion());
  maker.Perform();
  if (!maker.IsDone()) throw std::runtime_error("the blend's rows could not be interpolated");
  return maker.Curve();
}

}  // namespace

namespace occt_capi {

Continuity measure_continuity(const TopoDS_Edge& edge, const TopoDS_Face& a, const TopoDS_Face& b) {
  const BRepAdaptor_Curve curve(edge);
  const occ::handle<Geom_Surface> sa = BRep_Tool::Surface(a), sb = BRep_Tool::Surface(b);
  const double first = curve.FirstParameter(), last = curve.LastParameter();
  const double step = 1e-3 * curve_length(curve);
  BRepTopAdaptor_FClass2d class_a(a, 1e-6), class_b(b, 1e-6);
  Continuity result;
  int measured = 0;
  // Only interior samples: at a corner the edge ends and the question doesn't apply.
  for (int i = 1; i <= kMeasureSamples; ++i) {
    gp_Pnt p;
    gp_Vec t;
    curve.D1(first + (last - first) * i / (kMeasureSamples + 1.0), p, t);
    if (t.Magnitude() < 1e-12) continue;
    t.Normalize();
    Local la, lb;
    if (!local_at(sa, p, la) || !local_at(sb, p, lb)) continue;
    ++measured;
    result.gap = std::max({result.gap, la.gap, lb.gap});

    const gp_Vec wa = across(la, t), wb = across(lb, t);
    const int side_a = side_of(class_a, la, wa, step), side_b = side_of(class_b, lb, wb, step);
    double angle;
    if (side_a == 0 || side_b == 0) {
      // A seam has the face on both sides, so all that can be compared is the planes.
      angle = std::acos(std::min(1.0, std::abs(la.normal.Dot(lb.normal))));
    } else {
      // Smooth means one face continues where the other stops: opposite ways into them.
      const gp_Vec into_a = wa * double(side_a), into_b = wb * double(side_b);
      angle = into_a.Reversed().Angle(into_b);
    }
    result.angle = std::max(result.angle, angle);

    // Equal normal curvature in every direction, sampled, against one normal.
    const double sign = la.normal.Dot(lb.normal) < 0.0 ? -1.0 : 1.0;
    for (int k = 0; k < 4; ++k) {
      const double theta = k * M_PI / 4.0;
      const gp_Vec d = t * std::cos(theta) + wa * std::sin(theta);
      const gp_Vec on_b = d - lb.normal * d.Dot(lb.normal);
      if (on_b.Magnitude() < 1e-6) continue;
      result.curvature_gap =
          std::max(result.curvature_gap,
                   std::abs(normal_curvature(la, d) - sign * normal_curvature(lb, on_b)));
    }
  }
  if (measured == 0) throw std::runtime_error("no point of the edge could be measured");
  return result;
}

}  // namespace occt_capi

// Output: gap, angle in radians, curvature gap.
API uint32_t occt_measure_continuity(uint32_t a, uint32_t b, uint32_t edge) {
  return guard_handle([&]() -> uint32_t {
    const Continuity c = measure_continuity(as_edge(get_shape(edge)), as_face(get_shape(a)),
                                            as_face(get_shape(b)));
    Bytes out;
    append(out, c.gap);
    append(out, c.angle);
    append(out, c.curvature_gap);
    return put_buffer(std::move(out));
  });
}

// Builds the patch directly, because the plate solver behind occt_fill_boundary misses its
// curvature targets: on a sphere of curvature 0.1 it left errors of 0.35 and up.
//
// Across the blend each row is a quintic Bezier. Its first three control points fix the
// position, tangent and curvature leaving one face and its last three the other, so the
// conditions hold by construction wherever the row meets the edge. With S(0) = P0,
//   S'(0) = 5 (P1 - P0)  and  S''(0) = 20 (P2 - 2 P1 + P0),
// a tangent step `a` and a normal curvature k need P2 - 2 P1 + P0 = 1.25 a^2 k n.
//
// Input: u32 edge A, face A, edge B, face B, order (1 tangent, 2 curvature); f64 tension.
API uint32_t occt_blend_faces(const void* data, size_t len) {
  return guard_handle([&]() -> uint32_t {
    Reader r(data, len);
    const TopoDS_Edge edge_a = as_edge(r.shape());
    const TopoDS_Face face_a = as_face(r.shape());
    const TopoDS_Edge edge_b = as_edge(r.shape());
    const TopoDS_Face face_b = as_face(r.shape());
    const uint32_t order = r.u32();
    const double tension = r.f64();
    if (order != 1 && order != 2) throw std::invalid_argument("unknown blend continuity");
    if (!(tension > 0.0) || !std::isfinite(tension)) {
      throw std::invalid_argument("the tension must be positive");
    }
    if (edge_a.IsSame(edge_b)) throw std::invalid_argument("the blend needs two different edges");

    Rail a(edge_a, face_a), b(edge_b, face_b);
    const auto end = [](const Rail& rail, double s) { return rail.curve.Value(rail.at(s)); };
    const double straight = end(a, 0).Distance(end(b, 0)) + end(a, 1).Distance(end(b, 1));
    const double crossed = end(a, 0).Distance(end(b, 1)) + end(a, 1).Distance(end(b, 0));
    b.flipped = crossed < straight;

    const double length = std::max(curve_length(a.curve), curve_length(b.curve));
    const std::vector<Leaving> from_a = leaving(a, 1e-3 * length);
    const std::vector<Leaving> from_b = leaving(b, 1e-3 * length);

    std::array<std::vector<gp_Pnt>, 6> rows;
    for (auto& row : rows) row.resize(kBlendSpans + 1);
    const double curving = order == 2 ? 1.25 : 0.0;
    for (int i = 0; i <= kBlendSpans; ++i) {
      const Leaving& pa = from_a[i];
      const Leaving& pb = from_b[i];
      const double span = pa.point.Distance(pb.point);
      if (span < 1e-6) throw std::invalid_argument("the edges touch, so nothing separates them");
      const double step = tension * span / 3.0;
      const auto row = [&](const Leaving& l, int p0, int p1, int p2) {
        rows[p0][i] = l.point;
        rows[p1][i] = l.point.Translated(l.direction * step);
        rows[p2][i] = l.point.Translated(l.direction * (2.0 * step) +
                                         l.normal * (curving * step * step * l.curvature));
      };
      row(pa, 0, 1, 2);
      row(pb, 5, 4, 3);
    }

    std::array<occ::handle<Geom_BSplineCurve>, 6> curves;
    for (size_t j = 0; j < rows.size(); ++j) curves[j] = interpolated(rows[j]);
    const int nb_poles = curves[0]->NbPoles();
    for (const auto& curve : curves) {
      if (curve->NbPoles() != nb_poles || curve->NbKnots() != curves[0]->NbKnots()) {
        throw std::runtime_error("the blend's rows do not share a parameterisation");
      }
    }
    NCollection_Array2<gp_Pnt> poles(1, nb_poles, 1, 6);
    for (int i = 1; i <= nb_poles; ++i) {
      for (int j = 1; j <= 6; ++j) poles.SetValue(i, j, curves[j - 1]->Pole(i));
    }
    NCollection_Array1<double> v_knots(1, 2);
    v_knots.SetValue(1, 0.0);
    v_knots.SetValue(2, 1.0);
    NCollection_Array1<int> v_mults(1, 2);
    v_mults.Init(6);
    const occ::handle<Geom_BSplineSurface> surface =
        new Geom_BSplineSurface(poles, curves[0]->Knots(), v_knots, curves[0]->Multiplicities(),
                                v_mults, curves[0]->Degree(), 5);
    BRepBuilderAPI_MakeFace maker(surface, 1e-6);
    if (!maker.IsDone()) throw std::runtime_error("the blend did not make a face");
    const TopoDS_Face face = maker.Face();
    if (!BRepCheck_Analyzer(face).IsValid()) throw std::runtime_error("the blend is not valid");

    // By construction it holds. Measured because a sharp edge or a thin row could spoil it.
    for (const auto& [edge, side] : {std::pair(edge_a, face_a), std::pair(edge_b, face_b)}) {
      const Continuity c = measure_continuity(edge, face, side);
      if (c.gap > std::max(1e-3, 1e-3 * length)) {
        throw std::runtime_error("the blend drifted away from an edge");
      }
      if (c.angle > 0.5 * kDegree) {
        throw std::runtime_error("the blend is not tangent to a face along its edge");
      }
    }
    return put_shape(face);
  });
}

// Bits of `sides`: 1 U min, 2 U max, 4 V min, 8 V max.
API uint32_t occt_extend_face(uint32_t h, double length, uint32_t sides) {
  return guard_handle([&]() -> uint32_t {
    const TopoDS_Face face = as_face(get_shape(h));
    if (!(length > 0.0) || !std::isfinite(length)) {
      throw std::invalid_argument("the length must be positive");
    }
    if (sides == 0 || sides > 15) throw std::invalid_argument("no side to extend");
    TopoDS_Face extended;
    BRepLib::ExtendFace(face, length, (sides & 1) != 0, (sides & 2) != 0, (sides & 4) != 0,
                        (sides & 8) != 0, extended);
    if (extended.IsNull() || !BRepCheck_Analyzer(extended).IsValid()) {
      throw std::runtime_error("the extended face is not valid");
    }
    return put_shape(extended);
  });
}
