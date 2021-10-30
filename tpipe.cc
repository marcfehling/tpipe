#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/physics/transformations.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <string>
#include <utility>


using namespace dealii;


/**
 * Initialize the given triangulation with a T-pipe.
 *
 * @param tria An empty triangulation which will hold the T-pipe geometry.
 * @param openings Center point and radius of each opening.
 * @param bifurcation Center point of the bifurcation and radius of each pipe at
 *                    the bifurcation.
 */
void
tpipe(Triangulation<3, 3> &                             tria,
      const std::array<std::pair<Point<3>, double>, 3> &openings,
      const std::pair<Point<3>, double> &               bifurcation)
{
  constexpr unsigned int dim      = 3;
  constexpr unsigned int spacedim = 3;
  using vector                    = Tensor<1, spacedim, double>;

  constexpr unsigned int n_pipes   = 3;
  constexpr double       tolerance = 1.e-12;

  //
  // helper functions describing relations
  //

  // calculate angle between two vectors with the dot product
  const auto angle = [](const vector &a, const vector &b) -> double {
    Assert(a.norm() > tolerance, ExcInternalError());
    Assert(b.norm() > tolerance, ExcInternalError());
    auto argument = (a * b) / a.norm() / b.norm();

    // std::acos returns nan when out of bounds [-1,+1].
    // if argument slightly overshoots these bounds, set it to the bound.
    if ((1. - std::abs(argument)) < tolerance)
      argument = std::copysign(1., argument);

    return std::acos(argument);
  };

  // calculate angle between two vectors with atan2
  const auto angle_around_axis =
    [](const vector &a, const vector &b, const vector &n) -> double {
    Assert(std::abs(n.norm() - 1.) < tolerance, ExcInternalError());
    Assert(std::abs(n * a) < tolerance, ExcInternalError());
    Assert(std::abs(n * b) < tolerance, ExcInternalError());

    const double dot = a * b;
    const double det = n * cross_product_3d(a, b);

    return std::atan2(det, dot);
  };


  const auto cyclic = [n_pipes](const unsigned int i) -> unsigned int {
    return (i < (n_pipes - 1)) ? i + 1 : 0;
  };

  const auto anticyclic = [n_pipes](const unsigned int i) -> unsigned int {
    return (i > 0) ? i - 1 : n_pipes - 1;
  };


  //
  // helper variables describing the geometry
  //

  // unit vectors representing Cartesian base
  constexpr std::array<vector, dim> directions = {
    {vector({1., 0., 0.}), vector({0., 1., 0.}), vector({0., 0., 1.})}};

  // create reference hyper-ball domain in 2D that will act as a cross-section
  // for each pipe and extract components of this reference triangulation
  const auto tria_base = []() {
    Triangulation<dim - 1, spacedim - 1> tria_base;
    GridGenerator::hyper_ball_balanced(tria_base,
                                       /*center=*/Point<spacedim - 1>(),
                                       /*radius=*/1.);
    return tria_base;
  }();

  // skeleton corresponding to the axis of symmetry in the center of each pipe
  const std::array<vector, n_pipes> skeleton = [&]() {
    std::array<vector, n_pipes> skeleton;
    for (unsigned int p = 0; p < n_pipes; ++p)
      skeleton[p] = bifurcation.first - openings[p].first;
    return skeleton;
  }();

  const auto skeleton_length = [&]() {
    std::array<double, n_pipes> skeleton_length;
    for (unsigned int p = 0; p < n_pipes; ++p)
      {
        skeleton_length[p] = skeleton[p].norm();
        Assert(skeleton_length[p] > tolerance,
               ExcMessage("Invalid input: bifurcation matches opening."))
      }
    return skeleton_length;
  }();

  const auto skeleton_unit = [&]() {
    std::array<vector, n_pipes> skeleton_unit;
    for (unsigned int p = 0; p < n_pipes; ++p)
      skeleton_unit[p] = skeleton[p] / skeleton_length[p];
    return skeleton_unit;
  }();

  // to determine the orientation of the pipes to each other, we will construct
  // a plane. starting from the bifurcation point, we will move by the length
  // one in each of the skeleton directions and span a plane with the three
  // points we reached.
  // the normal vector then describes the axis at which the peak edge of each
  // pipe meets. if we would interpret the bifurcation as a ball joint, the
  // normal vector would correspond the polar axis of the ball.
  const auto normal = [&]() {
    std::array<Point<spacedim>, n_pipes> points;
    for (unsigned int p = 0; p < n_pipes; ++p)
      points[p] = bifurcation.first - skeleton_unit[p];

    const auto normal =
      cross_product_3d(points[1] - points[0], points[2] - points[0]);
    Assert(normal.norm() > tolerance,
           ExcMessage("Invalid input: all three openings "
                      "are located on one line."));

    return normal / normal.norm();
  }();

  // components of each skeleton vector that are perpendicular to the normal
  // vector, or in other words, are located on the plane described above.
  // we will use them to describe the azimuth angles.
  const auto skeleton_plane = [&]() {
    std::array<vector, n_pipes> skeleton_plane;
    for (unsigned int p = 0; p < n_pipes; ++p)
      {
        skeleton_plane[p] = skeleton[p] - (skeleton[p] * normal) * normal;
        Assert(std::abs(skeleton_plane[p] * normal) < tolerance,
               ExcInternalError());
        Assert(skeleton_plane[p].norm() > tolerance,
               ExcMessage("Invalid input."));
      }
    return skeleton_plane;
  }();


  //
  // build pipe
  //
  tria.clear();
  for (unsigned int p = 0; p < n_pipes; ++p)
    {
      Triangulation<dim, spacedim> pipe;

      //
      // step 1: create unit cylinder
      //
      // r in [0,1], phi in [0,2Pi], z in [0,1]
      // number of intersections
      /*
      const unsigned int n_slices =
        std::max(2.,
                 std::ceil(
                   skeleton_length[p] /
                   (0.5 * std::min(openings[p].second, bifurcation.second))));
      */
      const unsigned int n_slices = 2; // DEBUG
      GridGenerator::extrude_triangulation(tria_base,
                                           n_slices,
                                           /*height*/ 1.,
                                           pipe);

      //
      // step 2: transform to pipe segment
      //
      const double polar_angle = angle(skeleton[p], normal);
      Assert(std::abs(polar_angle) > tolerance &&
               std::abs(polar_angle - numbers::PI) > tolerance,
             ExcMessage("Invalid input."));
      const double cosecant_polar  = 1. / std::sin(polar_angle);
      const double cotangent_polar = std::cos(polar_angle) * cosecant_polar;

      // positive y -> right (cyclic) neighbor
      const double azimuth_angle_right =
        angle_around_axis(skeleton_plane[p], skeleton_plane[cyclic(p)], normal);
      Assert(std::abs(azimuth_angle_right) > tolerance,
             ExcMessage("Invalid input: at least two openings located "
                        "in same direction from bifurcation"));
      const double cotangent_azimuth_half_right =
        std::cos(.5 * azimuth_angle_right) / std::sin(.5 * azimuth_angle_right);

      // negative y -> left (anti-cyclic) neighbor
      const double azimuth_angle_left =
        angle_around_axis(skeleton_plane[p],
                          skeleton_plane[anticyclic(p)],
                          -normal);
      Assert(std::abs(azimuth_angle_left) > tolerance,
             ExcMessage("Invalid input: at least two openings located "
                        "in same direction from bifurcation"));
      const double cotangent_azimuth_half_left =
        std::cos(.5 * azimuth_angle_left) / std::sin(.5 * azimuth_angle_left);

      const auto pipe_segment = [&](const Point<spacedim> &pt) {
        const double r_factor =
          (bifurcation.second - openings[p].second) * pt[2] +
          openings[p].second;
        const double x_new = r_factor * pt[0];
        const double y_new = r_factor * pt[1];

        const double z_factor = skeleton_length[p] + x_new * cotangent_polar -
                                std::abs(y_new) * cosecant_polar *
                                  ((pt[1] > 0) ? cotangent_azimuth_half_right :
                                                 cotangent_azimuth_half_left);
        const double z_new = z_factor * pt[2];

        return Point<spacedim>(x_new, y_new, z_new);
      };
      GridTools::transform(pipe_segment, pipe);

      //
      // step 3: rotate to match skeleton
      //
      const auto rotation_angle = angle(directions[2], skeleton_unit[p]);
      const auto rotation_axis  = [&]() {
        const auto rotation_axis =
          cross_product_3d(directions[2], skeleton_unit[p]);
        const auto norm = rotation_axis.norm();
        if (norm < tolerance)
          return directions[1];
        else
          return rotation_axis / norm;
      }();
      GridTools::rotate(rotation_axis, rotation_angle, pipe);

      // also rotate directions to identify misplacement
      // if dir[2] and unit are collinear, then do not rotate x
      const auto rotation_matrix =
        Physics::Transformations::Rotations::rotation_matrix_3d(rotation_axis,
                                                                rotation_angle);
      const auto Rx = rotation_matrix * directions[0];

      //
      // step 4: lateral rotation
      //
      // project the normal vector of the plane of all openings into the
      // cross-section of the current opening
      const auto projected_normal =
        normal - (normal * skeleton_unit[p]) * skeleton_unit[p];

      // both vectors must be in the opening plane
      Assert(std::abs(skeleton_unit[p] * projected_normal) < tolerance,
             ExcInternalError());
      Assert(std::abs(skeleton_unit[p] * Rx) < tolerance, ExcInternalError());

      GridTools::rotate(skeleton_unit[p],
                        angle_around_axis(Rx,
                                          projected_normal,
                                          skeleton_unit[p]),
                        pipe);

      //
      // step 5: shift to position
      //
      GridTools::shift(openings[p].first, pipe);

#if false
      std::ofstream out("pipe-" + std::to_string(p) + ".vtk");
      GridOut().write_vtk(pipe, out);
#endif

      GridGenerator::merge_triangulations(
        pipe, tria, tria, tolerance, /*copy_manifold_ids=*/true);
    }
}



/**
 * Refines the provided triangulation globally by the specified amount of times.
 *
 * Writes the coarse mesh as well as the mesh after each global refinement
 * to the filesystem in VTK format. Also prints the volume of the mesh to the
 * console for debugging purposes.
 */
template <int dim, int spacedim>
void
refine_and_write(Triangulation<dim, spacedim> &tria,
                 const unsigned int            n_global_refinements = 0,
                 const std::string             filestem             = "grid")
{
  GridOut grid_out;

  auto output = [&](const unsigned int n) {
    std::ofstream output(filestem + "-" + std::to_string(n) + ".vtk");
    grid_out.write_vtk(tria, output);

    std::cout << "name: " << filestem << ", n: " << n
              << ", volume: " << GridTools::volume(tria) << std::endl;
  };

  output(0);
  for (unsigned int n = 1; n <= n_global_refinements; ++n)
    {
      tria.refine_global();
      output(n);
    }
}



/**
 * Exemplary application for a simple 3D T-pipe.
 */
int
main()
{
  constexpr unsigned int dim = 3;

  // ypipe in plane
  {
    const std::array<std::pair<Point<dim>, double>, 3> openings = {
      {{{2., 0., 0.}, 1.}, {{0., 2., 0.}, 1.}, {{0., 0., 2.}, 1.}}};

    const std::pair<Point<dim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim> tria;
    tpipe(tria, openings, bifurcation);

    refine_and_write(tria, 2, "ypipe_plane");
  }

  // tpipe in plane
  {
    const std::array<std::pair<Point<dim>, double>, 3> openings = {
      {{{2., 0., 0.}, 1.}, {{-2., 0., 0.}, 1.}, {{0., 2., 0.}, 1.}}};

    const std::pair<Point<dim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim> tria;
    tpipe(tria, openings, bifurcation);

    refine_and_write(tria, 2, "tpipe_plane");
  }

  // weird pipe
  {
    const std::array<std::pair<Point<dim>, double>, 3> openings = {
      {{{-4., 0., 0.}, 1.}, {{4., -8., -0.4}, 0.75}, {{0.1, 0., -4.}, 0.5}}};

    const std::pair<Point<dim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim> tria;
    tpipe(tria, openings, bifurcation);

    refine_and_write(tria, 2, "ypipe");
  }

  return 0;
}
