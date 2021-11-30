#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/physics/transformations.h>
#include <deal.II/physics/vector_relations.h>

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
tpipe(Triangulation<3, 3>                              &tria,
      const std::array<std::pair<Point<3>, double>, 3> &openings,
      const std::pair<Point<3>, double>                &bifurcation)
{
  constexpr unsigned int dim      = 3;
  constexpr unsigned int spacedim = 3;
  using vector                    = Tensor<1, spacedim, double>;

  constexpr unsigned int n_pipes   = 3;
  constexpr double       tolerance = 1.e-12;

  //
  // helper functions describing relations
  //

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
      const unsigned int n_slices =
        1 + std::ceil(skeleton_length[p] /
                      std::min(openings[p].second, bifurcation.second));
      // const unsigned int n_slices = 2; // DEBUG
      GridGenerator::extrude_triangulation(tria_base,
                                           n_slices,
                                           /*height*/ 1.,
                                           pipe);

      //
      // step 2: transform to pipe segment
      //
      const double polar_angle =
        Physics::VectorRelations::angle(skeleton[p], normal);
      Assert(std::abs(polar_angle) > tolerance &&
               std::abs(polar_angle - numbers::PI) > tolerance,
             ExcMessage("Invalid input."));
      const double cosecant_polar  = 1. / std::sin(polar_angle);
      const double cotangent_polar = std::cos(polar_angle) * cosecant_polar;

      // positive y -> right (cyclic) neighbor
      const double azimuth_angle_right =
        Physics::VectorRelations::signed_angle(skeleton_plane[p],
                                               skeleton_plane[cyclic(p)],
                                               /*axis=*/normal);
      Assert(std::abs(azimuth_angle_right) > tolerance,
             ExcMessage("Invalid input: at least two openings located "
                        "in same direction from bifurcation"));
      const double cotangent_azimuth_half_right =
        std::cos(.5 * azimuth_angle_right) / std::sin(.5 * azimuth_angle_right);

      // negative y -> left (anti-cyclic) neighbor
      const double azimuth_angle_left =
        Physics::VectorRelations::signed_angle(skeleton_plane[p],
                                               skeleton_plane[anticyclic(p)],
                                               /*axis=*/-normal);
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
        Assert(z_factor > 0,
               ExcMessage("Invalid input: at least one pipe segment "
                          "not long enough in this configuration"));
        const double z_new = z_factor * pt[2];

        return Point<spacedim>(x_new, y_new, z_new);
      };
      GridTools::transform(pipe_segment, pipe);

      //
      // step 3: rotate to match skeleton
      //
      const auto rotation_angle =
        Physics::VectorRelations::angle(directions[2], skeleton_unit[p]);
      const auto rotation_axis = [&]() {
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

      GridTools::rotate(
        skeleton_unit[p],
        Physics::VectorRelations::signed_angle(Rx,
                                               projected_normal,
                                               /*axis=*/skeleton_unit[p]),
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
  std::cout << "name: " << filestem << std::endl;

  GridOut    grid_out;
  const auto output = [&](const unsigned int n) {
    std::ofstream output(filestem + "-" + std::to_string(n) + ".vtk");
    grid_out.write_vtk(tria, output);

    std::cout << "  n: " << n << ", volume: " << GridTools::volume(tria)
              << std::endl;
  };

  output(0);
  for (unsigned int n = 1; n <= n_global_refinements; ++n)
    {
      tria.refine_global();
      output(n);
    }
}



/**
 * Tests a selection of different configurations.
 */
void
test_selection()
{
  constexpr unsigned int dim = 3;

  // ypipe in plane
  {
    const std::array<std::pair<Point<dim>, double>, 3> openings = {
      {{{-2., 0., 0.}, 1.},
       {{1., std::sqrt(3), 0.}, 1.},
       {{1., -std::sqrt(3), 0.}, 1.}}};

    const std::pair<Point<dim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim> tria;
    tpipe(tria, openings, bifurcation);

    refine_and_write(tria, 2, "ypipe");
  }

  // tpipe in plane
  {
    const std::array<std::pair<Point<dim>, double>, 3> openings = {
      {{{-2., 0., 0.}, 1.}, {{0., 2., 0.}, 1.}, {{2., 0., 0.}, 1.}}};

    const std::pair<Point<dim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim> tria;
    tpipe(tria, openings, bifurcation);

    refine_and_write(tria, 2, "tpipe");
  }

  // corner piece
  {
    const std::array<std::pair<Point<dim>, double>, 3> openings = {
      {{{2., 0., 0.}, 1.}, {{0., 2., 0.}, 1.}, {{0., 0., 2.}, 1.}}};

    const std::pair<Point<dim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim> tria;
    tpipe(tria, openings, bifurcation);

    refine_and_write(tria, 2, "corner");
  }

  // irregular configuration with arbitrary points
  {
    const std::array<std::pair<Point<dim>, double>, 3> openings = {
      {{{-4., 0., 0.}, 1.5}, {{4., -8., -0.4}, 0.75}, {{0.1, 0., -6.}, 0.5}}};

    const std::pair<Point<dim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim> tria;
    tpipe(tria, openings, bifurcation);

    refine_and_write(tria, 2, "irregular");
  }
}



/**
 * Returns the binomial coefficient (n choose k) via the multiplicative formula.
 */
unsigned int
n_choose_k(const unsigned int n, const unsigned int k)
{
  double result = 1.;
  for (unsigned int i = 1; i <= k; ++i)
    result *= (1. + n - i) / i;
  return static_cast<unsigned int>(result);
}



/**
 * Returns all possible permutations to choose k integers from the interval
 * [0,n-1].
 */
std::vector<std::vector<unsigned int>>
permutations(const unsigned int n, const unsigned int k)
{
  Assert(n >= k, ExcInternalError());

  if (k == 0)
    return std::vector<std::vector<unsigned int>>();

  // initialize mask
  // last k entries must be masked, otherwise std::next_permutation won't work
  std::vector<bool> mask(n);
  for (unsigned int i = 0; i < n; ++i)
    mask[i] = (i >= (n - k));

  const unsigned int                     n_permutations = n_choose_k(n, k);
  std::vector<std::vector<unsigned int>> permutations;
  permutations.reserve(n_permutations);
  do
    {
      // translate current mask permutation into indices
      std::vector<unsigned int> combination;
      combination.reserve(k);
      for (unsigned int i = 0; i < n; ++i)
        if (mask[i])
          combination.push_back(i);
      Assert(combination.size() == k, ExcInternalError());

      permutations.push_back(std::move(combination));
  } while (std::next_permutation(mask.begin(), mask.end()));
  Assert((permutations.size() == n_permutations), ExcInternalError());

  return permutations;
}



/**
 * Tests lots of permutations.
 */
void
test_permutations()
{
  // fixed constants
  constexpr unsigned int dim    = 3;
  constexpr unsigned int npipes = 3;

  // find n points in any coordinate direction
  constexpr int n_per_direction = 1;

  // parameters
  constexpr double radius  = 0.4;
  constexpr int    npoints = Utilities::pow(2 * n_per_direction + 1, dim) - 1;

  // set up all points for the test.
  // bifurcation will be in origin.
  std::pair<Point<dim>, double> bifurcation(Point<dim>(0, 0, 0), radius);

  // openings are located in a n_per_direction^3 box around the origin.
  std::vector<std::pair<Point<dim>, double>> points;
  points.reserve(npoints);
  for (int i = -n_per_direction; i <= n_per_direction; ++i)
    for (int j = -n_per_direction; j <= n_per_direction; ++j)
      for (int k = -n_per_direction; k <= n_per_direction; ++k)
        if (i != 0 || j != 0 || k != 0)
          points.emplace_back(Point<dim>(i, j, k), radius);
  Assert(points.size() == npoints, ExcInternalError());

  const auto perms = permutations(npoints, npipes);
  for (unsigned int c = 0; c < perms.size(); ++c)
    {
      std::cout << "Testing permutation " << c << " of " << perms.size()
                << std::endl;

      const auto &combination = perms[c];

      std::array<std::pair<Point<dim>, double>, npipes> openings;
      for (unsigned int i = 0; i < npipes; ++i)
        openings[i] = points[combination[i]];

      try
        {
          Triangulation<dim> tria;
          tpipe(tria, openings, bifurcation);
          tria.refine_global();
          GridTools::volume(tria);
        }
      catch (...)
        {
          std::cerr << "Exception on processing permutation " << c << " of "
                    << perms.size() << " with openings:";
          for (unsigned int i = 0; i < npipes; ++i)
            std::cerr << " (" << points[combination[i]].first << ")";
          std::cerr << std::endl;
        }
    }
}



/**
 * Exemplary application for a simple 3D T-pipe.
 */
int
main()
{
  test_selection();
  // test_permutations();

  return 0;
}
