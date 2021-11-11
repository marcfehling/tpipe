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
 * Initialize the given triangulation with a tee, which is the intersection of
 * three truncated cones.
 *
 * The geometry has four characteristic cross sections, located at the three
 * openings and the bifurcation. They need to be specified via the function's
 * arguments: each cross section is described by a characteristic point and a
 * radius. The cross sections at the openings are circles and are described by
 * their center point and radius. The bifurcation point describes where the
 * symmetry axes of all cones meet.
 *
 * Each truncated cone is transformed so that the three merge seamlessly into
 * each other. The bifurcation radius describes the radius that each original,
 * untransformed, truncated cone would have at the bifurcation. This radius is
 * necessary for the construction of the geometry and can, in general, no longer
 * be found in the final result.
 *
 * Each cone will be assigned a distinct <em>material ID</em> that matches the
 * index of their opening in the argument @p openings. For example, the cone
 * which connects to opening with index 0 in @p openings will have material ID 0.
 *
 * Similarly, <em>boundary IDs</em> are assigned to the cross-sections of each
 * opening to match their index. All other boundary faces will be assigned
 * boundary ID 3.
 *
 * <em>Manifold IDs</em> will be set on the mantles of each truncated cone in
 * the same way. Each cone will have a Manifold ??? assigned.
 *
 * @pre The triangulation passed as argument needs to be empty when calling this
 * function.
 *
 * @note Only implemented for `dim = 3` and `spacedim = 3`.
 *
 * @param tria An empty triangulation which will hold the tee geometry.
 * @param openings Center point and radius of each opening.
 * @param bifurcation Center point of the bifurcation and hypothetical radius of
 *                    each truncated cone at the bifurcation.
 */
void
tee(Triangulation<3, 3> &                             tria,
    const std::array<std::pair<Point<3>, double>, 3> &openings,
    const std::pair<Point<3>, double> &               bifurcation)
{
  constexpr unsigned int dim      = 3;
  constexpr unsigned int spacedim = 3;
  using vector                    = Tensor<1, spacedim, double>;

  constexpr unsigned int n_pipes   = 3;
  constexpr double       tolerance = 1.e-12;

  // Each pipe segment will be identified by the index of its opening in the
  // parameter array. To determine the next and previous entry in the array for
  // a given index, we create auxiliary functions.
  const auto cyclic = [n_pipes](const unsigned int i) -> unsigned int {
    return (i < (n_pipes - 1)) ? i + 1 : 0;
  };
  const auto anticyclic = [n_pipes](const unsigned int i) -> unsigned int {
    return (i > 0) ? i - 1 : n_pipes - 1;
  };

  // Cartesian base represented by unit vectors.
  constexpr std::array<vector, spacedim> directions = {
    {vector({1., 0., 0.}), vector({0., 1., 0.}), vector({0., 0., 1.})}};

  // The skeleton corresponds to the axis of symmetry in the center of each
  // pipe segment. Each skeleton vector points from the associated opening to
  // the common bifurcation point. For convenience, we also compute length and
  // unit vector of every skeleton vector here.
  const auto skeleton = [&]() {
    std::array<vector, n_pipes> skeleton;
    for (unsigned int p = 0; p < n_pipes; ++p)
      skeleton[p] = bifurcation.first - openings[p].first;
    return skeleton;
  }();

  const auto skeleton_length = [&]() {
    std::array<double, n_pipes> skeleton_length;
    for (unsigned int p = 0; p < n_pipes; ++p)
      skeleton_length[p] = skeleton[p].norm();
    return skeleton_length;
  }();

#ifdef DEBUG
  // In many assertions that come up below, we will verify the integrity of the
  // geometry. For this, we introduce a tolerance length which vectors must
  // exceed to avoid being considered "too short". We relate this length to the
  // longest pipe segment.
  const double tolerance_length =
    tolerance *
    *std::max_element(skeleton_length.begin(), skeleton_length.end());
#endif

  for (unsigned int p = 0; p < n_pipes; ++p)
    Assert(skeleton_length[p] > tolerance_length,
           ExcMessage("Invalid input: bifurcation matches opening."));

  const auto skeleton_unit = [&]() {
    std::array<vector, n_pipes> skeleton_unit;
    for (unsigned int p = 0; p < n_pipes; ++p)
      skeleton_unit[p] = skeleton[p] / skeleton_length[p];
    return skeleton_unit;
  }();

  // To determine the orientation of the pipe segments to each other, we will
  // construct a plane: starting from the bifurcation point, we will move by the
  // magnitude one in each of the skeleton directions and span a plane with the
  // three points we reached.
  //
  // The normal vector of this particular plane then describes the edge at which
  // all pipe segments meet. If we would interpret the bifurcation as a ball
  // joint, the normal vector would correspond to the polar axis of the ball.
  const auto normal = [&]() {
    std::array<Point<spacedim>, n_pipes> points;
    for (unsigned int p = 0; p < n_pipes; ++p)
      points[p] = bifurcation.first - skeleton_unit[p];

    const auto normal =
      cross_product_3d(points[1] - points[0], points[2] - points[0]);
    Assert(normal.norm() > tolerance_length,
           ExcMessage("Invalid input: all three openings "
                      "are located on one line."));

    return normal / normal.norm();
  }();

  // Projections of all skeleton vectors perpendicular to the normal vector, or
  // in other words, onto the plane described above.
  const auto skeleton_plane = [&]() {
    std::array<vector, n_pipes> skeleton_plane;
    for (unsigned int p = 0; p < n_pipes; ++p)
      {
        skeleton_plane[p] = skeleton[p] - (skeleton[p] * normal) * normal;
        Assert(std::abs(skeleton_plane[p] * normal) <
                 tolerance * skeleton_plane[p].norm(),
               ExcInternalError());
        Assert(skeleton_plane[p].norm() > tolerance_length,
               ExcMessage("Invalid input."));
      }
    return skeleton_plane;
  }();

  // Create a hyperball domain in 2D that will act as the reference cross
  // section for each pipe segment.
  const auto tria_base = []() {
    Triangulation<dim - 1, spacedim - 1> tria_base;
    GridGenerator::hyper_ball_balanced(tria_base,
                                       /*center=*/Point<spacedim - 1>(),
                                       /*radius=*/1.);
    return tria_base;
  }();

  // Now move on to actually build the tee geometry!
  //
  // For each pipe segment, we create a separate triangulation object which will
  // be merged with the parameter triangulation in the end.
  Assert(tria.n_cells() == 0,
         ExcMessage("The output triangulation object needs to be empty."));
  for (unsigned int p = 0; p < n_pipes; ++p)
    {
      Triangulation<dim, spacedim> pipe;

      //
      // Step 1: create unit cylinder
      //
      // We create a unit cylinder by extrusion from the base cross section.
      // The number of layers depends on the ratio of the length of the skeleton
      // and the minimal radius in the pipe segment.
      const unsigned int n_slices =
        1 + std::ceil(skeleton_length[p] /
                      std::min(openings[p].second, bifurcation.second));
      GridGenerator::extrude_triangulation(tria_base,
                                           n_slices,
                                           /*height*/ 1.,
                                           pipe);

      // Set all material, boundary, and manifold indicators on the unit
      // cylinder, simply because they are easier to handle in this geometry.
      // See general documentation of this function.
      for (const auto &cell : pipe.active_cell_iterators())
        {
          cell->set_material_id(p);

          for (const auto &face : cell->face_iterators())
            if (face->at_boundary())
              {
                const auto center_z = face->center()[2];

                if (std::abs(center_z) < tolerance)
                  {
                    // opening cross section
                    face->set_boundary_id(p);
                  }
                else if (std::abs(center_z - 1.) < tolerance)
                  {
                    // bifurcation cross section
                  }
                else
                  {
                    // cone mantle
                    face->set_boundary_id(n_pipes);
                    face->set_manifold_id(p);
                  }
              }
        }

      //
      // Step 2: transform unit cylinder to pipe segment
      //
      // For the given cylinder, we will interpret the base in the xy-plane as
      // the cross section of the opening, and the base at z=1 as the surface
      // where all pipe segments meet. On the latter surface, we assign the
      // section in positive y-direction to face the next (right/cyclic) pipe
      // segment, and allocate the domain in negative y-direction to border the
      // previous (left/anticyclic) pipe segment.
      //
      // In the end, the transformed pipe segment will look like this:
      //              z                   z
      //              ^                   ^
      //         left | right             |  /|
      //   anticyclic | cyclic            |/  |
      //             /|\                 /|   |
      //           /  |  \             /  |   |
      //          |   |   |           |   |   |
      //          |   |   |           |   |   |
      //        ------+----->y      ------+----->x

      // Before transforming the unit cylinder however, we compute angle
      // relations between the skeleton vectors viewed from the bifurcation
      // point. For this purpose, we interpret the bifurcation as a ball joint
      // as described above.
      //
      // In spherical coordinates, the polar angle describes the kink of the
      // skeleton vector with respect to the polar axis. If all openings and the
      // bifurcation are located on a plane, then this angle is pi/2 for every
      // pipe segment.
      const double polar_angle =
        Physics::VectorRelations::angle(skeleton[p], normal);
      Assert(std::abs(polar_angle) > tolerance &&
               std::abs(polar_angle - numbers::PI) > tolerance,
             ExcMessage("Invalid input."));
      const double cosecant_polar  = 1. / std::sin(polar_angle);
      const double cotangent_polar = std::cos(polar_angle) * cosecant_polar;

      // Further, we compute the angles between this pipe segment to the other
      // two. The angle corresponds to the azimuthal direction if we stick to
      // the picture of the ball joint.
      const double azimuth_angle_right =
        Physics::VectorRelations::signed_angle(skeleton_plane[p],
                                               skeleton_plane[cyclic(p)],
                                               /*axis=*/normal);
      Assert(std::abs(azimuth_angle_right) > tolerance,
             ExcMessage("Invalid input: at least two openings located "
                        "in same direction from bifurcation"));
      const double cotangent_azimuth_half_right =
        std::cos(.5 * azimuth_angle_right) / std::sin(.5 * azimuth_angle_right);

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
        // We transform the cylinder in x- and y-direction to become a truncated
        // cone, similarly to GridGenerator::truncated_cone().
        const double r_factor =
          (bifurcation.second - openings[p].second) * pt[2] +
          openings[p].second;
        const double x_new = r_factor * pt[0];
        const double y_new = r_factor * pt[1];

        // Further, to be able to smoothly merge all pipe segments at the
        // bifurcation, we also need to transform in z-direction.
        const double z_factor =
          // Scale the unit cylinder to the correct length.
          skeleton_length[p]
          // Next, adjust for the polar angle. This part will be zero if all
          // openings and the bifurcation are located on a plane.
          + x_new * cotangent_polar
          // Last, adjust for the azimuth angle.
          - std::abs(y_new) * cosecant_polar *
              ((y_new > 0) ? cotangent_azimuth_half_right :
                             cotangent_azimuth_half_left);
        Assert(z_factor > 0,
               ExcMessage("Invalid input: at least one pipe segment "
                          "is not long enough in this configuration"));
        const double z_new = z_factor * pt[2];

        return Point<spacedim>(x_new, y_new, z_new);
      };
      GridTools::transform(pipe_segment, pipe);

      //
      // Step 3: rotate pipe segment to match skeleton direction
      //
      // The symmetry axis of the pipe segment in its current state points in
      // positive z-direction. We rotate the pipe segment that its symmetry axis
      // matches the direction of the skeleton vector. For this purpose, we
      // rotate the pipe segment around the axis that is described by the cross
      // product of both vectors.
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

      //
      // Step 4: rotate laterally to align pipe segments
      //
      // On the unit cylinder, we find that the edge on which all pipe segments
      // meet is parallel to the x-axis. After the transformation to the pipe
      // segment, we notice that this statement still holds for the projection
      // of this edge onto the xy-plane, which corresponds to the cross section
      // of the opening.
      //
      // With the latest rotation however, this is no longer the case. We rotate
      // the unit vector in x-direction in the same fashion, which gives us the
      // current direction of the projected edge.
      const auto rotation_matrix =
        Physics::Transformations::Rotations::rotation_matrix_3d(rotation_axis,
                                                                rotation_angle);
      const auto Rx = rotation_matrix * directions[0];

      // To determine how far we need to rotate, we also need to project the
      // polar axis of the bifurcation ball joint into the same plane.
      const auto projected_normal =
        normal - (normal * skeleton_unit[p]) * skeleton_unit[p];

      // Both the projected normal and Rx must be in the opening plane.
      Assert(std::abs(skeleton_unit[p] * projected_normal) < tolerance,
             ExcInternalError());
      Assert(std::abs(skeleton_unit[p] * Rx) < tolerance, ExcInternalError());

      // Now we laterally rotate the pipe segment around its own symmetry axis
      // that the edge matches the polar axis.
      GridTools::rotate(
        skeleton_unit[p],
        Physics::VectorRelations::signed_angle(Rx,
                                               projected_normal,
                                               /*axis=*/skeleton_unit[p]),
        pipe);

      //
      // Step 5: shift to final position
      //
      GridTools::shift(openings[p].first, pipe);

#if false
      std::ofstream out("pipe-" + std::to_string(p) + ".vtk");
      GridOut().write_vtk(pipe, out);
#endif

      GridGenerator::merge_triangulations(
        pipe, tria, tria, tolerance, /*copy_manifold_ids=*/true);
    }

  // Since GridGenerator::merge_triangulation() does not copy boundary IDs, we
  // need to set them after the final geometry is created. Luckily, boundary IDs
  // match with manifold IDs, so we simply translate them.
  // TODO
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
 * Tests a selection of common configurations of tee geometries.
 */
void
test_selection()
{
  constexpr unsigned int dim      = 3;
  constexpr unsigned int spacedim = 3;

  // y-pipe in plane
  {
    const std::array<std::pair<Point<spacedim>, double>, 3> openings = {
      {{{-2., 0., 0.}, 1.},
       {{1., std::sqrt(3), 0.}, 1.},
       {{1., -std::sqrt(3), 0.}, 1.}}};

    const std::pair<Point<spacedim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim, spacedim> tria;
    tee(tria, openings, bifurcation);

    refine_and_write(tria, 2, "ypipe");
  }

  // t-pipe in plane
  {
    const std::array<std::pair<Point<spacedim>, double>, 3> openings = {
      {{{-2., 0., 0.}, 1.}, {{0., 2., 0.}, 1.}, {{2., 0., 0.}, 1.}}};

    const std::pair<Point<spacedim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim, spacedim> tria;
    tee(tria, openings, bifurcation);

    refine_and_write(tria, 2, "tpipe");
  }

  // corner piece
  {
    const std::array<std::pair<Point<spacedim>, double>, 3> openings = {
      {{{2., 0., 0.}, 1.}, {{0., 2., 0.}, 1.}, {{0., 0., 2.}, 1.}}};

    const std::pair<Point<spacedim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim, spacedim> tria;
    tee(tria, openings, bifurcation);

    refine_and_write(tria, 2, "corner");
  }

  // irregular configuration with arbitrary points
  {
    const std::array<std::pair<Point<spacedim>, double>, 3> openings = {
      {{{-4., 0., 0.}, 1.5}, {{4., -8., -0.4}, 0.75}, {{0.1, 0., -6.}, 0.5}}};

    const std::pair<Point<spacedim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim, spacedim> tria;
    tee(tria, openings, bifurcation);

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
  // last k entries must be masked, otherwise algorithm doesn't work
  // (either because of std::vector<bool> or std::next_permutation???)
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
  constexpr unsigned int dim      = 3;
  constexpr unsigned int spacedim = 3;
  constexpr unsigned int npipes   = 3;

  // find n points in any coordinate direction
  constexpr int n_per_direction = 1;

  // parameters
  constexpr double radius  = 0.4;
  constexpr int    npoints = Utilities::pow(2 * n_per_direction + 1, dim) - 1;

  // set up all points for the test.
  // bifurcation will be in origin.
  const std::pair<Point<spacedim>, double> bifurcation(
    Point<spacedim>(0., 0., 0.), radius);

  // openings are located in a n_per_direction^3 box around the origin.
  std::vector<Point<spacedim>> points;
  points.reserve(npoints);
  for (int i = -n_per_direction; i <= n_per_direction; ++i)
    for (int j = -n_per_direction; j <= n_per_direction; ++j)
      for (int k = -n_per_direction; k <= n_per_direction; ++k)
        if (i != 0 || j != 0 || k != 0)
          points.emplace_back(i, j, k);
  Assert(points.size() == npoints, ExcInternalError());

  const auto perms = permutations(npoints, npipes);
  for (unsigned int c = 0; c < perms.size(); ++c)
    {
      const auto &combination = perms[c];

      std::array<std::pair<Point<spacedim>, double>, npipes> openings;
      for (unsigned int i = 0; i < npipes; ++i)
        openings[i] = {points[combination[i]], radius};

      std::cout << "Testing permutation " << c << " of " << perms.size()
                << " with openings:";
      for (unsigned int i = 0; i < npipes; ++i)
        std::cout << " (" << openings[i].first << ")";
      std::cout << std::endl;

      try
        {
          Triangulation<dim, spacedim> tria;
          tee(tria, openings, bifurcation);
          tria.refine_global();
          GridTools::volume(tria);
        }
      catch (...)
        {
          std::cerr << "Exception on processing permutation " << c << " of "
                    << perms.size() << " with openings:";
          for (unsigned int i = 0; i < npipes; ++i)
            std::cerr << " (" << openings[i].first << ")";
          std::cerr << std::endl;
        }
    }
}



int
main()
{
  test_selection();
  // test_permutations();

  return 0;
}
