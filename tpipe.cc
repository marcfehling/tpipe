#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
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


namespace
{
  /**
   * Auxiliary constructs for the pipe junction geometry.
   *
   * Please refer to the in-source documentation of the pipe_junction function
   * below for more information about the individual contents.
   */
  namespace PipeSegment
  {
    /**
     * Selection of pipe segment properties to calculate its height with the
     * function below.
     */
    struct AdditionalData
    {
      double skeleton_length;

      double cosecant_polar;
      double cotangent_polar;
      double cotangent_azimuth_half_right;
      double cotangent_azimuth_half_left;
    };



    /**
     * Calculate the height of a pipe segment, depending on the location in the
     * x-y plane.
     */
    inline double
    compute_z_expansion(const double          x,
                        const double          y,
                        const AdditionalData &data)
    {
      return
        // Scale the unit cylinder to the correct length.
        data.skeleton_length
        // Next, adjust for the polar angle. This part will be zero if all
        // openings and the bifurcation are located on a plane.
        + x * data.cotangent_polar
        // Last, adjust for the azimuth angle.
        - std::abs(y) * data.cosecant_polar *
            ((y > 0) ? data.cotangent_azimuth_half_right :
                       data.cotangent_azimuth_half_left);
    }



    /**
     * Pipe segment manifold description.
     *
     * The manifold class is too specific to being of any other use than for the
     * pipe junction geometry.
     */
    template <int dim, int spacedim = dim>
    class Manifold : public ChartManifold<dim, spacedim, 3>
    {
    public:
      /**
       * Constructor. The manifold described is a pipe segment whose central
       * axis points in direction
       * @p direction and goes through the given @p point_on_axis.
       *
       *
       */
      Manifold(const Tensor<1, spacedim> &normal_direction,
               const Tensor<1, spacedim> &direction,
               const Point<spacedim>     &point_on_axis,
               const AdditionalData      &data,
               const double               tolerance = 1e-10);

      /**
       * Make a clone of this Manifold object.
       */
      virtual std::unique_ptr<dealii::Manifold<dim, spacedim>>
      clone() const override;

      /**
       * Compute the cylindrical coordinates $(r, \phi, \lambda)$ for the given
       * space point and map them to the unit cylinder, where $r$ denotes the
       * distance from the axis, $\phi$ the angle between the given point and
       * the computed normal direction, and $\lambda$ the axial position.
       */
      virtual Point<3>
      pull_back(const Point<spacedim> &space_point) const override;

      /**
       * Compute the Cartesian coordinates for a chart point given in
       * cylindrical coordinates $(r, \phi, \lambda)$ on a unit cylinder, where
       * $r$ denotes the distance from the axis, $\phi$ the angle between the
       * given point and the computed normal direction, and $\lambda$ the axial
       * position.
       */
      virtual Point<spacedim>
      push_forward(const Point<3> &chart_point) const override;

    protected:
      /**
       * A vector orthogonal to the normal direction.
       */
      const Tensor<1, spacedim> normal_direction;

      /**
       * The direction vector of the axis.
       */
      const Tensor<1, spacedim> direction;

      /**
       * An arbitrary point on the axis.
       */
      const Point<spacedim> point_on_axis;

    private:
      /**
       * Pipe segment properties to calculate its height.
       */
      const AdditionalData data;

      /**
       * Relative tolerance to measure zero distances.
       */
      const double tolerance;

      /**
       * The direction vector perpendicular to both direction and
       * normal_direction.
       */
      const Tensor<1, spacedim> dxn;
    };



    template <int dim, int spacedim>
    Manifold<dim, spacedim>::Manifold(
      const Tensor<1, spacedim> &normal_direction,
      const Tensor<1, spacedim> &direction,
      const Point<spacedim>     &point_on_axis,
      const AdditionalData      &data,
      const double               tolerance)
      : ChartManifold<dim, spacedim, 3>(Tensor<1, 3>({0, 2. * numbers::PI, 0}))
      , normal_direction(normal_direction)
      , direction(direction)
      , point_on_axis(point_on_axis)
      , data(data)
      , tolerance(tolerance)
      , dxn(cross_product_3d(direction, normal_direction))
    {
      Assert(spacedim == 3,
             ExcMessage(
               "PipeSegment::Manifold can only be used for spacedim==3!"));

      Assert(std::abs(normal_direction.norm() - 1) < tolerance,
             ExcMessage("Normal direction must be unit vector."));
      Assert(std::abs(direction.norm() - 1) < tolerance,
             ExcMessage("Direction must be unit vector."));
      Assert(normal_direction * direction < tolerance,
             ExcMessage(
               "Direction and normal direction must be perpendicular."));
    }



    template <int dim, int spacedim>
    std::unique_ptr<dealii::Manifold<dim, spacedim>>
    Manifold<dim, spacedim>::clone() const
    {
      return std::make_unique<Manifold<dim, spacedim>>(
        normal_direction, direction, point_on_axis, data, tolerance);
    }



    template <int dim, int spacedim>
    Point<3>
    Manifold<dim, spacedim>::pull_back(const Point<spacedim> &space_point) const
    {
      // First find the projection of the given point to the axis.
      const Tensor<1, spacedim> normalized_point = space_point - point_on_axis;
      double                    lambda           = normalized_point * direction;
      const Point<spacedim>     projection = point_on_axis + direction * lambda;
      const Tensor<1, spacedim> p_diff     = space_point - projection;
      const double              r          = p_diff.norm();

      Assert(r > tolerance * data.skeleton_length,
             ExcMessage(
               "This class won't handle points on the direction axis."));

      // Then compute the angle between the projection direction and
      // another vector orthogonal to the direction vector.
      const double phi =
        Physics::VectorRelations::signed_angle(normal_direction,
                                               p_diff,
                                               /*axis=*/direction);

      lambda /= compute_z_expansion(r * std::cos(phi), r * std::sin(phi), data);

      // Return distance from the axis, angle and signed distance on the axis.
      return Point<3>(r, phi, lambda);
    }



    template <int dim, int spacedim>
    Point<spacedim>
    Manifold<dim, spacedim>::push_forward(const Point<3> &chart_point) const
    {
      // Rotate the orthogonal direction by the given angle.
      const double sine_r   = chart_point(0) * std::sin(chart_point(1));
      const double cosine_r = chart_point(0) * std::cos(chart_point(1));
      const double lambda =
        chart_point(2) * compute_z_expansion(cosine_r, sine_r, data);

      const Tensor<1, spacedim> intermediate =
        normal_direction * cosine_r + dxn * sine_r;

      // Finally, put everything together.
      return point_on_axis + direction * lambda + intermediate;
    }
  } // namespace PipeSegment
} // namespace



/**
 * Initialize the given triangulation with a pipe junction, which is the
 * intersection of three truncated cones.
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
 * the same way. Each cone will have a special manifold object assigned, which
 * is based on the CylindricalManifold class. Further, all cells adjacent to the
 * mantle are given the manifold ID 3. If desired, you can assign an (expensive)
 * TransfiniteInterpolationManifold object to that particular layer of cells
 * with the following code snippet.
 * @code
 * TransfiniteInterpolationManifold<3> transfinite;
 * transfinite.initialize(triangulation);
 * triangulation.set_manifold(3, transfinite);
 * @endcode
 *
 * @pre The triangulation passed as argument needs to be empty when calling this
 * function.
 *
 * @note Only implemented for `dim = 3` and `spacedim = 3`.
 *
 * @param tria An empty triangulation which will hold the pipe junction geometry.
 * @param openings Center point and radius of each opening.
 * @param bifurcation Center point of the bifurcation and hypothetical radius of
 *                    each truncated cone at the bifurcation.
 */
void
pipe_junction(Triangulation<3, 3>                            &tria,
              const std::vector<std::pair<Point<3>, double>> &openings,
              const std::pair<Point<3>, double>              &bifurcation)
{
  constexpr unsigned int dim      = 3;
  constexpr unsigned int spacedim = 3;
  using vector                    = Tensor<1, spacedim, double>;

  constexpr unsigned int n_pipes   = 3;
  constexpr double       tolerance = 1.e-12;

  // TODO: MSVC can't capture const or constexpr values in lambda functions
  // (due to either a missing implementation or a bug). Instead, we need to
  // duplicate the declaration in all lambda functions below.
  //   See also: https://developercommunity.visualstudio.com/t/
  //             invalid-template-argument-expected-compile-time-co/187862

#ifdef DEBUG
  // Verify user input.
  Assert(bifurcation.second > 0, ExcMessage("Invalid input: negative radius."));
  Assert(openings.size() == n_pipes,
         ExcMessage("Invalid input: only 3 openings allowed."));
  for (const auto &opening : openings)
    Assert(opening.second > 0, ExcMessage("Invalid input: negative radius."));
#endif

  // Each pipe segment will be identified by the index of its opening in the
  // parameter array. To determine the next and previous entry in the array for
  // a given index, we create auxiliary functions.
  const auto cyclic = [](const unsigned int i) -> unsigned int {
    constexpr unsigned int n_pipes = 3;
    return (i < (n_pipes - 1)) ? i + 1 : 0;
  };
  const auto anticyclic = [](const unsigned int i) -> unsigned int {
    constexpr unsigned int n_pipes = 3;
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
    constexpr unsigned int      n_pipes = 3;
    std::array<vector, n_pipes> skeleton;
    for (unsigned int p = 0; p < n_pipes; ++p)
      skeleton[p] = bifurcation.first - openings[p].first;
    return skeleton;
  }();

  const auto skeleton_length = [&]() {
    constexpr unsigned int      n_pipes = 3;
    std::array<double, n_pipes> skeleton_length;
    for (unsigned int p = 0; p < n_pipes; ++p)
      skeleton_length[p] = skeleton[p].norm();
    return skeleton_length;
  }();

  // In many assertions that come up below, we will verify the integrity of the
  // geometry. For this, we introduce a tolerance length which vectors must
  // exceed to avoid being considered "too short". We relate this length to the
  // longest pipe segment.
  const double tolerance_length =
    tolerance *
    *std::max_element(skeleton_length.begin(), skeleton_length.end());

  for (unsigned int p = 0; p < n_pipes; ++p)
    Assert(skeleton_length[p] > tolerance_length,
           ExcMessage("Invalid input: bifurcation matches opening."));

  const auto skeleton_unit = [&]() {
    constexpr unsigned int      n_pipes = 3;
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
    const auto normal = cross_product_3d(skeleton_unit[1] - skeleton_unit[0],
                                         skeleton_unit[2] - skeleton_unit[0]);
    Assert(normal.norm() > tolerance_length,
           ExcMessage("Invalid input: all three openings "
                      "are located on one line."));

    return normal / normal.norm();
  }();

  // Projections of all skeleton vectors perpendicular to the normal vector, or
  // in other words, onto the plane described above.
  const auto skeleton_plane = [&]() {
    constexpr unsigned int      n_pipes = 3;
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
    constexpr unsigned int               dim      = 3;
    constexpr unsigned int               spacedim = 3;
    Triangulation<dim - 1, spacedim - 1> tria_base;
    GridGenerator::hyper_ball_balanced(tria_base,
                                       /*center=*/Point<spacedim - 1>(),
                                       /*radius=*/1.);
    return tria_base;
  }();

  // Now move on to actually build the pipe junction geometry!
  //
  // For each pipe segment, we create a separate triangulation object which will
  // be merged with the parameter triangulation in the end.
  Assert(tria.n_cells() == 0,
         ExcMessage("The output triangulation object needs to be empty."));

  std::vector<PipeSegment::Manifold<dim, spacedim>> manifolds;
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
        1 + static_cast<unsigned int>(
              std::ceil(skeleton_length[p] /
                        std::min(openings[p].second, bifurcation.second)));
      // const unsigned int n_slices = 2; // DEBUG
      GridGenerator::extrude_triangulation(tria_base,
                                           n_slices,
                                           /*height*/ 1.,
                                           pipe);

      // Set all material and manifold indicators on the unit cylinder, simply
      // because they are easier to handle in this geometry. We will set
      // boundary indicators at the end of the function. See general
      // documentation of this function.
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
                  }
                else if (std::abs(center_z - 1.) < tolerance)
                  {
                    // bifurcation cross section
                  }
                else
                  {
                    // cone mantle
                    cell->set_all_manifold_ids(n_pipes);
                    face->set_all_manifold_ids(p);
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

      const double azimuth_angle_left =
        Physics::VectorRelations::signed_angle(skeleton_plane[p],
                                               skeleton_plane[anticyclic(p)],
                                               /*axis=*/-normal);
      Assert(std::abs(azimuth_angle_left) > tolerance,
             ExcMessage("Invalid input: at least two openings located "
                        "in same direction from bifurcation"));

      // We compute some trigonometric relations with these angles, and store
      // them conveniently in a struct to be reused later.
      const auto data = [&]() {
        PipeSegment::AdditionalData data;
        data.skeleton_length = skeleton_length[p];
        data.cosecant_polar  = 1. / std::sin(polar_angle);
        data.cotangent_polar = std::cos(polar_angle) * data.cosecant_polar;
        data.cotangent_azimuth_half_right = std::cos(.5 * azimuth_angle_right) /
                                            std::sin(.5 * azimuth_angle_right);
        data.cotangent_azimuth_half_left =
          std::cos(.5 * azimuth_angle_left) / std::sin(.5 * azimuth_angle_left);
        return data;
      }();

      // Now transform the cylinder as described above.
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
          PipeSegment::compute_z_expansion(x_new, y_new, data);
        Assert(z_factor > 0,
               ExcMessage("Invalid input: at least one pipe segment "
                          "is not long enough in this configuration"));
        const double z_new = z_factor * pt[2];

        constexpr unsigned int spacedim = 3;
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
      const auto rotation_matrix =
        Physics::Transformations::Rotations::rotation_matrix_3d(rotation_axis,
                                                                rotation_angle);
      GridTools::transform(
        [&](const Point<spacedim> &pt) { return rotation_matrix * pt; }, pipe);

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
      const auto Rx = rotation_matrix * directions[0];

      // To determine how far we need to rotate, we also need to project the
      // polar axis of the bifurcation ball joint into the same plane.
      const auto normal_projected_on_opening =
        normal - (normal * skeleton_unit[p]) * skeleton_unit[p];

      // Both the projected normal and Rx must be in the opening plane.
      Assert(std::abs(skeleton_unit[p] * normal_projected_on_opening) <
               tolerance,
             ExcInternalError());
      Assert(std::abs(skeleton_unit[p] * Rx) < tolerance, ExcInternalError());

      // Now we laterally rotate the pipe segment around its own symmetry axis
      // that the edge matches the polar axis.
      const double lateral_angle =
        Physics::VectorRelations::signed_angle(Rx,
                                               normal_projected_on_opening,
                                               /*axis=*/skeleton_unit[p]);
      GridTools::rotate(skeleton_unit[p], lateral_angle, pipe);

      //
      // Step 5: shift to final position
      //
      GridTools::shift(openings[p].first, pipe);

#if false
      std::ofstream out("pipe-" + std::to_string(p) + ".vtk");
      GridOut().write_vtk(pipe, out);
#endif

      // Create a manifold object for the mantle of this particular pipe
      // segment. Since GridGenerator::merge_triangulations() does not copy
      // manifold objects, but just IDs if requested, we will copy them to
      // the final triangulation later.
      manifolds.emplace_back(
        /*normal_direction=*/normal_projected_on_opening /
          normal_projected_on_opening.norm(),
        /*direction=*/skeleton_unit[p],
        /*point_on_axis=*/openings[p].first,
        data,
        tolerance);

      GridGenerator::merge_triangulations(
        pipe, tria, tria, tolerance_length, /*copy_manifold_ids=*/true);
    }

  for (unsigned int p = 0; p < n_pipes; ++p)
    tria.set_manifold(p, manifolds[p]);

  // TransfiniteInterpolationManifold<dim, spacedim> transfinite;
  // transfinite.initialize(tria);
  // tria.set_manifold(n_pipes, transfinite);

  // Since GridGenerator::merge_triangulations() does not copy boundary IDs
  // either, we need to set them after the final geometry is created. Luckily,
  // boundary IDs match with manifold IDs, so we simply translate them.
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary())
        {
          if (face->manifold_id() == numbers::flat_manifold_id)
            // opening cross section
            face->set_boundary_id(cell->material_id());
          else
            // cone mantle
            face->set_boundary_id(n_pipes);
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
 * Tests a selection of common configurations of pipe junction geometries.
 */
void
test_selection()
{
  constexpr unsigned int dim      = 3;
  constexpr unsigned int spacedim = 3;

  // y-pipe in plane
  {
    const std::vector<std::pair<Point<spacedim>, double>> openings = {
      {{{-2., 0., 0.}, 1.},
       {{1., std::sqrt(3), 0.}, 1.},
       {{1., -std::sqrt(3), 0.}, 1.}}};

    const std::pair<Point<spacedim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim, spacedim> tria;
    pipe_junction(tria, openings, bifurcation);

    refine_and_write(tria, 2, "ypipe");
  }

  // t-pipe in plane
  {
    const std::vector<std::pair<Point<spacedim>, double>> openings = {
      {{{-2., 0., 0.}, 1.}, {{0., 2., 0.}, 1.}, {{2., 0., 0.}, 1.}}};

    const std::pair<Point<spacedim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim, spacedim> tria;
    pipe_junction(tria, openings, bifurcation);

    refine_and_write(tria, 2, "tpipe");
  }

  // corner piece
  {
    const std::vector<std::pair<Point<spacedim>, double>> openings = {
      {{{2., 0., 0.}, 1.}, {{0., 2., 0.}, 1.}, {{0., 0., 2.}, 1.}}};

    const std::pair<Point<spacedim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim, spacedim> tria;
    pipe_junction(tria, openings, bifurcation);

    refine_and_write(tria, 2, "corner");
  }

  // irregular configuration with arbitrary points
  {
    const std::vector<std::pair<Point<spacedim>, double>> openings = {
      {{{-4., 0., 0.}, 1.5}, {{4., -8., -0.4}, 0.75}, {{0.1, 0., -6.}, 0.5}}};

    const std::pair<Point<spacedim>, double> bifurcation = {{0., 0., 0.}, 1.};

    Triangulation<dim, spacedim> tria;
    pipe_junction(tria, openings, bifurcation);

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

      std::vector<std::pair<Point<spacedim>, double>> openings;
      for (unsigned int i = 0; i < npipes; ++i)
        openings.emplace_back(points[combination[i]], radius);

      std::cout << "Testing permutation " << c << " of " << perms.size()
                << " with openings:";
      for (unsigned int i = 0; i < npipes; ++i)
        std::cout << " (" << openings[i].first << ")";
      std::cout << std::endl;

      try
        {
          Triangulation<dim, spacedim> tria;
          pipe_junction(tria, openings, bifurcation);
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
