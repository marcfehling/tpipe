#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <array>
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
template <int dim, int spacedim>
void
tpipe(Triangulation<dim, spacedim> &                           tria,
      const std::array<std::pair<Point<spacedim>, double>, 3> &openings,
      const std::pair<Point<spacedim>, double> &               bifurcation)
{
  (void)tria;
  (void)openings;
  (void)bifurcation;
  Assert(false, ExcNotImplemented());
}



/**
 * 2D specialization.
 */
template <int spacedim>
void
tpipe(Triangulation<2, spacedim> &                             tria,
      const std::array<std::pair<Point<spacedim>, double>, 3> &openings,
      const std::pair<Point<spacedim>, double> &               bifurcation)
{
  constexpr const unsigned int dim = 2;

  // placeholder for mesh generation
  (void)dim;
  (void)openings;
  (void)bifurcation;
  GridGenerator::hyper_cube(tria);
}



/**
 * 3D specialization.
 */
template <>
void
tpipe(Triangulation<3, 3> &                             tria,
      const std::array<std::pair<Point<3>, double>, 3> &openings,
      const std::pair<Point<3>, double> &               bifurcation)
{
  constexpr const unsigned int dim = 3, spacedim = 3;

  // placeholder for mesh generation
  (void)dim;
  (void)spacedim;
  (void)openings;
  (void)bifurcation;
  GridGenerator::hyper_cube(tria);
}



/**
 * Refines the provided triangulation globally by the specified amount of times.
 *
 * Writes the coarse mesh as well as the mesh after each global refinement
 * to the filesystem in VTK format.
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
  };

  output(0);
  for (unsigned int n = 1; n <= n_global_refinements; ++n)
    {
      tria.refine_global();
      output(n);
    }
}



/**
 * Examplary application for a simple 3D T-pipe.
 */
int
main()
{
  constexpr const unsigned int dim = 3;

  const std::array<std::pair<Point<dim>, double>, 3> openings = {
    {{{-2., 0., 0.}, 1.}, {{0., -2., 0.}, 1.}, {{2., 0., 0.}, 1.}}};

  const std::pair<Point<dim>, double> bifurcation = {{0., 0., 0.}, 1.};

  Triangulation<dim> tria;
  tpipe(tria, openings, bifurcation);

  refine_and_write(tria, 2, "tpipe");

  return 0;
}
