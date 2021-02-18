#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <string>


using namespace dealii;


template <int dim>
void
tpipe(Triangulation<dim> &tria)
{
  // placeholder mesh generation
  GridGenerator::hyper_cube(tria);
}


template <int dim>
void
refine_and_write(Triangulation<dim> &tria,
                 const unsigned int  n_global_refinements = 0,
                 const std::string   filestem             = "grid")
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


int
main()
{
  Triangulation<3> tria;
  tpipe(tria);

  refine_and_write(tria, 2, "tpipe");

  return 0;
}
