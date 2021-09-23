#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <array>
#include <fstream>
#include <string>
#include <utility>

using namespace dealii;

template <typename T>
std::pair<std::pair<unsigned int, unsigned int>,
          std::pair<unsigned int, unsigned int>>
face_vertices(T face)
{
  std::pair<std::pair<unsigned int, unsigned int>,
            std::pair<unsigned int, unsigned int>>
    result;

  std::vector<unsigned int> v(4);

  for (unsigned int d = 0; d < 4; d++)
    v[d] = face->vertex_index(d);

  std::sort(v.begin(), v.end());

  result.first.first   = v[0];
  result.first.second  = v[1];
  result.second.first  = v[2];
  result.second.second = v[3];

  return result;
}

template <typename T, typename Map>
bool
mark(T cell, const unsigned int number, Map &map)
{
  // is not at boundary
  if (!cell->at_boundary(5))
    return false;

  // already visited
  if (cell->face(5)->boundary_id() > 0)
    return false;

  // set boundary id
  cell->face(5)->set_all_boundary_ids(number);
  map[face_vertices(cell->face(5))] = number;

  // mark all neighbors
  for (unsigned int d = 0; d < 6; d++)
    {
      // face is at boundary: there is no neighbor to mark
      if (cell->at_boundary(d))
        continue;

      // mark neighbor
      mark(cell->neighbor(d), number, map);
    }

  // cell has been marked
  return true;
}

Tensor<2, 3>
compute_rotation_matrix(Tensor<1, 3> a0,
                        Tensor<1, 3> a1,
                        Tensor<1, 3> b0,
                        Tensor<1, 3> b1)
{
  // assemble source system
  Tensor<2, 3> A;
  for (int i = 0; i < 3; i++)
    A[i][0] = a0[i];
  for (int i = 0; i < 3; i++)
    A[i][1] = a1[i];

  Tensor<1, 3> a2 = cross_product_3d(a0, a1);
  for (int i = 0; i < 3; i++)
    A[i][2] = a2[i];

  // assemble target system
  Tensor<2, 3> B;
  for (int i = 0; i < 3; i++)
    B[i][0] = b0[i];
  for (int i = 0; i < 3; i++)
    B[i][1] = b1[i];

  Tensor<1, 3> b2 = cross_product_3d(b0, b1);
  for (int i = 0; i < 3; i++)
    B[i][2] = b2[i];

  // compute rotation matrix: R = B*A^{-1}
  return B * invert(A);
}

bool
check_if_planar(Tensor<1, 3> v1, Tensor<1, 3> v2, Tensor<1, 3> v3)
{
  Tensor<2, 3> A;

  for (int i = 0; i < 3; i++)
    A[i][0] = v1[i];

  for (int i = 0; i < 3; i++)
    A[i][1] = v2[i];

  for (int i = 0; i < 3; i++)
    A[i][2] = v3[i];

  double det = determinant(A);

  return std::abs(det) < 1e-10;
}

double
get_degree(Tensor<1, 3> t1, Tensor<1, 3> t2)
{
  double argument = t1 * t2 / t1.norm() / t2.norm();

  if ((1.0 - std::abs(argument)) < 1e-10)
    argument = std::copysign(1.0, argument);

  return std::acos(argument);
}

Tensor<1, 3>
get_normal_vector(Tensor<1, 3> v1, Tensor<1, 3> v2)
{
  double scalar_product = v1 * v2;

  double deg = std::acos(scalar_product / v1.norm() / v2.norm());

  if (deg < 1e-10 || (numbers::PI - deg) < 1e-10)
    AssertThrow(false, ExcMessage("Given vectors are collinear!"))

      return cross_product_3d(v1, v2);
}

Tensor<1, 3>
get_normal_vector(Tensor<1, 3> v1, Tensor<1, 3> v2, Tensor<1, 3> v3)
{
  double argument = v1 * v2 / v1.norm() / v2.norm();

  if ((1.0 - std::abs(argument)) < 1e-10)
    argument = std::copysign(1.0, argument);

  double deg = std::acos(argument);

  Tensor<1, 3> normal;

  if (deg < 1e-10 || (numbers::PI - deg) < 1e-10)
    {
      normal = get_normal_vector(v1 / v1.norm(), v3 / v3.norm());
    }
  else
    {
      normal = get_normal_vector(v1 / v1.norm(), v2 / v2.norm());
    }

  return normal;
}

void
create_reference_cylinder(const bool                do_transition,
                          const unsigned int        n_sections,
                          std::vector<Point<3>> &   vertices_3d,
                          std::vector<CellData<3>> &cell_data_3d)
{
  if (do_transition)
    printf("WARNING: Transition has not been implemented yet (TODO)!\n");

  // position of auxiliary point to achieve an angle of 120 degrees in corner
  // of inner cell
  const double radius = 1;
  const double y_coord =
    0.55 * std::sqrt(0.5) * radius * std::cos(numbers::PI / 12) /
    (std::sin(numbers::PI / 12) + std::cos(numbers::PI / 12));
  // vertices for quarter of circle
  std::vector<Point<2>> vertices{{0, 0},
                                 {0.5 * radius, 0},
                                 {y_coord, y_coord},
                                 {radius, 0},
                                 {radius * 0.5, radius * 0.5}};


  // create additional vertices for other three quarters of circle -> gives 17
  // vertices in total
  for (unsigned int a = 1; a < 4; ++a)
    {
      Tensor<2, 2> transform;
      transform[0][0] = a == 2 ? -1. : 0;
      transform[1][0] = a == 2 ? 0 : (a == 1 ? 1 : -1);
      transform[0][1] = -transform[1][0];
      transform[1][1] = transform[0][0];
      for (unsigned int i = 1; i < 5; ++i)
        vertices.emplace_back(transform * vertices[i]);
    }

  // create 12 cells for 2d mesh on base; the first four elements are at the
  // center of the circle
  std::vector<CellData<2>> cell_data(12);
  cell_data[0].vertices[0] = 0;
  cell_data[0].vertices[1] = 1;
  cell_data[0].vertices[2] = 5;
  cell_data[0].vertices[3] = 2;
  cell_data[1].vertices[0] = 9;
  cell_data[1].vertices[1] = 0;
  cell_data[1].vertices[2] = 6;
  cell_data[1].vertices[3] = 5;
  cell_data[2].vertices[0] = 10;
  cell_data[2].vertices[1] = 13;
  cell_data[2].vertices[2] = 9;
  cell_data[2].vertices[3] = 0;
  cell_data[3].vertices[0] = 13;
  cell_data[3].vertices[1] = 14;
  cell_data[3].vertices[2] = 0;
  cell_data[3].vertices[3] = 1;

  // the next 8 elements describe the rim; we take one quarter of the circle
  // in each loop iteration
  for (unsigned int a = 0; a < 4; ++a)
    {
      cell_data[4 + a * 2].vertices[0] = 1 + a * 4;
      cell_data[4 + a * 2].vertices[1] = 3 + a * 4;
      cell_data[4 + a * 2].vertices[2] = 2 + a * 4;
      cell_data[4 + a * 2].vertices[3] = 4 + a * 4;
      cell_data[5 + a * 2].vertices[0] = 2 + a * 4;
      cell_data[5 + a * 2].vertices[1] = 4 + a * 4;
      AssertIndexRange(4 + a * 4, vertices.size());
      cell_data[5 + a * 2].vertices[2] = a == 3 ? 1 : 5 + a * 4;
      cell_data[5 + a * 2].vertices[3] = a == 3 ? 3 : 7 + a * 4;
    }
  SubCellData subcell_data;
  GridTools::consistently_order_cells(cell_data);

  Triangulation<2> tria_2d;
  tria_2d.create_triangulation(vertices, cell_data, subcell_data);

  vertices_3d.clear();
  vertices_3d.resize((n_sections + 1) * tria_2d.n_vertices());
  cell_data_3d.clear();
  cell_data_3d.resize(n_sections * cell_data.size());

  for (unsigned int s = 0; s <= n_sections; s++)
    {
      const double       beta  = (1.0 * s) / n_sections;
      const unsigned int shift = s * tria_2d.n_vertices();
      for (unsigned int i = 0; i < tria_2d.n_vertices(); ++i)
        {
          vertices_3d[shift + i][0] = tria_2d.get_vertices()[i][0];
          vertices_3d[shift + i][1] = tria_2d.get_vertices()[i][1];
          vertices_3d[shift + i][2] = beta;
        }
    }


  for (unsigned int s = 0; s < n_sections; s++)
    for (unsigned int i = 0; i < cell_data.size(); ++i)
      {
        for (unsigned int v = 0; v < 4; ++v)
          cell_data_3d[cell_data.size() * s + i].vertices[v + 0] =
            (s + 0) * vertices.size() + cell_data[i].vertices[v];
        for (unsigned int v = 0; v < 4; ++v)
          cell_data_3d[cell_data.size() * s + i].vertices[4 + v] =
            (s + 1) * vertices.size() + cell_data[i].vertices[v];
      }

#ifdef DEBUG
  // create triangulation
  Triangulation<3> tria(Triangulation<3>::MeshSmoothing::none, true);

  tria.create_triangulation(vertices_3d, cell_data_3d, subcell_data);
  GridOut gridout;

  std::ofstream out("mesh_reference_cylinder.vtu");

  gridout.write_vtu(tria, out);
#endif
}


void
create_cylinder(double                    radius1,
                double                    radius2,
                double                    length,
                Tensor<2, 3>              transform_top,
                Tensor<2, 3>              transform_bottom,
                Point<3>                  offset,
                std::vector<Point<3>> &   vertices_3d,
                std::vector<Point<3>> &   skeleton,
                std::vector<CellData<3>> &cell_data_3d,
                double                    deg0,
                double                    deg1,
                double                    deg2,
                double                    degree_parent_intersection,
                double                    degree_child_intersection,
                double                    degree_separation,
                bool                      is_left,
                bool                      left_right_mixed_up_bottom,
                bool                      left_right_mixed_up_top,
                bool                      do_rotate_bottom,
                unsigned int              n_sections = 1)
{
  if (left_right_mixed_up_bottom)
    std::swap(deg1, deg2);

  if (left_right_mixed_up_top)
    std::swap(deg0, degree_separation);

  // create some short cuts
  bool is_right = !is_left;

  /**************************************************************************
   * Create reference cylinder
   **************************************************************************/
  bool                  do_transition = false;
  std::vector<Point<3>> vertices_3d_temp;
  create_reference_cylinder(do_transition,
                            n_sections,
                            vertices_3d_temp,
                            cell_data_3d);
  vertices_3d.resize(vertices_3d_temp.size());

  skeleton.clear();
  skeleton.resize(8);

  /**************************************************************************
   * Loop over all points and transform
   **************************************************************************/
  for (unsigned int i = 0; i < vertices_3d_temp.size(); ++i)
    {
      // get reference to input and output
      auto &point_in = vertices_3d_temp[i], &point_out = vertices_3d[i];

      // transform point in both coordinate systems
      Point<3> point_out_alpha, point_out_beta;

      // get blending factor
      const double beta = point_in[2];

      /************************************************************************
       * Top part
       ************************************************************************/
      point_out_alpha[0] = point_in[0] * radius1;
      point_out_alpha[1] = point_in[1] * radius1;

      auto deg4 = degree_separation;

      if (is_right)
        {
          if (point_in[0] > 0)
            point_out_alpha[2] =
              length * beta +
              std::tan((numbers::PI_2 - deg4)) * std::abs(point_in[0]) *
                radius1 +
              std::tan(degree_child_intersection) * point_in[1] * radius1;
          else
            point_out_alpha[2] =
              length * beta +
              std::tan(numbers::PI_2 - deg0) * std::abs(point_in[0]) * radius1 +
              std::tan(degree_child_intersection) * point_in[1] * radius1;
        }
      else
        {
          if (point_in[0] > 0)
            point_out_alpha[2] =
              length * beta +
              std::tan((numbers::PI_2 - deg0)) * std::abs(point_in[0]) *
                radius1 +
              std::tan(degree_child_intersection) * point_in[1] * radius1;
          else
            point_out_alpha[2] =
              length * beta +
              std::tan(numbers::PI_2 - deg4) * std::abs(point_in[0]) * radius1 +
              std::tan(degree_child_intersection) * point_in[1] * radius1;
        }


      /************************************************************************
       * Bottom part
       ************************************************************************/

      point_out_beta[0] = point_in[0] * radius2;
      point_out_beta[1] = point_in[1] * radius2;


      if (!do_rotate_bottom)
        {
          if (point_in[0] > 0)
            point_out_beta[2] =
              length * beta -
              std::tan(numbers::PI_2 - deg1) * std::abs(point_in[0]) * radius2 -
              std::tan(degree_parent_intersection) * point_in[1] * radius2;
          else
            point_out_beta[2] =
              length * beta -
              std::tan(numbers::PI_2 - deg2) * std::abs(point_in[0]) * radius2 -
              std::tan(degree_parent_intersection) * point_in[1] * radius2;
        }
      else
        {
          if (point_in[1] < 0)
            point_out_beta[2] =
              length * beta -
              std::tan(numbers::PI_2 - deg1) * std::abs(point_in[1]) * radius2 -
              std::tan(degree_parent_intersection) * point_in[0] * radius2;
          else
            point_out_beta[2] =
              length * beta -
              std::tan(numbers::PI_2 - deg2) * std::abs(point_in[1]) * radius2 -
              std::tan(degree_parent_intersection) * point_in[0] * radius2;
        }

      /************************************************************************
       * Combine points and blend
       ************************************************************************/
      point_out =
        (1 - beta) * Point<3>(offset + transform_top * point_out_alpha) +
        beta * Point<3>(offset + transform_bottom * point_out_beta);

      /************************************************************************
       * Fill skeleton vector with corner nodes
       ************************************************************************/

      if ((beta == 0.0 || beta == 1.0) &&
          (std::abs(std::abs(point_in[0]) - 1.0) < 1e-8 ||
           std::abs(std::abs(point_in[1]) - 1.0) < 1e-8))
        {
          const unsigned int idz = beta == 0.0 ? 1 : 0;
          const unsigned int idy =
            (point_in[1] == -1 || point_in[0] == -1) ? 1 : 0;
          const unsigned int idx =
            (point_in[0] == +1 || point_in[1] == -1) ? 1 : 0;
          skeleton[idz * 4 + idy * 2 + idx] = point_out;
        }
    }
}

void
process_pipe(const std::array<std::pair<Point<3>, double>, 3> &openings,
             const std::pair<Point<3>, double> &               bifurcation,
             std::vector<CellData<3>> &cell_data_3d_global,
             std::vector<Point<3>> &   vertices_3d_global,
             std::vector<Point<3>> &   skeleton,
             const unsigned int        id,
             const unsigned int        parent_os         = 0,
             double                    degree_parent     = numbers::PI_2,
             double                    degree_separation = numbers::PI_2,
             double                    degree_child_intersection = 0.0,
             Tensor<1, 3>              normal_rotation_child = Tensor<1, 3>(),
             bool                      left_right_mixed_up_parent = false,
             bool                      is_child                   = false,
             bool                      is_left                    = true)
{
  // normal and tangential vector in the reference system
  Tensor<1, 3> src_n({0, 1, 0});
  Tensor<1, 3> src_t({0, 0, 1});

  unsigned int os = vertices_3d_global.size();

  // set shortcuts
  const auto bifurcation_point = bifurcation.first;
  const auto opening_0         = openings[0].first;
  const auto opening_1         = openings[1].first;
  const auto opening_2         = openings[2].first;

  // get tangential vectors
  Tensor<1, 3> tangent_left_child  = opening_1 - bifurcation_point;
  Tensor<1, 3> tangent_right_child = opening_2 - bifurcation_point;
  Tensor<1, 3> tangent_parent      = opening_0 - bifurcation_point;

  // get current tangent
  Tensor<1, 3> dst_t = !is_child ?
                         -tangent_parent :
                         (is_left ? tangent_left_child : tangent_right_child);

  dst_t /= dst_t.norm();

  // define default values

  Tensor<1, 3> normal_rotation_parent;
  Tensor<1, 3> normal_rotation_left_child;
  Tensor<1, 3> normal_rotation_right_child;

  double degree_parent_intersection      = 0.0;
  double degree_left_child_intersection  = 0.0;
  double degree_right_child_intersection = 0.0;

  bool left_right_mixed_up = false;

  double degree_parent_left_child      = numbers::PI_2;
  double degree_parent_right_child     = numbers::PI_2;
  double degree_left_child_right_child = numbers::PI_2;


  if (!is_child)
    {
      // plane normals
      auto normal_children     = get_normal_vector(tangent_right_child,
                                               tangent_left_child,
                                               tangent_parent);
      auto normal_parent_left  = get_normal_vector(tangent_parent,
                                                  tangent_left_child,
                                                  tangent_right_child);
      auto normal_parent_right = get_normal_vector(tangent_parent,
                                                   tangent_right_child,
                                                   tangent_left_child);

      // check if planar
      bool is_bifurcation_planar = check_if_planar(tangent_parent,
                                                   tangent_left_child,
                                                   tangent_right_child);

      Tensor<1, 3> normal_intersection_plane;

      // calculate normal of intersection plane
      if (is_bifurcation_planar)
        normal_intersection_plane = normal_children;
      else // 3D case
        {
          // calculate normal intersection plane

          std::vector<Tensor<1, 3>> normal_intersection_planes(8);
          normal_intersection_planes[0] =
            normal_children + normal_parent_left + normal_parent_right;
          normal_intersection_planes[1] =
            normal_children + normal_parent_left - normal_parent_right;
          normal_intersection_planes[2] =
            normal_children - normal_parent_left + normal_parent_right;
          normal_intersection_planes[3] =
            normal_children - normal_parent_left - normal_parent_right;
          normal_intersection_planes[4] =
            -normal_children + normal_parent_left + normal_parent_right;
          normal_intersection_planes[5] =
            -normal_children + normal_parent_left - normal_parent_right;
          normal_intersection_planes[6] =
            -normal_children - normal_parent_left + normal_parent_right;
          normal_intersection_planes[7] =
            -normal_children - normal_parent_left - normal_parent_right;

          std::vector<Tensor<1, 3>> normals_rotation_parent(8);
          for (int i = 0; i < 8; i++)
            normals_rotation_parent[i] =
              normal_intersection_planes[i] -
              normal_intersection_planes[i] * tangent_parent /
                tangent_parent.norm_square() * tangent_parent;

          std::vector<double> degree_intersection_normal(8);

          for (int i = 0; i < 8; i++)
            {
              degree_intersection_normal[i] =
                get_degree(normal_intersection_planes[i],
                           normals_rotation_parent[i]);
            }

          std::vector<double> sum_degree_intersection_normal_tangents(8);

          for (int i = 0; i < 8; i++)
            sum_degree_intersection_normal_tangents[i] =
              get_degree(normal_intersection_planes[i], tangent_parent) +
              get_degree(normal_intersection_planes[i], tangent_left_child) +
              get_degree(normal_intersection_planes[i], tangent_right_child);

          std::vector<double> product_degrees(8);

          for (int i = 0; i < 8; i++)
            product_degrees[i] = degree_intersection_normal[i] *
                                 sum_degree_intersection_normal_tangents[i];

          auto minimum_element =
            std::min_element(product_degrees.begin(), product_degrees.end());

          int minimum_element_at =
            std::distance(product_degrees.begin(), minimum_element);

          normal_intersection_plane =
            normal_intersection_planes[minimum_element_at];
        }

      normal_intersection_plane =
        normal_intersection_plane / normal_intersection_plane.norm();

      // calculate rotation normals
      normal_rotation_parent = normal_intersection_plane -
                               normal_intersection_plane * tangent_parent /
                                 tangent_parent.norm_square() * tangent_parent;

      normal_rotation_left_child =
        normal_intersection_plane -
        normal_intersection_plane * tangent_left_child /
          tangent_left_child.norm_square() * tangent_left_child;

      normal_rotation_right_child =
        normal_intersection_plane -
        normal_intersection_plane * tangent_right_child /
          tangent_right_child.norm_square() * tangent_right_child;

      // calculate degrees between rotation normals and intersection normal
      degree_parent_intersection =
        get_degree(normal_intersection_plane, normal_rotation_parent);
      degree_left_child_intersection =
        get_degree(normal_intersection_plane, normal_rotation_left_child);
      degree_right_child_intersection =
        get_degree(normal_intersection_plane, normal_rotation_right_child);

      // calculate degrees between cylinders
      if (is_bifurcation_planar)
        {
          degree_parent_left_child =
            get_degree(tangent_parent, tangent_left_child) / 2.0;
          degree_parent_right_child =
            get_degree(tangent_parent, tangent_right_child) / 2.0;
          degree_left_child_right_child =
            get_degree(tangent_left_child, tangent_right_child) / 2.0;
        }
      else // 3D case
        {
          // calculate direction of vectors

          auto direction_parent_left_child =
            cross_product_3d(normal_rotation_parent,
                             normal_rotation_left_child);
          auto direction_parent_right_child =
            cross_product_3d(normal_rotation_parent,
                             normal_rotation_right_child);
          auto direction_left_child_right_child =
            cross_product_3d(normal_rotation_left_child,
                             normal_rotation_right_child);

          if (direction_parent_left_child * tangent_parent > 0.0 &&
              direction_parent_left_child * tangent_left_child > 0)
            direction_parent_left_child =
              direction_parent_left_child / direction_parent_left_child.norm();
          else
            direction_parent_left_child =
              -direction_parent_left_child / direction_parent_left_child.norm();

          if (direction_parent_right_child * tangent_parent > 0.0 &&
              direction_parent_right_child * tangent_right_child > 0)
            direction_parent_right_child = direction_parent_right_child /
                                           direction_parent_right_child.norm();
          else
            direction_parent_right_child = -direction_parent_right_child /
                                           direction_parent_right_child.norm();

          if (direction_left_child_right_child * tangent_right_child > 0.0 &&
              direction_left_child_right_child * tangent_left_child > 0)
            direction_left_child_right_child =
              direction_left_child_right_child /
              direction_left_child_right_child.norm();
          else
            direction_left_child_right_child =
              -direction_left_child_right_child /
              direction_left_child_right_child.norm();

          // calculate degrees depending on direction vectors
          degree_parent_left_child =
            get_degree(tangent_parent, direction_parent_left_child);
          degree_parent_right_child =
            get_degree(tangent_parent, direction_parent_right_child);
          degree_left_child_right_child =
            get_degree(tangent_left_child, direction_left_child_right_child);
        }

      // check if right_children is right and left_children is left
      auto normal_right =
        cross_product_3d(normal_rotation_parent, tangent_parent);
      auto normal_left = -normal_right;

      bool left_child_is_left   = normal_left * tangent_left_child >= 0.0;
      bool right_child_is_right = normal_right * tangent_right_child >= 0.0;

      // switch parent children degrees if children tangents are mixed up
      if (!left_child_is_left && !right_child_is_right)
        {
          left_right_mixed_up = true;
        }
    }

  Tensor<2, 3> transform_top;
  Tensor<2, 3> transform_bottom;

  // compute rotation matrix

  if (is_child)
    {
      auto dst_n_top = normal_rotation_child;
      dst_n_top /= dst_n_top.norm();

      transform_top = compute_rotation_matrix(src_n, src_t, dst_n_top, dst_t);

      transform_bottom = transform_top;
    }
  else
    {
      auto dst_n_bottom = normal_rotation_parent;
      dst_n_bottom /= dst_n_bottom.norm();

      transform_bottom =
        compute_rotation_matrix(src_n, src_t, dst_n_bottom, dst_t);

      transform_top             = transform_bottom;
      degree_child_intersection = 0.0;
    }

  // define degrees
  double degree_1                   = degree_parent_left_child;
  double degree_2                   = degree_parent_right_child;
  double degree_separation_children = degree_left_child_right_child;

  // extract some parameters of this branch
  auto tangent = !is_child ?
                   -tangent_parent :
                   (is_left ? tangent_left_child : tangent_right_child);
  auto source  = !is_child ? opening_0 : bifurcation_point;

  double radius_top    = !is_child ? openings[0].second : bifurcation.second;
  double radius_bottom = !is_child ?
                           bifurcation.second :
                           (is_left ? openings[1].second : openings[2].second);

  double radius_mean = 0.5 * (radius_top + radius_bottom);

  // TODO adjust number of intersections
  unsigned int n_intersections =
    std::ceil(std::max(2.0, tangent.norm() / (2.0 * radius_mean)));

  // compute vertices and cells in reference system
  std::vector<CellData<3>> cell_data_3d;
  std::vector<Point<3>>    vertices_3d;
  create_cylinder(radius_top,
                  radius_bottom,
                  tangent.norm(),
                  transform_top,
                  transform_bottom,
                  source,
                  vertices_3d,
                  skeleton,
                  cell_data_3d,
                  degree_parent,
                  degree_1,
                  degree_2,
                  degree_parent_intersection,
                  degree_child_intersection,
                  degree_separation,
                  is_left,
                  left_right_mixed_up,
                  left_right_mixed_up_parent,
                  false,
                  n_intersections);

  // create triangulation
  SubCellData      subcell_data;
  Triangulation<3> tria(Triangulation<3>::MeshSmoothing::none, true);

  try
    {
      tria.create_triangulation(vertices_3d, cell_data_3d, subcell_data);

#ifdef DEBUG
      GridOut gridout;

      std::ofstream out("mesh_cylinder_" + std::to_string(id) + ".vtu");

      gridout.write_vtu(tria, out);
#endif
    }
  catch (const std::exception &e)
    {
      std::cout << e.what();

      std::ostringstream stream;
      stream << "Problematic branch:" << std::endl
             << "   " << opening_0 << std::endl
             << "   " << bifurcation_point;

      AssertThrow(false, ExcMessage(stream.str()));
    }

  // WARNING: This section is only reached if the creation of the
  // triangulation was successful (i.e. the cells are not too much deformed)

  unsigned int lung_number_of_vertices_2d = 17;

  unsigned int range_local = (n_intersections + 1) * lung_number_of_vertices_2d;
  unsigned int range_global =
    !is_child ? 0 :
                ((n_intersections + (is_left ? 0 : n_intersections)) + 2) *
                  lung_number_of_vertices_2d;

  // mark all vertices of local branch with -1
  std::map<unsigned int, unsigned int> map;
  for (unsigned int i = 0; i < range_local; i++)
    map[i] = numbers::invalid_unsigned_int;

  // check if vertex is already available (i.e. already created by parent or
  // left neighbor)
  for (unsigned int i = 0; i < range_local; i++)
    for (unsigned int j = parent_os;
         (j < parent_os + range_global) && (j < vertices_3d_global.size());
         j++)
      {
        auto t = vertices_3d[i];
        t -= vertices_3d_global[j];
        if (t.norm() < 1e-5)
          {
            map[i] = j;
            break;
          }
      }

  // assign actual new vertices new ids and save the position of these vertices
  unsigned int cou = os;
  for (unsigned int i = 0; i < range_local; i++)
    if (map[i] == numbers::invalid_unsigned_int)
      {
        vertices_3d_global.push_back(vertices_3d[i]);
        map[i] = cou++;
      }

  // save cell definition
  for (auto c : cell_data_3d)
    {
      for (int i = 0; i < 8; i++)
        c.vertices[i] = map[c.vertices[i]];
      c.material_id = id;
      cell_data_3d_global.push_back(c);
    }

  // process children
  if (!is_child)
    {
      // left child:
      try
        {
          process_pipe(openings,
                       bifurcation,
                       cell_data_3d_global,
                       vertices_3d_global,
                       skeleton,
                       1,
                       os,
                       degree_1,
                       degree_separation_children,
                       degree_left_child_intersection,
                       normal_rotation_left_child,
                       left_right_mixed_up,
                       true,
                       true);
        }
      catch (const std::exception &e)
        {
          std::cout << e.what();
        }

      // right child:
      try
        {
          process_pipe(openings,
                       bifurcation,
                       cell_data_3d_global,
                       vertices_3d_global,
                       skeleton,
                       2,
                       os,
                       degree_2,
                       degree_separation_children,
                       degree_right_child_intersection,
                       normal_rotation_right_child,
                       left_right_mixed_up,
                       true,
                       false);
        }
      catch (const std::exception &e)
        {
          std::cout << e.what();
        }
    }
}

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

  std::vector<Point<3>>    skeleton;
  std::vector<CellData<3>> cell_data_3d;
  std::vector<Point<3>>    vertices_3d;
  SubCellData              subcell_data;

  process_pipe(openings, bifurcation, cell_data_3d, vertices_3d, skeleton, 0);

  // collect faces and their ids for the non-reordered triangulation
  std::map<std::pair<std::pair<unsigned int, unsigned int>,
                     std::pair<unsigned int, unsigned int>>,
           unsigned int>
    map;

  unsigned int outlet_id_first = 1;
  unsigned int outlet_id_last;

  {
    Triangulation<3> tria_tmp(Triangulation<3>::MeshSmoothing::none, true);
    tria_tmp.create_triangulation(vertices_3d, cell_data_3d, subcell_data);

    // set boundary ids
    unsigned int counter = outlet_id_first; // counter for outlets
    for (auto cell : tria_tmp.active_cell_iterators())
      {
        // the mesh is generated in a way that inlet/outlets are one faces with
        // normal vector in positive or negative z-direction (faces 4/5)
        if (cell->at_boundary(4) && cell->material_id() == 1) // inlet
          map[face_vertices(cell->face(4))] = 1;

        if (cell->at_boundary(5)) // outlets (>1)
          if (mark(cell, counter, map))
            counter++;
      }
    // set outlet_id_last which is needed by the application setting the
    // boundary conditions
    outlet_id_last = counter;
  }

  (void)outlet_id_last; // TODO: use outlet id

  GridTools::consistently_order_cells(cell_data_3d);
  tria.create_triangulation(vertices_3d, cell_data_3d, subcell_data);

#ifdef DEBUG
  GridOut       grid_out;
  std::ofstream file("mesh-tria.vtu");
  grid_out.write_vtu(tria, file);
#endif

  // actually set boundary ids
  for (auto cell : tria.active_cell_iterators())
    for (unsigned int d = 0; d < 6; d++)
      if (cell->at_boundary(d) &&
          map.find(face_vertices(cell->face(d))) != map.end())
        cell->face(d)->set_all_boundary_ids(map[face_vertices(cell->face(d))]);
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
    {{{-4., 0., 0.}, 1.}, {{0., -4., -0.4}, 0.75}, {{0.1, 0., -4.}, 0.5}}};

  const std::pair<Point<dim>, double> bifurcation = {{0., 0., 0.}, 1.};

  Triangulation<dim> tria;
  tpipe(tria, openings, bifurcation);

  refine_and_write(tria, 0, "tpipe");

  return 0;
}
