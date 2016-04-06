#ifndef RECEIVERS_HPP
#define RECEIVERS_HPP

#include "config.hpp"
#include "utilities.hpp"

#include <fstream>
#include <vector>

namespace mfem { class Vertex; }

/**
 * Abstract class representing a set (a straight line, a circle, or other line)
 * of receivers (stations).
 */
class ReceiversSet
{
public:

  virtual ~ReceiversSet() {}

  /**
   * Find and save the numbers of cells containing the receivers.
   */
  void find_cells_containing_receivers(const mfem::Mesh &mesh);

  std::string get_variable() const { return _variable; }

  int n_receivers() const { return _n_receivers; }

  const std::vector<mfem::Vertex>& get_receivers() const
  { return _receivers; }

  const std::vector<int>& get_cells_containing_receivers() const
  { return _cells_containing_receivers; }

  /**
   * Initialize the parameters of the receivers set reading them from a given
   * and already open input stream (likely connected to a file).
   */
  virtual void init(std::ifstream &in) = 0;

  /**
   * Distribute the receivers along the corresponding line.
   */
  virtual void distribute_receivers() = 0;

  /**
   * Description of the set of receivers (to distinguish the derived sets).
   */
  virtual std::string description() const = 0;


protected:

  /**
   * Variable to be recorded at this receiver line.
   */
  std::string _variable;

  /**
   * Number of receivers (stations) in the set.
   */
  int _n_receivers;

  /**
   * The locations of the receivers (stations).
   */
  std::vector<mfem::Vertex> _receivers;

  /**
   * Numbers of grid cells containing the receivers.
   */
  std::vector<int> _cells_containing_receivers;

  /**
   * Dimension of the problem to be solved (affects some features of receivers)
   */
  int _dimension;

  ReceiversSet(int d);
  ReceiversSet(const ReceiversSet& rec);
  ReceiversSet& operator =(const ReceiversSet& rec);
};




/**
 * A class representing a straight line of receivers.
 */
class ReceiversLine: public ReceiversSet
{
public:
  ReceiversLine(int d);
  ~ReceiversLine() { }
  void init(std::ifstream &in);
  void distribute_receivers();
  std::string description() const;
protected:
  mfem::Vertex _start; ///< beginning of line of recievers
  mfem::Vertex _end;   ///<       end of line of receivers

  ReceiversLine(const ReceiversLine& rec);
  ReceiversLine& operator =(const ReceiversLine& rec);
};




/**
 * A class representing a plane of receivers.
 */
class ReceiversPlane: public ReceiversSet
{
public:
  ReceiversPlane(int d);
  ~ReceiversPlane() { }
  void init(std::ifstream &in);
  void distribute_receivers();
  std::string description() const;
protected:
  static const int N_VERTICES = 4; ///< Number of points to describe a plane
  mfem::Vertex _vertices[N_VERTICES];
  int _n_points_1; ///< Number of points in one direction
  int _n_points_2; ///< Number of points in another direction
  std::string _plane; ///< Description of the plane orientation

  void distribute_receivers_YZ_plane(double x, double y0, double y1,
                                     double z0, double z1);
  void distribute_receivers_XZ_plane(double x0, double x1, double y,
                                     double z0, double z1);
  void distribute_receivers_XY_plane(double x0, double x1, double y0,
                                     double y1, double z);

  ReceiversPlane(const ReceiversPlane& rec);
  ReceiversPlane& operator =(const ReceiversPlane& rec);
};


#endif // RECEIVERS_HPP
