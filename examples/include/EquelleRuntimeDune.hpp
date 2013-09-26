/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLERUNTIMEDUNE_HEADER_INCLUDED
#define EQUELLERUNTIMEDUNE_HEADER_INCLUDED

#include <dune/grid/CpGrid.hpp>
#include <Eigen/Eigen>
#include <vector>
#include <string>

/// Topological entities.
typedef Dune::CpGrid::Codim<0>::EntitySeed Cell;

struct Face
{
    Face(const Cell first, const Cell second, const int in)
        : first_cell(first), second_cell(second), intersection_number(in)
    {
    }
    Cell first_cell;
    Cell second_cell;
    int intersection_number;
    bool operator<(const Face& f) const {
        return (first_cell < f.first_cell) || (first_cell == f.first_cell && second_cell < f.second_cell);
    }
    bool operator=(const Face& f) const {
        return (first_cell == f.first_cell && second_cell == f.second_cell);
    }
};


/// Topological collections.
typedef std::vector<Cell> CollOfCells;
typedef std::vector<Face> CollOfFaces;

// Scalar type
typedef double Scalar;

/// Types from Eigen.
typedef Eigen::Array<double, Eigen::Dynamic, 1> CollOfScalars;
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> CollOfVectors;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> CollOfBooleans;



/// The Equelle runtime class.
/// Contains methods corresponding to Equelle built-ins to make
/// it easy to generate C++ code for an Equelle program.
class EquelleRuntimeDune
{
public:
    /// Constructor.
    EquelleRuntimeDune(const Opm::parameter::ParameterGroup& param);

    /// Topology and geometry related.
    CollOfCells allCells() const;
    // CollOfCells boundaryCells() const;
    // CollOfCells interiorCells() const;
    CollOfFaces allFaces() const;
    CollOfFaces boundaryFaces() const;
    CollOfFaces interiorFaces() const;
    CollOfCells firstCell(const CollOfFaces& faces) const;
    CollOfCells secondCell(const CollOfFaces& faces) const;
    CollOfScalars norm(const CollOfFaces& faces) const;
    CollOfScalars norm(const CollOfCells& cells) const;
    CollOfScalars norm(const CollOfVectors& vectors) const;
    CollOfVectors centroid(const CollOfFaces& faces) const;
    CollOfVectors centroid(const CollOfCells& cells) const;

    /// Operators.
    CollOfScalars gradient(const CollOfScalars& cell_scalarfield) const;
    CollOfScalars negGradient(const CollOfScalars& cell_scalarfield) const;
    CollOfScalars divergence(const CollOfScalars& face_fluxes) const;
    CollOfScalars interiorDivergence(const CollOfScalars& face_fluxes) const;
    CollOfBooleans isEmpty(const CollOfCells& cells) const;
    CollOfBooleans isEmpty(const CollOfFaces& faces) const;
    template <class EntityCollection>
    CollOfScalars operatorOn(const double data, const EntityCollection& to_set);
    template <class SomeCollection, class EntityCollection>
    SomeCollection operatorOn(const SomeCollection& data, const EntityCollection& from_set, const EntityCollection& to_set);

    template <class SomeCollection>
    SomeCollection trinaryIf(const CollOfBooleans& predicate,
                             const SomeCollection& iftrue,
                             const SomeCollection& iffalse) const;

    /// Output.
    void output(const std::string& tag, double val) const;
    void output(const std::string& tag, const CollOfScalars& vals) const;

    /// Input.
    CollOfFaces getUserSpecifiedCollectionOfFaceSubsetOf(const Opm::parameter::ParameterGroup& param,
                                                         const std::string& name,
                                                         const CollOfFaces& superset);
    static CollOfScalars getUserSpecifiedCollectionOfScalar(const Opm::parameter::ParameterGroup& param,
							    const std::string& name,
							    const int size);


private:
    /// We need a Cell object to signify the outside.
    Cell emptyCell() const;

    /// Norms.
    double twoNorm(const CollOfScalars& vals) const;

    void setupFaceSets();

    /// Data members.
    bool output_to_file_;
    Dune::CpGrid grid_;
    CollOfFaces allfaces_;
    CollOfFaces ifaces_;
    CollOfFaces bfaces_;
};

// Include the implementations of template members.
#include "EquelleRuntimeDune_impl.hpp"

#endif // EQUELLERUNTIMEDUNE_HEADER_INCLUDED
