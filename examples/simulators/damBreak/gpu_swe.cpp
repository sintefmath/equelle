
// This program was created by the Equelle compiler from SINTEF.

#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>
#include <opm/core/utility/ErrorMacros.hpp>
#include <opm/autodiff/AutoDiffBlock.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>
#include <array>

#include "EquelleRuntimeCUDA.hpp"

using namespace equelleCUDA;

void ensureRequirements(const EquelleRuntimeCUDA& er);

int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    EquelleRuntimeCUDA er(param);

    ensureRequirements(er);

    // ============= Generated code starts here ================

    const CollOfScalar h_init = er.inputCollectionOfScalar("h_init", er.allCells());
    const CollOfScalar u_init = er.inputCollectionOfScalar("u_init", er.allCells());
    const CollOfScalar v_init = er.inputCollectionOfScalar("v_init", er.allCells());
    const CollOfScalar b_north = er.inputCollectionOfScalar("b_north", er.allCells());
    const CollOfScalar b_south = er.inputCollectionOfScalar("b_south", er.allCells());
    const CollOfScalar b_east = er.inputCollectionOfScalar("b_east", er.allCells());
    const CollOfScalar b_west = er.inputCollectionOfScalar("b_west", er.allCells());
    const CollOfScalar b_mid = ((((b_north + b_south) + b_east) + b_west) / double(4));
    er.output("bottom", b_mid);
    const Scalar dx = er.inputScalarWithDefault("dx", double(10));
    const Scalar dy = er.inputScalarWithDefault("dy", double(10));
    const SeqOfScalar timesteps = er.inputSequenceOfScalar("timesteps");
    const CollOfFace int_faces = er.interiorFaces();
    const CollOfFace bound = er.boundaryFaces();
    const CollOfScalar vol = er.norm(er.allCells());
    const CollOfScalar area = er.norm(er.allFaces());
    const Scalar gravity = double(9.81);
    std::function<std::array<CollOfScalar, 3>(const std::array<CollOfScalar, 3>&, const CollOfScalar&)> f = [&](const std::array<CollOfScalar, 3>& q, const CollOfScalar& b) -> std::array<CollOfScalar, 3> {
        const CollOfScalar rawWaterHeight = (q[0] - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > double(0.05)), rawWaterHeight, er.operatorExtend(double(1000), int_faces));
        const CollOfScalar f0temp = q[1];
        const CollOfScalar f1temp = ((q[1] * (q[1] / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const CollOfScalar f2temp = ((q[1] * q[2]) / waterHeight);
        const CollOfScalar f0 = er.trinaryIf((rawWaterHeight > double(0.05)), f0temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar f1 = er.trinaryIf((rawWaterHeight > double(0.05)), f1temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar f2 = er.trinaryIf((rawWaterHeight > double(0.05)), f2temp, er.operatorExtend(double(0), int_faces));
        return makeArray(f0, f1, f2);
    };
    std::function<std::array<CollOfScalar, 3>(const std::array<CollOfScalar, 3>&, const CollOfScalar&)> g = [&](const std::array<CollOfScalar, 3>& q, const CollOfScalar& b) -> std::array<CollOfScalar, 3> {
        const CollOfScalar rawWaterHeight = (q[0] - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > double(0.05)), rawWaterHeight, er.operatorExtend(double(1000), int_faces));
        const CollOfScalar g0temp = q[2];
        const CollOfScalar g1temp = (q[1] * (q[2] / waterHeight));
        const CollOfScalar g2temp = ((q[2] * (q[2] / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const CollOfScalar g0 = er.trinaryIf((rawWaterHeight > double(0.05)), g0temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar g1 = er.trinaryIf((rawWaterHeight > double(0.05)), g1temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar g2 = er.trinaryIf((rawWaterHeight > double(0.05)), g2temp, er.operatorExtend(double(0), int_faces));
        return makeArray(g0, g1, g2);
    };
    std::function<std::array<CollOfScalar, 2>(const std::array<CollOfScalar, 3>&, const CollOfScalar&)> eigenvalueF = [&](const std::array<CollOfScalar, 3>& q, const CollOfScalar& b) -> std::array<CollOfScalar, 2> {
        const CollOfScalar rawWaterHeight = (q[0] - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > double(0.05)), rawWaterHeight, er.operatorExtend(double(1000), int_faces));
        const CollOfScalar eigF0temp = ((q[1] / waterHeight) - er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigF1temp = ((q[1] / waterHeight) + er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigF0 = er.trinaryIf((rawWaterHeight > double(0.05)), eigF0temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar eigF1 = er.trinaryIf((rawWaterHeight > double(0.05)), eigF1temp, er.operatorExtend(double(0), int_faces));
        return makeArray(eigF0, eigF1);
    };
    std::function<std::array<CollOfScalar, 2>(const std::array<CollOfScalar, 3>&, const CollOfScalar&)> eigenvalueG = [&](const std::array<CollOfScalar, 3>& q, const CollOfScalar& b) -> std::array<CollOfScalar, 2> {
        const CollOfScalar rawWaterHeight = (q[0] - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > double(0.05)), rawWaterHeight, er.operatorExtend(double(1000), int_faces));
        const CollOfScalar eigG0temp = ((q[2] / waterHeight) - er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigG1temp = ((q[2] / waterHeight) + er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigG0 = er.trinaryIf((rawWaterHeight > double(0.05)), eigG0temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar eigG1 = er.trinaryIf((rawWaterHeight > double(0.05)), eigG1temp, er.operatorExtend(double(0), int_faces));
        return makeArray(eigG0, eigG1);
    };
    std::function<std::array<CollOfScalar, 2>(const std::array<CollOfScalar, 3>&)> a_eval = [&](const std::array<CollOfScalar, 3>& q) -> std::array<CollOfScalar, 2> {
        const std::array<CollOfScalar, 3> qFirst = makeArray(er.operatorOn(q[0], er.allCells(), er.firstCell(int_faces)), er.operatorOn(q[1], er.allCells(), er.firstCell(int_faces)), er.operatorOn(q[2], er.allCells(), er.firstCell(int_faces)));
        const std::array<CollOfScalar, 3> qSecond = makeArray(er.operatorOn(q[0], er.allCells(), er.secondCell(int_faces)), er.operatorOn(q[1], er.allCells(), er.secondCell(int_faces)), er.operatorOn(q[2], er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::array<CollOfScalar, 2> eigsFirst = eigenvalueF(qFirst, bFirst);
        const std::array<CollOfScalar, 2> eigsSecond = eigenvalueF(qSecond, bSecond);
        const CollOfScalar smallest = er.trinaryIf((eigsFirst[0] < eigsSecond[0]), eigsFirst[0], eigsSecond[0]);
        const CollOfScalar aminus = er.trinaryIf((smallest < double(0)), smallest, er.operatorExtend(double(0), int_faces));
        const CollOfScalar largest = er.trinaryIf((eigsFirst[1] > eigsSecond[1]), eigsFirst[1], eigsSecond[1]);
        const CollOfScalar aplus = er.trinaryIf((largest > double(0)), largest, er.operatorExtend(double(0), int_faces));
        return makeArray(aminus, aplus);
    };
    std::function<std::array<CollOfScalar, 2>(const std::array<CollOfScalar, 3>&)> b_eval = [&](const std::array<CollOfScalar, 3>& q) -> std::array<CollOfScalar, 2> {
        const std::array<CollOfScalar, 3> qFirst = makeArray(er.operatorOn(q[0], er.allCells(), er.firstCell(int_faces)), er.operatorOn(q[1], er.allCells(), er.firstCell(int_faces)), er.operatorOn(q[2], er.allCells(), er.firstCell(int_faces)));
        const std::array<CollOfScalar, 3> qSecond = makeArray(er.operatorOn(q[0], er.allCells(), er.secondCell(int_faces)), er.operatorOn(q[1], er.allCells(), er.secondCell(int_faces)), er.operatorOn(q[2], er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::array<CollOfScalar, 2> eigsFirst = eigenvalueG(qFirst, bFirst);
        const std::array<CollOfScalar, 2> eigsSecond = eigenvalueG(qSecond, bSecond);
        const CollOfScalar smallest = er.trinaryIf((eigsFirst[0] < eigsSecond[0]), eigsFirst[0], eigsSecond[0]);
        const CollOfScalar bminus = er.trinaryIf((smallest < double(0)), smallest, er.operatorExtend(double(0), int_faces));
        const CollOfScalar largest = er.trinaryIf((eigsFirst[1] > eigsSecond[1]), eigsFirst[1], eigsSecond[1]);
        const CollOfScalar bplus = er.trinaryIf((largest > double(0)), largest, er.operatorExtend(double(0), int_faces));
        return makeArray(bminus, bplus);
    };
    std::function<std::array<CollOfScalar, 3>(const std::array<CollOfScalar, 3>&)> numF = [&](const std::array<CollOfScalar, 3>& q) -> std::array<CollOfScalar, 3> {
        const std::array<CollOfScalar, 3> qFirst = makeArray(er.operatorOn(q[0], er.allCells(), er.firstCell(int_faces)), er.operatorOn(q[1], er.allCells(), er.firstCell(int_faces)), er.operatorOn(q[2], er.allCells(), er.firstCell(int_faces)));
        const std::array<CollOfScalar, 3> qSecond = makeArray(er.operatorOn(q[0], er.allCells(), er.secondCell(int_faces)), er.operatorOn(q[1], er.allCells(), er.secondCell(int_faces)), er.operatorOn(q[2], er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::array<CollOfScalar, 2> a = a_eval(q);
        const CollOfScalar adiffRaw = (a[1] - a[0]);
        const CollOfScalar adiff = er.trinaryIf(((adiffRaw * adiffRaw) > (double(0.05) * double(0.05))), adiffRaw, er.operatorExtend(double(1000), int_faces));
        const std::array<CollOfScalar, 3> fFirst = f(qFirst, bFirst);
        const std::array<CollOfScalar, 3> fSecond = f(qSecond, bSecond);
        const CollOfScalar aFactor = ((a[1] * a[0]) / adiff);
        const CollOfScalar firstPart0 = (((a[1] * fFirst[0]) - (a[0] * fSecond[0])) / adiff);
        const CollOfScalar firstPart1 = (((a[1] * fFirst[1]) - (a[0] * fSecond[1])) / adiff);
        const CollOfScalar firstPart2 = (((a[1] * fFirst[2]) - (a[0] * fSecond[2])) / adiff);
        const CollOfScalar intFluxF0temp = (firstPart0 + (aFactor * (qSecond[0] - qFirst[0])));
        const CollOfScalar intFluxF1temp = (firstPart1 + (aFactor * (qSecond[1] - qFirst[1])));
        const CollOfScalar intFluxF2temp = (firstPart2 + (aFactor * (qSecond[2] - qFirst[2])));
        const CollOfScalar intFluxF0 = er.trinaryIf(((adiffRaw * adiffRaw) > (double(0.05) * double(0.05))), intFluxF0temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar intFluxF1 = er.trinaryIf(((adiffRaw * adiffRaw) > (double(0.05) * double(0.05))), intFluxF1temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar intFluxF2 = er.trinaryIf(((adiffRaw * adiffRaw) > (double(0.05) * double(0.05))), intFluxF2temp, er.operatorExtend(double(0), int_faces));
        return makeArray(intFluxF0, intFluxF1, intFluxF2);
    };
    std::function<std::array<CollOfScalar, 3>(const std::array<CollOfScalar, 3>&)> numG = [&](const std::array<CollOfScalar, 3>& q) -> std::array<CollOfScalar, 3> {
        const std::array<CollOfScalar, 3> qFirst = makeArray(er.operatorOn(q[0], er.allCells(), er.firstCell(int_faces)), er.operatorOn(q[1], er.allCells(), er.firstCell(int_faces)), er.operatorOn(q[2], er.allCells(), er.firstCell(int_faces)));
        const std::array<CollOfScalar, 3> qSecond = makeArray(er.operatorOn(q[0], er.allCells(), er.secondCell(int_faces)), er.operatorOn(q[1], er.allCells(), er.secondCell(int_faces)), er.operatorOn(q[2], er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::array<CollOfScalar, 2> b = b_eval(q);
        const CollOfScalar bdiffRaw = (b[1] - b[0]);
        const CollOfScalar bdiff = er.trinaryIf(((bdiffRaw * bdiffRaw) > (double(0.05) * double(0.05))), bdiffRaw, er.operatorExtend(double(1000), int_faces));
        const std::array<CollOfScalar, 3> gFirst = g(qFirst, bFirst);
        const std::array<CollOfScalar, 3> gSecond = g(qSecond, bSecond);
        const CollOfScalar bFactor = ((b[1] * b[0]) / bdiff);
        const CollOfScalar firstPart0 = (((b[1] * gFirst[0]) - (b[0] * gSecond[0])) / bdiff);
        const CollOfScalar firstPart1 = (((b[1] * gFirst[1]) - (b[0] * gSecond[1])) / bdiff);
        const CollOfScalar firstPart2 = (((b[1] * gFirst[2]) - (b[0] * gSecond[2])) / bdiff);
        const CollOfScalar intFluxG0temp = (firstPart0 + (bFactor * (qSecond[0] - qFirst[0])));
        const CollOfScalar intFluxG1temp = (firstPart1 + (bFactor * (qSecond[1] - qFirst[1])));
        const CollOfScalar intFluxG2temp = (firstPart2 + (bFactor * (qSecond[2] - qFirst[2])));
        const CollOfScalar intFluxG0 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (double(0.05) * double(0.05))), intFluxG0temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar intFluxG1 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (double(0.05) * double(0.05))), intFluxG1temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar intFluxG2 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (double(0.05) * double(0.05))), intFluxG2temp, er.operatorExtend(double(0), int_faces));
        return makeArray(intFluxG0, intFluxG1, intFluxG2);
    };
    std::function<std::array<CollOfScalar, 3>(const std::array<CollOfScalar, 3>&)> get_flux = [&](const std::array<CollOfScalar, 3>& q) -> std::array<CollOfScalar, 3> {
        const CollOfVector int_orientation = er.normal(int_faces);
        const std::array<CollOfScalar, 2> pos_normal = makeArray(er.sqrt((CollOfScalar(int_orientation.col(0)) * CollOfScalar(int_orientation.col(0)))), er.sqrt((CollOfScalar(int_orientation.col(1)) * CollOfScalar(int_orientation.col(1)))));
        const std::array<CollOfScalar, 3> int_numF = numF(q);
        const std::array<CollOfScalar, 3> int_numG = numG(q);
        const CollOfScalar int_fluxes0 = ((pos_normal[0] * int_numF[0]) + (pos_normal[1] * int_numG[0]));
        const CollOfScalar int_fluxes1 = ((pos_normal[0] * int_numF[1]) + (pos_normal[1] * int_numG[1]));
        const CollOfScalar int_fluxes2 = ((pos_normal[0] * int_numF[2]) + (pos_normal[1] * int_numG[2]));
        const std::array<CollOfScalar, 3> intFlux = makeArray(int_fluxes0, int_fluxes1, int_fluxes2);
        const CollOfVector bound_orientation = er.normal(bound);
        const CollOfCell bound_cells = er.trinaryIf(er.isEmpty(er.firstCell(bound)), er.secondCell(bound), er.firstCell(bound));
        const CollOfScalar bound_q0 = er.operatorOn(q[0], er.allCells(), bound_cells);
        const CollOfScalar bound_b = er.operatorOn(b_mid, er.allCells(), bound_cells);
        const CollOfScalar bound_height = (bound_q0 - bound_b);
        const CollOfScalar bound_signX = er.trinaryIf((CollOfScalar(bound_orientation.col(0)) > double(0)), er.operatorExtend(double(1), bound), er.operatorExtend(-double(1), bound));
        const CollOfScalar bound_signY = er.trinaryIf((CollOfScalar(bound_orientation.col(1)) > double(0)), er.operatorExtend(double(1), bound), er.operatorExtend(-double(1), bound));
        const CollOfScalar b_fluxtemp = (((double(0.5) * gravity) * bound_height) * bound_height);
        const CollOfScalar b_flux = er.trinaryIf((bound_height > double(0.05)), b_fluxtemp, er.operatorExtend(double(0), bound));
        const CollOfScalar boundFlux0 = er.operatorExtend(double(0), bound);
        const CollOfScalar boundFlux1 = ((er.sqrt((CollOfScalar(bound_orientation.col(0)) * CollOfScalar(bound_orientation.col(0)))) * b_flux) * bound_signX);
        const CollOfScalar boundFlux2 = ((er.sqrt((CollOfScalar(bound_orientation.col(1)) * CollOfScalar(bound_orientation.col(1)))) * b_flux) * bound_signY);
        const std::array<CollOfScalar, 3> boundFlux = makeArray(boundFlux0, boundFlux1, boundFlux2);
        const std::array<CollOfScalar, 3> allFluxes = makeArray(((er.operatorExtend(double(0), er.allFaces()) + er.operatorExtend(boundFlux[0], er.boundaryFaces(), er.allFaces())) + er.operatorExtend(intFlux[0], er.interiorFaces(), er.allFaces())), ((er.operatorExtend(double(0), er.allFaces()) + er.operatorExtend(boundFlux[1], er.boundaryFaces(), er.allFaces())) + er.operatorExtend(intFlux[1], er.interiorFaces(), er.allFaces())), ((er.operatorExtend(double(0), er.allFaces()) + er.operatorExtend(boundFlux[2], er.boundaryFaces(), er.allFaces())) + er.operatorExtend(intFlux[2], er.interiorFaces(), er.allFaces())));
        return allFluxes;
    };
    std::function<std::array<CollOfScalar, 3>(const std::array<CollOfScalar, 3>&)> evalSourceTerm = [&](const std::array<CollOfScalar, 3>& q) -> std::array<CollOfScalar, 3> {
        const CollOfScalar bx = ((b_east - b_west) / dx);
        const CollOfScalar by = ((b_north - b_south) / dy);
        const CollOfScalar secondTerm_x = (((q[0] - b_east) + (q[0] - b_west)) / double(2));
        const CollOfScalar secondTerm_y = (((q[0] - b_north) + (q[0] - b_south)) / double(2));
        const CollOfScalar dryTerm = er.trinaryIf(((q[0] - b_mid) > double(0.05)), er.operatorExtend(double(1), er.allCells()), er.operatorExtend(double(0), er.allCells()));
        return makeArray(er.operatorExtend(double(0), er.allCells()), (((-gravity * bx) * secondTerm_x) * dryTerm), (((-gravity * by) * secondTerm_y) * dryTerm));
    };
    std::function<std::array<CollOfScalar, 3>(const std::array<CollOfScalar, 3>&, const Scalar&)> rungeKutta = [&](const std::array<CollOfScalar, 3>& q, const Scalar& dt) -> std::array<CollOfScalar, 3> {
        const std::array<CollOfScalar, 3> flux = get_flux(q);
        const std::array<CollOfScalar, 3> source = evalSourceTerm(q);
        const CollOfScalar q_star0 = (q[0] + ((dt / vol) * (-er.divergence(flux[0]) + (vol * source[0]))));
        const CollOfScalar q_star1 = (q[1] + ((dt / vol) * (-er.divergence(flux[1]) + (vol * source[1]))));
        const CollOfScalar q_star2 = (q[2] + ((dt / vol) * (-er.divergence(flux[2]) + (vol * source[2]))));
        const std::array<CollOfScalar, 3> flux_star = get_flux(makeArray(q_star0, q_star1, q_star2));
        const std::array<CollOfScalar, 3> source_star = evalSourceTerm(makeArray(q_star0, q_star1, q_star2));
        const CollOfScalar newQ0 = ((double(0.5) * q[0]) + (double(0.5) * (q_star0 + ((dt / vol) * (-er.divergence(flux_star[0]) + (vol * source_star[0]))))));
        const CollOfScalar newQ1 = ((double(0.5) * q[1]) + (double(0.5) * (q_star1 + ((dt / vol) * (-er.divergence(flux_star[1]) + (vol * source_star[1]))))));
        const CollOfScalar newQ2 = ((double(0.5) * q[2]) + (double(0.5) * (q_star2 + ((dt / vol) * (-er.divergence(flux_star[2]) + (vol * source_star[2]))))));
        return makeArray(newQ0, newQ1, newQ2);
    };
    std::array<CollOfScalar, 3> q0;
    q0 = makeArray((h_init + b_mid), (h_init * u_init), (h_init * v_init));
    er.output("q1", q0[0]);
    er.output("q2", q0[1]);
    er.output("q3", q0[2]);
    for (const Scalar& dt : timesteps) {
        const std::array<CollOfScalar, 3> q = rungeKutta(q0, dt);
        //er.output("q1", q[0]);
        //er.output("q2", q[1]);
        //er.output("q3", q[2]);
        q0 = q;
    }

    // ============= Generated code ends here ================

    return 0;
}

void ensureRequirements(const EquelleRuntimeCUDA& er)
{
    er.ensureGridDimensionMin(1);
    er.ensureGridDimensionMin(2);
}
