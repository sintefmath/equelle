
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

#include "equelle/EquelleRuntimeCPU.hpp"
#include "equelle/CartesianGrid.hpp"//Should be renamed EquelleCartesianRuntimeCPU

void ensureRequirements(const equelle::EquelleRuntimeCPU& er);
void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er, equelle::CartesianEquelleRuntime& er_cart);

#ifndef EQUELLE_NO_MAIN
int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    equelle::CartesianEquelleRuntime er_cart(param);
    equelle::EquelleRuntimeCPU er(param);
    equelleGeneratedCode(er, er_cart);
    return 0;
}
#endif // EQUELLE_NO_MAIN

void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er,
                          equelle::CartesianEquelleRuntime& er_cart) {
    using namespace equelle;
    ensureRequirements(er);
    (void)er_cart; // To suppress compile warnings if not used below.

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
    auto f = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > double(0.05)), rawWaterHeight, er.operatorExtend(double(1000), int_faces));
        const CollOfScalar f0temp = std::get<1>(q);
        const CollOfScalar f1temp = ((std::get<1>(q) * (std::get<1>(q) / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const CollOfScalar f2temp = ((std::get<1>(q) * std::get<2>(q)) / waterHeight);
        const CollOfScalar f0 = er.trinaryIf((rawWaterHeight > double(0.05)), f0temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar f1 = er.trinaryIf((rawWaterHeight > double(0.05)), f1temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar f2 = er.trinaryIf((rawWaterHeight > double(0.05)), f2temp, er.operatorExtend(double(0), int_faces));
        return makeArray(f0, f1, f2);
    };
    auto g = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > double(0.05)), rawWaterHeight, er.operatorExtend(double(1000), int_faces));
        const CollOfScalar g0temp = std::get<2>(q);
        const CollOfScalar g1temp = (std::get<1>(q) * (std::get<2>(q) / waterHeight));
        const CollOfScalar g2temp = ((std::get<2>(q) * (std::get<2>(q) / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const CollOfScalar g0 = er.trinaryIf((rawWaterHeight > double(0.05)), g0temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar g1 = er.trinaryIf((rawWaterHeight > double(0.05)), g1temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar g2 = er.trinaryIf((rawWaterHeight > double(0.05)), g2temp, er.operatorExtend(double(0), int_faces));
        return makeArray(g0, g1, g2);
    };
    auto eigenvalueF = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > double(0.05)), rawWaterHeight, er.operatorExtend(double(1000), int_faces));
        const CollOfScalar eigF0temp = ((std::get<1>(q) / waterHeight) - er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigF1temp = ((std::get<1>(q) / waterHeight) + er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigF0 = er.trinaryIf((rawWaterHeight > double(0.05)), eigF0temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar eigF1 = er.trinaryIf((rawWaterHeight > double(0.05)), eigF1temp, er.operatorExtend(double(0), int_faces));
        return makeArray(eigF0, eigF1);
    };
    auto eigenvalueG = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > double(0.05)), rawWaterHeight, er.operatorExtend(double(1000), int_faces));
        const CollOfScalar eigG0temp = ((std::get<2>(q) / waterHeight) - er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigG1temp = ((std::get<2>(q) / waterHeight) + er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigG0 = er.trinaryIf((rawWaterHeight > double(0.05)), eigG0temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar eigG1 = er.trinaryIf((rawWaterHeight > double(0.05)), eigG1temp, er.operatorExtend(double(0), int_faces));
        return makeArray(eigG0, eigG1);
    };
    auto a_eval = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::tuple<CollOfScalar, CollOfScalar> eigsFirst = eigenvalueF(qFirst, bFirst);
        const std::tuple<CollOfScalar, CollOfScalar> eigsSecond = eigenvalueF(qSecond, bSecond);
        const CollOfScalar smallest = er.trinaryIf((std::get<0>(eigsFirst) < std::get<0>(eigsSecond)), std::get<0>(eigsFirst), std::get<0>(eigsSecond));
        const CollOfScalar aminus = er.trinaryIf((smallest < double(0)), smallest, er.operatorExtend(double(0), int_faces));
        const CollOfScalar largest = er.trinaryIf((std::get<1>(eigsFirst) > std::get<1>(eigsSecond)), std::get<1>(eigsFirst), std::get<1>(eigsSecond));
        const CollOfScalar aplus = er.trinaryIf((largest > double(0)), largest, er.operatorExtend(double(0), int_faces));
        return makeArray(aminus, aplus);
    };
    auto b_eval = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::tuple<CollOfScalar, CollOfScalar> eigsFirst = eigenvalueG(qFirst, bFirst);
        const std::tuple<CollOfScalar, CollOfScalar> eigsSecond = eigenvalueG(qSecond, bSecond);
        const CollOfScalar smallest = er.trinaryIf((std::get<0>(eigsFirst) < std::get<0>(eigsSecond)), std::get<0>(eigsFirst), std::get<0>(eigsSecond));
        const CollOfScalar bminus = er.trinaryIf((smallest < double(0)), smallest, er.operatorExtend(double(0), int_faces));
        const CollOfScalar largest = er.trinaryIf((std::get<1>(eigsFirst) > std::get<1>(eigsSecond)), std::get<1>(eigsFirst), std::get<1>(eigsSecond));
        const CollOfScalar bplus = er.trinaryIf((largest > double(0)), largest, er.operatorExtend(double(0), int_faces));
        return makeArray(bminus, bplus);
    };
    auto numF = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::tuple<CollOfScalar, CollOfScalar> a = a_eval(q);
        const CollOfScalar adiffRaw = (std::get<1>(a) - std::get<0>(a));
        const CollOfScalar adiff = er.trinaryIf(((adiffRaw * adiffRaw) > (double(0.05) * double(0.05))), adiffRaw, er.operatorExtend(double(1000), int_faces));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> fFirst = f(qFirst, bFirst);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> fSecond = f(qSecond, bSecond);
        const CollOfScalar aFactor = ((std::get<1>(a) * std::get<0>(a)) / adiff);
        const CollOfScalar firstPart0 = (((std::get<1>(a) * std::get<0>(fFirst)) - (std::get<0>(a) * std::get<0>(fSecond))) / adiff);
        const CollOfScalar firstPart1 = (((std::get<1>(a) * std::get<1>(fFirst)) - (std::get<0>(a) * std::get<1>(fSecond))) / adiff);
        const CollOfScalar firstPart2 = (((std::get<1>(a) * std::get<2>(fFirst)) - (std::get<0>(a) * std::get<2>(fSecond))) / adiff);
        const CollOfScalar intFluxF0temp = (firstPart0 + (aFactor * (std::get<0>(qSecond) - std::get<0>(qFirst))));
        const CollOfScalar intFluxF1temp = (firstPart1 + (aFactor * (std::get<1>(qSecond) - std::get<1>(qFirst))));
        const CollOfScalar intFluxF2temp = (firstPart2 + (aFactor * (std::get<2>(qSecond) - std::get<2>(qFirst))));
        const CollOfScalar intFluxF0 = er.trinaryIf(((adiffRaw * adiffRaw) > (double(0.05) * double(0.05))), intFluxF0temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar intFluxF1 = er.trinaryIf(((adiffRaw * adiffRaw) > (double(0.05) * double(0.05))), intFluxF1temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar intFluxF2 = er.trinaryIf(((adiffRaw * adiffRaw) > (double(0.05) * double(0.05))), intFluxF2temp, er.operatorExtend(double(0), int_faces));
        return makeArray(intFluxF0, intFluxF1, intFluxF2);
    };
    auto numG = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::tuple<CollOfScalar, CollOfScalar> b = b_eval(q);
        const CollOfScalar bdiffRaw = (std::get<1>(b) - std::get<0>(b));
        const CollOfScalar bdiff = er.trinaryIf(((bdiffRaw * bdiffRaw) > (double(0.05) * double(0.05))), bdiffRaw, er.operatorExtend(double(1000), int_faces));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> gFirst = g(qFirst, bFirst);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> gSecond = g(qSecond, bSecond);
        const CollOfScalar bFactor = ((std::get<1>(b) * std::get<0>(b)) / bdiff);
        const CollOfScalar firstPart0 = (((std::get<1>(b) * std::get<0>(gFirst)) - (std::get<0>(b) * std::get<0>(gSecond))) / bdiff);
        const CollOfScalar firstPart1 = (((std::get<1>(b) * std::get<1>(gFirst)) - (std::get<0>(b) * std::get<1>(gSecond))) / bdiff);
        const CollOfScalar firstPart2 = (((std::get<1>(b) * std::get<2>(gFirst)) - (std::get<0>(b) * std::get<2>(gSecond))) / bdiff);
        const CollOfScalar intFluxG0temp = (firstPart0 + (bFactor * (std::get<0>(qSecond) - std::get<0>(qFirst))));
        const CollOfScalar intFluxG1temp = (firstPart1 + (bFactor * (std::get<1>(qSecond) - std::get<1>(qFirst))));
        const CollOfScalar intFluxG2temp = (firstPart2 + (bFactor * (std::get<2>(qSecond) - std::get<2>(qFirst))));
        const CollOfScalar intFluxG0 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (double(0.05) * double(0.05))), intFluxG0temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar intFluxG1 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (double(0.05) * double(0.05))), intFluxG1temp, er.operatorExtend(double(0), int_faces));
        const CollOfScalar intFluxG2 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (double(0.05) * double(0.05))), intFluxG2temp, er.operatorExtend(double(0), int_faces));
        return makeArray(intFluxG0, intFluxG1, intFluxG2);
    };
    auto get_flux = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfVector int_orientation = er.normal(int_faces);
        const std::tuple<CollOfScalar, CollOfScalar> pos_normal = makeArray(er.sqrt((CollOfScalar(int_orientation.col(0)) * CollOfScalar(int_orientation.col(0)))), er.sqrt((CollOfScalar(int_orientation.col(1)) * CollOfScalar(int_orientation.col(1)))));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> int_numF = numF(q);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> int_numG = numG(q);
        const CollOfScalar int_fluxes0 = ((std::get<0>(pos_normal) * std::get<0>(int_numF)) + (std::get<1>(pos_normal) * std::get<0>(int_numG)));
        const CollOfScalar int_fluxes1 = ((std::get<0>(pos_normal) * std::get<1>(int_numF)) + (std::get<1>(pos_normal) * std::get<1>(int_numG)));
        const CollOfScalar int_fluxes2 = ((std::get<0>(pos_normal) * std::get<2>(int_numF)) + (std::get<1>(pos_normal) * std::get<2>(int_numG)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> intFlux = makeArray(int_fluxes0, int_fluxes1, int_fluxes2);
        const CollOfVector bound_orientation = er.normal(bound);
        const CollOfCell bound_cells = er.trinaryIf(er.isEmpty(er.firstCell(bound)), er.secondCell(bound), er.firstCell(bound));
        const CollOfScalar bound_q0 = er.operatorOn(std::get<0>(q), er.allCells(), bound_cells);
        const CollOfScalar bound_b = er.operatorOn(b_mid, er.allCells(), bound_cells);
        const CollOfScalar bound_height = (bound_q0 - bound_b);
        const CollOfScalar bound_signX = er.trinaryIf((CollOfScalar(bound_orientation.col(0)) > double(0)), er.operatorExtend(double(1), bound), er.operatorExtend(-double(1), bound));
        const CollOfScalar bound_signY = er.trinaryIf((CollOfScalar(bound_orientation.col(1)) > double(0)), er.operatorExtend(double(1), bound), er.operatorExtend(-double(1), bound));
        const CollOfScalar b_fluxtemp = (((double(0.5) * gravity) * bound_height) * bound_height);
        const CollOfScalar b_flux = er.trinaryIf((bound_height > double(0.05)), b_fluxtemp, er.operatorExtend(double(0), bound));
        const CollOfScalar boundFlux0 = er.operatorExtend(double(0), bound);
        const CollOfScalar boundFlux1 = ((er.sqrt((CollOfScalar(bound_orientation.col(0)) * CollOfScalar(bound_orientation.col(0)))) * b_flux) * bound_signX);
        const CollOfScalar boundFlux2 = ((er.sqrt((CollOfScalar(bound_orientation.col(1)) * CollOfScalar(bound_orientation.col(1)))) * b_flux) * bound_signY);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> boundFlux = makeArray(boundFlux0, boundFlux1, boundFlux2);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> allFluxes = makeArray(((er.operatorExtend(double(0), er.allFaces()) + er.operatorExtend(std::get<0>(boundFlux), er.boundaryFaces(), er.allFaces())) + er.operatorExtend(std::get<0>(intFlux), er.interiorFaces(), er.allFaces())), ((er.operatorExtend(double(0), er.allFaces()) + er.operatorExtend(std::get<1>(boundFlux), er.boundaryFaces(), er.allFaces())) + er.operatorExtend(std::get<1>(intFlux), er.interiorFaces(), er.allFaces())), ((er.operatorExtend(double(0), er.allFaces()) + er.operatorExtend(std::get<2>(boundFlux), er.boundaryFaces(), er.allFaces())) + er.operatorExtend(std::get<2>(intFlux), er.interiorFaces(), er.allFaces())));
        return allFluxes;
    };
    auto evalSourceTerm = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar bx = ((b_east - b_west) / dx);
        const CollOfScalar by = ((b_north - b_south) / dy);
        const CollOfScalar secondTerm_x = (((std::get<0>(q) - b_east) + (std::get<0>(q) - b_west)) / double(2));
        const CollOfScalar secondTerm_y = (((std::get<0>(q) - b_north) + (std::get<0>(q) - b_south)) / double(2));
        const CollOfScalar dryTerm = er.trinaryIf(((std::get<0>(q) - b_mid) > double(0.05)), er.operatorExtend(double(1), er.allCells()), er.operatorExtend(double(0), er.allCells()));
        return makeArray(er.operatorExtend(double(0), er.allCells()), (((-gravity * bx) * secondTerm_x) * dryTerm), (((-gravity * by) * secondTerm_y) * dryTerm));
    };
    auto rungeKutta = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const Scalar& dt) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> flux = get_flux(q);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> source = evalSourceTerm(q);
        const CollOfScalar q_star0 = (std::get<0>(q) + ((dt / vol) * (-er.divergence(std::get<0>(flux)) + (vol * std::get<0>(source)))));
        const CollOfScalar q_star1 = (std::get<1>(q) + ((dt / vol) * (-er.divergence(std::get<1>(flux)) + (vol * std::get<1>(source)))));
        const CollOfScalar q_star2 = (std::get<2>(q) + ((dt / vol) * (-er.divergence(std::get<2>(flux)) + (vol * std::get<2>(source)))));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> flux_star = get_flux(makeArray(q_star0, q_star1, q_star2));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> source_star = evalSourceTerm(makeArray(q_star0, q_star1, q_star2));
        const CollOfScalar newQ0 = ((double(0.5) * std::get<0>(q)) + (double(0.5) * (q_star0 + ((dt / vol) * (-er.divergence(std::get<0>(flux_star)) + (vol * std::get<0>(source_star)))))));
        const CollOfScalar newQ1 = ((double(0.5) * std::get<1>(q)) + (double(0.5) * (q_star1 + ((dt / vol) * (-er.divergence(std::get<1>(flux_star)) + (vol * std::get<1>(source_star)))))));
        const CollOfScalar newQ2 = ((double(0.5) * std::get<2>(q)) + (double(0.5) * (q_star2 + ((dt / vol) * (-er.divergence(std::get<2>(flux_star)) + (vol * std::get<2>(source_star)))))));
        return makeArray(newQ0, newQ1, newQ2);
    };
    std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> q0 = makeArray((h_init + b_mid), (h_init * u_init), (h_init * v_init));
    er.output("q1", std::get<0>(q0));
    er.output("q2", std::get<1>(q0));
    er.output("q3", std::get<2>(q0));
    for (const Scalar& dt : timesteps) {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> q = rungeKutta(q0, dt);
        er.output("q1", std::get<0>(q));
        er.output("q2", std::get<1>(q));
        er.output("q3", std::get<2>(q));
        q0 = q;
    }

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    er.ensureGridDimensionMin(1);
    er.ensureGridDimensionMin(2);
}
