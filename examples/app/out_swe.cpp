
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

    const auto h_init = er.inputCollectionOfScalar("h_init", er.allCells());
    const auto u_init = er.inputCollectionOfScalar("u_init", er.allCells());
    const auto v_init = er.inputCollectionOfScalar("v_init", er.allCells());
    const auto b_north = er.inputCollectionOfScalar("b_north", er.allCells());
    const auto b_south = er.inputCollectionOfScalar("b_south", er.allCells());
    const auto b_east = er.inputCollectionOfScalar("b_east", er.allCells());
    const auto b_west = er.inputCollectionOfScalar("b_west", er.allCells());
    const auto b_mid = ((((b_north + b_south) + b_east) + b_west) / double(4));
    er.output("bottom", b_mid);
    const auto dx = er.inputScalarWithDefault("dx", double(10));
    const auto dy = er.inputScalarWithDefault("dy", double(10));
    const auto timesteps = er.inputSequenceOfScalar("timesteps");
    const auto int_faces = er.interiorFaces();
    const auto bound = er.boundaryFaces();
    const auto vol = er.norm(er.allCells());
    const auto area = er.norm(er.allFaces());
    const auto gravity = double(9.81);
    auto f = [&](const auto& q, const auto& b) {
        const auto rawWaterHeight = (std::get<0>(q) - b);
        const auto waterHeight = er.trinaryIf((rawWaterHeight > double(0.05)), rawWaterHeight, er.operatorExtend(double(1000), int_faces));
        const auto f0temp = std::get<1>(q);
        const auto f1temp = ((std::get<1>(q) * (std::get<1>(q) / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const auto f2temp = ((std::get<1>(q) * std::get<2>(q)) / waterHeight);
        const auto f0 = er.trinaryIf((rawWaterHeight > double(0.05)), f0temp, er.operatorExtend(double(0), int_faces));
        const auto f1 = er.trinaryIf((rawWaterHeight > double(0.05)), f1temp, er.operatorExtend(double(0), int_faces));
        const auto f2 = er.trinaryIf((rawWaterHeight > double(0.05)), f2temp, er.operatorExtend(double(0), int_faces));
        return makeArray(f0, f1, f2);
    };
    auto g = [&](const auto& q, const auto& b) {
        const auto rawWaterHeight = (std::get<0>(q) - b);
        const auto waterHeight = er.trinaryIf((rawWaterHeight > double(0.05)), rawWaterHeight, er.operatorExtend(double(1000), int_faces));
        const auto g0temp = std::get<2>(q);
        const auto g1temp = (std::get<1>(q) * (std::get<2>(q) / waterHeight));
        const auto g2temp = ((std::get<2>(q) * (std::get<2>(q) / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const auto g0 = er.trinaryIf((rawWaterHeight > double(0.05)), g0temp, er.operatorExtend(double(0), int_faces));
        const auto g1 = er.trinaryIf((rawWaterHeight > double(0.05)), g1temp, er.operatorExtend(double(0), int_faces));
        const auto g2 = er.trinaryIf((rawWaterHeight > double(0.05)), g2temp, er.operatorExtend(double(0), int_faces));
        return makeArray(g0, g1, g2);
    };
    auto eigenvalueF = [&](const auto& q, const auto& b) {
        const auto rawWaterHeight = (std::get<0>(q) - b);
        const auto waterHeight = er.trinaryIf((rawWaterHeight > double(0.05)), rawWaterHeight, er.operatorExtend(double(1000), int_faces));
        const auto eigF0temp = ((std::get<1>(q) / waterHeight) - er.sqrt((gravity * waterHeight)));
        const auto eigF1temp = ((std::get<1>(q) / waterHeight) + er.sqrt((gravity * waterHeight)));
        const auto eigF0 = er.trinaryIf((rawWaterHeight > double(0.05)), eigF0temp, er.operatorExtend(double(0), int_faces));
        const auto eigF1 = er.trinaryIf((rawWaterHeight > double(0.05)), eigF1temp, er.operatorExtend(double(0), int_faces));
        return makeArray(eigF0, eigF1);
    };
    auto eigenvalueG = [&](const auto& q, const auto& b) {
        const auto rawWaterHeight = (std::get<0>(q) - b);
        const auto waterHeight = er.trinaryIf((rawWaterHeight > double(0.05)), rawWaterHeight, er.operatorExtend(double(1000), int_faces));
        const auto eigG0temp = ((std::get<2>(q) / waterHeight) - er.sqrt((gravity * waterHeight)));
        const auto eigG1temp = ((std::get<2>(q) / waterHeight) + er.sqrt((gravity * waterHeight)));
        const auto eigG0 = er.trinaryIf((rawWaterHeight > double(0.05)), eigG0temp, er.operatorExtend(double(0), int_faces));
        const auto eigG1 = er.trinaryIf((rawWaterHeight > double(0.05)), eigG1temp, er.operatorExtend(double(0), int_faces));
        return makeArray(eigG0, eigG1);
    };
    auto a_eval = [&](const auto& q) {
        const auto qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const auto qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const auto bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const auto bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const auto eigsFirst = eigenvalueF(qFirst, bFirst);
        const auto eigsSecond = eigenvalueF(qSecond, bSecond);
        const auto smallest = er.trinaryIf((std::get<0>(eigsFirst) < std::get<0>(eigsSecond)), std::get<0>(eigsFirst), std::get<0>(eigsSecond));
        const auto aminus = er.trinaryIf((smallest < double(0)), smallest, er.operatorExtend(double(0), int_faces));
        const auto largest = er.trinaryIf((std::get<1>(eigsFirst) > std::get<1>(eigsSecond)), std::get<1>(eigsFirst), std::get<1>(eigsSecond));
        const auto aplus = er.trinaryIf((largest > double(0)), largest, er.operatorExtend(double(0), int_faces));
        return makeArray(aminus, aplus);
    };
    auto b_eval = [&](const auto& q) {
        const auto qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const auto qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const auto bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const auto bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const auto eigsFirst = eigenvalueG(qFirst, bFirst);
        const auto eigsSecond = eigenvalueG(qSecond, bSecond);
        const auto smallest = er.trinaryIf((std::get<0>(eigsFirst) < std::get<0>(eigsSecond)), std::get<0>(eigsFirst), std::get<0>(eigsSecond));
        const auto bminus = er.trinaryIf((smallest < double(0)), smallest, er.operatorExtend(double(0), int_faces));
        const auto largest = er.trinaryIf((std::get<1>(eigsFirst) > std::get<1>(eigsSecond)), std::get<1>(eigsFirst), std::get<1>(eigsSecond));
        const auto bplus = er.trinaryIf((largest > double(0)), largest, er.operatorExtend(double(0), int_faces));
        return makeArray(bminus, bplus);
    };
    auto numF = [&](const auto& q) {
        const auto qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const auto qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const auto bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const auto bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const auto a = a_eval(q);
        const auto adiffRaw = (std::get<1>(a) - std::get<0>(a));
        const auto adiff = er.trinaryIf(((adiffRaw * adiffRaw) > (double(0.05) * double(0.05))), adiffRaw, er.operatorExtend(double(1000), int_faces));
        const auto fFirst = f(qFirst, bFirst);
        const auto fSecond = f(qSecond, bSecond);
        const auto aFactor = ((std::get<1>(a) * std::get<0>(a)) / adiff);
        const auto firstPart0 = (((std::get<1>(a) * std::get<0>(fFirst)) - (std::get<0>(a) * std::get<0>(fSecond))) / adiff);
        const auto firstPart1 = (((std::get<1>(a) * std::get<1>(fFirst)) - (std::get<0>(a) * std::get<1>(fSecond))) / adiff);
        const auto firstPart2 = (((std::get<1>(a) * std::get<2>(fFirst)) - (std::get<0>(a) * std::get<2>(fSecond))) / adiff);
        const auto intFluxF0temp = (firstPart0 + (aFactor * (std::get<0>(qSecond) - std::get<0>(qFirst))));
        const auto intFluxF1temp = (firstPart1 + (aFactor * (std::get<1>(qSecond) - std::get<1>(qFirst))));
        const auto intFluxF2temp = (firstPart2 + (aFactor * (std::get<2>(qSecond) - std::get<2>(qFirst))));
        const auto intFluxF0 = er.trinaryIf(((adiffRaw * adiffRaw) > (double(0.05) * double(0.05))), intFluxF0temp, er.operatorExtend(double(0), int_faces));
        const auto intFluxF1 = er.trinaryIf(((adiffRaw * adiffRaw) > (double(0.05) * double(0.05))), intFluxF1temp, er.operatorExtend(double(0), int_faces));
        const auto intFluxF2 = er.trinaryIf(((adiffRaw * adiffRaw) > (double(0.05) * double(0.05))), intFluxF2temp, er.operatorExtend(double(0), int_faces));
        return makeArray(intFluxF0, intFluxF1, intFluxF2);
    };
    auto numG = [&](const auto& q) {
        const auto qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const auto qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const auto bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const auto bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const auto b = b_eval(q);
        const auto bdiffRaw = (std::get<1>(b) - std::get<0>(b));
        const auto bdiff = er.trinaryIf(((bdiffRaw * bdiffRaw) > (double(0.05) * double(0.05))), bdiffRaw, er.operatorExtend(double(1000), int_faces));
        const auto gFirst = g(qFirst, bFirst);
        const auto gSecond = g(qSecond, bSecond);
        const auto bFactor = ((std::get<1>(b) * std::get<0>(b)) / bdiff);
        const auto firstPart0 = (((std::get<1>(b) * std::get<0>(gFirst)) - (std::get<0>(b) * std::get<0>(gSecond))) / bdiff);
        const auto firstPart1 = (((std::get<1>(b) * std::get<1>(gFirst)) - (std::get<0>(b) * std::get<1>(gSecond))) / bdiff);
        const auto firstPart2 = (((std::get<1>(b) * std::get<2>(gFirst)) - (std::get<0>(b) * std::get<2>(gSecond))) / bdiff);
        const auto intFluxG0temp = (firstPart0 + (bFactor * (std::get<0>(qSecond) - std::get<0>(qFirst))));
        const auto intFluxG1temp = (firstPart1 + (bFactor * (std::get<1>(qSecond) - std::get<1>(qFirst))));
        const auto intFluxG2temp = (firstPart2 + (bFactor * (std::get<2>(qSecond) - std::get<2>(qFirst))));
        const auto intFluxG0 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (double(0.05) * double(0.05))), intFluxG0temp, er.operatorExtend(double(0), int_faces));
        const auto intFluxG1 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (double(0.05) * double(0.05))), intFluxG1temp, er.operatorExtend(double(0), int_faces));
        const auto intFluxG2 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (double(0.05) * double(0.05))), intFluxG2temp, er.operatorExtend(double(0), int_faces));
        return makeArray(intFluxG0, intFluxG1, intFluxG2);
    };
    auto get_flux = [&](const auto& q) {
        const auto int_orientation = er.normal(int_faces);
        const auto pos_normal = makeArray(er.sqrt((CollOfScalar(int_orientation.col(0)) * CollOfScalar(int_orientation.col(0)))), er.sqrt((CollOfScalar(int_orientation.col(1)) * CollOfScalar(int_orientation.col(1)))));
        const auto int_numF = numF(q);
        const auto int_numG = numG(q);
        const auto int_fluxes0 = ((std::get<0>(pos_normal) * std::get<0>(int_numF)) + (std::get<1>(pos_normal) * std::get<0>(int_numG)));
        const auto int_fluxes1 = ((std::get<0>(pos_normal) * std::get<1>(int_numF)) + (std::get<1>(pos_normal) * std::get<1>(int_numG)));
        const auto int_fluxes2 = ((std::get<0>(pos_normal) * std::get<2>(int_numF)) + (std::get<1>(pos_normal) * std::get<2>(int_numG)));
        const auto intFlux = makeArray(int_fluxes0, int_fluxes1, int_fluxes2);
        const auto bound_orientation = er.normal(bound);
        const auto bound_cells = er.trinaryIf(er.isEmpty(er.firstCell(bound)), er.secondCell(bound), er.firstCell(bound));
        const auto bound_q0 = er.operatorOn(std::get<0>(q), er.allCells(), bound_cells);
        const auto bound_b = er.operatorOn(b_mid, er.allCells(), bound_cells);
        const auto bound_height = (bound_q0 - bound_b);
        const auto bound_signX = er.trinaryIf((CollOfScalar(bound_orientation.col(0)) > double(0)), er.operatorExtend(double(1), bound), er.operatorExtend(-double(1), bound));
        const auto bound_signY = er.trinaryIf((CollOfScalar(bound_orientation.col(1)) > double(0)), er.operatorExtend(double(1), bound), er.operatorExtend(-double(1), bound));
        const auto b_fluxtemp = (((double(0.5) * gravity) * bound_height) * bound_height);
        const auto b_flux = er.trinaryIf((bound_height > double(0.05)), b_fluxtemp, er.operatorExtend(double(0), bound));
        const auto boundFlux0 = er.operatorExtend(double(0), bound);
        const auto boundFlux1 = ((er.sqrt((CollOfScalar(bound_orientation.col(0)) * CollOfScalar(bound_orientation.col(0)))) * b_flux) * bound_signX);
        const auto boundFlux2 = ((er.sqrt((CollOfScalar(bound_orientation.col(1)) * CollOfScalar(bound_orientation.col(1)))) * b_flux) * bound_signY);
        const auto boundFlux = makeArray(boundFlux0, boundFlux1, boundFlux2);
        const auto allFluxes = makeArray(((er.operatorExtend(double(0), er.allFaces()) + er.operatorExtend(std::get<0>(boundFlux), er.boundaryFaces(), er.allFaces())) + er.operatorExtend(std::get<0>(intFlux), er.interiorFaces(), er.allFaces())), ((er.operatorExtend(double(0), er.allFaces()) + er.operatorExtend(std::get<1>(boundFlux), er.boundaryFaces(), er.allFaces())) + er.operatorExtend(std::get<1>(intFlux), er.interiorFaces(), er.allFaces())), ((er.operatorExtend(double(0), er.allFaces()) + er.operatorExtend(std::get<2>(boundFlux), er.boundaryFaces(), er.allFaces())) + er.operatorExtend(std::get<2>(intFlux), er.interiorFaces(), er.allFaces())));
        return allFluxes;
    };
    auto evalSourceTerm = [&](const auto& q) {
        const auto bx = ((b_east - b_west) / dx);
        const auto by = ((b_north - b_south) / dy);
        const auto secondTerm_x = (((std::get<0>(q) - b_east) + (std::get<0>(q) - b_west)) / double(2));
        const auto secondTerm_y = (((std::get<0>(q) - b_north) + (std::get<0>(q) - b_south)) / double(2));
        const auto dryTerm = er.trinaryIf(((std::get<0>(q) - b_mid) > double(0.05)), er.operatorExtend(double(1), er.allCells()), er.operatorExtend(double(0), er.allCells()));
        return makeArray(er.operatorExtend(double(0), er.allCells()), (((-gravity * bx) * secondTerm_x) * dryTerm), (((-gravity * by) * secondTerm_y) * dryTerm));
    };
    auto rungeKutta = [&](const auto& q, const auto& dt) {
        const auto flux = get_flux(q);
        const auto source = evalSourceTerm(q);
        const auto q_star0 = (std::get<0>(q) + ((dt / vol) * (-er.divergence(std::get<0>(flux)) + (vol * std::get<0>(source)))));
        const auto q_star1 = (std::get<1>(q) + ((dt / vol) * (-er.divergence(std::get<1>(flux)) + (vol * std::get<1>(source)))));
        const auto q_star2 = (std::get<2>(q) + ((dt / vol) * (-er.divergence(std::get<2>(flux)) + (vol * std::get<2>(source)))));
        const auto flux_star = get_flux(makeArray(q_star0, q_star1, q_star2));
        const auto source_star = evalSourceTerm(makeArray(q_star0, q_star1, q_star2));
        const auto newQ0 = ((double(0.5) * std::get<0>(q)) + (double(0.5) * (q_star0 + ((dt / vol) * (-er.divergence(std::get<0>(flux_star)) + (vol * std::get<0>(source_star)))))));
        const auto newQ1 = ((double(0.5) * std::get<1>(q)) + (double(0.5) * (q_star1 + ((dt / vol) * (-er.divergence(std::get<1>(flux_star)) + (vol * std::get<1>(source_star)))))));
        const auto newQ2 = ((double(0.5) * std::get<2>(q)) + (double(0.5) * (q_star2 + ((dt / vol) * (-er.divergence(std::get<2>(flux_star)) + (vol * std::get<2>(source_star)))))));
        return makeArray(newQ0, newQ1, newQ2);
    };
    auto q0 = makeArray((h_init + b_mid), (h_init * u_init), (h_init * v_init));
    er.output("q1", std::get<0>(q0));
    er.output("q2", std::get<1>(q0));
    er.output("q3", std::get<2>(q0));
    for (const Scalar& dt : timesteps) {
        const auto q = rungeKutta(q0, dt);
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
