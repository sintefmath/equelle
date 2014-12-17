
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

void ensureRequirements(const equelle::EquelleRuntimeCPU& er);
void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er);

#ifndef EQUELLE_NO_MAIN
int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    equelle::EquelleRuntimeCPU er(param);
    equelleGeneratedCode(er);
    return 0;
}
#endif // EQUELLE_NO_MAIN

void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er) {
    using namespace equelle;
    ensureRequirements(er);

    // ============= Generated code starts here ================

    const CollOfScalar h_init = (er.inputCollectionOfScalar("h_init", er.allCells()) * double(1));
    const CollOfScalar u_init = (er.inputCollectionOfScalar("u_init", er.allCells()) * double(1));
    const CollOfScalar v_init = (er.inputCollectionOfScalar("v_init", er.allCells()) * double(1));
    const CollOfScalar b_north = (er.inputCollectionOfScalar("b_north", er.allCells()) * double(1));
    const CollOfScalar b_south = (er.inputCollectionOfScalar("b_south", er.allCells()) * double(1));
    const CollOfScalar b_east = (er.inputCollectionOfScalar("b_east", er.allCells()) * double(1));
    const CollOfScalar b_west = (er.inputCollectionOfScalar("b_west", er.allCells()) * double(1));
    const CollOfScalar b_mid = ((((b_north + b_south) + b_east) + b_west) / double(4));
    er.output("bottom", b_mid);
    er.output("b_north", b_north);
    er.output("b_south", b_south);
    er.output("b_east", b_east);
    er.output("b_west", b_west);
    const Scalar dx = (er.inputScalarWithDefault("dx", double(10)) * double(1));
    const Scalar dy = (er.inputScalarWithDefault("dy", double(10)) * double(1));
    const SeqOfScalar timesteps = (er.inputSequenceOfScalar("timesteps") * double(1));
    const CollOfFace int_faces = er.interiorFaces();
    const CollOfFace bound = er.boundaryFaces();
    const CollOfScalar vol = er.norm(er.allCells());
    const CollOfScalar area = er.norm(er.allFaces());
    const Scalar gravity = (double(9.81) * double(1));
    const Scalar dry = (double(0.05) * double(1));
    const Scalar dummy = (double(1000) * double(1));
    const std::tuple<Scalar, Scalar, Scalar> zeroFlux = makeArray((double(0) * double(1)), (double(0) * double(1)), (double(0) * double(1)));
    const Scalar zeroEigen = (double(0) * double(1));
    const Scalar zeroWater = (double(0) * double(1));
    const std::tuple<Scalar, Scalar, Scalar> zeroSource = makeArray((double(0) * double(1)), (double(0) * double(1)), (double(0) * double(1)));
    const Scalar ab_dry = (double(0.05) * double(1));
    const Scalar ab_dummy = (double(1000) * double(1));
    auto f_i3_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar f0temp = std::get<1>(q);
        const CollOfScalar f1temp = ((std::get<1>(q) * (std::get<1>(q) / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const CollOfScalar f2temp = ((std::get<1>(q) * std::get<2>(q)) / waterHeight);
        const CollOfScalar f0 = er.trinaryIf((rawWaterHeight > dry), f0temp, er.operatorExtend(std::get<0>(zeroFlux), int_faces));
        const CollOfScalar f1 = er.trinaryIf((rawWaterHeight > dry), f1temp, er.operatorExtend(std::get<1>(zeroFlux), int_faces));
        const CollOfScalar f2 = er.trinaryIf((rawWaterHeight > dry), f2temp, er.operatorExtend(std::get<2>(zeroFlux), int_faces));
        return makeArray(f0, f1, f2);
    };
    auto f_i4_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar f0temp = std::get<1>(q);
        const CollOfScalar f1temp = ((std::get<1>(q) * (std::get<1>(q) / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const CollOfScalar f2temp = ((std::get<1>(q) * std::get<2>(q)) / waterHeight);
        const CollOfScalar f0 = er.trinaryIf((rawWaterHeight > dry), f0temp, er.operatorExtend(std::get<0>(zeroFlux), int_faces));
        const CollOfScalar f1 = er.trinaryIf((rawWaterHeight > dry), f1temp, er.operatorExtend(std::get<1>(zeroFlux), int_faces));
        const CollOfScalar f2 = er.trinaryIf((rawWaterHeight > dry), f2temp, er.operatorExtend(std::get<2>(zeroFlux), int_faces));
        return makeArray(f0, f1, f2);
    };
    auto f_i17_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar f0temp = std::get<1>(q);
        const CollOfScalar f1temp = ((std::get<1>(q) * (std::get<1>(q) / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const CollOfScalar f2temp = ((std::get<1>(q) * std::get<2>(q)) / waterHeight);
        const CollOfScalar f0 = er.trinaryIf((rawWaterHeight > dry), f0temp, er.operatorExtend(std::get<0>(zeroFlux), int_faces));
        const CollOfScalar f1 = er.trinaryIf((rawWaterHeight > dry), f1temp, er.operatorExtend(std::get<1>(zeroFlux), int_faces));
        const CollOfScalar f2 = er.trinaryIf((rawWaterHeight > dry), f2temp, er.operatorExtend(std::get<2>(zeroFlux), int_faces));
        return makeArray(f0, f1, f2);
    };
    auto f_i18_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar f0temp = std::get<1>(q);
        const CollOfScalar f1temp = ((std::get<1>(q) * (std::get<1>(q) / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const CollOfScalar f2temp = ((std::get<1>(q) * std::get<2>(q)) / waterHeight);
        const CollOfScalar f0 = er.trinaryIf((rawWaterHeight > dry), f0temp, er.operatorExtend(std::get<0>(zeroFlux), int_faces));
        const CollOfScalar f1 = er.trinaryIf((rawWaterHeight > dry), f1temp, er.operatorExtend(std::get<1>(zeroFlux), int_faces));
        const CollOfScalar f2 = er.trinaryIf((rawWaterHeight > dry), f2temp, er.operatorExtend(std::get<2>(zeroFlux), int_faces));
        return makeArray(f0, f1, f2);
    };
    auto g_i9_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar g0temp = std::get<2>(q);
        const CollOfScalar g1temp = (std::get<1>(q) * (std::get<2>(q) / waterHeight));
        const CollOfScalar g2temp = ((std::get<2>(q) * (std::get<2>(q) / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const CollOfScalar g0 = er.trinaryIf((rawWaterHeight > dry), g0temp, er.operatorExtend(std::get<0>(zeroFlux), int_faces));
        const CollOfScalar g1 = er.trinaryIf((rawWaterHeight > dry), g1temp, er.operatorExtend(std::get<1>(zeroFlux), int_faces));
        const CollOfScalar g2 = er.trinaryIf((rawWaterHeight > dry), g2temp, er.operatorExtend(std::get<2>(zeroFlux), int_faces));
        return makeArray(g0, g1, g2);
    };
    auto g_i10_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar g0temp = std::get<2>(q);
        const CollOfScalar g1temp = (std::get<1>(q) * (std::get<2>(q) / waterHeight));
        const CollOfScalar g2temp = ((std::get<2>(q) * (std::get<2>(q) / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const CollOfScalar g0 = er.trinaryIf((rawWaterHeight > dry), g0temp, er.operatorExtend(std::get<0>(zeroFlux), int_faces));
        const CollOfScalar g1 = er.trinaryIf((rawWaterHeight > dry), g1temp, er.operatorExtend(std::get<1>(zeroFlux), int_faces));
        const CollOfScalar g2 = er.trinaryIf((rawWaterHeight > dry), g2temp, er.operatorExtend(std::get<2>(zeroFlux), int_faces));
        return makeArray(g0, g1, g2);
    };
    auto g_i23_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar g0temp = std::get<2>(q);
        const CollOfScalar g1temp = (std::get<1>(q) * (std::get<2>(q) / waterHeight));
        const CollOfScalar g2temp = ((std::get<2>(q) * (std::get<2>(q) / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const CollOfScalar g0 = er.trinaryIf((rawWaterHeight > dry), g0temp, er.operatorExtend(std::get<0>(zeroFlux), int_faces));
        const CollOfScalar g1 = er.trinaryIf((rawWaterHeight > dry), g1temp, er.operatorExtend(std::get<1>(zeroFlux), int_faces));
        const CollOfScalar g2 = er.trinaryIf((rawWaterHeight > dry), g2temp, er.operatorExtend(std::get<2>(zeroFlux), int_faces));
        return makeArray(g0, g1, g2);
    };
    auto g_i24_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar g0temp = std::get<2>(q);
        const CollOfScalar g1temp = (std::get<1>(q) * (std::get<2>(q) / waterHeight));
        const CollOfScalar g2temp = ((std::get<2>(q) * (std::get<2>(q) / waterHeight)) + (((double(0.5) * gravity) * waterHeight) * waterHeight));
        const CollOfScalar g0 = er.trinaryIf((rawWaterHeight > dry), g0temp, er.operatorExtend(std::get<0>(zeroFlux), int_faces));
        const CollOfScalar g1 = er.trinaryIf((rawWaterHeight > dry), g1temp, er.operatorExtend(std::get<1>(zeroFlux), int_faces));
        const CollOfScalar g2 = er.trinaryIf((rawWaterHeight > dry), g2temp, er.operatorExtend(std::get<2>(zeroFlux), int_faces));
        return makeArray(g0, g1, g2);
    };
    auto eigenvalueF_i0_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar eigF0temp = ((std::get<1>(q) / waterHeight) - er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigF1temp = ((std::get<1>(q) / waterHeight) + er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigF0 = er.trinaryIf((rawWaterHeight > dry), eigF0temp, er.operatorExtend(zeroEigen, int_faces));
        const CollOfScalar eigF1 = er.trinaryIf((rawWaterHeight > dry), eigF1temp, er.operatorExtend(zeroEigen, int_faces));
        return makeArray(eigF0, eigF1);
    };
    auto eigenvalueF_i1_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar eigF0temp = ((std::get<1>(q) / waterHeight) - er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigF1temp = ((std::get<1>(q) / waterHeight) + er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigF0 = er.trinaryIf((rawWaterHeight > dry), eigF0temp, er.operatorExtend(zeroEigen, int_faces));
        const CollOfScalar eigF1 = er.trinaryIf((rawWaterHeight > dry), eigF1temp, er.operatorExtend(zeroEigen, int_faces));
        return makeArray(eigF0, eigF1);
    };
    auto eigenvalueF_i14_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar eigF0temp = ((std::get<1>(q) / waterHeight) - er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigF1temp = ((std::get<1>(q) / waterHeight) + er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigF0 = er.trinaryIf((rawWaterHeight > dry), eigF0temp, er.operatorExtend(zeroEigen, int_faces));
        const CollOfScalar eigF1 = er.trinaryIf((rawWaterHeight > dry), eigF1temp, er.operatorExtend(zeroEigen, int_faces));
        return makeArray(eigF0, eigF1);
    };
    auto eigenvalueF_i15_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar eigF0temp = ((std::get<1>(q) / waterHeight) - er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigF1temp = ((std::get<1>(q) / waterHeight) + er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigF0 = er.trinaryIf((rawWaterHeight > dry), eigF0temp, er.operatorExtend(zeroEigen, int_faces));
        const CollOfScalar eigF1 = er.trinaryIf((rawWaterHeight > dry), eigF1temp, er.operatorExtend(zeroEigen, int_faces));
        return makeArray(eigF0, eigF1);
    };
    auto eigenvalueG_i6_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar eigG0temp = ((std::get<2>(q) / waterHeight) - er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigG1temp = ((std::get<2>(q) / waterHeight) + er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigG0 = er.trinaryIf((rawWaterHeight > dry), eigG0temp, er.operatorExtend(zeroEigen, int_faces));
        const CollOfScalar eigG1 = er.trinaryIf((rawWaterHeight > dry), eigG1temp, er.operatorExtend(zeroEigen, int_faces));
        return makeArray(eigG0, eigG1);
    };
    auto eigenvalueG_i7_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar eigG0temp = ((std::get<2>(q) / waterHeight) - er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigG1temp = ((std::get<2>(q) / waterHeight) + er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigG0 = er.trinaryIf((rawWaterHeight > dry), eigG0temp, er.operatorExtend(zeroEigen, int_faces));
        const CollOfScalar eigG1 = er.trinaryIf((rawWaterHeight > dry), eigG1temp, er.operatorExtend(zeroEigen, int_faces));
        return makeArray(eigG0, eigG1);
    };
    auto eigenvalueG_i20_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar eigG0temp = ((std::get<2>(q) / waterHeight) - er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigG1temp = ((std::get<2>(q) / waterHeight) + er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigG0 = er.trinaryIf((rawWaterHeight > dry), eigG0temp, er.operatorExtend(zeroEigen, int_faces));
        const CollOfScalar eigG1 = er.trinaryIf((rawWaterHeight > dry), eigG1temp, er.operatorExtend(zeroEigen, int_faces));
        return makeArray(eigG0, eigG1);
    };
    auto eigenvalueG_i21_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const CollOfScalar& b) -> std::tuple<CollOfScalar, CollOfScalar> {
        const CollOfScalar rawWaterHeight = (std::get<0>(q) - b);
        const CollOfScalar waterHeight = er.trinaryIf((rawWaterHeight > dry), rawWaterHeight, er.operatorExtend(dummy, int_faces));
        const CollOfScalar eigG0temp = ((std::get<2>(q) / waterHeight) - er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigG1temp = ((std::get<2>(q) / waterHeight) + er.sqrt((gravity * waterHeight)));
        const CollOfScalar eigG0 = er.trinaryIf((rawWaterHeight > dry), eigG0temp, er.operatorExtend(zeroEigen, int_faces));
        const CollOfScalar eigG1 = er.trinaryIf((rawWaterHeight > dry), eigG1temp, er.operatorExtend(zeroEigen, int_faces));
        return makeArray(eigG0, eigG1);
    };
    auto a_eval_i2_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::tuple<CollOfScalar, CollOfScalar> eigsFirst = eigenvalueF_i14_(qFirst, bFirst);
        const std::tuple<CollOfScalar, CollOfScalar> eigsSecond = eigenvalueF_i15_(qSecond, bSecond);
        const CollOfScalar smallest = er.trinaryIf((std::get<0>(eigsFirst) < std::get<0>(eigsSecond)), std::get<0>(eigsFirst), std::get<0>(eigsSecond));
        const CollOfScalar aminus = er.trinaryIf((smallest < zeroEigen), smallest, er.operatorExtend(zeroEigen, int_faces));
        const CollOfScalar largest = er.trinaryIf((std::get<1>(eigsFirst) > std::get<1>(eigsSecond)), std::get<1>(eigsFirst), std::get<1>(eigsSecond));
        const CollOfScalar aplus = er.trinaryIf((largest > zeroEigen), largest, er.operatorExtend(zeroEigen, int_faces));
        return makeArray(aminus, aplus);
    };
    auto a_eval_i16_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::tuple<CollOfScalar, CollOfScalar> eigsFirst = eigenvalueF_i14_(qFirst, bFirst);
        const std::tuple<CollOfScalar, CollOfScalar> eigsSecond = eigenvalueF_i15_(qSecond, bSecond);
        const CollOfScalar smallest = er.trinaryIf((std::get<0>(eigsFirst) < std::get<0>(eigsSecond)), std::get<0>(eigsFirst), std::get<0>(eigsSecond));
        const CollOfScalar aminus = er.trinaryIf((smallest < zeroEigen), smallest, er.operatorExtend(zeroEigen, int_faces));
        const CollOfScalar largest = er.trinaryIf((std::get<1>(eigsFirst) > std::get<1>(eigsSecond)), std::get<1>(eigsFirst), std::get<1>(eigsSecond));
        const CollOfScalar aplus = er.trinaryIf((largest > zeroEigen), largest, er.operatorExtend(zeroEigen, int_faces));
        return makeArray(aminus, aplus);
    };
    auto b_eval_i8_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::tuple<CollOfScalar, CollOfScalar> eigsFirst = eigenvalueG_i20_(qFirst, bFirst);
        const std::tuple<CollOfScalar, CollOfScalar> eigsSecond = eigenvalueG_i21_(qSecond, bSecond);
        const CollOfScalar smallest = er.trinaryIf((std::get<0>(eigsFirst) < std::get<0>(eigsSecond)), std::get<0>(eigsFirst), std::get<0>(eigsSecond));
        const CollOfScalar bminus = er.trinaryIf((smallest < zeroEigen), smallest, er.operatorExtend(zeroEigen, int_faces));
        const CollOfScalar largest = er.trinaryIf((std::get<1>(eigsFirst) > std::get<1>(eigsSecond)), std::get<1>(eigsFirst), std::get<1>(eigsSecond));
        const CollOfScalar bplus = er.trinaryIf((largest > zeroEigen), largest, er.operatorExtend(zeroEigen, int_faces));
        return makeArray(bminus, bplus);
    };
    auto b_eval_i22_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::tuple<CollOfScalar, CollOfScalar> eigsFirst = eigenvalueG_i20_(qFirst, bFirst);
        const std::tuple<CollOfScalar, CollOfScalar> eigsSecond = eigenvalueG_i21_(qSecond, bSecond);
        const CollOfScalar smallest = er.trinaryIf((std::get<0>(eigsFirst) < std::get<0>(eigsSecond)), std::get<0>(eigsFirst), std::get<0>(eigsSecond));
        const CollOfScalar bminus = er.trinaryIf((smallest < zeroEigen), smallest, er.operatorExtend(zeroEigen, int_faces));
        const CollOfScalar largest = er.trinaryIf((std::get<1>(eigsFirst) > std::get<1>(eigsSecond)), std::get<1>(eigsFirst), std::get<1>(eigsSecond));
        const CollOfScalar bplus = er.trinaryIf((largest > zeroEigen), largest, er.operatorExtend(zeroEigen, int_faces));
        return makeArray(bminus, bplus);
    };
    auto numF_i5_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::tuple<CollOfScalar, CollOfScalar> a = a_eval_i16_(q);
        const CollOfScalar adiffRaw = (std::get<1>(a) - std::get<0>(a));
        const CollOfScalar adiff = er.trinaryIf(((adiffRaw * adiffRaw) > (ab_dry * ab_dry)), adiffRaw, er.operatorExtend(ab_dummy, int_faces));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> fFirst = f_i17_(qFirst, bFirst);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> fSecond = f_i18_(qSecond, bSecond);
        const CollOfScalar aFactor = ((std::get<1>(a) * std::get<0>(a)) / adiff);
        const CollOfScalar firstPart0 = (((std::get<1>(a) * std::get<0>(fFirst)) - (std::get<0>(a) * std::get<0>(fSecond))) / adiff);
        const CollOfScalar firstPart1 = (((std::get<1>(a) * std::get<1>(fFirst)) - (std::get<0>(a) * std::get<1>(fSecond))) / adiff);
        const CollOfScalar firstPart2 = (((std::get<1>(a) * std::get<2>(fFirst)) - (std::get<0>(a) * std::get<2>(fSecond))) / adiff);
        const CollOfScalar intFluxF0temp = (firstPart0 + (aFactor * (std::get<0>(qSecond) - std::get<0>(qFirst))));
        const CollOfScalar intFluxF1temp = (firstPart1 + (aFactor * (std::get<1>(qSecond) - std::get<1>(qFirst))));
        const CollOfScalar intFluxF2temp = (firstPart2 + (aFactor * (std::get<2>(qSecond) - std::get<2>(qFirst))));
        const CollOfScalar intFluxF0 = er.trinaryIf(((adiffRaw * adiffRaw) > (ab_dry * ab_dry)), intFluxF0temp, er.operatorExtend(std::get<0>(zeroFlux), int_faces));
        const CollOfScalar intFluxF1 = er.trinaryIf(((adiffRaw * adiffRaw) > (ab_dry * ab_dry)), intFluxF1temp, er.operatorExtend(std::get<1>(zeroFlux), int_faces));
        const CollOfScalar intFluxF2 = er.trinaryIf(((adiffRaw * adiffRaw) > (ab_dry * ab_dry)), intFluxF2temp, er.operatorExtend(std::get<2>(zeroFlux), int_faces));
        return makeArray(intFluxF0, intFluxF1, intFluxF2);
    };
    auto numF_i19_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::tuple<CollOfScalar, CollOfScalar> a = a_eval_i16_(q);
        const CollOfScalar adiffRaw = (std::get<1>(a) - std::get<0>(a));
        const CollOfScalar adiff = er.trinaryIf(((adiffRaw * adiffRaw) > (ab_dry * ab_dry)), adiffRaw, er.operatorExtend(ab_dummy, int_faces));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> fFirst = f_i17_(qFirst, bFirst);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> fSecond = f_i18_(qSecond, bSecond);
        const CollOfScalar aFactor = ((std::get<1>(a) * std::get<0>(a)) / adiff);
        const CollOfScalar firstPart0 = (((std::get<1>(a) * std::get<0>(fFirst)) - (std::get<0>(a) * std::get<0>(fSecond))) / adiff);
        const CollOfScalar firstPart1 = (((std::get<1>(a) * std::get<1>(fFirst)) - (std::get<0>(a) * std::get<1>(fSecond))) / adiff);
        const CollOfScalar firstPart2 = (((std::get<1>(a) * std::get<2>(fFirst)) - (std::get<0>(a) * std::get<2>(fSecond))) / adiff);
        const CollOfScalar intFluxF0temp = (firstPart0 + (aFactor * (std::get<0>(qSecond) - std::get<0>(qFirst))));
        const CollOfScalar intFluxF1temp = (firstPart1 + (aFactor * (std::get<1>(qSecond) - std::get<1>(qFirst))));
        const CollOfScalar intFluxF2temp = (firstPart2 + (aFactor * (std::get<2>(qSecond) - std::get<2>(qFirst))));
        const CollOfScalar intFluxF0 = er.trinaryIf(((adiffRaw * adiffRaw) > (ab_dry * ab_dry)), intFluxF0temp, er.operatorExtend(std::get<0>(zeroFlux), int_faces));
        const CollOfScalar intFluxF1 = er.trinaryIf(((adiffRaw * adiffRaw) > (ab_dry * ab_dry)), intFluxF1temp, er.operatorExtend(std::get<1>(zeroFlux), int_faces));
        const CollOfScalar intFluxF2 = er.trinaryIf(((adiffRaw * adiffRaw) > (ab_dry * ab_dry)), intFluxF2temp, er.operatorExtend(std::get<2>(zeroFlux), int_faces));
        return makeArray(intFluxF0, intFluxF1, intFluxF2);
    };
    auto numG_i11_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::tuple<CollOfScalar, CollOfScalar> b = b_eval_i22_(q);
        const CollOfScalar bdiffRaw = (std::get<1>(b) - std::get<0>(b));
        const CollOfScalar bdiff = er.trinaryIf(((bdiffRaw * bdiffRaw) > (ab_dry * ab_dry)), bdiffRaw, er.operatorExtend(ab_dummy, int_faces));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> gFirst = g_i23_(qFirst, bFirst);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> gSecond = g_i24_(qSecond, bSecond);
        const CollOfScalar bFactor = ((std::get<1>(b) * std::get<0>(b)) / bdiff);
        const CollOfScalar firstPart0 = (((std::get<1>(b) * std::get<0>(gFirst)) - (std::get<0>(b) * std::get<0>(gSecond))) / bdiff);
        const CollOfScalar firstPart1 = (((std::get<1>(b) * std::get<1>(gFirst)) - (std::get<0>(b) * std::get<1>(gSecond))) / bdiff);
        const CollOfScalar firstPart2 = (((std::get<1>(b) * std::get<2>(gFirst)) - (std::get<0>(b) * std::get<2>(gSecond))) / bdiff);
        const CollOfScalar intFluxG0temp = (firstPart0 + (bFactor * (std::get<0>(qSecond) - std::get<0>(qFirst))));
        const CollOfScalar intFluxG1temp = (firstPart1 + (bFactor * (std::get<1>(qSecond) - std::get<1>(qFirst))));
        const CollOfScalar intFluxG2temp = (firstPart2 + (bFactor * (std::get<2>(qSecond) - std::get<2>(qFirst))));
        const CollOfScalar intFluxG0 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (ab_dry * ab_dry)), intFluxG0temp, er.operatorExtend(std::get<0>(zeroFlux), int_faces));
        const CollOfScalar intFluxG1 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (ab_dry * ab_dry)), intFluxG1temp, er.operatorExtend(std::get<1>(zeroFlux), int_faces));
        const CollOfScalar intFluxG2 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (ab_dry * ab_dry)), intFluxG2temp, er.operatorExtend(std::get<2>(zeroFlux), int_faces));
        return makeArray(intFluxG0, intFluxG1, intFluxG2);
    };
    auto numG_i25_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qFirst = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.firstCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.firstCell(int_faces)));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> qSecond = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<1>(q), er.allCells(), er.secondCell(int_faces)), er.operatorOn(std::get<2>(q), er.allCells(), er.secondCell(int_faces)));
        const CollOfScalar bFirst = er.operatorOn(b_mid, er.allCells(), er.firstCell(int_faces));
        const CollOfScalar bSecond = er.operatorOn(b_mid, er.allCells(), er.secondCell(int_faces));
        const std::tuple<CollOfScalar, CollOfScalar> b = b_eval_i22_(q);
        const CollOfScalar bdiffRaw = (std::get<1>(b) - std::get<0>(b));
        const CollOfScalar bdiff = er.trinaryIf(((bdiffRaw * bdiffRaw) > (ab_dry * ab_dry)), bdiffRaw, er.operatorExtend(ab_dummy, int_faces));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> gFirst = g_i23_(qFirst, bFirst);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> gSecond = g_i24_(qSecond, bSecond);
        const CollOfScalar bFactor = ((std::get<1>(b) * std::get<0>(b)) / bdiff);
        const CollOfScalar firstPart0 = (((std::get<1>(b) * std::get<0>(gFirst)) - (std::get<0>(b) * std::get<0>(gSecond))) / bdiff);
        const CollOfScalar firstPart1 = (((std::get<1>(b) * std::get<1>(gFirst)) - (std::get<0>(b) * std::get<1>(gSecond))) / bdiff);
        const CollOfScalar firstPart2 = (((std::get<1>(b) * std::get<2>(gFirst)) - (std::get<0>(b) * std::get<2>(gSecond))) / bdiff);
        const CollOfScalar intFluxG0temp = (firstPart0 + (bFactor * (std::get<0>(qSecond) - std::get<0>(qFirst))));
        const CollOfScalar intFluxG1temp = (firstPart1 + (bFactor * (std::get<1>(qSecond) - std::get<1>(qFirst))));
        const CollOfScalar intFluxG2temp = (firstPart2 + (bFactor * (std::get<2>(qSecond) - std::get<2>(qFirst))));
        const CollOfScalar intFluxG0 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (ab_dry * ab_dry)), intFluxG0temp, er.operatorExtend(std::get<0>(zeroFlux), int_faces));
        const CollOfScalar intFluxG1 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (ab_dry * ab_dry)), intFluxG1temp, er.operatorExtend(std::get<1>(zeroFlux), int_faces));
        const CollOfScalar intFluxG2 = er.trinaryIf(((bdiffRaw * bdiffRaw) > (ab_dry * ab_dry)), intFluxG2temp, er.operatorExtend(std::get<2>(zeroFlux), int_faces));
        return makeArray(intFluxG0, intFluxG1, intFluxG2);
    };
    auto get_flux_i12_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfVector int_orientation = er.normal(int_faces);
        const std::tuple<CollOfScalar, CollOfScalar> pos_normal = makeArray(er.sqrt((CollOfScalar(int_orientation.col(0)) * CollOfScalar(int_orientation.col(0)))), er.sqrt((CollOfScalar(int_orientation.col(1)) * CollOfScalar(int_orientation.col(1)))));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> int_numF = numF_i19_(q);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> int_numG = numG_i25_(q);
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
        const CollOfScalar b_flux = er.trinaryIf((bound_height > dry), b_fluxtemp, er.operatorExtend(std::get<2>(zeroFlux), bound));
        const CollOfScalar boundFlux0 = er.operatorExtend(std::get<0>(zeroFlux), bound);
        const CollOfScalar boundFlux1 = ((er.sqrt((CollOfScalar(bound_orientation.col(0)) * CollOfScalar(bound_orientation.col(0)))) * b_flux) * bound_signX);
        const CollOfScalar boundFlux2 = ((er.sqrt((CollOfScalar(bound_orientation.col(1)) * CollOfScalar(bound_orientation.col(1)))) * b_flux) * bound_signY);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> boundFlux = makeArray(boundFlux0, boundFlux1, boundFlux2);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> allFluxes = makeArray((((er.operatorExtend(std::get<0>(zeroFlux), er.allFaces()) + er.operatorExtend(std::get<0>(boundFlux), er.boundaryFaces(), er.allFaces())) + er.operatorExtend(std::get<0>(intFlux), er.interiorFaces(), er.allFaces())) * area), (((er.operatorExtend(std::get<1>(zeroFlux), er.allFaces()) + er.operatorExtend(std::get<1>(boundFlux), er.boundaryFaces(), er.allFaces())) + er.operatorExtend(std::get<1>(intFlux), er.interiorFaces(), er.allFaces())) * area), (((er.operatorExtend(std::get<2>(zeroFlux), er.allFaces()) + er.operatorExtend(std::get<2>(boundFlux), er.boundaryFaces(), er.allFaces())) + er.operatorExtend(std::get<2>(intFlux), er.interiorFaces(), er.allFaces())) * area));
        return allFluxes;
    };
    auto get_flux_i26_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfVector int_orientation = er.normal(int_faces);
        const std::tuple<CollOfScalar, CollOfScalar> pos_normal = makeArray(er.sqrt((CollOfScalar(int_orientation.col(0)) * CollOfScalar(int_orientation.col(0)))), er.sqrt((CollOfScalar(int_orientation.col(1)) * CollOfScalar(int_orientation.col(1)))));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> int_numF = numF_i19_(q);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> int_numG = numG_i25_(q);
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
        const CollOfScalar b_flux = er.trinaryIf((bound_height > dry), b_fluxtemp, er.operatorExtend(std::get<2>(zeroFlux), bound));
        const CollOfScalar boundFlux0 = er.operatorExtend(std::get<0>(zeroFlux), bound);
        const CollOfScalar boundFlux1 = ((er.sqrt((CollOfScalar(bound_orientation.col(0)) * CollOfScalar(bound_orientation.col(0)))) * b_flux) * bound_signX);
        const CollOfScalar boundFlux2 = ((er.sqrt((CollOfScalar(bound_orientation.col(1)) * CollOfScalar(bound_orientation.col(1)))) * b_flux) * bound_signY);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> boundFlux = makeArray(boundFlux0, boundFlux1, boundFlux2);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> allFluxes = makeArray((((er.operatorExtend(std::get<0>(zeroFlux), er.allFaces()) + er.operatorExtend(std::get<0>(boundFlux), er.boundaryFaces(), er.allFaces())) + er.operatorExtend(std::get<0>(intFlux), er.interiorFaces(), er.allFaces())) * area), (((er.operatorExtend(std::get<1>(zeroFlux), er.allFaces()) + er.operatorExtend(std::get<1>(boundFlux), er.boundaryFaces(), er.allFaces())) + er.operatorExtend(std::get<1>(intFlux), er.interiorFaces(), er.allFaces())) * area), (((er.operatorExtend(std::get<2>(zeroFlux), er.allFaces()) + er.operatorExtend(std::get<2>(boundFlux), er.boundaryFaces(), er.allFaces())) + er.operatorExtend(std::get<2>(intFlux), er.interiorFaces(), er.allFaces())) * area));
        return allFluxes;
    };
    auto evalSourceTerm_i13_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar bx = ((b_east - b_west) / dx);
        const CollOfScalar by = ((b_north - b_south) / dy);
        const CollOfScalar secondTerm_x = (((std::get<0>(q) - b_east) + (std::get<0>(q) - b_west)) / double(2));
        const CollOfScalar secondTerm_y = (((std::get<0>(q) - b_north) + (std::get<0>(q) - b_south)) / double(2));
        const CollOfScalar dryTerm = er.trinaryIf(((std::get<0>(q) - b_mid) > dry), er.operatorExtend(double(1), er.allCells()), er.operatorExtend(double(0), er.allCells()));
        return makeArray(er.operatorExtend(std::get<0>(zeroSource), er.allCells()), (((-gravity * bx) * secondTerm_x) * dryTerm), (((-gravity * by) * secondTerm_y) * dryTerm));
    };
    auto evalSourceTerm_i27_ = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const CollOfScalar bx = ((b_east - b_west) / dx);
        const CollOfScalar by = ((b_north - b_south) / dy);
        const CollOfScalar secondTerm_x = (((std::get<0>(q) - b_east) + (std::get<0>(q) - b_west)) / double(2));
        const CollOfScalar secondTerm_y = (((std::get<0>(q) - b_north) + (std::get<0>(q) - b_south)) / double(2));
        const CollOfScalar dryTerm = er.trinaryIf(((std::get<0>(q) - b_mid) > dry), er.operatorExtend(double(1), er.allCells()), er.operatorExtend(double(0), er.allCells()));
        return makeArray(er.operatorExtend(std::get<0>(zeroSource), er.allCells()), (((-gravity * bx) * secondTerm_x) * dryTerm), (((-gravity * by) * secondTerm_y) * dryTerm));
    };
    auto rungeKutta = [&](const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar>& q, const Scalar& dt) -> std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> {
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> flux = get_flux_i12_(q);
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> source = evalSourceTerm_i13_(q);
        const CollOfScalar unitVol = er.operatorExtend((double(1) * double(1)), er.allCells());
        const CollOfScalar temp_vol = (vol + unitVol);
        const CollOfScalar unitSource = er.operatorExtend((double(1) * double(1)), er.allCells());
        const CollOfScalar tmp_source = (std::get<0>(source) + unitSource);
        const CollOfScalar unitDiv = er.operatorExtend((double(1) * double(1)), er.allCells());
        const CollOfScalar temp = (er.divergence(std::get<0>(flux)) + (vol * std::get<0>(source)));
        const CollOfScalar q_star0 = (std::get<0>(q) + ((dt / vol) * (-er.divergence(std::get<0>(flux)) + (vol * std::get<0>(source)))));
        const CollOfScalar q_star1 = (std::get<1>(q) + ((dt / vol) * (-er.divergence(std::get<1>(flux)) + (vol * std::get<1>(source)))));
        const CollOfScalar q_star2 = (std::get<2>(q) + ((dt / vol) * (-er.divergence(std::get<2>(flux)) + (vol * std::get<2>(source)))));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> flux_star = get_flux_i26_(makeArray(q_star0, q_star1, q_star2));
        const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> source_star = evalSourceTerm_i27_(makeArray(q_star0, q_star1, q_star2));
        const CollOfScalar newQ0 = ((double(0.5) * std::get<0>(q)) + (double(0.5) * (q_star0 + ((dt / vol) * (-er.divergence(std::get<0>(flux_star)) + (vol * std::get<0>(source_star)))))));
        const CollOfScalar newQ1 = ((double(0.5) * std::get<1>(q)) + (double(0.5) * (q_star1 + ((dt / vol) * (-er.divergence(std::get<1>(flux_star)) + (vol * std::get<1>(source_star)))))));
        const CollOfScalar newQ2 = ((double(0.5) * std::get<2>(q)) + (double(0.5) * (q_star2 + ((dt / vol) * (-er.divergence(std::get<2>(flux_star)) + (vol * std::get<2>(source_star)))))));
        return makeArray(newQ0, newQ1, newQ2);
    };
    auto q0 = makeArray((h_init + b_mid), (h_init * u_init), (h_init * v_init));
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
