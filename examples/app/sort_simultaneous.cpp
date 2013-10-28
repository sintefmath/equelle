/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>

// A small program intended to create input files for simulators using
// the EquelleRuntimeCPU backend. That backend requires all input of
// entity sets to be sorted, so if our entity sets and corresponding
// data living on those sets are not initially sorted, we must make
// new files in which the entity sets are sorted and the data sets
// reordered by the same permutation, so that the mapping between
// entities and data is preserved.

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " entityfile datafile\n";
        return EXIT_FAILURE;
    }

    // Read entity indices.
    std::string entity_filename(argv[1]);
    std::ifstream es(entity_filename.c_str());
    std::istream_iterator<int> ebeg(es);
    std::istream_iterator<int> eend;
    std::vector<int> e(ebeg, eend);

    // Read data.
    std::string data_filename(argv[2]);
    std::ifstream ds(data_filename.c_str());
    std::istream_iterator<double> dbeg(ds);
    std::istream_iterator<double> dend;
    std::vector<double> d(dbeg, dend);

    // Check sizes.
    if (e.size() != d.size()) {
        std::cerr << "Files must contain the same number of elements.\n";
        return EXIT_FAILURE;
    }

    // Make a vector of paired (entity, data), and sort it.
    std::vector<std::pair<int, double> > ed(e.size());
    for (std::size_t i = 0; i < ed.size(); ++i) {
        ed[i] = std::make_pair(e[i], d[i]);
    }
    std::sort(ed.begin(), ed.end());

    // Write entity indices to file.
    entity_filename += "-sorted";
    std::ofstream eout(entity_filename.c_str());
    for (std::size_t i = 0; i < ed.size(); ++i) {
        eout << ed[i].first << '\n';
    }

    // Write data to file.
    data_filename += "-sorted";
    std::ofstream dout(data_filename.c_str());
    for (std::size_t i = 0; i < ed.size(); ++i) {
        dout << ed[i].second << '\n';
    }
}
