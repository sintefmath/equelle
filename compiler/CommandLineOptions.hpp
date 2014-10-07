/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>

#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

/**
 * Class which handles parsing of argv using boost::program_options
 */
class CommandLineOptions {
public:
    CommandLineOptions() {
        options.add_options()
            ("help,h", "produce help message")
            ("verbose", "Verbose output")
            ("config,c", boost::program_options::value<std::string>(), "Configuration filename (specify command line parameters in file)")
            ("input,i", boost::program_options::value<std::string>()->required(), "Input Equelle file to compile")
            ("backend", boost::program_options::value<std::string>()->default_value("cpu"), "Backend of compiler to use (ast, ast_equelle, cpu, cuda, mrst)")
            ("dump", boost::program_options::value<std::string>()->default_value("none"), "Dump compiler internals (symboltable, io)");
    }

    void printOptions() {
        std::cout << options << std::endl;
    }

    boost::program_options::variables_map parse(int argc, char** argv) {
        //Create command line parser
        boost::program_options::command_line_parser cli_parser(argc, argv);
        cli_parser.options(options);
        cli_parser.allow_unregistered();

        //Parse, and store in map
        boost::program_options::variables_map cli_vars;
        boost::program_options::parsed_options cli_po = cli_parser.run();
        boost::program_options::store(cli_po, cli_vars);

        printUnrecognized(cli_po);

        if (cli_vars.count("config")) {
            std::string config_file = cli_vars["config"].as<std::string>();
            std::ifstream ifs(config_file.c_str());
            if (ifs.good()) {
                boost::program_options::parsed_options cf_po = parse_config_file(ifs, options, true);
                store(cf_po, cli_vars);

                printUnrecognized(cf_po);
            }
            else {
                throw std::runtime_error(std::string("Could not open config file '") + config_file);
            }
        }
        notify(cli_vars);

        return cli_vars;
    }


    void printVars(boost::program_options::variables_map &cli_vars) {
        const unsigned int field_one_width = 25;
        const unsigned int field_two_width = 10;
        std::cout << "Options on command line:" << std::endl;
        for (boost::program_options::variables_map::iterator it=cli_vars.begin(); it!=cli_vars.end(); ++it) {
            std::stringstream tmp;
            std::cout << std::setw(field_one_width) << std::left << it->first;
            bool success=false;

            if (success==false) {
                try {
                    tmp << std::setw(field_two_width) << std::right << "[string] '" << it->second.as<std::string>();
                    success=true;
                }
                catch(const boost::bad_any_cast &) {
                    success=false;
                }
            }

            if (success==false) {
                try {
                    tmp << std::setw(field_two_width) << std::right << "[float] '" << it->second.as<float>();
                    success=true;
                }
                catch(const boost::bad_any_cast &) {
                    success=false;
                }
            }

            if (success==false) {
                try {
                    tmp << std::setw(field_two_width) << std::right << "[int] '" << it->second.as<int>();
                    success=true;
                }
                catch(const boost::bad_any_cast &) {
                    success=false;
                }
            }

            if (success==false) {
                try {
                    tmp << std::setw(field_two_width) << std::right << "[bool] '" << it->second.as<bool>();
                    success=true;
                }
                catch(const boost::bad_any_cast &) {
                    success=false;
                }
            }

            if (success==false) {
                try {
                    std::vector<int> vec = it->second.as<std::vector<int> >();
                    tmp << std::setw(field_two_width) << std::right << "<int> '[";
                    for (unsigned int i=0; i<vec.size(); ++i) {
                        if (i>0) tmp << ", ";
                        tmp << "'" << vec.at(i) << "'";
                    }
                    tmp << "]";
                    success = true;
                }
                catch (const boost::bad_any_cast &) {
                    success = false;
                }
            }

            if (success==false) {
                try {
                    std::vector<std::string> vec = it->second.as<std::vector<std::string> >();
                    tmp << std::setw(field_two_width) << std::right << "<string> '[";
                    for (unsigned int i=0; i<vec.size(); ++i) {
                        if (i>0) tmp << ", ";
                        tmp << "'" << vec.at(i) << "'";
                    }
                    tmp << "]";
                    success = true;
                }
                catch (const boost::bad_any_cast &) {
                    success = false;
                }
            }

            if (success==false) {
                try {
                    std::vector<float> vec = it->second.as<std::vector<float> >();
                    tmp << std::setw(field_two_width) << std::right << "<float> '[";
                    for (unsigned int i=0; i<vec.size(); ++i) {
                        if (i>0) tmp << ", ";
                        tmp << "'" << vec.at(i) << "'";
                    }
                    tmp << "]";
                    success = true;
                }
                catch (const boost::bad_any_cast &) {
                    success = false;
                }
            }

            if (success==false) {
                tmp << std::setw(field_two_width) << std::right << "{UNKNOWN} " << "!!!";
            }

            std::cout << tmp.str() << "'" << std::endl;
        }
    }

    void printUnrecognized(boost::program_options::parsed_options &cli_po) {
        std::vector<std::string> unrecognized = boost::program_options::collect_unrecognized(cli_po.options, boost::program_options::exclude_positional);

        for (unsigned int i=0; i<unrecognized.size(); ++i) {
            std::cerr << "Warning: Unrecognized option '" << unrecognized.at(i) << "'. Ignoring..." << std::endl;
        }
    }
private:
    boost::program_options::options_description options;
};
