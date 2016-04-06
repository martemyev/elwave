#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>

// MFEM config file
#include "/u/artemyev/projects/mfem/serial/include/config/config.hpp"

#ifndef nullptr
  #define nullptr NULL
#endif

const std::string SEISMOGRAMS_DIR = "seismograms/";
const std::string SNAPSHOTS_DIR   = "snapshots/";

const double VERY_SMALL_NUMBER = 1e-32;
const double FLOAT_NUMBERS_EQUALITY_TOLERANCE = 1e-12;
const double FLOAT_NUMBERS_EQUALITY_REDUCED_TOLERANCE = 1e-6;
const double FIND_CELL_TOLERANCE = 1e-12;

#endif // CONFIG_HPP
