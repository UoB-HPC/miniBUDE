#pragma once

#include "make-bude-input.h"

#include <vector>

#define WGSIZE 256

namespace bude::kernel {
	void fasten_main(int natlig,
	                 int natpro,
	                 const std::vector <bude::Atom> &protein,
	                 const std::vector <bude::Atom> &ligand,
	                 const std::vector<float> &transforms_0,
	                 const std::vector<float> &transforms_1,
	                 const std::vector<float> &transforms_2,
	                 const std::vector<float> &transforms_3,
	                 const std::vector<float> &transforms_4,
	                 const std::vector<float> &transforms_5,
	                 std::vector<float> &results,
	                 const std::vector <bude::FFParams> &forcefield,
	                 int pose);


}

