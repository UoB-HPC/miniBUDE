#include <iostream>

#include <vector>
#include <functional>
#include <filesystem>
#include <iterator>
#include <cmath>

#include "input-utils.h"
#include "ref-kernel.h"

const std::vector<float> DEFAULT_DEGS{
		-170.f, -160.f, -150.f, -140.f, -130.f, -120.f, -110.f, -100.f, -90.f, -80.f, -70.f, -60.f, -50.f, -40.f, -30.f, -20.f, -10.f,
		0.f,
		10.f, 20.f, 30.f, 40.f, 50.f, 60.f, 70.f, 80.f, 90.f, 100.f, 110.f, 120.f, 130.f, 140.f, 150.f, 160.f, 170.f, 180.f
};

const std::vector<float> DEFAULT_TRANS{
		-7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f
};

const size_t DEFAULT_POSE_SEED = 42;
const size_t DEFAULT_POSE_SIZE = 65536;

namespace fs = std::filesystem;

struct DeckConfig {
	fs::path forceFieldBhff;
	fs::path proteinMol2;
	fs::path ligandMol2;

	fs::path deckDir;
	bool overwriteDeck = false;

	size_t poseSize = DEFAULT_POSE_SIZE;
	size_t poseSeed = DEFAULT_POSE_SEED;
	bude::Pose<std::vector<float>> poseRanges = {
			DEFAULT_DEGS, DEFAULT_DEGS, DEFAULT_DEGS,
			DEFAULT_TRANS, DEFAULT_TRANS, DEFAULT_TRANS
	};
};

[[noreturn]] void fail(const std::string &reason) {
	std::cerr << reason << std::endl;
	std::exit(EXIT_FAILURE);
}


int main(int argc, char *argv[]) {

	auto args = std::vector<std::string>(argv + 1, argv + argc);

	const auto readParam = [&args](size_t &current,
	                               const std::string &arg,
	                               const std::initializer_list<std::string> &matches,
	                               const std::function<void(std::string)> &handle) {
		if (matches.size() == 0) return false;
		if (std::find(matches.begin(), matches.end(), arg) != matches.end()) {
			if (current + 1 < args.size()) {
				current++;
				handle(args[current]);
			} else {
				std::string err = "[";
				for (const auto &m : matches) err += m;
				err += "] specified but no value was given";
				fail(err);
			}
			return true;
		}
		return false;
	};


	const auto bindPath = [](const std::string &param, fs::path &dest, const std::string &name) {
		std::ifstream in(param);
		if (!in.good()) fail("Unable to open " + name + " file: `" + param + "`, is the path valid?");
		dest = param;
	};

	const auto bindInt = [](const std::string &param, size_t &dest, const std::string &name) {
		try {
			auto parsed = std::stol(param);
			if (parsed < 0) fail("positive integer required for " + name + ": `" + std::to_string(parsed) + "`");
			dest = parsed;
		} catch (...) { fail("malformed value, integer required for <" + name + ">: `" + param + "`"); }
	};


	DeckConfig config;

	for (size_t i = 0; i < args.size(); ++i) {
		using namespace std::placeholders;
		const auto arg = args[i];
		if (readParam(i, arg, {"--forcefield", "-f"}, std::bind(bindPath, _1, std::ref(config.forceFieldBhff), "forceFieldBhff"))) continue;
		if (readParam(i, arg, {"--protein", "-p"}, std::bind(bindPath, _1, std::ref(config.proteinMol2), "proteinMol2"))) continue;
		if (readParam(i, arg, {"--ligand", "-l"}, std::bind(bindPath, _1, std::ref(config.ligandMol2), "ligandMol2"))) continue;
		if (readParam(i, arg, {"--pose-seed", "-s"}, std::bind(bindInt, _1, std::ref(config.poseSeed), "poseSeed"))) continue;
		if (readParam(i, arg, {"--pose-length", "-n"}, std::bind(bindInt, _1, std::ref(config.poseSize), "nPoses"))) continue;
		if (readParam(i, arg, {"--out", "-o"}, [&](const std::string &param) { config.deckDir = param; })) { continue; }

		if (arg == "--force") {
			config.overwriteDeck = true;
			continue;
		}
		if (arg == "--help" || arg == "-h") {
			std::cout << "This program generates input decks for the bude benchmark program.\n";
			std::cout << "Usage: ./makedeck [OPTIONS]\n\n"
					<< "Options:\n"
					<< "  -h  --help         Print this message\n"
					<< "  -f  --forcefield   Path to a forcefield bhff file\n"
					<< "  -p  --protein      Path to a protein mol2 file\n"
					<< "  -l  --ligand       Path to a ligand mol2 file\n"
					<< "  -s  --pose-seed    The random seed used to generate pose combinations (default: " << DEFAULT_POSE_SEED << ")\n"
					<< "  -n  --pose-length  The amount of poses to generate (default: " << DEFAULT_POSE_SIZE << ")\n"
					<< "  -o  --out          The output directory (containing {protein,ligand,forcefield,poses}.in,params.txt,energies.out) name of the decks\n"
					<< "      --force        If specified, any file/directory that matches the output dir name will be deleted/overwritten\n"
					<< std::endl;
			std::exit(EXIT_SUCCESS);
		}
		fail("Unrecognized argument '" + arg + "' (try '--help')");
	}

	if (config.poseSize % WGSIZE != 0) {
		fail("poseSize(" + std::to_string(config.poseSize) + ") cannot be divided by WGSIZE(" + std::to_string(WGSIZE) + ")");
	}

	auto nonEmpty = [](const fs::path &p, const std::string &name) {
		if (!p.empty()) return;
		fail("--" + name + " is a required parameter");
	};

	nonEmpty(config.forceFieldBhff, "forcefield");
	nonEmpty(config.proteinMol2, "protein");
	nonEmpty(config.ligandMol2, "ligand");
	nonEmpty(config.deckDir, "out");


	if (fs::is_regular_file(config.deckDir) || fs::is_directory(config.deckDir)) {
		if (config.overwriteDeck) fs::remove_all(config.deckDir);
		else fail("Output path already exists (either a file or directory), pass --force to delete/overwrite them");
	}

	if (!fs::create_directories(config.deckDir)) fail("Unable to create output directory: " + config.deckDir.string());

	auto forcefield = bude::readForceField(config.forceFieldBhff);

	if (forcefield.size() != 1) {
		std::cout << "The forcefield file contains more that one residue which the original script does not support.\n"
					 "For the atoms in mol2 to find the correct atom type in the forcefield, "
					 "the current implementation concatenates all residues (groups of atom) so the kernel has reference to all forcefields"
				<< std::endl;
	}

	std::vector<bude::FFParams> ffParams;
	for (auto &&[k, xs] : forcefield) {
		for (const auto &x : xs) ffParams.push_back(x.params);
	}

	auto protein = bude::readMol2(config.proteinMol2, forcefield);
	auto ligand = bude::readMol2(config.ligandMol2, forcefield);

	bude::utils::writeNStruct(config.deckDir / "forcefield.in", ffParams);
	bude::utils::writeNStruct(config.deckDir / "protein.in", protein.first);
	bude::utils::writeNStruct(config.deckDir / "ligand.in", ligand.first);

	auto posesPath = config.deckDir / "poses.in";
	auto poses = bude::generatePoses(config.poseSize, config.poseSeed, config.poseRanges);
	for (const auto &f : poses.fields()) bude::utils::writeNStruct(posesPath, f);

	std::vector<float> energies(config.poseSize);

	std::cout << "Launching kernel ..." << std::endl;
	auto kernelStart = std::chrono::high_resolution_clock::now();
	size_t completed = 0;
	size_t totalPoses = config.poseSize;
	#pragma omp parallel for default(none) shared(ligand, protein, ffParams, poses, energies, totalPoses, completed, std::cout)
	for (size_t pose = 0; pose < totalPoses; pose++) {
		bude::kernel::fasten_main(
				ligand.first.size(), protein.first.size(),
				protein.first, ligand.first,
				poses.tilt, poses.roll, poses.pan,
				poses.xTrans, poses.yTrans, poses.zTrans,
				energies, ffParams, pose);
		#pragma omp critical
		{
			completed++;
			if (completed % 10 == 0) {
				auto pct = static_cast<int>((static_cast<double>(completed) / totalPoses) * 100.0);
				std::cout << "["
						<< std::string(pct, '|') << std::string(100 - pct, ' ')
						<< "] (" << totalPoses << "/" << completed << ") " << pct << "%\r" << std::flush;
			}
		};
	}
	auto kernelEnd = std::chrono::high_resolution_clock::now();
	std::cout << std::endl;
	std::cout << "Kernel completed" << std::endl;

	std::ofstream reference(config.deckDir / "ref_energies.out");
	reference << std::fixed << std::setprecision(5);
	for (const auto &e : energies) reference << e << "\n";
	reference.close();

	std::ofstream input(config.deckDir / "params.txt");
	for (const auto &arg : args) input << arg << " ";
	input << std::endl;

	std::cout << "Output:\n";


	std::cout << "- Forcefield: " << forcefield.size() << " residue(s)\n";
	for (auto&&[residue, atoms] : forcefield)
		std::cout << "\t" << std::setw(5) << residue << " : " << atoms.size() << " atom(s)\n";

	std::cout << "- Protein: " << protein.first.size() << " atom(s), " << protein.second.size() << " conformations(s)\n";
	for (size_t i = 0; i < protein.second.size(); ++i)
		std::cout << "\tConformation " << std::setw(5) << i << " : " << protein.second[i].size() << " atom(s)\n";

	std::cout << "- Ligand: " << ligand.first.size() << " atom(s), " << ligand.second.size() << " conformations(s)\n";
	for (size_t i = 0; i < ligand.second.size(); ++i)
		std::cout << "\tConformation " << std::setw(5) << i << " : " << ligand.second[i].size() << " atom(s)\n";

	std::cout << "- Poses: " << poses.pan.size() << " (seed=" << config.poseSeed << ")\n";

	std::cout << "- Reference energy: " << energies.size() << " lines"
			<< " (kernel elapsed: " << (std::chrono::duration_cast<std::chrono::milliseconds>(kernelEnd - kernelStart).count())
			<< " ms)";


	std::cout << std::endl;

	std::cout << "Done" << std::endl;
	return EXIT_SUCCESS;
}
