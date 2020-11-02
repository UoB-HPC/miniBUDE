
#include "make-bude-input.h"
#include <iostream>
#include <functional>
#include <numeric>
#include <iterator>
#include <unordered_set>
#include <random>
#include <cmath>


std::string &bude::utils::ltrim(std::string &str) {
	if (str.empty()) return str;
	str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](char ch) {
		return !std::isspace<char>(ch, std::locale::classic());
	}));
	return str;
}

std::string &bude::utils::rtrim(std::string &str) {
	if (str.empty()) return str;
	str.erase(std::find_if(str.rbegin(), str.rend(), [](char ch) {
		return !std::isspace<char>(ch, std::locale::classic());
	}).base(), str.end());
	return str;
}

std::string bude::utils::trim(const std::string &str) {
	auto s = str;
	return ltrim(rtrim(s));
}

std::vector<std::string> bude::utils::splitWs(const std::string &s) {
	std::istringstream iss(s);
	return std::vector<std::string>(std::istream_iterator<std::string>(iss), {});
}


namespace fs = std::filesystem;

typedef std::pair<size_t, std::string> BudeLine;

// read in file and return vector of trimmed lines without any bude comments
static std::vector<BudeLine> readBudeLines(const fs::path &path, const bool log) {
	if (log) std::cout << "Reading " << path << "...";
	std::ifstream in(path);
	std::vector<BudeLine> xs;
	std::string line;
	size_t lineNum = 0;
	while (std::getline(in, line)) {
		lineNum++;
		auto trimmed = bude::utils::trim(line);
		if (trimmed.empty()) continue;
		if (trimmed[0] == '#' || trimmed[0] == '%') continue;
		xs.emplace_back(lineNum, trimmed);
	}
	if (log) std::cout << " " << lineNum << " lines (" << (lineNum - xs.size()) << " line(s) of comments/header)" << std::endl;
	return xs;
}

// splits a line by whitespace and then formats any exceptions caused by the parsing function passed in
static void parseBudeColumn(const fs::path &path, const std::vector<BudeLine> &lines, size_t i,
                            const std::function<void(std::vector<std::string>)> &f) {
	auto[lineNum, line] = lines[i];
	try { f(bude::utils::splitWs(line)); }
	catch (const std::exception &e) {
		throw std::runtime_error("Parse error at " + path.string() + ":" + std::to_string(lineNum) + ": " + e.what() + "(Source line: `" + line + "`)");
	}
}

// read a bhff forcefield file
bude::BudeForceField bude::readForceField(const fs::path &bhff, const bool log) {
	auto lines = readBudeLines(bhff, log);
	std::map<std::string, std::vector<bude::FFEntry>> data;

	// parse a atom line (i.e `C.3    -      1.4200   -0.4736   38.0000    5.5000    1.0000    1.0000    0.0000`)
	auto parseBalls = [&](std::string residueId, size_t b) {
		parseBudeColumn(bhff, lines, b, [&](auto ballCols) {
			if (ballCols.size() != 9)
				throw std::runtime_error("ball row requires 9 columns, got `" + std::to_string(ballCols.size()) + "`");
			if (ballCols[1].size() != 1)
				throw std::runtime_error("column 2 should be a single character column, got `" + ballCols[1] + "`");
			auto electType = static_cast<int32_t>(ballCols[1][0]);

			auto atomType = ballCols[0];
			auto radius = std::stof(ballCols[2]);
			auto hphb = std::stof(ballCols[3]);
			auto scaling = std::stof(ballCols[7]);
			auto elsc = std::stof(ballCols[8]);
			data[residueId].push_back(bude::FFEntry{
					data[residueId].size(),
					residueId,
					atomType,
					bude::FFParams{
							electType,
							radius * scaling, // apply radius scaling as per the original script
							hphb,
							elsc
					}
			});
		});
	};

	// parse a residue line (i.e `WLD 34`)
	auto parseResidue = [&](size_t r) {
		parseBudeColumn(bhff, lines, r, [&](auto residueCols) {
			if (residueCols.size() == 2) {
				auto residueId = residueCols[0];
				auto count = std::stoul(residueCols[1]);
				// for each residue, we parse atoms below it
				for (size_t b = 0; b < count; ++b) parseBalls(residueId, r + 1 + b);
			}
		});
	};

	for (size_t r = 0; r < lines.size(); r++) parseResidue(r);

	return data;
}


bude::BudeMol2 bude::readMol2(const fs::path &mol2, const std::map<std::string, std::vector<bude::FFEntry>> &forcefield, const bool log) {

	// atoms are enclosed between  @<TRIPOS>ATOM and @<TRIPOS>
	const std::string BeginAtomMarker = "@<TRIPOS>ATOM";
	const std::string EndAtomMarker = "@<TRIPOS>";

	auto lines = readBudeLines(mol2, log);

	// get the slice of lines between the atom markers
	auto beginAtom = std::find_if(lines.begin(), lines.end(), [&](const auto &p) { return p.second == BeginAtomMarker; }) + 1;
	if (beginAtom == lines.end()) throw std::runtime_error("Begin atom marker(`" + BeginAtomMarker + "`) missing");
	auto endAtom = std::find_if(beginAtom, lines.end(), [&](const auto &p) { return p.second.find(EndAtomMarker) == 0; });
	if (endAtom == lines.end()) throw std::runtime_error("End atom marker(`" + EndAtomMarker + "`) missing");

	// switch to atom type mode when forcefield contains only one residue
	auto byAtomType = forcefield.size() == 1;

	std::vector<bude::Atom> atoms;

	// parse the atoms
	auto parseAtom = [&](size_t i) {
		parseBudeColumn(mol2, lines, i, [&](const auto &cols) {

			if (byAtomType && cols.size() != 9)
				throw std::runtime_error("forcefield contains only 1 group but " + mol2.string() + " doesn't have 9 columns");
			if (!byAtomType && cols.size() != 10)
				throw std::runtime_error("forcefield contains > 1 group but " + mol2.string() + " doesn't have 10 columns");

			// skip hydrogen atoms
			if (cols[5] != "H" && cols[5] != "h") {

				auto atomType = byAtomType ? cols[5] : cols[1];
				auto residueId = (byAtomType ? forcefield.begin()->first : cols[7]).substr(0, 3);

				auto x = std::stof(cols[2]);
				auto y = std::stof(cols[3]);
				auto z = std::stof(cols[4]);

				if (forcefield.count(residueId) < 1)
					throw std::runtime_error("Cannot match key " + residueId + "." + atomType + " in forcefield.");

				// cross reference atom index in forcefield
				auto residueGroup = forcefield.at(residueId);

				auto entry = std::find_if(residueGroup.begin(), residueGroup.end(), [&](const auto &a) { return a.atomType == atomType; });
				if (entry == residueGroup.end())
					throw std::runtime_error("Cannot match key " + residueId + "." + atomType + " in forcefield.");
				atoms.push_back(bude::Atom{
						x, y, z,
						static_cast<int32_t>(entry->index)
				});
			}
		});
	};


	std::vector<std::vector<bude::Atom>> conformations;
	// parse the conformations
	auto parseConformation = [&](size_t i) {
		parseBudeColumn(mol2, lines, i, [&](const auto &confCols) {
			if (confCols.size() == 2 && confCols[0] == "@<BUDE>CONF") {
				std::vector<bude::Atom> conformation;
				for (size_t a = 0; a < atoms.size(); ++a)
					parseBudeColumn(mol2, lines, a + 1 + i, [&](const auto &confCols) {
						if (confCols.size() != 3)
							throw std::runtime_error("Conformation requires 3 columns, got " + std::to_string(confCols.size()));
						auto x = std::stof(confCols[0]);
						auto y = std::stof(confCols[1]);
						auto z = std::stof(confCols[2]);
						conformation.push_back(bude::Atom{x, y, z, atoms[a].type});
					});
				conformations.push_back(conformation);
			}
		});
	};

	for (auto it = beginAtom; it != endAtom; ++it) parseAtom(std::distance(lines.begin(), it));
	for (auto it = endAtom; it != lines.end(); ++it) parseConformation(std::distance(lines.begin(), it));

	return std::make_pair(atoms, conformations);
}


bude::Pose<std::vector<float>> bude::generatePoses(
		size_t poseSize,
		size_t poseSeed,
		const bude::Pose<std::vector<float>> &poseRanges, const bool log) {

	auto poseRangeFields = poseRanges.fields();
	auto maxPoseCombinations = std::transform_reduce(poseRangeFields.begin(), poseRangeFields.end(), 1ul,
	                                                 std::multiplies<>(), [](const auto &xs) { return xs.size(); });

	if (maxPoseCombinations < poseSize)
		throw std::invalid_argument("poseSize exceeds maximum possible pose combinations of " + std::to_string(maxPoseCombinations));

	typedef std::tuple<size_t, size_t, size_t, size_t, size_t, size_t> PoseParam;
	auto hash = [](const PoseParam &p) -> size_t {
		return std::get<0>(p) * 100000 + std::get<1>(p) * 10000 + std::get<2>(p) * 1000 +
		       std::get<3>(p) * 100 + std::get<4>(p) * 10 + std::get<5>(p);
	};
	auto equal = [](const PoseParam &l, const PoseParam &r) -> bool { return l == r; };

	// store poses in a set so we don't get duplicates
	std::unordered_set<PoseParam, decltype(hash), decltype(equal)> xs(poseSize, hash, equal);

	std::mt19937 gen(poseSeed);

	// create randomly distributed indices for each field
	std::uniform_int_distribution<size_t> tDegDist(0, poseRanges.tilt.size() - 1);
	std::uniform_int_distribution<size_t> rDegDist(0, poseRanges.roll.size() - 1);
	std::uniform_int_distribution<size_t> pDegDist(0, poseRanges.pan.size() - 1);
	std::uniform_int_distribution<size_t> xTransDist(0, poseRanges.xTrans.size() - 1);
	std::uniform_int_distribution<size_t> yTransDist(0, poseRanges.yTrans.size() - 1);
	std::uniform_int_distribution<size_t> zTransDist(0, poseRanges.zTrans.size() - 1);

	int generated = 0;
	while (xs.size() < poseSize) {
		xs.emplace(tDegDist(gen), rDegDist(gen), pDegDist(gen),
		           xTransDist(gen), yTransDist(gen), zTransDist(gen));
		generated++;
	}

	bude::Pose<std::vector<float>> transposedPoses;

	constexpr double DEG_TO_RAD = M_PI / 180.0;
	std::for_each(xs.begin(), xs.end(), [&](const auto &param) {
		auto &[t, r, p, x, y, z] = param;
		transposedPoses.tilt.push_back(poseRanges.tilt[t] * DEG_TO_RAD);
		transposedPoses.roll.push_back(poseRanges.roll[r] * DEG_TO_RAD);
		transposedPoses.pan.push_back(poseRanges.pan[p] * DEG_TO_RAD);
		transposedPoses.xTrans.push_back(poseRanges.xTrans[x]);
		transposedPoses.yTrans.push_back(poseRanges.yTrans[y]);
		transposedPoses.zTrans.push_back(poseRanges.zTrans[z]);
	});

	if (log) {
		std::cout << "Generated " << generated << " poses with "
		          << generated - xs.size() << " duplicates (removed)" << std::endl;
	}

	return transposedPoses;
}