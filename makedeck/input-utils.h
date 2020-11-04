#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <map>

namespace bude {

	namespace fs = std::filesystem;

	namespace utils {
		std::string &ltrim(std::string &str);
		std::string &rtrim(std::string &str);
		std::string trim(const std::string &str);
		std::vector<std::string> splitWs(const std::string &s);


		template<typename T>
		void writeNStruct(const fs::path &path, const std::vector<T> &xs) {
			std::ofstream out(path, std::ios_base::binary | std::ios_base::out | std::ios_base::app);
			out.write(reinterpret_cast<const char *>(xs.data()), xs.size() * sizeof(T));
		}

	}

	struct __attribute__((__packed__)) Atom {
		float x, y, z;
		int32_t type;
	};

	struct __attribute__((__packed__)) FFParams {
		int32_t hbtype;
		float radius;
		float hphb;
		float elsc;
	};

	struct FFEntry {
		size_t index;
		std::string residueId;
		std::string atomType;
		FFParams params;
	};

	template<typename T>
	struct Pose {
		T tilt, roll, pan;
		T xTrans, yTrans, zTrans;

		std::array<T, 6> fields() const {
			return {tilt, roll, pan, xTrans, yTrans, zTrans};
		}

	};

	typedef std::map<
			std::string,
			std::vector<FFEntry>
	> BudeForceField;

	typedef std::pair<
			std::vector<Atom>,
			std::vector<std::vector<Atom>>
	> BudeMol2;

	BudeForceField readForceField(const fs::path &bhff, bool log = true);

	BudeMol2 readMol2(const fs::path &mol2, const std::map<std::string, std::vector<FFEntry>> &forcefield, bool log = true);

	Pose<std::vector<float>> generatePoses(
			size_t xs,
			size_t poseSeed,
			const bude::Pose<std::vector<float>> &poseRanges,
			bool log = true
	);


}