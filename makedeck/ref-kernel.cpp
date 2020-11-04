#include "ref-kernel.h"
#include <cfloat>
#include <cmath>

// Energy evaluation parameters
#define CNSTNT   45.0f
#define HBTYPE_F 70
#define HBTYPE_E 69
#define HARDNESS 38.0f
#define NPNPDIST  5.5f
#define NPPDIST   1.0f



void bude::kernel::fasten_main(
		int natlig, int natpro,
		const std::vector<bude::Atom> &protein,
		const std::vector<bude::Atom> &ligand,
		const std::vector<float> &transforms_0,
		const std::vector<float> &transforms_1,
		const std::vector<float> &transforms_2,
		const std::vector<float> &transforms_3,
		const std::vector<float> &transforms_4,
		const std::vector<float> &transforms_5,
		std::vector<float> &results,
		const std::vector<bude::FFParams> &forcefield,
		int pose) {

	float etot = 0;
	// Compute transformation matrix
	const float sx = sinf(transforms_0[pose]);
	const float cx = cosf(transforms_0[pose]);
	const float sy = sinf(transforms_1[pose]);
	const float cy = cosf(transforms_1[pose]);
	const float sz = sinf(transforms_2[pose]);
	const float cz = cosf(transforms_2[pose]);

	float transform[3][4];
	transform[0][0] = cy * cz;
	transform[0][1] = sx * sy * cz - cx * sz;
	transform[0][2] = cx * sy * cz + sx * sz;
	transform[0][3] = transforms_3[pose];
	transform[1][0] = cy * sz;
	transform[1][1] = sx * sy * sz + cx * cz;
	transform[1][2] = cx * sy * sz - sx * cz;
	transform[1][3] = transforms_4[pose];
	transform[2][0] = -sy;
	transform[2][1] = sx * cy;
	transform[2][2] = cx * cy;
	transform[2][3] = transforms_5[pose];

	// Loop over ligand atoms
	int il = 0;
	do {
		// Load ligand atom data
		const bude::Atom l_atom = ligand[il];
		const bude::FFParams l_params = forcefield[l_atom.type];
		const int lhphb_ltz = l_params.hphb < 0.f;
		const int lhphb_gtz = l_params.hphb > 0.f;

		// Transform ligand atom
		float lpos_x = transform[0][3]
		               + l_atom.x * transform[0][0]
		               + l_atom.y * transform[0][1]
		               + l_atom.z * transform[0][2];
		float lpos_y = transform[1][3]
		               + l_atom.x * transform[1][0]
		               + l_atom.y * transform[1][1]
		               + l_atom.z * transform[1][2];
		float lpos_z = transform[2][3]
		               + l_atom.x * transform[2][0]
		               + l_atom.y * transform[2][1]
		               + l_atom.z * transform[2][2];

		// Loop over protein atoms
		int ip = 0;
		do {
			// Load protein atom data
			const bude::Atom p_atom = protein[ip];
			const bude::FFParams p_params = forcefield[p_atom.type];

			const float radij = p_params.radius + l_params.radius;
			const float r_radij = 1.f / radij;

			const float elcdst =
					(p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F)
					? 4.f : 2.f;
			const float elcdst1 =
					(p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F)
					? 0.25f : 0.5f;
			const int type_E =
					((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

			const int phphb_ltz = p_params.hphb < 0.f;
			const int phphb_gtz = p_params.hphb > 0.f;
			const int phphb_nz = p_params.hphb != 0.f;
			const float p_hphb =
					p_params.hphb * (phphb_ltz && lhphb_gtz ? -1.f : 1.f);
			const float l_hphb =
					l_params.hphb * (phphb_gtz && lhphb_ltz ? -1.f : 1.f);
			const float distdslv =
					(phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST)
					           : (lhphb_ltz ? NPPDIST : -FLT_MAX));
			const float r_distdslv = 1.f / distdslv;

			const float chrg_init = l_params.elsc * p_params.elsc;
			const float dslv_init = p_hphb + l_hphb;

			// Calculate distance between atoms
			const float x = lpos_x - p_atom.x;
			const float y = lpos_y - p_atom.y;
			const float z = lpos_z - p_atom.z;
			const float distij = sqrtf(x * x + y * y + z * z);

			// Calculate the sum of the sphere radii
			const float distbb = distij - radij;
			const int zone1 = (distbb < 0.f);

			// Calculate steric energy
			etot += (1.f - (distij * r_radij)) * (zone1 ? 2 * HARDNESS : 0.f);

			// Calculate formal and dipole charge interactions
			float chrg_e = chrg_init
			               * ((zone1 ? 1 : (1.f - distbb * elcdst1))
			                  * (distbb < elcdst ? 1 : 0.f));
			float neg_chrg_e = -fabs(chrg_e);
			chrg_e = type_E ? neg_chrg_e : chrg_e;
			etot += chrg_e * CNSTNT;

			// Calculate the two cases for Nonpolar-Polar repulsive interactions
			float coeff = (1.f - (distbb * r_distdslv));
			float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? 1 : 0.f);
			dslv_e *= (zone1 ? 1 : coeff);
			etot += dslv_e;

		} while (++ip < natpro); // loop over protein atoms
	} while (++il < natlig); // loop over ligand atoms
	results[pose] = etot * 0.5f;
}
