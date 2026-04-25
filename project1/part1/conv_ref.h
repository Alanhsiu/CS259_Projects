/*
 * conv_ref.h  –  CPU reference implementation for 2-D convolution.
 *
 * Matches conv.cpp from the course repo exactly:
 *   synapse  layout : [KY][KX][Nn][Ni]
 *   neuron_i layout : [B*NYPAD][NXPAD][Ni]   (calloc → zero-padded)
 *   neuron_n layout : [B*Ny][Nx][Nn]
 *
 * Include this header once, from conv.cu.
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

/* ── Convolution constants ───────────────────────────────────────────────── */
#define KY 3
#define KX 3
#define SY 1
#define SX 1

typedef float VTYPE;

/* ── Data initialisation (deterministic, matches reference conv.cpp) ─────── */

static void fill_synapse(VTYPE *m, long long n)
{
    for (long long i = 0; i < n; i++)
        m[i] = 0.01f * sinf((float)(i * 3 + 7));
}

/*
 * Fill the valid region of the already-zero-initialised padded input.
 * Valid data occupies rows 0..Ny-1, cols 0..Nx-1 inside [B*NYPAD][NXPAD][Ni].
 * Rows Ny..NYPAD-1 and cols Nx..NXPAD-1 stay zero (right/bottom padding).
 */
static void fill_input(VTYPE *neuron_i,
                       int B, int Ny, int Nx, int Ni)
{
    const int NYPAD = Ny + KY - 1;
    const int NXPAD = Nx + KX - 1;

    for (int b = 0; b < B; b++)
    for (int y = 0; y < Ny; y++)
    for (int x = 0; x < Nx; x++)
    for (int i = 0; i < Ni; i++) {
        long long idx = ((long long)(b * NYPAD + y) * NXPAD + x) * Ni + i;
        neuron_i[idx] = 0.01f * sinf((float)(
            (long long)b*Ny*Nx*Ni + (long long)y*Nx*Ni + x*Ni + i));
    }
}

/* ── CPU reference convolution (OpenMP-parallel) ─────────────────────────── */

static void conv_cpu_reference(
    const VTYPE *syn,   /* [KY * KX * Nn * Ni]       */
    const VTYPE *inp,   /* [B * NYPAD * NXPAD * Ni]  */
    VTYPE       *out,   /* [B * Ny * Nx * Nn]         */
    int B, int Ny, int Nx, int Ni, int Nn)
{
    const int NYPAD = Ny + KY - 1;
    const int NXPAD = Nx + KX - 1;

#pragma omp parallel for schedule(static) collapse(3)
    for (int b  = 0; b  < B;  b++)
    for (int y  = 0; y  < Ny; y++)
    for (int x  = 0; x  < Nx; x++) {
        for (int nn = 0; nn < Nn; nn++) {
            VTYPE sum = 0.f;
            for (int ky = 0; ky < KY; ky++)
            for (int kx = 0; kx < KX; kx++)
            for (int ni = 0; ni < Ni; ni++)
                sum += syn[((ky * KX + kx) * Nn + nn) * Ni + ni] // syn[ky][kx][nn][ni]
                     * inp[((long long)(b * NYPAD + ky + y) * NXPAD // inp[b][y+ky][x+kx][ni]
                            + kx + x) * Ni + ni];

            out[((long long)(b * Ny + y) * Nx + x) * Nn + nn] =
                (sum > 0.f ? sum : 0.f);   /* ReLU */
        }
    }
}

/* ── Verification helper ──────────────────────────────────────────────────── */

static float max_abs_error(const VTYPE *gpu, const VTYPE *cpu, long long n)
{
    float mx = 0.f;
    for (long long i = 0; i < n; i++) {
        float e = fabsf(gpu[i] - cpu[i]);
        if (e > mx) mx = e;
    }
    return mx;
}
