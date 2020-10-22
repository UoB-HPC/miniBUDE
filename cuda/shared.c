#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include "shared.h"

Cuda _cuda = {0};
Params params = {0};

double getTimestamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_usec + tv.tv_sec*1e6;
}

void printTimings(double start, double end, double poses_per_wi)
{
    double ms = ((end-start)/params.iterations)*1e-3;

    // Compute FLOP/s
    double runtime   = ms*1e-3;
    double ops_per_wi = 27*poses_per_wi
        + params.natlig*(3 + 18*poses_per_wi + params.natpro*(11 + 30*poses_per_wi))
        + poses_per_wi;
    double total_ops     = ops_per_wi * (params.nposes/poses_per_wi);
    double flops      = total_ops / runtime;
    double gflops     = flops / 1e9;

    double interactions         =
        (double)params.nposes
        * (double)params.natlig
        * (double)params.natpro;
    double interactions_per_sec = interactions / runtime;

    // Print stats
    printf("- Total time:     %7.2lf ms\n", (end-start)*1e-3);
    printf("- Average time:   %7.2lf ms\n", ms);
    printf("- Interactions/s: %7.2lf billion\n", (interactions_per_sec / 1e9));
    printf("- GFLOP/s:        %7.2lf\n", gflops);
}

