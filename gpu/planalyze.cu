/* planalyze.cu - kernel performance analysis removed.
 * Debug profiling functions (planalyze_short_kernel, planalyze_long_kernel)
 * have been removed as they introduced unnecessary cudaStreamSynchronize
 * and cudaMemcpy calls that degraded performance.
 */
