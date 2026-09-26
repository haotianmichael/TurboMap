/*
 * pllog.h — Compile-time gated info logging for the GPU pipeline.
 *
 * Set PRINT=1 (via `make PRINT=1`) to enable [Info...] stderr lines emitted
 * from plmem.cu and plalign.cu.  Default is OFF for low-overhead production
 * runs (the long-phase per-batch log line, the arena summary, etc. all add
 * up — measurable when running thousands of batches).
 *
 * [FATAL] / [ERROR] / [WARNING] / [DEBUG] messages are NOT gated by PRINT —
 * they always fire.  Only the routine "[Info...]" reporting is suppressed.
 *
 * Usage:
 *   PLOG_INFO(stderr, "[Info] arena: %.2f GB\n", x);   // gated
 *   fprintf (stderr, "[ERROR] ...\n", ...);             // always fires
 */
#ifndef __PLLOG_H__
#define __PLLOG_H__

#ifndef PRINT
#define PRINT 0
#endif

#if PRINT
#include <stdio.h>
#define PLOG_INFO(...) fprintf(__VA_ARGS__)
#else
#define PLOG_INFO(...) ((void)0)
#endif

#endif /* __PLLOG_H__ */
