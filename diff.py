#!/usr/bin/env python3
"""
Alignment-fidelity evaluation for the TurboMap precision study
(paper section "Alignment Fidelity and SV Sensitivity", Tier-1).

Compares a GPU SAM against CPU minimap2 (the correctness reference) and reports
exactly the Tier-1 metrics that go into the paper's fidelity table:

    * Primary concordance   - does the GPU primary land where CPU put it?
    * Split-read retention   - does the GPU keep minimap2's z-drop split reads?

These two numbers cleanly separate the two TurboMap builds:

    turbomap-raw   (make RAW=1) - single forward pass, z-drop split disabled.
                                   Discards post-z-drop remainders -> no
                                   supplementary alignments, primary over-
                                   extends through SV breakpoints.
    turbomap-full  (make)       - split re-extension preserved -> output is
                                   essentially identical to CPU minimap2.

Usage:
    python diff.py <cpu.sam> <gpu.sam> [--label NAME] [--pos-tol BP] [--tsv]

    python diff.py cpu.sam full.sam --label turbomap-full
    python diff.py cpu.sam raw.sam  --label turbomap-raw  --tsv

What this script does NOT do
----------------------------
It does NOT compute SV Recall / Precision / F1. Those require downstream SV
calling (Sniffles2 / cuteSV) on a sorted BAM, benchmarked with Truvari against
the GIAB HG002 v4.2.1 truth set. Using minimap2 as "truth" for SV recall would
be circular, so the SV rows of the table are produced by that separate
pipeline, not here.
"""

import sys
import re
import argparse
from collections import defaultdict

# ─── Tuneable thresholds ────────────────────────────────────────────────────
POSITION_TOL  = 50      # bp: max allowed start / end coordinate difference
MAPQ_THRESH   = 20      # standard high-confidence cutoff (for MAPQ agreement)
OVEREXT_FRAC  = 1.30    # GPU primary ref_span >= this x CPU -> ran through a
                        # breakpoint where CPU z-drop-split (raw signature)
# ────────────────────────────────────────────────────────────────────────────

CIGAR_RE  = re.compile(r'(\d+)([MIDNSHP=X])')
REF_OPS   = set('MDN=X')


def parse_cigar(cigar):
    if cigar == '*' or not cigar:
        return []
    return [(int(n), op) for n, op in CIGAR_RE.findall(cigar)]


def ref_span(ops):
    return sum(n for n, op in ops if op in REF_OPS)


def parse_primaries(path):
    """{qname: primary_record}. Primary = FLAG & 0x900 == 0. First one wins."""
    pri = {}
    with open(path) as f:
        for line in f:
            if not line or line[0] == '@':
                continue
            ff = line.rstrip('\n').split('\t')
            if len(ff) < 11:
                continue
            flag = int(ff[1])
            if flag & 0x900:            # skip supplementary / secondary
                continue
            qname = ff[0]
            if qname in pri:
                continue
            ops = parse_cigar(ff[5])
            pri[qname] = dict(
                rname=ff[2], pos=int(ff[3]), mapq=int(ff[4]), cigar=ff[5],
                unmapped=bool(flag & 0x4), ref_span=ref_span(ops),
            )
    return pri


def supplementary_reads(path):
    """Set of read names carrying at least one supplementary (0x800) record."""
    sup = set()
    with open(path) as f:
        for line in f:
            if not line or line[0] == '@':
                continue
            ff = line.rstrip('\n').split('\t')
            if len(ff) < 11:
                continue
            flag = int(ff[1])
            if flag & 0x800:
                sup.add(ff[0])
    return sup


def evaluate(cpu_path, gpu_path, pos_tol):
    cpu = parse_primaries(cpu_path)
    gpu = parse_primaries(gpu_path)

    # Denominator: reads CPU actually mapped (primary present & mapped).
    cpu_mapped = {q: c for q, c in cpu.items() if not c['unmapped']}
    n = len(cpu_mapped)

    start_conc = 0      # chrom + start within tol
    full_conc  = 0      # chrom + start + end within tol
    strict     = 0      # identical rname + pos + cigar
    mapq_agree = 0      # same high-conf status at MAPQ_THRESH
    gpu_missing = 0     # CPU mapped but GPU absent or unmapped
    overext    = 0      # GPU primary ran through a CPU breakpoint

    for q, c in cpu_mapped.items():
        g = gpu.get(q)
        if g is None or g['unmapped']:
            gpu_missing += 1
            if c['mapq'] < MAPQ_THRESH:   # CPU also low-conf -> filter status agrees
                mapq_agree += 1
            continue

        same_chrom = (c['rname'] == g['rname'])
        dstart = abs(c['pos'] - g['pos'])
        dend   = abs((c['pos'] + c['ref_span']) - (g['pos'] + g['ref_span']))

        if same_chrom and dstart <= pos_tol:
            start_conc += 1
            if dend <= pos_tol:
                full_conc += 1
            if c['ref_span'] > 0 and g['ref_span'] >= OVEREXT_FRAC * c['ref_span']:
                overext += 1

        if (same_chrom and c['pos'] == g['pos'] and c['cigar'] == g['cigar']):
            strict += 1

        if (c['mapq'] >= MAPQ_THRESH) == (g['mapq'] >= MAPQ_THRESH):
            mapq_agree += 1

    # ── Split-read retention ────────────────────────────────────────────────
    cpu_sup = supplementary_reads(cpu_path)
    gpu_sup = supplementary_reads(gpu_path)
    kept    = len(cpu_sup & gpu_sup)
    missed  = len(cpu_sup) - kept

    return dict(
        n=n, gpu_missing=gpu_missing,
        start_conc=start_conc, full_conc=full_conc, strict=strict,
        mapq_agree=mapq_agree, overext=overext,
        n_cpu_sup=len(cpu_sup), n_gpu_sup=len(gpu_sup),
        split_kept=kept, split_missed=missed,
    )


def pct(num, den):
    return f'{100.0 * num / den:.2f}%' if den else 'n/a'


def report(r, cpu_path, gpu_path, label, pos_tol, tsv):
    n = r['n']
    s = r['n_cpu_sup']
    W = 70
    print('=' * W)
    print('  ALIGNMENT FIDELITY   (GPU vs CPU minimap2 reference)')
    print(f'  reference (truth) : {cpu_path}')
    print(f'  evaluated         : {gpu_path}' + (f'   [{label}]' if label else ''))
    print(f'  tolerance         : start & end within {pos_tol} bp')
    print('=' * W)
    print(f'  CPU primary-mapped reads (denominator)   : {n}')
    print(f'  GPU absent / unmapped on those reads      : {r["gpu_missing"]}')
    print('  ── Primary alignment fidelity ────────────────────────────────')
    print(f'  Concordance (chrom + start)               : {pct(r["start_conc"], n):>8}   ({r["start_conc"]}/{n})')
    print(f'  Concordance (chrom + start + end)         : {pct(r["full_conc"], n):>8}   ({r["full_conc"]}/{n})')
    print(f'    strict CIGAR identity                   : {pct(r["strict"], n):>8}   ({r["strict"]}/{n})')
    print(f'    MAPQ filter-status agreement (@{MAPQ_THRESH})       : {pct(r["mapq_agree"], n):>8}')
    print('  ── Split-read retention (z-drop split signal) ────────────────')
    print(f'  CPU reads with supplementary alignment    : {s}')
    print(f'  GPU reads with supplementary alignment    : {r["n_gpu_sup"]}')
    print(f'  Split retention (GPU also split)          : {pct(r["split_kept"], s):>8}   ({r["split_kept"]}/{s})')
    print(f'  Missed splits (CPU split, GPU did not)    : {r["split_missed"]}')
    print(f'  Over-extension (GPU >={OVEREXT_FRAC:.1f}x CPU ref_span)   : {r["overext"]} reads')
    print('=' * W)
    print('  Expectation:  turbomap-full  -> split retention ~100%, overext ~0')
    print('                turbomap-raw   -> split retention ~0,   overext high')
    print('=' * W)

    if tsv:
        # Machine-readable row for aggregating across datasets / builds.
        cols = ['label', 'n_reads', 'conc_start', 'conc_start_end', 'strict_cigar',
                'mapq_agree', 'split_retention', 'missed_splits', 'overext']
        vals = [label or gpu_path, n,
                f'{100.0*r["start_conc"]/n:.2f}' if n else '0',
                f'{100.0*r["full_conc"]/n:.2f}' if n else '0',
                f'{100.0*r["strict"]/n:.2f}' if n else '0',
                f'{100.0*r["mapq_agree"]/n:.2f}' if n else '0',
                f'{100.0*r["split_kept"]/s:.2f}' if s else '0',
                r['split_missed'], r['overext']]
        print('TSV\t' + '\t'.join(str(v) for v in [*cols]))
        print('TSV\t' + '\t'.join(str(v) for v in vals))


# ─── Diagnostic: WHY are CPU split reads not split on GPU? ───────────────────
def parse_all_records(path):
    """{qname: [record,...]} for every mapped record (incl. secondary/supp)."""
    recs = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line or line[0] == '@':
                continue
            ff = line.rstrip('\n').split('\t')
            if len(ff) < 11:
                continue
            flag = int(ff[1])
            if flag & 0x4:
                continue
            recs[ff[0]].append(dict(flag=flag, rname=ff[2], pos=int(ff[3]),
                                    mapq=int(ff[4]), cigar=ff[5],
                                    ref_span=ref_span(parse_cigar(ff[5]))))
    return recs


def diagnose_missed_splits(cpu_path, gpu_path, max_ex=6):
    """For every read CPU split (has supplementary) but GPU did NOT, classify
    what the GPU produced instead — pinpoints the failure mode."""
    cpu = parse_all_records(cpu_path)
    gpu = parse_all_records(gpu_path)
    cats = defaultdict(int)
    ex   = defaultdict(list)
    n_cpu_split = 0
    for q, crecs in cpu.items():
        if not any(r['flag'] & 0x800 for r in crecs):
            continue
        n_cpu_split += 1
        grecs = gpu.get(q, [])
        if any(r['flag'] & 0x800 for r in grecs):
            continue  # GPU also split — not missed
        cpri = next((r for r in crecs if not (r['flag'] & 0x900)), None)
        gpri = next((r for r in grecs if not (r['flag'] & 0x900)), None)
        g_sec = any(r['flag'] & 0x100 for r in grecs)
        if gpri is None:
            cat = 'gpu_unmapped_or_absent'
        elif g_sec:
            cat = 'gpu_emits_secondary_not_supp'   # split exists but flagged 0x100
        elif cpri and cpri['ref_span'] > 0 and gpri['ref_span'] >= 1.3 * cpri['ref_span']:
            cat = 'gpu_overextended_single'        # ran THROUGH breakpoint (no z-drop split)
        else:
            cat = 'gpu_single_no_remainder'        # primary ~CPU, remainder just gone
        cats[cat] += 1
        if len(ex[cat]) < max_ex:
            ex[cat].append((q, cpri, gpri, len(crecs), len(grecs)))

    missed = sum(cats.values())
    W = 74
    print('=' * W)
    print('  MISSED-SPLIT DIAGNOSIS  (CPU split, GPU did not — what did GPU do?)')
    print('=' * W)
    print(f'  CPU split reads: {n_cpu_split}   missed by GPU: {missed}')
    print('  ── failure mode breakdown ───────────────────────────────────')
    order = ['gpu_overextended_single', 'gpu_single_no_remainder',
             'gpu_emits_secondary_not_supp', 'gpu_unmapped_or_absent']
    labels = {
        'gpu_overextended_single':     'ran THROUGH breakpoint (no z-drop split)   ',
        'gpu_single_no_remainder':     'primary ~CPU but remainder dropped         ',
        'gpu_emits_secondary_not_supp':'split as SECONDARY(0x100) not SUPP(0x800)  ',
        'gpu_unmapped_or_absent':      'GPU primary unmapped/absent                ',
    }
    for c in order:
        if cats[c]:
            print(f'    {labels[c]}: {cats[c]:5d}  ({100.0*cats[c]/missed:.1f}%)')
    print('=' * W)
    for c in order:
        if not ex[c]:
            continue
        print(f'\n--- {c} (showing <={max_ex}) ---')
        for q, cpri, gpri, ncr, ngr in ex[c]:
            cp = f"{cpri['rname']}:{cpri['pos']} span={cpri['ref_span']}" if cpri else '<none>'
            gp = f"{gpri['rname']}:{gpri['pos']} span={gpri['ref_span']}" if gpri else '<none>'
            print(f'  {q}  CPU[{ncr}rec] pri={cp}   GPU[{ngr}rec] pri={gp}')


def main():
    ap = argparse.ArgumentParser(description='TurboMap Tier-1 alignment fidelity (GPU vs CPU minimap2).')
    ap.add_argument('cpu_sam', help='CPU minimap2 SAM (correctness reference / truth)')
    ap.add_argument('gpu_sam', help='GPU SAM to evaluate (turbomap-raw or turbomap-full)')
    ap.add_argument('--label', default='', help='label for this run (e.g. turbomap-full)')
    ap.add_argument('--pos-tol', type=int, default=POSITION_TOL,
                    help=f'start/end coordinate tolerance in bp (default {POSITION_TOL})')
    ap.add_argument('--tsv', action='store_true', help='also emit a machine-readable TSV row')
    ap.add_argument('--diagnose', action='store_true',
                    help='classify why CPU-split reads are not split on GPU')
    args = ap.parse_args()

    r = evaluate(args.cpu_sam, args.gpu_sam, args.pos_tol)
    report(r, args.cpu_sam, args.gpu_sam, args.label, args.pos_tol, args.tsv)
    if args.diagnose:
        diagnose_missed_splits(args.cpu_sam, args.gpu_sam)


if __name__ == '__main__':
    main()
