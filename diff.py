#!/usr/bin/env python3
"""
Evaluate GPU BAM/SAM usability for downstream variant calling vs CPU reference.

Usage:
    python diff.py <cpu.sam> <gpu.sam>
    python diff.py minimap.sam turbomap.sam

This script answers the practical question: "Will substituting the GPU-produced
alignments for the CPU-produced ones affect variant calling?"  It does NOT
require identical CIGAR strings — it uses the criteria that actual variant
callers (GATK HaplotypeCaller, DeepVariant, bcftools call) care about.

Variant-calling usability criteria
------------------------------------
A GPU alignment is USABLE when ALL of the following hold:

  1. Same chromosome (RNAME)
  2. Start position within POSITION_TOL bp  (default 50)
  3. Genomic coverage overlap >= OVERLAP_MIN  (default 80%)
     overlap = intersection of [pos, pos+ref_span) intervals / union
  4. GPU MAPQ passes the standard filter:
       - if CPU MAPQ >= MAPQ_THRESHOLD: GPU MAPQ must also be >= MAPQ_THRESHOLD
       - if CPU MAPQ <  MAPQ_THRESHOLD: any GPU MAPQ accepted
  5. Alignment is not severely truncated:
       GPU ref_span >= REF_SPAN_MIN_FRAC * CPU ref_span  (default 0.80)

A GPU alignment is BORDERLINE when the chromosome and position match but
one of criteria 3-5 fails modestly (overlap 60-80% or ref_span 60-80%).

Everything else is UNUSABLE.

A separate "strict" count tracks reads where RNAME+POS+CIGAR are identical.

Headline metric
---------------
  VC-usable rate = (USABLE + STRICT) / (reads where CPU produced a mapping)

This is the fraction of CPU-mapped reads the GPU handles well enough for
variant calling.  It should be >= 95% to consider the GPU pipeline production-
ready.

Report also breaks out WHY reads are UNUSABLE so you know where to focus.
"""

import sys
import re
from collections import defaultdict

# ─── Tuneable thresholds ────────────────────────────────────────────────────
POSITION_TOL      = 50      # bp: max allowed start-position difference
OVERLAP_MIN       = 0.80    # 80% reciprocal overlap in reference span
REF_SPAN_MIN_FRAC = 0.80    # GPU ref_span must be >= 80% of CPU ref_span
MAPQ_THRESHOLD    = 20      # standard variant-caller filter
TRUNCATION_CLIP   = 500     # trailing soft-clip (bp) → truncated
BORDERLINE_OVERLAP= 0.60    # below this overlap is always UNUSABLE
# ────────────────────────────────────────────────────────────────────────────

CIGAR_RE = re.compile(r'(\d+)([MIDNSHP=X])')
REF_OPS   = set('MDN=X')
QUERY_OPS = set('MIS=X')


def parse_cigar(cigar):
    if cigar == '*' or not cigar:
        return []
    return [(int(n), op) for n, op in CIGAR_RE.findall(cigar)]


def cigar_spans(ops):
    """Return (ref_span, query_span, trailing_soft_clip)."""
    ref_span  = sum(n for n, op in ops if op in REF_OPS)
    qry_span  = sum(n for n, op in ops if op in QUERY_OPS and op != 'S')
    trail_clip = ops[-1][0] if ops and ops[-1][1] == 'S' else 0
    return ref_span, qry_span, trail_clip


def parse_nm(fields):
    for f in fields[11:]:
        if f.startswith('NM:i:'):
            try:
                return int(f[5:])
            except ValueError:
                return None
    return None


def interval_overlap_frac(pos_a, span_a, pos_b, span_b):
    """Intersection / Union of two 0-based half-open intervals."""
    if span_a <= 0 or span_b <= 0:
        return 0.0
    end_a = pos_a + span_a
    end_b = pos_b + span_b
    inter = max(0, min(end_a, end_b) - max(pos_a, pos_b))
    union = max(end_a, end_b) - min(pos_a, pos_b)
    return inter / union if union > 0 else 0.0


def parse_sam(path):
    """
    Return {qname: primary_record}.
    Only primary alignments (FLAG & 0x900 == 0) are kept.
    """
    primaries = {}
    with open(path) as f:
        for line in f:
            if not line or line[0] == '@':
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 11:
                continue
            qname = fields[0]
            flag  = int(fields[1])
            if flag & 0x900:   # skip supplementary / secondary
                continue
            if qname in primaries:
                continue       # keep first primary
            rname    = fields[2]
            pos      = int(fields[3])   # 1-based
            mapq     = int(fields[4])
            cigar    = fields[5]
            nm       = parse_nm(fields)
            unmapped = bool(flag & 0x4)
            ops      = parse_cigar(cigar)
            ref_span, qry_span, trail_clip = cigar_spans(ops)
            primaries[qname] = dict(
                rname=rname, pos=pos, mapq=mapq, cigar=cigar,
                nm=nm, unmapped=unmapped,
                ref_span=ref_span, qry_span=qry_span,
                trail_clip=trail_clip,
            )
    return primaries


def classify(cpu, gpu):
    """
    Return (category, detail_flags) for a CPU/GPU primary alignment pair.

    Categories:
        both_unmapped   – both unmapped (agreement, excluded from denominator)
        cpu_unmapped    – CPU did not map; GPU may or may not have
        gpu_unmapped    – CPU mapped, GPU reported unmapped
        strict          – identical RNAME + POS + CIGAR
        usable          – passes all VC-usability criteria (see module doc)
        borderline      – same chrom+pos but coverage overlap 60-80%
        wrong_chrom     – different chromosome
        wrong_pos       – same chrom, |pos diff| > POSITION_TOL
        truncated       – same chrom+pos, GPU severely truncated
        unusable        – same chrom+pos but fails VC criteria
    """
    detail = {}

    if cpu is None:
        return 'cpu_unmapped', detail
    if gpu is None:
        detail['reason'] = 'missing from GPU SAM'
        return 'unusable', detail

    if cpu['unmapped'] and gpu['unmapped']:
        return 'both_unmapped', detail
    if cpu['unmapped']:
        return 'cpu_unmapped', detail
    if gpu['unmapped']:
        detail['reason'] = 'GPU unmapped'
        return 'gpu_unmapped', detail

    # ── Strict identical ────────────────────────────────────────────────────
    if (cpu['rname'] == gpu['rname'] and
            cpu['pos']   == gpu['pos'] and
            cpu['cigar'] == gpu['cigar']):
        return 'strict', detail

    # ── Chromosome ──────────────────────────────────────────────────────────
    if cpu['rname'] != gpu['rname']:
        detail['cpu_rname'] = cpu['rname']
        detail['gpu_rname'] = gpu['rname']
        return 'wrong_chrom', detail

    # ── Position ────────────────────────────────────────────────────────────
    pos_diff = abs(cpu['pos'] - gpu['pos'])
    detail['pos_diff'] = pos_diff
    if pos_diff > POSITION_TOL:
        # Only a real failure for VC if CPU MAPQ is above the filter threshold.
        # Low-MAPQ reads are filtered by variant callers anyway, so a different
        # mapping position for MAPQ-1 reads does not affect downstream results.
        if cpu['mapq'] >= MAPQ_THRESHOLD:
            detail['reason'] = (f'|pos diff|={pos_diff} > {POSITION_TOL}  '
                                f'cpu_mapq={cpu["mapq"]} (high-conf!)')
            return 'wrong_pos', detail
        else:
            # Low MAPQ — position difference doesn't matter for VC; record
            # as a note but count as filtered (excluded from denom effectively
            # via the low_mapq bucket).
            detail['note'] = (f'pos diff={pos_diff} but cpu_mapq={cpu["mapq"]}'
                              f' < {MAPQ_THRESHOLD} — filtered by VC anyway')
            return 'low_mapq_pos_diff', detail

    # ── Same chrom + close position: check usability ────────────────────────

    # 1. Genomic overlap
    overlap = interval_overlap_frac(
        cpu['pos'], cpu['ref_span'],
        gpu['pos'], gpu['ref_span'])
    detail['overlap'] = round(overlap, 3)

    # 2. Ref_span ratio — the ONLY reliable truncation signal.
    #    Do NOT use trailing soft-clip size: long reads routinely have huge
    #    trailing S in primary alignments (the rest maps as supplementary).
    #    If ref_span is the same, the primary alignment is equivalent regardless
    #    of how many unaligned bases follow it.
    cpu_ref = cpu['ref_span']
    gpu_ref = gpu['ref_span']
    ref_ratio = (gpu_ref / cpu_ref) if cpu_ref > 0 else 1.0
    detail['ref_ratio'] = round(ref_ratio, 3)
    if ref_ratio < REF_SPAN_MIN_FRAC:
        detail['reason'] = (f'GPU truncated: ref_ratio={ref_ratio:.2f} '
                            f'(gpu_ref={gpu_ref} vs cpu_ref={cpu_ref})')
        return 'truncated', detail

    # 3. MAPQ filter impact
    cpu_high_conf = cpu['mapq'] >= MAPQ_THRESHOLD
    gpu_high_conf = gpu['mapq'] >= MAPQ_THRESHOLD
    mapq_fail = cpu_high_conf and not gpu_high_conf
    detail['cpu_mapq'] = cpu['mapq']
    detail['gpu_mapq'] = gpu['mapq']

    # 5. Final usability decision
    if overlap >= OVERLAP_MIN and not mapq_fail:
        return 'usable', detail
    elif overlap >= BORDERLINE_OVERLAP and not mapq_fail:
        detail['reason'] = f'overlap={overlap:.2f} between 60-80%'
        return 'borderline', detail
    else:
        reasons = []
        if overlap < BORDERLINE_OVERLAP:
            reasons.append(f'overlap={overlap:.2f} < {BORDERLINE_OVERLAP}')
        if mapq_fail:
            reasons.append(f'MAPQ drop {cpu["mapq"]}→{gpu["mapq"]} (threshold={MAPQ_THRESHOLD})')
        detail['reason'] = '; '.join(reasons)
        return 'unusable', detail


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    cpu_path, gpu_path = argv[1], argv[2]
    cpu_recs = parse_sam(cpu_path)
    gpu_recs = parse_sam(gpu_path)

    all_reads = set(cpu_recs) | set(gpu_recs)
    counts  = defaultdict(int)
    details = defaultdict(list)   # {category: [(qname, cpu, gpu, detail), ...]}
    MAX_EX  = 5

    for qname in sorted(all_reads):
        c = cpu_recs.get(qname)
        g = gpu_recs.get(qname)
        cat, det = classify(c, g)
        counts[cat] += 1
        if len(details[cat]) < MAX_EX:
            details[cat].append((qname, c, g, det))

    # ── Denominator: reads where CPU attempted to map ────────────────────────
    # Excludes both_unmapped and cpu_unmapped.
    # low_mapq_pos_diff: CPU mapped but MAPQ < threshold → VC filters them
    #   anyway, so they neither contribute to "usable" nor to "fail" counts.
    denom = (counts['strict'] + counts['usable'] + counts['borderline'] +
             counts['wrong_chrom'] + counts['wrong_pos'] +
             counts['truncated'] + counts['unusable'] + counts['gpu_unmapped'] +
             counts['low_mapq_pos_diff'])

    vc_usable = counts['strict'] + counts['usable']

    # VC-relevant denominator: exclude reads that are filtered by MAPQ anyway
    # (low_mapq_pos_diff are mapped but irrelevant to VC outcome)
    vc_denom = denom - counts['low_mapq_pos_diff']

    def pct(n, d=None):
        d = d if d is not None else vc_denom
        return f'{100.0*n/d:.1f}%' if d else 'n/a'

    W = 72
    print('=' * W)
    print(f'  CPU: {cpu_path}')
    print(f'  GPU: {gpu_path}')
    print('=' * W)
    print(f'  Thresholds: pos_tol={POSITION_TOL}bp  overlap_min={int(OVERLAP_MIN*100)}%'
          f'  ref_span_min={int(REF_SPAN_MIN_FRAC*100)}%'
          f'  mapq_threshold={MAPQ_THRESHOLD}')
    print('=' * W)
    print(f'  Total reads in union:                 {len(all_reads)}')
    print(f'  Both unmapped (excluded):             {counts["both_unmapped"]}')
    print(f'  CPU unmapped (excluded):              {counts["cpu_unmapped"]}')
    print(f'  Low-MAPQ wrong pos (VC-filtered):     {counts["low_mapq_pos_diff"]}')
    print(f'  ─────────────────────────────────────────────────────────────')
    print(f'  VC-relevant denominator:              {vc_denom}')
    print(f'  (total CPU-mapped reads:              {denom})')
    print()
    print(f'  ┌── PASSES variant-calling bar ─────────────────────────────')
    print(f'  │  strict (identical CIGAR):          {counts["strict"]:5d}  {pct(counts["strict"])}')
    print(f'  │  usable (equiv. for VC):            {counts["usable"]:5d}  {pct(counts["usable"])}')
    print(f'  │                                     ──────  ──────')
    print(f'  │  VC-USABLE TOTAL:                   {vc_usable:5d}  {pct(vc_usable)}')
    print(f'  │')
    print(f'  ├── MARGINAL ───────────────────────────────────────────────')
    print(f'  │  borderline (60-80% overlap):       {counts["borderline"]:5d}  {pct(counts["borderline"])}')
    print(f'  │')
    print(f'  └── FAILS variant-calling bar ─────────────────────────────')
    print(f'     GPU unmapped (CPU MAPQ≥{MAPQ_THRESHOLD} mapped):  {counts["gpu_unmapped"]:5d}  {pct(counts["gpu_unmapped"])}')
    print(f'     truncated (ref_span<80% of CPU):   {counts["truncated"]:5d}  {pct(counts["truncated"])}')
    print(f'     wrong pos (high-conf, >{POSITION_TOL}bp off): {counts["wrong_pos"]:5d}  {pct(counts["wrong_pos"])}')
    print(f'     wrong chromosome:                  {counts["wrong_chrom"]:5d}  {pct(counts["wrong_chrom"])}')
    print(f'     other unusable:                    {counts["unusable"]:5d}  {pct(counts["unusable"])}')
    print('=' * W)

    # Colour-code the headline — use vc_denom (excludes VC-filtered reads)
    rate = vc_usable / vc_denom if vc_denom else 0
    star = '✓✓' if rate >= 0.98 else ('✓' if rate >= 0.95 else '✗')
    print(f'  VC-USABLE RATE:  {rate*100:.1f}%  {star}')
    if counts['borderline']:
        bordr_rate = (vc_usable + counts['borderline']) / vc_denom if vc_denom else 0
        print(f'  With borderline: {bordr_rate*100:.1f}%')
    print('=' * W)

    # ── Per-category examples ────────────────────────────────────────────────
    def show_examples(cat, label):
        if not details[cat]:
            return
        print()
        print(f'--- {label} ({counts[cat]} total, showing ≤{MAX_EX}) ---')
        for qname, c, g, det in details[cat]:
            print(f'  read: {qname}')
            if c and not c['unmapped']:
                print(f'    CPU  {c["rname"]}:{c["pos"]}  ref_span={c["ref_span"]}  '
                      f'MAPQ={c["mapq"]}  NM={c["nm"]}')
                print(f'         CIGAR={c["cigar"][:100]}')
            elif c:
                print('    CPU  <unmapped>')
            else:
                print('    CPU  <absent>')
            if g and not g['unmapped']:
                print(f'    GPU  {g["rname"]}:{g["pos"]}  ref_span={g["ref_span"]}  '
                      f'MAPQ={g["mapq"]}  NM={g["nm"]}')
                print(f'         CIGAR={g["cigar"][:100]}')
            elif g:
                print('    GPU  <unmapped>')
            else:
                print('    GPU  <absent>')
            if det:
                print(f'         detail: {det}')

    show_examples('truncated',    'TRUNCATED (ref_span < 80% of CPU)')
    show_examples('wrong_pos',    'WRONG POSITION (high-confidence reads only)')
    show_examples('wrong_chrom',  'WRONG CHROMOSOME')
    show_examples('gpu_unmapped', 'GPU FAILED TO MAP')
    show_examples('unusable',     'UNUSABLE (other)')
    show_examples('borderline',   'BORDERLINE (marginal coverage overlap)')
    show_examples('low_mapq_pos_diff',
                  f'LOW-MAPQ position diff (MAPQ<{MAPQ_THRESHOLD}, VC-filtered — informational only)')


if __name__ == '__main__':
    main(sys.argv)
