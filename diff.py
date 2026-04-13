#!/usr/bin/env python3
"""
Compare two SAM files (CPU minimap2 reference vs GPU TurboMap) and report
precision with a tolerant metric that treats DP tie-break differences as
equivalent alignments.

Usage:
    python diff.py <cpu.sam> <gpu.sam>
    python diff.py minimap.sam test.sam

Definitions:
    A "tie-break" occurs when multiple DP traceback paths yield the same
    optimal score — the two aligners land on the same locus, consume the
    same reference and query spans, but pick slightly different CIGAR ops
    internally. Such alignments are biologically equivalent: they cover
    the same bases on the genome. This script counts them as matches.

    Strict match (exact CIGAR equality) is also reported separately so
    you can see how many reads are byte-identical.

Categories per primary alignment pair (matched by QNAME):

    - strict:          RNAME + POS + CIGAR all identical
    - tiebreak:        same RNAME + POS + ref_span + query_span,
                       NM difference within tolerance (default 5%),
                       but CIGAR string differs
    - drifted:         same RNAME + POS but the alignment shape differs
                       substantially (ref_span differs, or NM drift > 5%)
    - truncated:       same RNAME + POS, GPU ends early (trailing
                       soft-clip > 1000 bp or ref_span shortfall > 20%)
    - wrong_pos:       same RNAME, POS differs by > 100 bp
    - wrong_chrom:     RNAME differs
    - gpu_unmapped:    mapped in CPU, unmapped in GPU
    - cpu_unmapped:    unmapped in CPU, mapped in GPU
    - both_unmapped:   unmapped in both

Two headline numbers:

    Strict accuracy  = strict / (strict + tiebreak + drifted + truncated
                                 + wrong_pos + wrong_chrom + gpu_unmapped)
    Effective accuracy = (strict + tiebreak) / (same denominator)

Effective accuracy is the real-world precision: it credits GPU for
producing biologically equivalent alignments even when the CIGAR string
does not match CPU byte-for-byte.
"""

import sys
import re
from collections import defaultdict

CIGAR_RE = re.compile(r'(\d+)([MIDNSHP=X])')

# Reference-consuming ops: M, D, N, =, X
REF_OPS = set('MDN=X')
# Query-consuming ops: M, I, S, =, X
QUERY_OPS = set('MIS=X')
# Match-ish ops (count toward aligned length)
ALIGNED_OPS = set('MID=X')

NM_TOLERANCE = 0.05         # 5% NM drift still counted as tie-break
TRUNCATION_CLIP = 1000      # trailing soft-clip threshold in bp
TRUNCATION_REF_FRAC = 0.20  # ref_span shortfall fraction for truncation
POS_TOLERANCE = 100         # bp - "same POS" within this many bases


def parse_cigar(cigar):
    """Return list of (length, op) tuples. Empty for '*'."""
    if cigar == '*' or not cigar:
        return []
    return [(int(n), op) for n, op in CIGAR_RE.findall(cigar)]


def cigar_spans(ops):
    """Return (ref_span, query_span, aligned_len, leading_clip, trailing_clip)."""
    ref_span = 0
    query_span = 0
    aligned = 0
    for n, op in ops:
        if op in REF_OPS:
            ref_span += n
        if op in QUERY_OPS and op != 'S':
            query_span += n
        if op in ALIGNED_OPS:
            aligned += n
    leading = ops[0][0] if ops and ops[0][1] == 'S' else 0
    trailing = ops[-1][0] if ops and ops[-1][1] == 'S' else 0
    return ref_span, query_span, aligned, leading, trailing


def parse_nm(fields):
    """Extract NM tag value from SAM optional fields. Returns None if absent."""
    for f in fields[11:]:
        if f.startswith('NM:i:'):
            try:
                return int(f[5:])
            except ValueError:
                return None
    return None


def parse_sam(path):
    """
    Return {qname: primary_record} where primary_record is a dict or None
    (unmapped). Only primary alignments (FLAG & 0x900 == 0) are kept —
    supplementary and secondary are skipped.
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
            flag = int(fields[1])
            # Skip supplementary (0x800) and secondary (0x100)
            if flag & 0x900:
                continue
            rname = fields[2]
            pos = int(fields[3])
            mapq = int(fields[4])
            cigar = fields[5]
            nm = parse_nm(fields)
            is_unmapped = bool(flag & 0x4)
            ops = parse_cigar(cigar)
            ref_span, qry_span, aligned, lead_clip, trail_clip = cigar_spans(ops)
            if qname in primaries:
                # Duplicate primary — shouldn't happen, keep first
                continue
            primaries[qname] = dict(
                flag=flag,
                rname=rname,
                pos=pos,
                mapq=mapq,
                cigar=cigar,
                nm=nm,
                unmapped=is_unmapped,
                ref_span=ref_span,
                qry_span=qry_span,
                aligned=aligned,
                lead_clip=lead_clip,
                trail_clip=trail_clip,
            )
    return primaries


def classify(cpu, gpu):
    """Classify a matched pair of primary alignments."""
    if cpu is None and gpu is None:
        return 'both_unmapped'
    if cpu is None:
        return 'cpu_unmapped'      # only in gpu SAM's read set
    if gpu is None:
        return 'gpu_only_missing'  # read not emitted by gpu at all
    if cpu['unmapped'] and gpu['unmapped']:
        return 'both_unmapped'
    if not cpu['unmapped'] and gpu['unmapped']:
        return 'gpu_unmapped'
    if cpu['unmapped'] and not gpu['unmapped']:
        return 'cpu_unmapped'

    # Both mapped: compare loci
    if cpu['rname'] != gpu['rname']:
        return 'wrong_chrom'
    if abs(cpu['pos'] - gpu['pos']) > POS_TOLERANCE:
        return 'wrong_pos'

    # Same locus — now compare alignment shape
    if cpu['cigar'] == gpu['cigar']:
        return 'strict'

    # Check for truncation first — GPU ending significantly earlier
    cpu_ref = cpu['ref_span']
    gpu_ref = gpu['ref_span']
    if cpu_ref > 0:
        shortfall = (cpu_ref - gpu_ref) / cpu_ref
    else:
        shortfall = 0.0
    if gpu['trail_clip'] > TRUNCATION_CLIP or shortfall > TRUNCATION_REF_FRAC:
        return 'truncated'

    # Same locus, similar extent — is it a tie-break?
    same_ref_span = cpu['ref_span'] == gpu['ref_span']
    same_qry_span = cpu['qry_span'] == gpu['qry_span']
    nm_ok = True
    if cpu['nm'] is not None and gpu['nm'] is not None:
        if cpu['nm'] == 0 and gpu['nm'] == 0:
            nm_ok = True
        else:
            denom = max(1, cpu['nm'])
            nm_ok = abs(gpu['nm'] - cpu['nm']) / denom <= NM_TOLERANCE

    if same_ref_span and same_qry_span and nm_ok:
        return 'tiebreak'
    return 'drifted'


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    cpu_path, gpu_path = argv[1], argv[2]
    cpu = parse_sam(cpu_path)
    gpu = parse_sam(gpu_path)

    all_reads = set(cpu) | set(gpu)
    counts = defaultdict(int)
    examples = defaultdict(list)

    for qname in all_reads:
        c = cpu.get(qname)
        g = gpu.get(qname)
        cat = classify(c, g)
        counts[cat] += 1
        if len(examples[cat]) < 3:
            examples[cat].append((qname, c, g))

    total = sum(counts.values())
    mapped_denom = (total
                    - counts['both_unmapped']
                    - counts['cpu_unmapped']
                    - counts['gpu_only_missing'])

    strict = counts['strict']
    tiebreak = counts['tiebreak']
    effective = strict + tiebreak

    def pct(n, d):
        return f'{100.0*n/d:.2f}%' if d else 'n/a'

    print('=' * 68)
    print(f'CPU file: {cpu_path}')
    print(f'GPU file: {gpu_path}')
    print('=' * 68)
    print(f'Total reads compared:              {total}')
    print(f'  both unmapped (agree, no op):    {counts["both_unmapped"]}')
    print(f'  only in GPU SAM (no CPU entry):  {counts["cpu_unmapped"]}')
    print(f'  missing from GPU SAM:            {counts["gpu_only_missing"]}')
    print()
    print(f'Denominator (CPU attempted to map): {mapped_denom}')
    print('-' * 68)
    print(f'  strict (identical CIGAR):        {strict:5d}  ({pct(strict, mapped_denom)})')
    print(f'  tiebreak-equivalent:             {tiebreak:5d}  ({pct(tiebreak, mapped_denom)})')
    print(f'  drifted (same locus, shape diff):{counts["drifted"]:5d}  ({pct(counts["drifted"], mapped_denom)})')
    print(f'  truncated (GPU extension early): {counts["truncated"]:5d}  ({pct(counts["truncated"], mapped_denom)})')
    print(f'  wrong POS (>{POS_TOLERANCE} bp off):          {counts["wrong_pos"]:5d}  ({pct(counts["wrong_pos"], mapped_denom)})')
    print(f'  wrong chromosome:                {counts["wrong_chrom"]:5d}  ({pct(counts["wrong_chrom"], mapped_denom)})')
    print(f'  GPU failed to map:               {counts["gpu_unmapped"]:5d}  ({pct(counts["gpu_unmapped"], mapped_denom)})')
    print('=' * 68)
    print(f'  Strict accuracy:     {pct(strict, mapped_denom)}')
    print(f'  Effective accuracy:  {pct(effective, mapped_denom)}'
          f'   <-- real-world precision (tolerates tie-break)')
    print('=' * 68)

    # Show a few examples for the actionable failure buckets
    def show(cat, label):
        if not examples[cat]:
            return
        print()
        print(f'--- {label} examples ({counts[cat]} total) ---')
        for qname, c, g in examples[cat]:
            print(f'  {qname}')
            if c:
                print(f'    CPU: {c["rname"]}:{c["pos"]} ref={c["ref_span"]} '
                      f'NM={c["nm"]} MAPQ={c["mapq"]}')
                print(f'         CIGAR[:120]={c["cigar"][:120]}')
            else:
                print('    CPU: <missing>')
            if g:
                print(f'    GPU: {g["rname"]}:{g["pos"]} ref={g["ref_span"]} '
                      f'NM={g["nm"]} MAPQ={g["mapq"]}')
                print(f'         CIGAR[:120]={g["cigar"][:120]}')
            else:
                print('    GPU: <missing>')

    show('truncated', 'Truncated (GPU extension stopped early)')
    show('drifted', 'Drifted (same locus, substantial shape difference)')
    show('wrong_pos', 'Wrong POS')
    show('wrong_chrom', 'Wrong chromosome')
    show('gpu_unmapped', 'GPU failed to map')


if __name__ == '__main__':
    main(sys.argv)
