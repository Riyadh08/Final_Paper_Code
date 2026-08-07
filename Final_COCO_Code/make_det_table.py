"""Assemble the COCO detection / segmentation results table.

Scans the run tree written by ``run_mixed_precision_det.py`` (one
``run_meta.json`` per run) and emits the comparison table in the layout used
by the ViT-PTQ detection literature:

                     Mask R-CNN              Cascade Mask R-CNN
    Method   Prec.   Swin-T      Swin-S      Swin-T      Swin-S
                     box  mask   box  mask   box  mask   box  mask

Published reference rows are printed above your own, so the comparison is
readable in one place.  Those reference numbers are transcribed literature
values, not something this code computes -- edit ``REFERENCE_ROWS`` below if
you need to correct or extend them.

Usage
-----
    python make_det_table.py                       # markdown to stdout
    python make_det_table.py --format latex        # LaTeX tabular
    python make_det_table.py --format both --out results_table
    python make_det_table.py --include-partial     # also show smoke-test runs
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------
# Table layout
# ----------------------------------------------------------------------

# Column order: (detector, backbone) -> the --model value that produces it.
COLUMNS: List[Tuple[str, str, Optional[str]]] = [
    ('Mask R-CNN',         'Swin-T', 'mask_rcnn_swin_t_3x'),
    ('Mask R-CNN',         'Swin-S', 'mask_rcnn_swin_s_3x'),
    ('Cascade Mask R-CNN', 'Swin-T', 'cascade_mask_rcnn_swin_t_3x'),
    ('Cascade Mask R-CNN', 'Swin-S', 'cascade_mask_rcnn_swin_s_3x'),
]

# Row order for the precision cells, so 32/32 sorts first, then 3/3 group,
# then 4/4 group -- matching how the published tables are laid out.
PRECISION_ORDER = ['32/32', '3/3', 'MP3/MP3', '4/4', 'MP4/MP4', '6/6', 'MP6/MP6']

# Literature values transcribed from the published table, for side-by-side
# reading only. Nothing here is computed by this repository.
# Format: (method label, precision) -> [mrcnn_t_box, mrcnn_t_mask,
#          mrcnn_s_box, mrcnn_s_mask, casc_t_box, casc_t_mask,
#          casc_s_box, casc_s_mask]
REFERENCE_ROWS: Dict[Tuple[str, str], List[Optional[float]]] = {
    ('Full-Precision',   '32/32'):  [46.0, 41.6, 48.5, 43.3, 50.4, 43.7, 51.9, 45.0],
    ('RepQ-ViT',         '3/3'):    [0.5, 0.5, 1.9, 1.3, 0.7, 0.7, 1.3, 1.2],
    ('AdaLog',           '3/3'):    [12.6, 11.4, 21.0, 19.4, 20.8, 15.6, 25.6, 19.7],
    ('RQViT (+CL)',      '3/3'):    [2.8, 2.2, 11.8, 10.1, 4.1, 3.3, 12.2, 12.9],
    ('AQViT (+CL)',      '3/3'):    [21.1, 20.3, 30.7, 24.6, 30.9, 26.2, 32.3, 27.4],
    ('LRP-RQViT',        'MP3/MP3'): [5.4, 3.9, 11.4, 10.8, 7.0, 7.1, 13.7, 13.1],
    ('LRP-AQViT',        'MP3/MP3'): [28.2, 26.1, 33.2, 29.4, 33.2, 31.1, 37.9, 31.5],
    ('RepQ-ViT',         '4/4'):    [36.1, 36.0, 44.2, 40.2, 47.0, 41.4, 49.3, 43.1],
    ('AdaLog',           '4/4'):    [39.1, 37.7, 44.3, 41.2, 48.2, 42.3, 50.6, 44.0],
    ('RQViT (+CL)',      '4/4'):    [38.6, 37.4, 44.3, 40.6, 47.4, 41.5, 49.5, 43.4],
    ('AQViT (+CL)',      '4/4'):    [41.8, 39.6, 45.4, 41.8, 48.8, 42.5, 50.9, 44.2],
    ('LRP-RQViT',        'MP4/MP4'): [42.8, 39.1, 46.8, 41.3, 48.0, 42.1, 50.2, 44.3],
    ('LRP-AQViT',        'MP4/MP4'): [42.9, 39.9, 46.8, 42.2, 49.3, 43.0, 51.1, 44.4],
}

REFERENCE_ORDER = [
    ('Full-Precision', '32/32'),
    ('RepQ-ViT', '3/3'), ('AdaLog', '3/3'),
    ('RQViT (+CL)', '3/3'), ('AQViT (+CL)', '3/3'),
    ('LRP-RQViT', 'MP3/MP3'), ('LRP-AQViT', 'MP3/MP3'),
    ('RepQ-ViT', '4/4'), ('AdaLog', '4/4'),
    ('RQViT (+CL)', '4/4'), ('AQViT (+CL)', '4/4'),
    ('LRP-RQViT', 'MP4/MP4'), ('LRP-AQViT', 'MP4/MP4'),
]


# ----------------------------------------------------------------------
# Loading runs
# ----------------------------------------------------------------------

def load_runs(runs_dir: str, include_partial: bool = False) -> List[dict]:
    """Load every ``run_meta.json`` under ``runs_dir``.

    Runs evaluated with ``--max-eval-images`` are skipped by default: their AP
    is computed over a handful of images and is not a meaningful table entry.
    """
    metas = []
    for path in sorted(glob.glob(os.path.join(runs_dir, '**', 'run_meta.json'),
                                 recursive=True)):
        with open(path) as f:
            meta = json.load(f)
        meta['_path'] = path
        if meta.get('partial_eval') and not include_partial:
            continue
        metas.append(meta)
    return metas


def method_label(meta: dict) -> str:
    """The 'Method' cell for one of our own runs."""
    if meta['mode'] == 'fp32':
        return 'Full-Precision'
    if meta['mode'] == 'uniform':
        return 'AdaLog (repro)'
    label = 'Ours'
    if meta.get('refined') is False:
        label += ' (no refine)'
    return label


def collect(metas: List[dict]) -> Dict[Tuple[str, str], List[Optional[float]]]:
    """Fold runs into ``(method, precision) -> 8 AP cells``.

    Later runs win on collision, so re-running one cell updates the table
    without needing to delete the old output directory.
    """
    model_to_col = {m: i for i, (_, _, m) in enumerate(COLUMNS) if m}
    rows: Dict[Tuple[str, str], List[Optional[float]]] = {}
    for meta in metas:
        col = model_to_col.get(meta['model'])
        if col is None:
            continue
        key = (method_label(meta), meta['precision'])
        row = rows.setdefault(key, [None] * (2 * len(COLUMNS)))
        metrics = meta.get('metrics') or {}
        box = metrics.get('coco/bbox_mAP')
        mask = metrics.get('coco/segm_mAP')
        # mmdet reports mAP as a 0-1 fraction; the tables are in percent.
        row[2 * col] = round(box * 100, 1) if box is not None else None
        row[2 * col + 1] = round(mask * 100, 1) if mask is not None else None
    return rows


def sort_key(key: Tuple[str, str]) -> Tuple[int, str]:
    _, prec = key
    order = PRECISION_ORDER.index(prec) if prec in PRECISION_ORDER else 99
    return (order, key[0])


# ----------------------------------------------------------------------
# Single-run summary, in the published table's format
# ----------------------------------------------------------------------

def _precision_group(prec: str) -> Optional[str]:
    """'4/4' and 'MP4/MP4' both belong to group '4'; '32/32' to no group."""
    if prec == '32/32':
        return None
    head = prec.split('/')[0]
    digits = ''.join(ch for ch in head if ch.isdigit())
    return digits or None


def format_run_summary(model: str, precision: str,
                       metrics: Optional[dict] = None,
                       box: Optional[float] = None,
                       mask: Optional[float] = None,
                       method: str = 'Ours (this run)',
                       actual_avg_bit: Optional[float] = None,
                       partial_eval: bool = False) -> str:
    """Render one run's AP as a paper-format row, above the matching refs.

    Prints the published Full-Precision row and every published row at the
    same bit-width for this exact detector/backbone, then this run underneath,
    so the comparison the table is meant to support is readable the moment a
    run finishes -- no cross-referencing the paper by hand.

    ``metrics`` is mmdet's metric dict (mAP as a 0-1 fraction); ``box`` /
    ``mask`` accept already-percent values instead.
    """
    if metrics is not None:
        b = metrics.get('coco/bbox_mAP')
        m = metrics.get('coco/segm_mAP')
        box = b * 100 if b is not None else None
        mask = m * 100 if m is not None else None

    col = next((i for i, (_, _, mo) in enumerate(COLUMNS) if mo == model), None)
    if col is not None:
        det, backbone, _ = COLUMNS[col]
        title = f'{det} w. {backbone}'
    else:
        det = backbone = None
        title = model

    W = 66
    out = ['', '=' * W, f' COCO RESULT  --  {title}', '=' * W]

    if col is None:
        out.append(f' (no published reference column for --model {model})')
    out.append(f' {"Method":<26}{"Prec. (W/A)":>13}{"AP^box":>11}{"AP^mask":>11}')
    out.append('-' * W)

    if col is not None:
        group = _precision_group(precision)
        for key in REFERENCE_ORDER:
            ref_method, ref_prec = key
            if ref_prec != '32/32' and _precision_group(ref_prec) != group:
                continue
            vals = REFERENCE_ROWS.get(key)
            if not vals:
                continue
            out.append(f' {ref_method + " [ref]":<26}{ref_prec:>13}'
                       f'{_fmt(vals[2 * col]):>11}{_fmt(vals[2 * col + 1]):>11}')
        out.append('-' * W)

    out.append(f' {method:<26}{precision:>13}{_fmt(box):>11}{_fmt(mask):>11}')
    out.append('=' * W)

    if actual_avg_bit is not None:
        out.append(f' achieved average backbone bit-width: {actual_avg_bit:.4f}')
    if partial_eval:
        out.append(' WARNING: partial evaluation (--max-eval-images). This AP is'
                   ' NOT comparable')
        out.append('          to the reference rows, which use all 5000 val'
                   ' images.')
    return '\n'.join(out)


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def _fmt(v: Optional[float]) -> str:
    return '--' if v is None else f'{v:.1f}'


def render_markdown(ours, show_reference=True) -> str:
    head1 = '| Method | Prec. (W/A) |'
    head2 = '| --- | --- |'
    for det, bb, _ in COLUMNS:
        head1 += f' {det} {bb} APbox | {det} {bb} APmask |'
        head2 += ' ---: | ---: |'
    lines = [head1, head2]

    if show_reference:
        for key in REFERENCE_ORDER:
            if key not in REFERENCE_ROWS:
                continue
            method, prec = key
            cells = ' | '.join(_fmt(v) for v in REFERENCE_ROWS[key])
            lines.append(f'| {method} [ref] | {prec} | {cells} |')
        lines.append('| | | ' + ' | '.join([''] * (2 * len(COLUMNS))) + ' |')

    for key in sorted(ours, key=sort_key):
        method, prec = key
        cells = ' | '.join(_fmt(v) for v in ours[key])
        lines.append(f'| **{method}** | {prec} | {cells} |')
    return '\n'.join(lines)


def render_latex(ours, show_reference=True) -> str:
    n = len(COLUMNS)
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{COCO object detection and instance segmentation results. '
        r'$AP^{box}$ and $AP^{mask}$ denote box and mask average precision. '
        r"`Prec. (W/A)' indicates weight/activation bit precision and `MP' "
        r'represents mixed precision.}',
        r'\begin{tabular}{ll' + 'cc' * n + '}',
        r'\toprule',
    ]
    # two-level header
    groups, seen = [], []
    for det, bb, _ in COLUMNS:
        if det not in seen:
            seen.append(det)
        groups.append(det)
    top = r'\multirow{2}{*}{Method} & \multirow{2}{*}{Prec. (W/A)}'
    for det in seen:
        span = 2 * groups.count(det)
        top += r' & \multicolumn{%d}{c}{%s}' % (span, det)
    lines.append(top + r' \\')
    sub = ' & '
    for det, bb, _ in COLUMNS:
        sub += r' & \multicolumn{2}{c}{%s}' % bb
    lines.append(sub + r' \\')
    third = ' & '
    for _ in COLUMNS:
        third += r' & $AP^{box}$ & $AP^{mask}$'
    lines.append(third + r' \\')
    lines.append(r'\midrule')

    if show_reference:
        for key in REFERENCE_ORDER:
            if key not in REFERENCE_ROWS:
                continue
            method, prec = key
            cells = ' & '.join(_fmt(v) for v in REFERENCE_ROWS[key])
            lines.append(f'{method} & {prec} & {cells}' + r' \\')
        lines.append(r'\midrule')

    for key in sorted(ours, key=sort_key):
        method, prec = key
        cells = ' & '.join(
            (r'\textbf{%s}' % _fmt(v)) if v is not None else '--'
            for v in ours[key])
        lines.append(r'\textbf{%s} & %s & %s \\' % (method, prec, cells))

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


def render_coverage(metas, ours) -> str:
    """Report which table cells are still empty and what would fill them."""
    model_to_col = {m: (d, b) for d, b, m in COLUMNS if m}
    have = {}
    for meta in metas:
        if meta['model'] in model_to_col:
            have.setdefault(meta['model'], []).append(meta['precision'])
    lines = ['', 'Coverage:']
    for det, bb, model in COLUMNS:
        if model is None:
            continue
        got = sorted(set(have.get(model, [])), key=lambda p: (
            PRECISION_ORDER.index(p) if p in PRECISION_ORDER else 99))
        status = ', '.join(got) if got else 'NO RUNS YET'
        lines.append(f'  {det:20s} {bb:7s} (--model {model:28s}) : {status}')
    skipped = [m['_path'] for m in metas if m.get('partial_eval')]
    if skipped:
        lines.append(f'  ({len(skipped)} partial-eval run(s) included)')
    return '\n'.join(lines)


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs-dir', default='./checkpoints/mixed_precision_det',
                   help='directory holding the per-run output folders')
    p.add_argument('--format', choices=['markdown', 'latex', 'both'],
                   default='markdown')
    p.add_argument('--out', default=None,
                   help='write to <out>.md / <out>.tex instead of stdout')
    p.add_argument('--include-partial', action='store_true',
                   help='include runs evaluated with --max-eval-images '
                        '(smoke tests; their AP is not table-worthy)')
    p.add_argument('--no-reference', action='store_true',
                   help='omit the published reference rows')
    args = p.parse_args()

    metas = load_runs(args.runs_dir, args.include_partial)
    if not metas:
        print(f"No completed runs found under {args.runs_dir}.")
        print("Runs evaluated with --max-eval-images are skipped by default; "
              "pass --include-partial to see them.")
        return
    ours = collect(metas)

    outputs = {}
    if args.format in ('markdown', 'both'):
        outputs['md'] = render_markdown(ours, not args.no_reference)
    if args.format in ('latex', 'both'):
        outputs['tex'] = render_latex(ours, not args.no_reference)

    for ext, text in outputs.items():
        if args.out:
            path = f'{args.out}.{ext}'
            with open(path, 'w') as f:
                f.write(text + '\n')
            print(f'Wrote {path}')
        else:
            print(text)
            print()

    print(render_coverage(metas, ours))


if __name__ == '__main__':
    main()
