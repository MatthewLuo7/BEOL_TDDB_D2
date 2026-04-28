#!/usr/bin/env python3
"""Simple CLI wrapper to run the wafer-mapping pipeline for a single lot/wafer or an entire lot.

Usage examples:
  python Code/run_engine.py --lot 1 --wafer 2
  python Code/run_engine.py --lot 1 --all
  python Code/run_engine.py --all-lots
"""
from pathlib import Path
import sys
import argparse

code_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(code_dir))

import Wafer_Mapping_Single_Structure as wms
import numpy as np


def process_wafer(lot_number: int, wafer_number: int):
    wafer_name = f"wafer_{wafer_number:02d}"
    print(f"Processing lot_{lot_number:03d}/{wafer_name}...")
    dataset = wms.load_data(lot_number=lot_number, wafer_number=wafer_number)

    eta_matrix_via, beta_matrix_via = wms.obtain_eta_beta_via(dataset['space'])
    eta_matrix_line, beta_matrix_line = wms.obtain_eta_beta_line(dataset['ms'])

    reliability_via = wms.reliability_prediction(eta_matrix_via, beta_matrix_via)
    reliability_line = wms.reliability_prediction(eta_matrix_line, beta_matrix_line)
    joint_reliability = wms.joint_reliability_prediction(
        eta_matrix_line, beta_matrix_line, eta_matrix_via, beta_matrix_via
    )

    TTF_via = wms.time_to_failure_prediction(eta_matrix_via, beta_matrix_via)
    TTF_line = wms.time_to_failure_prediction(eta_matrix_line, beta_matrix_line)
    joint_TTF = wms.joint_time_to_failure_prediction(
        eta_matrix_line, beta_matrix_line, eta_matrix_via, beta_matrix_via
    )

    lot_output_root = wms.OUTPUT_ROOT / f"lot_{lot_number:03d}"
    wafer_output_root = lot_output_root / wafer_name
    wafer_output_root.mkdir(parents=True, exist_ok=True)

    np.savetxt(wafer_output_root / 'Predicted_eta_tBD_via.csv', eta_matrix_via, delimiter=',')
    np.savetxt(wafer_output_root / 'Predicted_beta_tBD_via.csv', beta_matrix_via, delimiter=',')
    np.savetxt(wafer_output_root / 'Predicted_eta_tBD_line.csv', eta_matrix_line, delimiter=',')
    np.savetxt(wafer_output_root / 'Predicted_beta_tBD_line.csv', beta_matrix_line, delimiter=',')
    np.savetxt(wafer_output_root / 'Predicted_joint_reliability.csv', joint_reliability, delimiter=',')
    np.savetxt(wafer_output_root / 'Predicted_joint_TTF.csv', joint_TTF, delimiter=',')
    np.savetxt(wafer_output_root / 'Predicted_reliability_via.csv', reliability_via, delimiter=',')
    np.savetxt(wafer_output_root / 'Predicted_reliability_line.csv', reliability_line, delimiter=',')

    wms.save_2x4_map_figure(
        wafer_output_root / f'{wafer_name}.png',
        wafer_name,
        [
            ('Eta tBD via', eta_matrix_via, 'magma'),
            ('Beta tBD via', beta_matrix_via, 'plasma'),
            ('Eta tBD line', eta_matrix_line, 'magma'),
            ('Beta tBD line', beta_matrix_line, 'plasma'),
            ('TTF via', TTF_via, 'spectral'),
            ('TTF line', TTF_line, 'spectral'),
            ('Joint TTF', joint_TTF, 'spectral'),
        ],
        existence_matrix=dataset.get('existence'),
    )

    wms.save_2x3_map_figure(
        wafer_output_root / f'{wafer_name}_reliability.png',
        wafer_name,
        [
            ('Existence class', dataset['existence'], 'gray'),
            ('MS spacing', dataset['ms'], 'viridis'),
            ('Space', dataset['space'], 'plasma'),
            ('Class map', wms.class_map(dataset['ms'], dataset['space']), wms.CLASS_MAP_CMAP),
            ('Spacing dominance', wms.spacing_dominance(dataset['ms'], dataset['space']), wms.DOMINANCE_CMAP),
            ('TTF via', TTF_via, 'spectral'),
        ],
        existence_matrix=dataset.get('existence'),
    )

    print(f"Saved outputs to {wafer_output_root}")


def process_lot(lot_number: int):
    wafer_paths = wms.iter_wafer_folders(lot_number)
    if not wafer_paths:
        print(f"No wafer folders found for lot_{lot_number:03d}")
        return
    for wafer_path in wafer_paths:
        wafer_number = int(wafer_path.name.split('_')[1])
        process_wafer(lot_number, wafer_number)


def main():
    parser = argparse.ArgumentParser(description='Run wafer mapping predictions for a lot/wafer')
    parser.add_argument('--lot', '-l', type=int, help='Lot number (integer)')
    parser.add_argument('--wafer', '-w', type=int, help='Wafer number (integer). If omitted and --all not set, defaults to 1')
    parser.add_argument('--all', action='store_true', help='Process all wafers in the lot')
    parser.add_argument('--all-lots', action='store_true', help='Process all lots found under data/')

    args = parser.parse_args()

    if args.all_lots:
        lots = wms.iter_lot_numbers()
        if not lots:
            print('No lot folders found under data/')
            return
        for lot in lots:
            process_lot(lot)
        return

    if args.lot is None:
        parser.error('Either --lot or --all-lots must be provided')

    if args.all:
        process_lot(args.lot)
    else:
        wafer_num = args.wafer if args.wafer is not None else 1
        process_wafer(args.lot, wafer_num)


if __name__ == '__main__':
    main()
