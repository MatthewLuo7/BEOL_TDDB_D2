from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from Global_Params import T_TARGET, F_TARGET, S_MAX
from VERIFIED_Unit_Level_DPM_Based import calc_eta_tBD, calc_beta_tBD

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"

CMAP_NAME_LOOKUP = {name.lower(): name for name in plt.colormaps()}

EXISTENCE_CMAP = ListedColormap(['black', 'red', 'yellow', 'green'])
EXISTENCE_BOUNDS = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
EXISTENCE_NORM = BoundaryNorm(EXISTENCE_BOUNDS, EXISTENCE_CMAP.N)
EXISTENCE_CLASSES = [0, 1, 2, 3]
EXISTENCE_LABELS = [
    '0: non-existence',
    '1: <2 nm spacing',
    '2: 2-10 nm spacing',
    '3: >=10 nm spacing',
]

DEBUG_POINTS_XY = [(12, 12), (18, 18)]


def prepare_matrix_for_imshow(matrix):
    """Returns an imshow-ready matrix and optional colorbar labels for categorical data."""
    array = np.asarray(matrix)

    if np.issubdtype(array.dtype, np.number):
        return np.ma.masked_where((array == 0) | ~np.isfinite(array), array), None

    encoded = np.full(array.shape, np.nan, dtype=float)
    category_values = [value for value in np.unique(array) if str(value).strip() not in {'', '0'}]

    labels = []
    for index, value in enumerate(category_values):
        encoded[array == value] = float(index)
        labels.append(str(value))

    return encoded, labels


def load_data(lot_number, wafer_number=1):
    """Loads the via spacing and existence matrices for one wafer."""
    lot_folder = f"lot_{lot_number:03d}"
    wafer_folder = f"wafer_{wafer_number:02d}"
    csv_dir = DATA_ROOT / lot_folder / "csv" / wafer_folder

    files = {
        'space': 'Space.csv',
        'existence': 'ExistenceClass.csv',
    }

    return {name: np.loadtxt(csv_dir / filename, delimiter=',') for name, filename in files.items()}


def iter_wafer_folders(lot_number):
    """Yields wafer directories for the requested lot in sorted order."""
    lot_folder = DATA_ROOT / f"lot_{lot_number:03d}" / "csv"
    return sorted(path for path in lot_folder.glob("wafer_*") if path.is_dir())


def iter_lot_numbers():
    """Yields available lot numbers under DATA_ROOT in sorted order."""
    lot_numbers = []
    for path in DATA_ROOT.glob("lot_*"):
        if not path.is_dir():
            continue
        try:
            lot_numbers.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(lot_numbers)


def _compute_eta_beta(spacing_matrix):
    """Applies the DPM physics engine cell-by-cell using the via spacing only."""
    eta = np.zeros_like(spacing_matrix, dtype=float)
    beta = np.zeros_like(spacing_matrix, dtype=float)
    rows, cols = spacing_matrix.shape
    for row_index in range(rows):
        for col_index in range(cols):
            spacing = spacing_matrix[row_index, col_index]
            if spacing <= 0:
                continue
            if spacing <= S_MAX:
                eta[row_index, col_index] = calc_eta_tBD(spacing)
                beta[row_index, col_index] = calc_beta_tBD(spacing)
            else:
                eta[row_index, col_index] = np.nan
                beta[row_index, col_index] = np.nan
    return eta, beta


def obtain_eta_beta_via(space_matrix):
    return _compute_eta_beta(space_matrix)


def reliability_prediction(eta_tBD, beta_tBD, time=T_TARGET):
    """Calculates cumulative failure probability for the via model."""
    result = np.zeros_like(eta_tBD, dtype=float)
    valid = (eta_tBD > 0) & (beta_tBD > 0) & np.isfinite(eta_tBD)
    ratio = np.divide(time, eta_tBD, out=np.zeros_like(eta_tBD, dtype=float), where=valid)
    result[valid] = 1 - np.exp(-(ratio[valid] ** beta_tBD[valid]))
    return result


def time_to_failure_prediction(eta_tBD, beta_tBD, target_reliability=1 - F_TARGET):
    """Calculates the via time to failure for a target reliability."""
    result = np.full_like(eta_tBD, np.inf, dtype=float)
    valid = (eta_tBD > 0) & (beta_tBD > 0) & np.isfinite(eta_tBD)
    scale = -np.log(target_reliability)
    exponent = np.divide(1.0, beta_tBD, out=np.zeros_like(beta_tBD, dtype=float), where=valid)
    result[valid] = eta_tBD[valid] * (scale ** exponent[valid])
    return result


def print_debug_points(wafer_name, space_matrix, ttf_via, points_xy):
    """Print debug values at specified grid points for the via model."""
    print(f"Debug points for {wafer_name}:")
    for row_index, col_index in points_xy:
        ttf_value = ttf_via[row_index, col_index]
        in_range = space_matrix[row_index, col_index] <= S_MAX
        print(
            f"  ({row_index},{col_index}): Space={space_matrix[row_index, col_index]:.2f} nm "
            f"({'in range' if in_range else f'> S_MAX={S_MAX}'})"
        )
        print(f"         TTF Via={ttf_value:.2e} s")


def _overlay_no_metal(axis, existence_matrix):
    """Overlays black pixels on all cells where ExistenceClass == 0."""
    no_metal = np.ma.masked_where(existence_matrix != 0, np.ones_like(existence_matrix, dtype=float))
    axis.imshow(no_metal, cmap='binary', vmin=0, vmax=1)


def _draw_subplot(fig, axis, title, matrix, cmap, existence_matrix):
    resolved_cmap = CMAP_NAME_LOOKUP.get(cmap.lower(), cmap) if isinstance(cmap, str) else cmap

    if title == 'Existence class':
        image = axis.imshow(np.asarray(matrix), cmap=EXISTENCE_CMAP, norm=EXISTENCE_NORM)
        category_labels = EXISTENCE_LABELS
    else:
        imshow_matrix, category_labels = prepare_matrix_for_imshow(matrix)
        if category_labels is not None:
            vmax = max(len(category_labels) - 0.5, 0.5)
            image = axis.imshow(imshow_matrix, cmap=resolved_cmap, vmin=-0.5, vmax=vmax)
        else:
            image = axis.imshow(imshow_matrix, cmap=resolved_cmap)
        if existence_matrix is not None and title != 'Existence class':
            _overlay_no_metal(axis, existence_matrix)

    axis.set_title(title)
    axis.set_xlabel('X Grid')
    axis.set_ylabel('Y Grid')
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    if title == 'Existence class':
        colorbar.set_ticks(EXISTENCE_CLASSES)
        colorbar.set_ticklabels(EXISTENCE_LABELS)
    elif category_labels is not None and category_labels:
        colorbar.set_ticks(np.arange(len(category_labels)))
        colorbar.set_ticklabels(category_labels)


def save_2x3_map_figure(output_path, wafer_name, maps, existence_matrix=None):
    """Saves six heatmaps in a 2x3 layout."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for axis, (title, matrix, cmap) in zip(axes.ravel(), maps):
        _draw_subplot(fig, axis, title, matrix, cmap, existence_matrix)
    fig.suptitle(f'Wafer Maps for {wafer_name}', fontsize=16)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    print("Loading data...")
    lot_numbers = iter_lot_numbers()

    if not lot_numbers:
        print(f"No lot folders found under {DATA_ROOT}")

    for lot_number in lot_numbers:
        print(f"Processing lot_{lot_number:03d}...")
        wafer_folders = iter_wafer_folders(lot_number)
        if not wafer_folders:
            print(f"No wafer folders found for lot_{lot_number:03d}; skipping.")
            continue

        lot_output_root = OUTPUT_ROOT / f"lot_{lot_number:03d}"
        lot_output_root.mkdir(parents=True, exist_ok=True)

        for wafer_folder in wafer_folders:
            wafer_number = int(wafer_folder.name.split("_")[1])
            dataset = load_data(lot_number=lot_number, wafer_number=wafer_number)

            eta_matrix_via, beta_matrix_via = obtain_eta_beta_via(dataset['space'])
            reliability_via = reliability_prediction(eta_matrix_via, beta_matrix_via)
            ttf_via = time_to_failure_prediction(eta_matrix_via, beta_matrix_via)

            wafer_output_root = lot_output_root / wafer_folder.name
            wafer_output_root.mkdir(parents=True, exist_ok=True)

            np.savetxt(wafer_output_root / 'Predicted_eta_tBD_via.csv', eta_matrix_via, delimiter=',')
            np.savetxt(wafer_output_root / 'Predicted_beta_tBD_via.csv', beta_matrix_via, delimiter=',')
            np.savetxt(wafer_output_root / 'Predicted_reliability_via.csv', reliability_via, delimiter=',')
            np.savetxt(wafer_output_root / 'Predicted_TTF_via.csv', ttf_via, delimiter=',')

            # print_debug_points(wafer_folder.name, dataset['space'], ttf_via, DEBUG_POINTS_XY)

            save_2x3_map_figure(
                wafer_output_root / f'{wafer_folder.name}.png',
                wafer_folder.name,
                [
                    ('Existence class', dataset['existence'], 'gray'),
                    ('Space', dataset['space'], 'plasma'),
                    ('Eta tBD via', eta_matrix_via, 'spectral'),
                    ('Beta tBD via', beta_matrix_via, 'plasma'),
                    ('Reliability via', reliability_via, 'magma'),
                    ('TTF via', ttf_via, 'spectral'),
                ],
                existence_matrix=dataset['existence'],
            )

        sample_dataset = load_data(lot_number=lot_number, wafer_number=int(wafer_folders[0].name.split("_")[1]))
        sample_eta_matrix, sample_beta_matrix = obtain_eta_beta_via(sample_dataset['space'])
        sample_reliability_via = reliability_prediction(sample_eta_matrix, sample_beta_matrix)
        sample_ttf_via = time_to_failure_prediction(sample_eta_matrix, sample_beta_matrix)

        figures_dir = lot_output_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_path = figures_dir / "predicted_via_sample.png"
        save_2x3_map_figure(
            plot_path,
            'wafer_01 sample',
            [
                ('Existence class', sample_dataset['existence'], 'gray'),
                ('Space', sample_dataset['space'], 'plasma'),
                ('Eta tBD via', sample_eta_matrix, 'viridis'),
                ('Beta tBD via', sample_beta_matrix, 'plasma'),
                ('Reliability via', sample_reliability_via, 'magma'),
                ('TTF via', sample_ttf_via, 'spectral'),
            ],
            existence_matrix=sample_dataset['existence'],
        )
        print(f"Plot saved to {plot_path}")

    plt.show()
