from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from Global_Params import T_TARGET, F_TARGET, S_MAX, S_MIN

import VERIFIED_Unit_Level_DPM_Based as verified_model
import Unit_Level_1_over_E_V2 as one_over_e_model
import Unit_Level_sqrt_E as sqrt_e_model

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"

DEBUG_POINTS_XY = [(12, 12), (18, 18)]
CMAP_NAME_LOOKUP = {name.lower(): name for name in plt.colormaps()}

PHYSICS_MODELS = {
    'verified': verified_model,
    '1e_v2': one_over_e_model,
    'sqrt_e': sqrt_e_model,
}

ACTIVE_MODEL_NAME = 'verified'
calc_eta_tBD = verified_model.calc_eta_tBD
calc_beta_tBD = verified_model.calc_beta_tBD


def set_physics_model(model_name):
    """Selects which unit-level physics model is used by the wafer mapper."""
    normalized = model_name.strip().lower()
    aliases = {
        'verified': 'verified',
        'verfied': 'verified',
        '1e': '1e_v2',
        '1/e': '1e_v2',
        '1e_v2': '1e_v2',
        'sqrt': 'sqrt_e',
        'sqrt(e)': 'sqrt_e',
        'sqrt_e': 'sqrt_e',
    }
    resolved_name = aliases.get(normalized, normalized)
    if resolved_name not in PHYSICS_MODELS:
        available = ', '.join(sorted(PHYSICS_MODELS))
        raise ValueError(f"Unknown physics model '{model_name}'. Available: {available}")

    model = PHYSICS_MODELS[resolved_name]
    global ACTIVE_MODEL_NAME, calc_eta_tBD, calc_beta_tBD
    ACTIVE_MODEL_NAME = resolved_name
    calc_eta_tBD = model.calc_eta_tBD
    calc_beta_tBD = model.calc_beta_tBD
    return ACTIVE_MODEL_NAME


def get_physics_model_names():
    """Returns the selectable physics model names."""
    return tuple(sorted(PHYSICS_MODELS))

# Categorical palettes tuned for clearer, less saturated dominance/classification plots.
DOMINANCE_CMAP = ListedColormap([
    '#4C78A8',  # line
    '#F58518',  # via
])
CLASS_MAP_CMAP = ListedColormap([
    'black',   # non-existence
    'red',     # fail
    'orange',  # via risk
    'blue',    # line risk
    'purple',  # joint risk
    'green',   # fully reliable
])
CLASS_MAP_CLASSES = ['non-existence', 'fail', 'via risk', 'line risk', 'joint risk', 'no risk']
CLASS_MAP_LABELS = [
    'Non-existence',
    'Failure',
    'Via risk',
    'Line risk',
    'Joint risk',
    'Fully reliable',
]
CLASS_MAP_BOUNDS = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
CLASS_MAP_NORM = BoundaryNorm(CLASS_MAP_BOUNDS, CLASS_MAP_CMAP.N)
EXISTENCE_CLASSES = [0, 1, 2, 3]
EXISTENCE_LABELS = [
    '0: non-existence',
    '1: <2 nm spacing',
    '2: 2-10 nm spacing',
    '3: >=10 nm spacing',
]
EXISTENCE_CMAP = ListedColormap(['black', 'red', 'yellow', 'green'])
EXISTENCE_BOUNDS = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
EXISTENCE_NORM = BoundaryNorm(EXISTENCE_BOUNDS, EXISTENCE_CMAP.N)

def prepare_matrix_for_imshow(matrix):
    """Returns an imshow-ready matrix and optional colorbar labels for categorical data."""
    array = np.asarray(matrix)

    if np.issubdtype(array.dtype, np.number):
        return np.ma.masked_where((array == 0) | ~np.isfinite(array), array), None

    encoded = np.full(array.shape, np.nan, dtype=float)
    category_values = [
        value
        for value in np.unique(array)
        if str(value).strip() not in {'', '0'}
    ]

    labels = []
    for index, value in enumerate(category_values):
        encoded[array == value] = float(index)
        labels.append(str(value))

    return encoded, labels

def load_data(lot_number, wafer_number=1):
    """Loads the spatial matrices from the provided CSV files."""
    lot_folder = f"lot_{lot_number:03d}"
    wafer_folder = f"wafer_{wafer_number:02d}"
    csv_dir = DATA_ROOT / lot_folder / "csv" / wafer_folder

    files = {
        # Space obtained iwth Space=MS-(1/2)*(VCD-CBCD)-OVL
        'space': 'Space.csv',

        # width of the metal line
        'cbcd': 'CBCD.csv',

        # existence class of the metal line (3 for 10+nm spacing, 2 for 2-10nm spacing, 1 for <2nm spacing, 0 for non-existence)
        'existence': 'ExistenceClass.csv',

        # Spacing between the metal line and its nearest neighbor
        'ms': 'MS.csv',

        # offset of via center from the center of the nearest metal line
        'ovl': 'OVL.csv',

        # width of the via
        'vcd': 'VCD.csv',
    }

    return {
        name: np.loadtxt(csv_dir / filename, delimiter=',')
        for name, filename in files.items()
    }

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
    """
    Applies the DPM physics engine cell-by-cell.
      S <= 0:          no metal — eta=0, beta=0
      0 < S <= S_MAX:  valid range — compute normally
      S > S_MAX:       outside valid E-field range at 0.7 V — eta=NaN, beta=NaN (assume passes)
    """
    eta = np.zeros_like(spacing_matrix, dtype=float)
    beta = np.zeros_like(spacing_matrix, dtype=float)
    rows, cols = spacing_matrix.shape
    for i in range(rows):
        for j in range(cols):
            S = spacing_matrix[i, j]
            if S <= 0:
                pass  # no metal; leave as 0
            elif S <= S_MAX:
                eta[i, j] = calc_eta_tBD(S)
                beta[i, j] = calc_beta_tBD(S)
            else:
                eta[i, j] = np.nan   # spacing too wide — model non-physical here
                beta[i, j] = np.nan  # downstream functions treat NaN as "passes"
    return eta, beta

def obtain_eta_beta_line(MS_matrix):
    return _compute_eta_beta(MS_matrix)

def obtain_eta_beta_via(space_matrix):
    return _compute_eta_beta(space_matrix)

def reliability_prediction(eta_tBD, beta_tBD, time=T_TARGET):
    """Calculates the cumulative failure probability at a given time using the Weibull distribution.
    Cells with eta=0/NaN (no metal or out-of-range spacing) return 0.0 (passes spec)."""
    result = np.zeros_like(eta_tBD, dtype=float)
    valid = (eta_tBD > 0) & (beta_tBD > 0) & np.isfinite(eta_tBD)
    ratio = np.divide(time, eta_tBD, out=np.zeros_like(eta_tBD, dtype=float), where=valid)
    result[valid] = 1 - np.exp(-(ratio[valid] ** beta_tBD[valid]))
    return result

def joint_reliability_prediction(eta_tBD_line, beta_tBD_line, eta_tBD_via, beta_tBD_via, time=T_TARGET):
    """Combines line and via reliability predictions to get an overall failure probability."""
    # Calculate individual reliabilities only where parameters are valid.
    reliability_line = np.ones_like(eta_tBD_line, dtype=float)
    reliability_via = np.ones_like(eta_tBD_via, dtype=float)

    valid_line = (eta_tBD_line > 0) & (beta_tBD_line > 0) & np.isfinite(eta_tBD_line)
    valid_via = (eta_tBD_via > 0) & (beta_tBD_via > 0) & np.isfinite(eta_tBD_via)

    ratio_line = np.divide(time, eta_tBD_line, out=np.zeros_like(eta_tBD_line, dtype=float), where=valid_line)
    ratio_via = np.divide(time, eta_tBD_via, out=np.zeros_like(eta_tBD_via, dtype=float), where=valid_via)

    reliability_line[valid_line] = np.exp(-(ratio_line[valid_line] ** beta_tBD_line[valid_line]))
    reliability_via[valid_via] = np.exp(-(ratio_via[valid_via] ** beta_tBD_via[valid_via]))
    
    # Assuming independence, the joint reliability is the product of individual reliabilities
    joint_reliability = reliability_line * reliability_via
    
    # Convert back to failure probability
    return 1 - joint_reliability

def time_to_failure_prediction(eta_tBD, beta_tBD, target_reliability= 1 - F_TARGET):
    """Calculates the time to failure for a given target reliability using the Weibull distribution.
    Cells with eta=0 (no metal) return np.inf. Cells with eta=NaN (spacing > S_MAX) also return
    np.inf — those cells are assumed to pass the spec; np.minimum picks the other component."""
    result = np.full_like(eta_tBD, np.inf, dtype=float)
    valid = (eta_tBD > 0) & (beta_tBD > 0) & np.isfinite(eta_tBD)
    scale = -np.log(target_reliability)
    exponent = np.divide(1.0, beta_tBD, out=np.zeros_like(beta_tBD, dtype=float), where=valid)
    result[valid] = eta_tBD[valid] * (scale ** exponent[valid])
    return result

def joint_time_to_failure_prediction(eta_tBD_line, beta_tBD_line, eta_tBD_via, beta_tBD_via, target_reliability= 1 - F_TARGET):
    """Calculates the joint time to failure for line and via based on a target reliability."""
    # Calculate individual time to failure predictions
    ttf_line = time_to_failure_prediction(eta_tBD_line, beta_tBD_line, target_reliability)
    ttf_via = time_to_failure_prediction(eta_tBD_via, beta_tBD_via, target_reliability)
    
    # Assuming independence, the joint time to failure can be approximated by the minimum of the two
    return np.minimum(ttf_line, ttf_via)

def spacing_dominance(ms_matrix, space_matrix):
    """Determines whether line or via spacing is more critical at each point."""
    dominance_matrix = np.where(space_matrix < ms_matrix, 'via', 'line')
    return dominance_matrix

def tff_dominance(ttf_line, ttf_via):
    """Determines whether line or via time to failure is more critical at each point."""
    dominance_matrix = np.where(ttf_via < ttf_line, 'via', 'line')
    return dominance_matrix

def class_map(ms_matrix, space_matrix):
    """Classifies each cell into reliability categories based on spacing thresholds."""
    conditions = [
        (space_matrix <= 0) & (ms_matrix <= 0),
        (space_matrix > 0) & (ms_matrix > 0) & ((space_matrix < S_MIN) | (ms_matrix < S_MIN)),
        (space_matrix >= S_MIN) & (space_matrix <= S_MAX) & (ms_matrix > S_MAX),
        (ms_matrix >= S_MIN) & (ms_matrix <= S_MAX) & (space_matrix > S_MAX),
        (space_matrix >= S_MIN) & (space_matrix <= S_MAX) & (ms_matrix >= S_MIN) & (ms_matrix <= S_MAX),
        (space_matrix > S_MAX) & (ms_matrix > S_MAX)
    ]
    choices = ['non-existence', 'fail', 'via risk', 'line risk', 'joint risk', 'no risk']
    return np.select(conditions, choices, default='unknown')

def print_debug_points(wafer_name, ms_matrix, space_matrix, ttf_line, ttf_via, joint_ttf, points_xy):
    """Print debug values at specified (row, col) grid points."""
    print(f"Debug points for {wafer_name}:")
    for x, y in points_xy:
        tl, tv = ttf_line[x, y], ttf_via[x, y]
        ttf_dominant = "via" if tv < tl else "line"
        spacing_tight = "via" if space_matrix[x, y] < ms_matrix[x, y] else "line"
        in_range_line = ms_matrix[x, y] <= S_MAX
        in_range_via  = space_matrix[x, y] <= S_MAX
        print(
            f"  ({x},{y}): MS={ms_matrix[x,y]:.2f} nm ({'in range' if in_range_line else f'> S_MAX={S_MAX}'})  "
            f"Space={space_matrix[x,y]:.2f} nm ({'in range' if in_range_via else f'> S_MAX={S_MAX}'})"
        )
        print(
            f"         TTF Line={tl:.2e} s  TTF Via={tv:.2e} s  Joint={joint_ttf[x,y]:.2e} s"
        )
        print(
            f"         TTF dominated by: {ttf_dominant}  |  spacing-tight: {spacing_tight}"
        )


def _overlay_no_metal(axis, existence_matrix):
    """Overlays black pixels on all cells where ExistenceClass == 0 (no metal present)."""
    no_metal = np.ma.masked_where(existence_matrix != 0, np.ones_like(existence_matrix, dtype=float))
    axis.imshow(no_metal, cmap='binary', vmin=0, vmax=1)

def _draw_subplot(fig, axis, title, matrix, cmap, existence_matrix):
    resolved_cmap = CMAP_NAME_LOOKUP.get(cmap.lower(), cmap) if isinstance(cmap, str) else cmap

    if title == 'Existence class':
        image = axis.imshow(np.asarray(matrix), cmap=EXISTENCE_CMAP, norm=EXISTENCE_NORM)
        category_labels = EXISTENCE_LABELS
    elif title in {'MS spacing', 'Space'}:
        spacing_matrix = np.ma.masked_where(~np.isfinite(np.asarray(matrix, dtype=float)), np.asarray(matrix, dtype=float))
        if title == 'MS spacing':
            image = axis.imshow(spacing_matrix, cmap='plasma_r', vmin=15, vmax=21)
        else:
            image = axis.imshow(spacing_matrix, cmap='plasma_r', vmin=2.5, vmax=20)
        category_labels = None
        if existence_matrix is not None:
            _overlay_no_metal(axis, existence_matrix)
    elif title == 'Class map':
        class_matrix = np.asarray(matrix)
        encoded = np.full(class_matrix.shape, np.nan, dtype=float)
        for index, class_name in enumerate(CLASS_MAP_CLASSES):
            encoded[class_matrix == class_name] = float(index)
        image = axis.imshow(encoded, cmap=CLASS_MAP_CMAP, norm=CLASS_MAP_NORM)
        category_labels = CLASS_MAP_LABELS
    else:
        imshow_matrix, category_labels = prepare_matrix_for_imshow(matrix)
        if category_labels is not None:
            vmax = max(len(category_labels) - 0.5, 0.5)
            image = axis.imshow(imshow_matrix, cmap=resolved_cmap, vmin=-0.5, vmax=vmax)
        else:
            image = axis.imshow(imshow_matrix, cmap=resolved_cmap)
        if existence_matrix is not None:
            _overlay_no_metal(axis, existence_matrix)

    axis.set_title(title)
    axis.set_xlabel('X Grid')
    axis.set_ylabel('Y Grid')
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    if title == 'Existence class':
        colorbar.set_ticks(EXISTENCE_CLASSES)
        colorbar.set_ticklabels(EXISTENCE_LABELS)
    elif title == 'Class map':
        colorbar.set_ticks(np.arange(len(CLASS_MAP_LABELS)))
        colorbar.set_ticklabels(CLASS_MAP_LABELS)
    elif category_labels is not None and category_labels:
        colorbar.set_ticks(np.arange(len(category_labels)))
        colorbar.set_ticklabels(category_labels)

def save_2x4_map_figure(output_path, wafer_name, maps, existence_matrix=None):
    """Saves eight heatmaps in a 2x4 layout."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for axis, (title, matrix, cmap) in zip(axes.ravel(), maps):
        _draw_subplot(fig, axis, title, matrix, cmap, existence_matrix)
    fig.suptitle(f'Wafer Maps for {wafer_name}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_2x3_map_figure(output_path, wafer_name, maps, existence_matrix=None):
    """Saves six heatmaps in a 2x3 layout."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for axis, (title, matrix, cmap) in zip(axes.ravel(), maps):
        _draw_subplot(fig, axis, title, matrix, cmap, existence_matrix)
    fig.suptitle(f'Wafer Maps for {wafer_name}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    print(f"Using physics model: {ACTIVE_MODEL_NAME}")

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
            # print(f"Running DPM engine predictions for {wafer_folder.name}...")
            dataset = load_data(lot_number=lot_number, wafer_number=wafer_number)
            eta_matrix_via, beta_matrix_via = obtain_eta_beta_via(dataset['space'])

            eta_matrix_line, beta_matrix_line = obtain_eta_beta_line(dataset['ms'])

            reliability_via = reliability_prediction(eta_matrix_via, beta_matrix_via)
            reliability_line = reliability_prediction(eta_matrix_line, beta_matrix_line)

            joint_reliability= joint_reliability_prediction(eta_matrix_line, beta_matrix_line, eta_matrix_via, beta_matrix_via)

            TTF_via = time_to_failure_prediction(eta_matrix_via, beta_matrix_via)
            TTF_line = time_to_failure_prediction(eta_matrix_line, beta_matrix_line)

            joint_TTF = joint_time_to_failure_prediction(eta_matrix_line, beta_matrix_line, eta_matrix_via, beta_matrix_via)

            wafer_output_root = lot_output_root / wafer_folder.name
            wafer_output_root.mkdir(parents=True, exist_ok=True)

            np.savetxt(wafer_output_root / 'Predicted_eta_tBD_via.csv', eta_matrix_via, delimiter=',')
            np.savetxt(wafer_output_root / 'Predicted_beta_tBD_via.csv', beta_matrix_via, delimiter=',')
            np.savetxt(wafer_output_root / 'Predicted_eta_tBD_line.csv', eta_matrix_line, delimiter=',')
            np.savetxt(wafer_output_root / 'Predicted_beta_tBD_line.csv', beta_matrix_line, delimiter=',')
            np.savetxt(wafer_output_root / 'Predicted_joint_reliability.csv', joint_reliability, delimiter=',')
            np.savetxt(wafer_output_root / 'Predicted_joint_TTF.csv', joint_TTF, delimiter=',')
            np.savetxt(wafer_output_root / 'Predicted_reliability_via.csv', reliability_via, delimiter=',')
            np.savetxt(wafer_output_root / 'Predicted_reliability_line.csv', reliability_line, delimiter=',')
            

            # print_debug_points(wafer_folder.name, dataset['ms'], dataset['space'], TTF_line, TTF_via, joint_TTF, DEBUG_POINTS_XY)

            save_2x4_map_figure(
                wafer_output_root / f'{wafer_folder.name}.png',
                wafer_folder.name,
                [
                    ('Eta tBD via', eta_matrix_via, 'spectral'),
                    ('Beta tBD via', beta_matrix_via, 'plasma'),
                    ('Eta tBD line', eta_matrix_line, 'spectral'),
                    ('Beta tBD line', beta_matrix_line, 'plasma'),
                    ('TTF via', TTF_via, 'spectral'),
                    ('TTF line', TTF_line, 'spectral'),
                    ('Joint TTF', joint_TTF, 'spectral'),
                ],
                existence_matrix=dataset['existence'],
            )
            save_2x3_map_figure(
                wafer_output_root / f'{wafer_folder.name}_reliability.png',
                wafer_folder.name,
                [   ('Existence class', dataset['existence'], 'gray'),
                    ('MS spacing', dataset['ms'], 'viridis'),
                    ('Space', dataset['space'], 'plasma'),
                    ('Class map', class_map(dataset['ms'], dataset['space']), CLASS_MAP_CMAP),
                    ('Spacing dominance', spacing_dominance(dataset['ms'], dataset['space']), DOMINANCE_CMAP),
                    ('TTF via', TTF_via, 'spectral'),
                  
                ],
                existence_matrix=dataset['existence'],
            )


        sample_dataset = load_data(lot_number=lot_number, wafer_number=int(wafer_folders[0].name.split("_")[1]))
        sample_eta_matrix, sample_beta_matrix = obtain_eta_beta_via(sample_dataset['space'])
        sample_eta_matrix_line, sample_beta_matrix_line = obtain_eta_beta_line(sample_dataset['ms'])
        sample_reliability_via = reliability_prediction(sample_eta_matrix, sample_beta_matrix)
        sample_reliability_line = reliability_prediction(sample_eta_matrix_line, sample_beta_matrix_line)
        sample_joint_reliability = joint_reliability_prediction(
            sample_eta_matrix_line,
            sample_beta_matrix_line,
            sample_eta_matrix,
            sample_beta_matrix,
        )
        sample_joint_ttf = joint_time_to_failure_prediction(
            sample_eta_matrix_line,
            sample_beta_matrix_line,
            sample_eta_matrix,
            sample_beta_matrix,
        )

        figures_dir = lot_output_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_path = figures_dir / "predicted_eta_tBD_sample.png"
        save_2x4_map_figure(
            plot_path,
            'wafer_01 sample',
            [
                ('Eta tBD via', sample_eta_matrix, 'viridis'),
                ('Beta tBD via', sample_beta_matrix, 'plasma'),
                ('Eta tBD line', sample_eta_matrix_line, 'viridis'),
                ('Beta tBD line', sample_beta_matrix_line, 'plasma'),
                ('Reliability via', sample_reliability_via, 'magma'),
                ('Reliability line', sample_reliability_line, 'magma'),
                ('Joint reliability', sample_joint_reliability, 'inferno'),
                ('Joint TTF', sample_joint_ttf, 'spectral'),
            ],
            existence_matrix=sample_dataset['existence'],
        )
        print(f"Plot saved to {plot_path}")
    plt.show()
