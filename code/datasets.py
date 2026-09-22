from torch.utils.data import Dataset
import numpy as np
import os


# These are the only datasets treated as undirected in the current protocol.
# FB-Forum and unknown datasets remain directed in ``auto`` mode.
UNDIRECTED_DATASET_ALIASES = (
    "hypertext",
    "contact",
    "ia-contact",
    "ia_contact",
    "iacontact",
)


def resolve_graph_mode(dataset_name=None, graph_mode="auto", file_name=None):
    """Resolve directed/undirected semantics without inferring them from rows.

    Edge direction is a property of the dataset semantics and cannot be
    recovered reliably from a three-column edge list.  ``auto`` therefore uses
    the explicit dataset registry above; callers may always override it with
    ``directed`` or ``undirected``.
    """
    requested = str(graph_mode).strip().lower()
    if requested not in {"auto", "directed", "undirected"}:
        raise ValueError("graph_mode must be one of: auto, directed, undirected")
    if requested != "auto":
        return requested

    # Inspect the declared dataset name and the data-file basename only.  The
    # parent directory is deliberately ignored: placing a directed dataset in
    # a folder whose name contains "contact" must not change graph semantics.
    identifiers = [
        str(dataset_name or "").lower(),
        os.path.basename(str(file_name or "")).lower(),
    ]
    if any(alias in identifier for identifier in identifiers for alias in UNDIRECTED_DATASET_ALIASES):
        return "undirected"
    return "directed"


def _resolve_delimiter(file_name, delimiter, skip_rows):
    if delimiter is None:
        return None
    normalized = str(delimiter).strip().lower()
    if normalized == "auto":
        with open(file_name, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f):
                if line_number < skip_rows:
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                if "," in stripped:
                    return ","
                if "\t" in stripped:
                    return "\t"
                return None
        return None
    if normalized in {"space", "whitespace", "none"}:
        return None
    if normalized in {"comma", "csv"}:
        return ","
    if normalized in {"tab", "tsv"}:
        return "\t"
    return delimiter


class Temporal_Dataset(Dataset):
    def __init__(self, file_name, starting=0, skip_rows=0, div=3600,
                 delimiter="auto", source_col=0, target_col=1, time_col=-1,
                 dataset_name=None, graph_mode="auto"):
        self.graph_mode = resolve_graph_mode(dataset_name, graph_mode, file_name)
        self.is_undirected = self.graph_mode == "undirected"
        delimiter = _resolve_delimiter(file_name, delimiter, skip_rows)
        raw_data = np.loadtxt(fname=file_name, skiprows=skip_rows, delimiter=delimiter)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        num_cols = raw_data.shape[1]
        time_col = int(time_col)
        if time_col < 0:
            time_col = num_cols + time_col
        selected_cols = [int(source_col), int(target_col), time_col]
        if min(selected_cols) < 0 or max(selected_cols) >= num_cols:
            raise ValueError(
                f"Invalid column selection {selected_cols} for {num_cols}-column data file: {file_name}"
            )

        self.data = raw_data[:, selected_cols]
        if not np.isfinite(self.data).all():
            raise ValueError(f"Non-finite source/target/timestamp value in data file: {file_name}")
        if not np.all(self.data[:, :2] == np.floor(self.data[:, :2])):
            raise ValueError(f"Node identifiers must be integers in data file: {file_name}")

        # Temporal datasets commonly contain many events with an identical timestamp.
        # NumPy's default quicksort is not stable, so using it here can silently change
        # the causal order of tied events across platforms/versions.  Preserve the
        # source-file order for ties to make training and evaluation reproducible.
        order = np.argsort(self.data[:, 2], kind="mergesort")
        self.data = self.data[order]
        self.time = self.data[:, 2]
        self.raw_time_span = float(self.time[-1] - self.time[0]) if len(self.time) > 0 else 0.0
        self.trans_time = (self.time - self.time[0]) / div
        self.time_div = float(div)
        self.time_span = float(self.trans_time[-1] - self.trans_time[0]) if len(self.trans_time) > 0 else 0.0
        self.data[:, 2] = self.trans_time
        self.data[:, [0, 1]] = self.data[:, [0, 1]] - starting
        if np.min(self.data[:, :2]) < 0:
            raise ValueError(
                f"--starting={starting} produces negative node identifiers for data file: {file_name}"
            )

        # An undirected event is the unordered pair {u, v}.  Store a unique,
        # deterministic representation without duplicating the event.  This
        # removes any dependence on which endpoint happened to be written in
        # the first input column while preserving event counts and timestamps.
        if self.is_undirected:
            self.data[:, :2] = np.sort(self.data[:, :2], axis=1)

        max_src = int(np.max(self.data[:, 0]))
        max_dst = int(np.max(self.data[:, 1]))
        self.max_node = max(max_src, max_dst)
        print(f"Max node id: {self.max_node}")
        print(f"Raw time span: {self.raw_time_span:.6g} | time_div: {self.time_div:.6g} | "
              f"model time span: {self.time_span:.6g}")
        print(f"Graph mode: {self.graph_mode}")

    def __len__(self):
        return self.time.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx, :]
        return sample
