import pathlib
import warnings

def format_delta_time(delta, i, I, micro=False):
    microseconds = delta % 1
    delta_seconds = delta - microseconds
    hours, remainder = divmod(delta_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    r_delta = (I * delta / (i + 1)) - delta
    r_microseconds = r_delta % 1
    r_delta_seconds = r_delta - r_microseconds
    r_hours, r_remainder = divmod(r_delta_seconds, 3600)
    r_minutes, r_seconds = divmod(r_remainder, 60)

    if micro:
        return (
            f"TP: {int(hours)}:{int(minutes)}:{int(seconds)}"
            f".{int(microseconds * 1e6)}"
            f"\tETA: {int(r_hours)}:{int(r_minutes)}:{int(r_seconds)}"
            f".{int(r_microseconds * 1e6)}")
    else:
        return (
            f"TP: {int(hours)}:{int(minutes)}:{int(seconds)}"
            f"\tETA: {int(r_hours)}:{int(r_minutes)}:{int(r_seconds)}"
        )


class VideoGenerator:
    def __init__(self):
        self._path = pathlib.Path("./")
        self._name = "video.mp4"
        self._final_path = None
        self._frames = []

    def _set_final_path(self):
        self._final_path = self._path / self._name

    def _set_path(self, value):
        if value is None:
            return
        self._path = pathlib.Path(value)
        self._set_final_path()

    def set_file_name(self, path, value):
        self._name = value
        self._set_path(path)

    def add_frame(self, frame):
        self._frames.append(frame)

    def reset_frame_buffer(self):
        self._frames = []

    def save_video(self):
        warnings.warn(
            "The save_video method is deprecated and not implemented. "
            "Please implement it in a subclass.",
            DeprecationWarning
        )
