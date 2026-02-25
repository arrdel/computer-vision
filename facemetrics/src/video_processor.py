"""Video Processor — frame-by-frame iterator with optional resizing."""

import cv2


class VideoProcessor:
    """
    Opens a video file and yields RGB frames one at a time.

    Usage
    -----
        vp = VideoProcessor("video.mp4", resize_width=960)
        for idx, frame_rgb in vp:
            ...
        vp.release()
    """

    def __init__(self, video_path: str, resize_width: int | None = None):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.resize_width = resize_width
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._index = 0

        print(f"  Video: {video_path}")
        print(f"  Resolution: {self.width}×{self.height} @ {self.fps:.1f} fps")
        print(f"  Total frames: {self.total_frames}  "
              f"(~{self.total_frames / self.fps / 60:.1f} min)")

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, "np.ndarray"]:
        ret, frame_bgr = self.cap.read()
        if not ret:
            raise StopIteration

        # Resize if requested
        if self.resize_width and frame_bgr.shape[1] != self.resize_width:
            ratio = self.resize_width / frame_bgr.shape[1]
            new_h = int(frame_bgr.shape[0] * ratio)
            frame_bgr = cv2.resize(frame_bgr, (self.resize_width, new_h))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        idx = self._index
        self._index += 1
        return idx, frame_rgb

    def release(self):
        self.cap.release()

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0
