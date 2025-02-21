import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from subprocess import Popen, PIPE
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, DummyVecEnv

# Disable eager execution for TF v1 compatibility.
tf1.disable_eager_execution()


class TensorboardVideoRecorder(VecEnvWrapper):
    """
    A VecEnv wrapper that records video frames from one of the vectorized environments
    and logs them to TensorBoard as an animated GIF using TensorFlow’s summary API.

    If the provided environment is not vectorized, it will be automatically wrapped in a DummyVecEnv.

    :param env: The environment to wrap (gymnasium.Env or VecEnv).
    :param video_trigger: A function that takes the current global step (int) and returns True
                          when a video should be recorded (e.g., lambda step: step % 10000 == 0).
    :param video_length: The max number of frames to record for the video.
    :param record_video_env_idx: The index of the environment within the vectorized env to record.
    :param tag: Video tag name in TensorBoard.
    :param fps: Frames per second to encode the video.
    :param tb_log_dir: The directory path where TensorBoard logs (summaries) will be saved.
    """

    def __init__(
            self,
            env,
            video_trigger,
            video_length,
            record_video_env_idx=0,
            tag="policy_rollout",
            fps=30,
            tb_log_dir="./logs/tensorboard"
    ):
        # Automatically wrap non-vectorized envs.
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        super().__init__(env)

        self._video_trigger = video_trigger
        self._video_length = video_length
        self._record_video_env_idx = record_video_env_idx
        self._tag = tag
        self._fps = fps

        self._global_step = 0
        self._recording = False
        self._recording_step_count = 0
        self._recorded_frames = []

        self._record_on_reset_pending = False

        self._tb_log_dir = tb_log_dir
        self._file_writer = tf1.summary.FileWriter(tb_log_dir)

    @staticmethod
    def _encode_gif(frames, fps):
        h, w, c = frames[0].shape
        pxfmt = {1: 'gray', 3: 'rgb24'}[c]
        cmd = ' '.join([
            'ffmpeg -y -f rawvideo -vcodec rawvideo',
            f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
            '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
            f'-r {fps:.02f} -f gif -'
        ])
        proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        for image in frames:
            proc.stdin.write(image.tobytes())
        out, err = proc.communicate()
        if proc.returncode:
            raise IOError('\n'.join([' '.join(cmd.split(" ")), err.decode('utf8')]))
        del proc
        return out

    def _log_video_to_tensorboard(self, tag, video, step):
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(255 * video, 0, 255).astype(np.uint8)
        try:
            T, H, W, C = video.shape
            summary = tf1.Summary()
            image = tf1.Summary.Image(height=H, width=W, colorspace=C)
            image.encoded_image_string = self._encode_gif(list(video), self._fps)
            summary.value.add(tag=tag, image=image)
            self._file_writer.add_summary(summary, step)
        except (IOError, OSError) as e:
            print('GIF summaries require ffmpeg in $PATH.', e)
            tf.summary.image(tag, video, step)

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        if self._recording:
            self._record_frame()
            self._recording_step_count += 1
        return obs

    def _record_frame(self):
        frames = self.venv.env_method("render")
        frame = frames[self._record_video_env_idx]
        self._recorded_frames.append(frame)

    def _finalize_video(self):
        if not self._recorded_frames:
            return
        video_np = np.array(self._recorded_frames)  # Shape: (T, H, W, C)
        self._log_video_to_tensorboard(self._tag, video_np, self._global_step)
        self._recording = False
        self._recording_step_count = 0
        self._recorded_frames = []

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self._global_step += self.venv.num_envs

        if not self._recording and not self._record_on_reset_pending and self._video_trigger(self._global_step):
            self._record_on_reset_pending = True

        if self._recording:
            self._record_frame()
            self._recording_step_count += 1

            if self._recording_step_count >= self._video_length or dones[self._record_video_env_idx]:
                self._finalize_video()

        if self._record_on_reset_pending and dones[self._record_video_env_idx]:
            self._recording = True
            self._record_on_reset_pending = False
            self._recording_step_count = 0
            self._recorded_frames = []

        return obs, rewards, dones, infos
