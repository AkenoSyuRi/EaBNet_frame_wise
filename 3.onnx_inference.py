import wave

import librosa
import numpy as np
import onnxruntime as ort
from tqdm import trange


class Stft:
    def __init__(self, win_size, hop_size, in_channels, out_channels):
        self.win_size = win_size
        self.hop_size = hop_size
        self.overlap = win_size - hop_size
        self.fft_bins = win_size // 2 + 1

        self.window = np.hanning(win_size + 1)[1:]
        self.win_sum = self.get_win_sum_of_1frame(self.window, win_size, hop_size)

        self.in_win_data = np.zeros([in_channels, win_size])
        self.out_ola_data = np.zeros([out_channels, win_size])
        ...

    @staticmethod
    def get_win_sum_of_1frame(window, win_len, win_inc):
        assert win_len % win_inc == 0, "win_len must be equally divided by win_inc"
        win_square = window**2
        overlap = win_len - win_inc
        win_tmp = np.zeros(overlap + win_len)

        loop_cnt = win_len // win_inc
        for i in range(loop_cnt):
            win_tmp[i * win_inc : i * win_inc + win_len] += win_square
        win_sum = win_tmp[overlap : overlap + win_inc]
        assert np.min(win_sum) > 0, "the nonzero overlap-add constraint is not satisfied"
        return win_sum

    def transform(self, input_data):
        self.in_win_data[:, : self.overlap] = self.in_win_data[:, self.hop_size :]
        self.in_win_data[:, self.overlap :] = input_data

        spec_data = np.fft.rfft(self.in_win_data * self.window, axis=-1)
        return spec_data.squeeze()

    def inverse(self, input_spec):
        syn_data = np.fft.irfft(input_spec, axis=-1) * self.window

        self.out_ola_data += syn_data
        output_data = self.out_ola_data[:, : self.hop_size] / self.win_sum

        self.out_ola_data[:, : self.overlap] = self.out_ola_data[:, self.hop_size :]
        self.out_ola_data[:, self.overlap :] = 0
        return output_data.squeeze()


def init_and_check_states(sess: ort.InferenceSession):
    enc_states = [
        np.zeros([1, 2 * 8, 1, 161]).astype(np.float32),
        np.zeros([1, 64, 1, 79]).astype(np.float32),
        np.zeros([1, 64, 1, 39]).astype(np.float32),
        np.zeros([1, 64, 1, 19]).astype(np.float32),
        np.zeros([1, 64, 1, 9]).astype(np.float32),
    ]

    squ_states = [[np.zeros([1, 64, (5 - 1) * 2**i, 2]).astype(np.float32) for i in range(6)] for _ in range(3)]
    dec_states = [
        np.zeros([1, 128, 1, 4]).astype(np.float32),
        np.zeros([1, 128, 1, 9]).astype(np.float32),
        np.zeros([1, 128, 1, 19]).astype(np.float32),
        np.zeros([1, 128, 1, 39]).astype(np.float32),
        np.zeros([1, 128, 1, 79]).astype(np.float32),
    ]
    rnn_state = np.zeros([2, 161, 64, 2]).astype(np.float32)

    states = enc_states
    for group in squ_states:
        states += group
    states += dec_states
    states += [rnn_state]

    i = 0
    for inp, out in zip(sess.get_inputs(), sess.get_outputs()):
        if "state" in inp.name:
            assert inp.shape == out.shape, f"out state shape mismatch: {inp.shape} vs {out.shape}"
            assert inp.shape == list(states[i].shape), f"in state shape mismatch: {inp.shape} vs {states[i].shape}"
            i += 1
        print(f"{inp.name}: {inp.shape} -> {out.name}: {out.shape}")
    return states


def main():
    in_wav_path = r"D:\Temp\athena_test_out\[real]M16_demo_1_inp.wav"
    out_wav_path = r"D:\Temp\athena_test_out\test_out.wav"
    onnx_model_path = "data/output/EaBNet_iLN_epoch67.onnx"
    session = ort.InferenceSession(onnx_model_path)

    # Initialize states
    output_names = [out.name for out in session.get_outputs()]
    states = init_and_check_states(session)

    in_data, sr = librosa.load(in_wav_path, sr=None, mono=False)

    win_size, hop_size, in_channels, out_channels = 320, 160, 8, 1
    stft = Stft(win_size, hop_size, in_channels, out_channels)

    with wave.Wave_write(out_wav_path) as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(sr)
        for i in trange(0, in_data.shape[1], hop_size):
            frame = in_data[:, i : i + hop_size]
            in_spec = stft.transform(frame).T
            in_spec = np.stack([in_spec.real, in_spec.imag], axis=-1)
            in_spec = in_spec.astype(np.float32)[None, None]

            ort_inputs = {"input": in_spec}
            for j, state in enumerate(states):
                ort_inputs[f"in_state{j}"] = state

            ort_outs = session.run(output_names, ort_inputs)
            out_spec, states = ort_outs[0].squeeze(), ort_outs[1:]
            out_spec = out_spec[0] + 1j * out_spec[1]
            out_data = stft.inverse(out_spec)

            out_data = (out_data * 32768).astype(np.int16)
            fp.writeframes(out_data.tobytes())
            ...
    ...


if __name__ == "__main__":
    main()
    ...
