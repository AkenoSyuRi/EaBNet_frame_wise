import wave

import librosa
import numpy as np
from openvino.runtime import Core
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


def init_states(model_input_shapes, data_type):
    states = []
    for shape in model_input_shapes:
        states.append(np.zeros(shape, dtype=data_type))
    return states


def main():
    in_wav_path = r"D:\Temp\athena_test_out\[real]M16_demo_1_inp.wav"
    out_wav_path = r"D:\Temp\athena_test_out\test_out.wav"
    onnx_model_path = "data/output/EaBNet_iLN_epoch67.xml"
    data_type = [np.float32, np.float16][0]

    # Initialize OpenVINO Core and load model
    core = Core()
    model = core.read_model(onnx_model_path)
    compiled_model = core.compile_model(model, "CPU")  # Use "AUTO" or "GPU" if hardware available
    infer_request = compiled_model.create_infer_request()

    # Get model input and state shapes
    input_tensor = compiled_model.inputs[0]
    state_tensors = compiled_model.inputs[1:]
    output_tensors = compiled_model.outputs
    state_shapes = [state_tensor.shape for state_tensor in state_tensors]

    # Initialize states
    states = init_states(state_shapes, data_type)

    # Read input audio
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
            in_spec = in_spec.astype(data_type)[None, None]

            # Prepare input dictionary
            inputs = {input_tensor: in_spec}
            for state_tensor, state in zip(state_tensors, states):
                inputs[state_tensor] = state

            # Run inference
            infer_request.infer(inputs)
            out_spec = infer_request.get_tensor(output_tensors[0]).data.squeeze()
            states = [infer_request.get_tensor(state_tensor).data for state_tensor in output_tensors[1:]]

            # Convert output spectrogram back to time-domain audio
            out_spec = out_spec[0] + 1j * out_spec[1]
            out_data = stft.inverse(out_spec)

            out_data = (out_data * 32768).astype(np.int16)
            fp.writeframes(out_data.tobytes())


if __name__ == "__main__":
    main()
