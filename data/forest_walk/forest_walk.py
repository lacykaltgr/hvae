import numpy as np
import cv2
import os
from hvae_backbone.elements.dataset import DataSet

root = "/Users/laszlofreund/code/ai/hvae/data/forest_walk/"


class ForestVideoDataset(DataSet):
    def __init__(self, seq_len, n_frames, frame_rate=2, patch_size=20, whiten=False, n_downsamples=2, with_labels=False):
        self.patch_size = patch_size
        self.frame_rate = frame_rate
        self.n_frames = n_frames
        self.whiten = whiten
        self.n_downsamples = n_downsamples
        self.sequence_len = seq_len
        super(ForestVideoDataset, self).__init__(with_labels=with_labels)

    def load(self):
        if os.path.exists(os.path.join(root, 'sliced.npy')):
            return self.split(np.load(os.path.join(root, 'sliced.npy')))

        video_path = os.path.join(root, 'forest_video.mp4')
        assert os.path.exists(video_path), "Video file does not exist."
        dataset = load_video(video_path, n_frames=self.n_frames, frame_rate=self.frame_rate)
        np.save(os.path.join(root, 'original.npy'), dataset)

        dataset = np.load(os.path.join(root, 'original.npy'))
        downsampled = downsample(dataset, factor=self.n_downsamples)
        np.save(os.path.join(root, 'downsampled.npy'), downsampled)

        if self.whiten:
            downsampled = np.load(os.path.join(root, 'downsampled.npy'))
            whitened = whitening(downsampled)
            np.save(os.path.join(root, 'whitened.npy'), whitened)

        normalized = normalize(downsampled)
        np.save(os.path.join(root, 'normalized.npy'), normalized)

        sliced = slice_sequence(normalized, self.sequence_len, self.patch_size)
        np.save(os.path.join(root, 'sliced.npy'), sliced)

        return self.split(sliced)


    def split(self, dataset):
        val_split = int(0.8*len(dataset))
        test_split = int(0.9*len(dataset))
        train = dataset[:val_split]
        val = dataset[val_split:test_split]
        test = dataset[test_split:]
        return train, val, test


def slice_sequence(sequence, sequence_len, patch_size):
    n_frames, height, width = sequence.shape
    edge_buffer = 20
    assert n_frames >= sequence_len
    assert height >= patch_size
    assert width >= patch_size
    n_sliced_sequences = n_frames // sequence_len
    sliced_sequences = []
    for i in range(n_sliced_sequences):
        start_frame = i * sequence_len
        end_frame = start_frame + sequence_len
        start_height = np.random.randint(edge_buffer, height - patch_size + 1 - edge_buffer)
        start_width  = np.random.randint(edge_buffer, width  - patch_size + 1 - edge_buffer)
        sliced_sequences.append(sequence[
                                start_frame:end_frame,
                                start_height:start_height+patch_size,
                                start_width:start_width+patch_size
                                ])
    return np.array(sliced_sequences)


def extract_frames(file_path, start_frame, n_frames, frame_rate=1):
    cap = cv2.VideoCapture(file_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    # Read the frames of the video
    frames = []
    for i in range(start_frame+n_frames):
        ret, frame = cap.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        if i < start_frame or i % frame_rate != 0:
            continue

        # Convert the frame to grayscale or perform any other preprocessing
        # For example, you can use cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for grayscale
        # or apply other image processing techniques
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Append the frame to the list
        frames.append(gray_frame)

    # Release the video capture object
    cap.release()

    # Convert the list of frames to a NumPy array
    frames_array = np.array(frames)

    return frames_array


def load_video(video_path, n_frames, frame_rate=1, original_w=1280, original_h=720):
    video = extract_frames(video_path, 1000, n_frames, frame_rate)[
            :, :, int(original_w/2 - original_h/2): int(original_w/2 + original_h/2)]
    return video


def downsample(data, algorithm="gaussian_pyramid", factor=2):

    if algorithm == 'steerable_pyramid':
        # steerable pyramid downsample
        # https://pyrtools.readthedocs.io/en/latest/tutorials/03_steerable_pyramids.html
        #filt = 'sp3_filters' # There are 4 orientations for this filter
        #pyr = pt.pyramids.SteerablePyramidSpace(video[0], height=4, order=3)
        raise NotImplementedError()

    elif algorithm == 'gaussian_pyramid':
        downsampled = []
        for frame in data:
            for i in range(factor):
                frame = cv2.pyrDown(frame)
            downsampled.append(frame)
        downsampled = np.array(downsampled)
        print("Original shape: ", data.shape,
              "Downsampled shape: ", downsampled.shape)
        return downsampled

    else:
        raise ValueError("Unknown algorithm: ", algorithm)


def normalize(data, method='rgb'):
    if method == 'minmax':
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'mean':
        data = (data - np.mean(data)) / np.std(data)
    elif method == 'rgb':
        data = (data.astype(np.float32) - 127.5) / 127.5
    else:
        raise ValueError("Unknown normalization method: ", method)
    return data


def whiten_frame(frame):
    # Take Fourier transform
    f_transform = np.fft.fft2(frame)

    # Create frequency grid
    rows, cols = frame.shape
    u = np.fft.fftfreq(rows)
    v = np.fft.fftfreq(cols)

    # Create meshgrid from frequency grid
    u, v = np.meshgrid(u, v)

    # Calculate frequency radius
    r = np.sqrt(u**2 + v**2)

    # Define linear ramp function w1(u, v) = r
    w1 = r

    # Define low-pass windowing function w2(u, v) = e^(-(r/r0)^4)
    r0 = 48
    w2 = np.exp(-(r/r0)**4)

    # Calculate whitening mask function w(u, v) = w1(u, v) * w2(u, v)
    whitening_mask = w1 * w2

    # Modulate amplitude by whitening mask
    f_transform_whitened = f_transform * whitening_mask

    # Take inverse Fourier transform
    frame_whitened = np.fft.ifft2(f_transform_whitened).real

    return frame_whitened


def whitening(data, method='fft'):
    whitened = []
    for frame in data:
        whitened.append(whiten_frame(frame))
    whitened = np.array(whitened)
    return whitened


