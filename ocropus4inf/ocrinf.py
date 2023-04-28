import os
import os, random, shutil
import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy.ndimage as ndi
import torch
from torchmore2.ctc import ctc_decode
import urllib

from . import nlbin

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")

default_device = "?cuda:0" if torch.cuda.is_available() else "cpu"
default_device = os.environ.get("OCROPUS4_DEVICE", default_device)


class DefaultCharset:
    def __init__(self, chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"):
        if isinstance(chars, str):
            chars = list(chars)
        self.chars = [""] + chars
    def __len__(self):
        return len(self.chars)
    def encode_char(self, c):
        try:
            index = self.chars.index(c)
        except ValueError:
            index = len(self.chars)-1
        return max(index, 1)
    def encode(self, s):
        assert isinstance(s, str)
        return [self.encode_char(c) for c in s]
    def decode(self, l):
        assert isinstance(l, list)
        return "".join([self.chars[k] for k in l])        


class OnDevice:
    """Performs inference on device.

    The device string can be any valid device string for PyTorch.
    If it starts with a "?", the model is moved to the device before
    inference and moved back to the CPU afterwards.
    """

    def __init__(self, model, device):
        if device is None:
            self.device = device
            self.unload = False
        elif isinstance(device, str):
            if device.startswith("?mps"):
                device = device[1:]
            self.unload = device[0] == "?"
            self.device = device.strip("?")
        else:
            self.unload = False
            self.device = device
        self.model = model

    def __call__(self, inputs):
        if "cuda" not in self.device:
            print("warning: running on CPU")
        return self.model(inputs.to(self.device))

    def __enter__(self, *args):
        if self.device is not None:
            self.model = self.model.to(self.device)
        return self

    def __exit__(self, *args):
        if self.device is not None and self.unload:
            self.model = self.model.to("cpu")


def usm_filter(image):
    return image - ndi.gaussian_filter(image, 16.0)

class PageSegmenter:
    def __init__(self, murl, device=default_device):
        self.model = get_model(murl).to(device)
        self.device = device

    def inference(self, image):
        assert isinstance(image, np.ndarray)
        # print("segmenter:", np.amin(image), np.median(image), np.mean(image), np.amax(image))
        if image.ndim == 3:
            assert image.shape[2] in [1, 3]
            image = np.mean(image, axis=2)
        assert np.amin(image) >= 0
        assert np.amax(image) <= 1
        image = usm_filter(image)
        h, w = image.shape
        h32, w32 = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
        input = torch.zeros((h32, w32)).unsqueeze(0).unsqueeze(0)
        input[:, :, :h, :w] = torch.tensor(image)
        with OnDevice(self.model, self.device) as model:
            with torch.no_grad():
                output = model(input)
        probs = output.softmax(1)[0].cpu().permute(1, 2, 0).detach().numpy()
        return probs


def batch_images(images, maxheight=48.0):
    images = [torch.tensor(im) if not torch.is_tensor(im) else im for im in images]
    images = [im.unsqueeze(0) if im.ndim == 2 else im for im in images]
    d, h, w = map(max, zip(*[x.shape for x in images]))
    assert h <= maxheight, [im.shape for im in images]
    result = torch.zeros((len(images), d, h, w), dtype=torch.float32)
    for i, im in enumerate(images):
        d, h, w = im.shape
        if im.dtype == torch.uint8:
            im = im.float() / 255.0
        result[i, :d, :h, :w] = im
    return result


def make_ascii_charset():
    chars = [chr(i) for i in range(32, 127)]
    charset = DefaultCharset(chars)
    return charset


def scale_to_maxheight(image, maxheight=48.0):
    assert isinstance(image, np.ndarray)
    assert image.ndim == 2, image.shape
    h, w = image.shape
    scale = float(maxheight) / h
    if scale >= 1.0:
        return image
    return ndi.zoom(image, scale, order=1)


def get_model(url):
    # parse the path as a url
    scheme, netloc, path, params, query, fragment = urllib.parse.urlparse(url)
    # if the scheme is file or empty, then it is a local file
    if scheme in ["", "file"]:
        return load_model(path)
    elif scheme in ["http", "https"]:
        # download the file to $HOME/.cache/ocropus4
        cache = os.path.expanduser("~/.cache/ocropus4")
        os.makedirs(cache, exist_ok=True)
        fname = os.path.basename(path)
        local = os.path.join(cache, fname)
        if not os.path.exists(local):
            print("downloading", url, "to", local)
            with open(local, "wb") as stream:
                stream.write(requests.get(url).content)
        return load_model(local)
    elif scheme in ["gs"]:
        # download using the gsutil command line program
        cache = os.path.expanduser("~/.cache/ocropus4")
        os.makedirs(cache, exist_ok=True)
        fname = os.path.basename(path)
        local = os.path.join(cache, fname)
        if not os.path.exists(local):
            print("downloading", url, "to", local)
            os.system("gsutil cp {} {}".format(url, local))
        return load_model(local)
    else:
        raise Exception("unknown url scheme: " + url)

def load_model(path):
    print("loading model", path)
    if path.endswith(".jit"):
        import torch.jit

        return torch.jit.load(path)
    elif path.endswith(".pth"):
        import torch
        import ocrlib.ocrmodels as models

        mname = os.path.basename(path).split("-")[0]
        model = models.make(mname, device="cpu")
        mdict = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(mdict)
        return model
    else:
        raise Exception("unknown model type: " + path)

class WordRecognizer:
    def __init__(self, murl, charset=None, device=default_device, maxheight=48.0):
        charset = charset or make_ascii_charset()
        self.device = device
        self.charset = charset
        self.model = get_model(murl).to(device)
        self.maxheight = maxheight

    def inference(self, images):
        assert all(isinstance(im, np.ndarray) for im in images)
        images = [scale_to_maxheight(im, self.maxheight) for im in images]
        images = [usm_filter(im) for im in images]
        assert all(im.shape[0] <= self.maxheight for im in images)
        input = batch_images(images) # BDHW
        assert torch.is_tensor(input)
        with OnDevice(self.model, self.device) as model:
            with torch.no_grad():
                assert input.shape[-2] <= self.maxheight
                outputs = model(input)
        outputs = outputs.detach().cpu().softmax(1)
        seqs = [ctc_decode(pred.numpy()) for pred in outputs]
        texts = [self.charset.decode(seq) for seq in seqs]
        return texts


def show_seg(a, ax=None):
    ax = ax or plt.gca()
    ax.imshow(np.where(a == 0, 0, 0.3 + np.abs(np.sin(a))), cmap="gnuplot")


def compute_segmentation(probs, show=True):
    word_markers = probs[:, :, 3] > 0.5
    word_markers = ndi.minimum_filter(ndi.maximum_filter(word_markers, (1, 3)), (1, 3))
    # plt.imshow(word_markers)

    word_labels, n = ndi.label(word_markers)

    _, sources = ndi.distance_transform_edt(1 - word_markers, return_indices=True)
    word_sources = word_labels[sources[0], sources[1]]
    # show_seg(word_sources)

    word_boundaries = np.maximum(
        (np.roll(word_sources, 1, 0) != word_sources),
        np.roll(word_sources, 1, 1) != word_sources,
    )
    word_boundaries = ndi.minimum_filter(ndi.maximum_filter(word_boundaries, 4), 2)
    # plt.imshow(word_boundaries)

    # separators = maximum(probs[:,:,1]>0.3, word_boundaries)
    separators = np.maximum(probs[:, :, 1] > 0.3, probs[:, :, 0] > 0.5, word_boundaries)
    # plt.imshow(separators)
    all_components, n = ndi.label(1 - separators)
    # show_seg(all_components)

    # word_markers = (probs[:,:,3] > 0.5) * (1-separators)
    word_markers = (np.maximum(probs[:, :, 2], probs[:, :, 3]) > 0.5) * (1 - separators)
    word_markers = ndi.minimum_filter(ndi.maximum_filter(word_markers, (1, 3)), (1, 3))
    word_labels, n = ndi.label(word_markers)
    # show_seg(word_labels)

    correspondence = 1000000 * word_labels + all_components
    nwords = np.amax(word_sources) + 1
    ncomponents = np.amax(all_components) + 1

    wordmap = np.zeros(ncomponents, dtype=int)
    for word, comp in [
        (k // 1000000, k % 1000000) for k in np.unique(correspondence.ravel())
    ]:
        if comp == 0:
            continue
        if word == 0:
            continue
        if wordmap[comp] > 0:
            # FIXME do something about ambiguous assignments
            # print(word, comp)
            pass
        wordmap[comp] = word

    result = wordmap[all_components]
    return locals()


def compute_slices(wordmap):
    for s in ndi.find_objects(wordmap):
        if s is not None:
            yield s


def compute_bboxes(wordmap, pad=10, padr=0):
    if isinstance(pad, int):
        pad = (pad, pad, pad, pad)
    if isinstance(padr, (int, float)):
        padr = (padr, padr, padr, padr)
    for ys, xs in compute_slices(wordmap):
        h, w = ys.stop - ys.start, xs.stop - xs.start
        yield dict(
            t=max(ys.start - max(pad[0], int(padr[0] * h)), 0),
            l=max(xs.start - max(pad[1], int(padr[1] * h)), 0),
            b=ys.stop + max(pad[2], int(padr[2] * h)),
            r=xs.stop + max(pad[3], int(padr[3] * h)),
        )


# bboxes = list(compute_bboxes(probs, pad=10))

import matplotlib.patches as patches


def draw_bboxes(boxes, ax=None):
    ax = ax or plt.gca()
    for box in boxes:
        t, l, b, r = [box[c] for c in "tlbr"]
        ax.add_patch(
            patches.Rectangle(
                (l, t), r - l, b - t, linewidth=1, edgecolor="r", facecolor="none"
            )
        )


def show_extracts(image, bboxes, nrows=4, ncols=3):
    # from IPython.display import display
    fig, axs = plt.subplots(nrows, ncols)
    axs = axs.ravel()

    for i in range(nrows * ncols):
        if i >= len(bboxes):
            break
        t, l, b, r = [bboxes[i][c] for c in "tlbr"]
        axs[i].imshow(image[t:b, l:r])
        # axs[i].set_xticks([])
        # axs[i].set_yticks([])
        # axs[i].axis()

    # display(fig)


def download_file(url, filename, overwrite=False):
    if os.path.exists(filename) and not overwrite:
        return filename
    assert 0 == os.system("curl -L -o '{}' '{}'".format(filename, url))
    return
    print(f"Downloading {url} to {filename}")
    with requests.get(url, stream=True) as r:
        with open(filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return filename


cache_dir = os.environ.get("OCROPUS4_CACHE", None) or os.path.expanduser(
    "~/.cache/ocropus4"
)
model_bucket = "http://storage.googleapis.com/ocro-models/v1/"
default_textmodel = model_bucket + "lstm_resnet_f-007-000198062.pth"
default_segmodel = model_bucket + "seg_unet2f-102-000215483.pth"


def autoinvert(image):
    if image.shape[0] < 2 or image.shape[1] < 2:
        return image
    middle = (np.amax(image) + np.amin(image)) / 2
    if np.mean(image) > middle:
        return 1 - image
    else:
        return image


class PageRecognizer:
    def __init__(self, segmodel=None, textmodel=None, device=default_device):
        self.device = device
        os.makedirs(cache_dir, exist_ok=True)
        segmodel = segmodel or default_segmodel
        textmodel = textmodel or default_textmodel
        self.segmenter = PageSegmenter(segmodel, device=device)
        self.textmodel = WordRecognizer(textmodel, device=device)
        self.words_per_batch = 64

    def to(self, device):
        self.device = device

    def valid_binary_image(self, binarized, verbose=False):
        h, w = binarized.shape
        if h < 10 and w < 10:
            return False
        if h > 200:
            return False
        if not (np.amin(binarized) < 0.1 and np.amax(binarized) > 0.9):
            return False
        if False:  # FIXME
            if np.mean(binarized) > 0.5:
                binarized = 1 - binarized
            if np.sum(binarized > 0.9) < 0.05 * h * w:
                if verbose:
                    print("insufficient white")
                return False
        return True

    def recognize(self, image, keep_images=False, preproc="none"):
        self.image = image
        self.bin = nlbin.nlbin(image, deskew=False)
        if preproc == "none":
            srcimg = self.image
        elif preproc == "binarize":
            srcimg = self.bin
        elif preproc == "threshold":
            srcimg = (self.bin > 0.5).astype(np.float32)
        else:
            raise ValueError("preproc must be one of none, binarize, threshold")
        self.srcimg = srcimg
        self.seg_probs = self.segmenter.inference(srcimg)
        self.segmentation = compute_segmentation(self.seg_probs)
        self.wordmap = self.segmentation["result"]
        self.bboxes = list(compute_bboxes(self.wordmap))
        for i in range(len(self.bboxes)):
            t, l, b, r = [self.bboxes[i][c] for c in "tlbr"]
            box = self.bboxes[i]
            box["image"] = autoinvert(srcimg[t:b, l:r])
            box["binarized"] = autoinvert(self.bin[t:b, l:r])
        self.bboxes = bboxes = [
            b for b in self.bboxes if self.valid_binary_image(b["binarized"])
        ]
        for i in range(0, len(self.bboxes), self.words_per_batch):
            bboxes = self.bboxes[i : i + self.words_per_batch]
            images = [b["image"] for b in self.bboxes[i : i + self.words_per_batch]]
            pred = self.textmodel.inference(images)
            assert len(pred) == len(bboxes)
            for i in range(len(bboxes)):
                bboxes[i]["text"] = pred[i]
        if not keep_images:
            for b in self.bboxes:
                del b["image"]
                del b["binarized"]
        return self.bboxes

    def draw_overlaid(
        self, fontsize=6, offset=(5, 10), color="red", rcolor="red", alpha=0.25, ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(self.srcimg)
        for i in range(len(self.bboxes)):
            box = self.bboxes[i]
            text = box["text"]
            t, l, b, r = [box[c] for c in "tlbr"]
            ax.text(l + offset[0], t + offset[1], text, fontsize=fontsize, color=color)
            # draw a rectangle around the word
            ax.add_patch(
                patches.Rectangle(
                    (l, t),
                    r - l,
                    b - t,
                    linewidth=1,
                    edgecolor=rcolor,
                    facecolor="none",
                    alpha=alpha,
                )
            )

    def draw_words(self, nrows=6, ncols=4, ax=None):
        bboxes = list(self.bboxes)
        random.shuffle(bboxes)
        n = min(nrows * ncols, len(bboxes))
        for i in range(n):
            box = bboxes[i]
            image = box["image"]
            text = box["text"]
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(1 - image.numpy()[0])
            plt.xticks([])
            plt.yticks([])
            h, w = image.shape[-2:]
            plt.gca().text(5, 12, text, fontsize=9, color="red")
            # title(pred[i], fontsize=6)
