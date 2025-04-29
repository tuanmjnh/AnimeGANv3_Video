from os import path # makedirs
from cv2 import resize, cvtColor, imread, imwrite, COLOR_BGR2RGB, COLOR_RGB2BGR
import time
from glob import glob
from numpy import squeeze, clip, expand_dims, float32, uint8
from onnxruntime import get_device, InferenceSession
import threading

pic_form = [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]


# def checkFolder(path):
#     if not path.exists(path):
#         makedirs(path)
#     return path

# select style model


def getStyleModel(resources_path):
    styles = ["AnimeGANv3_Hayao_36", "animeganv3_H40_model", "animeganv3_H64_model0", "AnimeGANv3_JP_face_v1.0",
              "AnimeGANv3_PortraitSketch_25", "AnimeGANv3_Shinkai_37", "AnimeGANv3_Shinkai_40", "AnimeGANv3_tiny_Cute"]
    # select style model
    print(f"Available styles to set: (default: {styles[0]})")
    for idx, key in enumerate(styles, start=1):
        print(f"{idx}. {key}")

    selections = input("Enter the numbers of the styles you want to set (e.g. 1 8): ")
    indexes = [int(i) for i in selections.split() if i.isdigit()]

    if (len(selections) > 0 and int(selections[0]) > 0 and int(selections[0]) <= len(styles)):
        for i in indexes:
            if 1 <= i <= len(styles):
                key = styles[i - 1]
                value = f"{resources_path}\{key}.onnx"
                # print(f"Selected style model: {value}")
                return value
    else:
        print(f"Selected style model default: {styles[0]}")
        return f"{resources_path}\{styles[0]}.onnx"


def processImage(img, model_name):
    h, w = img.shape[:2]
    # resize image to multiple of 8s

    def to_8s(x):
        # If using the tiny model, the multiple should be 16 instead of 8.
        if "tiny" in path.basename(model_name):
            return 256 if x < 256 else x - x % 16
        else:
            return 256 if x < 256 else x - x % 8
    img = resize(img, (to_8s(w), to_8s(h)))
    img = cvtColor(img, COLOR_BGR2RGB).astype(float32) / 127.5 - 1.0
    return img


def loadData(image_path, model_name):
    img0 = imread(image_path).astype(float32)
    img = processImage(img0, model_name)
    img = expand_dims(img, axis=0)
    return img, img0.shape


def saveImages(images, image_path, size):
    images = (squeeze(images) + 1.) / 2 * 255
    images = clip(images, 0, 255).astype(uint8)
    images = resize(images, size)
    imwrite(image_path, cvtColor(images, COLOR_RGB2BGR))


def convertImages(input_imgs_path, output_path, onnx="model.onnx", device="cpu"):
    # checkFolder(output_path)
    test_files = glob("{}/*.*".format(input_imgs_path))
    test_files = [x for x in test_files if path.splitext(x)[-1] in pic_form]
    if get_device() == "GPU" and device == "gpu":
        session = InferenceSession(onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    else:
        session = InferenceSession(onnx, providers=["CPUExecutionProvider"])
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name

    begin = time.time()
    for i, sample_file in enumerate(test_files):
        t = time.time()
        sample_image, shape = loadData(sample_file, onnx)
        image_path = path.join(output_path, "{0}".format(path.basename(sample_file)))
        fake_img = session.run(None, {x: sample_image})
        saveImages(fake_img[0], image_path, (shape[1], shape[0]))
        print(f"Processing image: {i}, image size: {shape[1], shape[0]}, " + sample_file, f" time: {time.time() - t:.3f} s")
    end = time.time()
    print(f"Average time per image : {(end-begin)/len(test_files)} s")


def RunConvertImages(frame_dir, style_dir, selected_style, device):
    process = threading.Thread(target=convertImages, args=(frame_dir, style_dir, selected_style, device))
    process.start()
    return process
