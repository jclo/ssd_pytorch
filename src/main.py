# ******************************************************************************
"""
Detects objects inside a video or an image.


Private Functions:
    . _parse                    parses the script arguments,
    . _create_net               creates the SSD neural network,


Public Functions:
    . detect                    detects the objects in a frame,
    . process_image             detects objects in an image,
    . process_video             detects objects in a video,


@namespace      _
@author         -
@since          0.0.0
@version        0.0.0
@licence        MIT. Copyright (c) 2020 Mobilabs <contact@mobilabs.fr>
"""
# ******************************************************************************
import argparse
import imageio
import torch
import cv2

from models.ssd import build_ssd
from data import BaseTransform, VOC_CLASSES as labelmap

WEIGHTS = './weights/ssd300_mAP_77.43_v2.pth'
THRESHOLD = 0.6
RECT_COLOR = (255, 255, 0)
RECT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_COLOR = (0, 0, 0)
FONT_SIZE = 0.5
FONT_THICKNESS = 1


# -- Private Functions ---------------------------------------------------------

def _parse():
    """Parses the script arguments.

    ### Parameters:
        param1 ():          none,

    ### Returns:
        (str):              returns the option values,

    ### Raises:
        none
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, help='the path of the passed-in image or video')
    parser.add_argument('-t', '--type', type=str, help='image or video')
    parser.add_argument('-o', '--output', type=str, help='the output path')
    args = parser.parse_args()
    return args.filename, args.type, args.output


def _create_net():
    """Creates the SSD neural network.

    ### Parameters:
        -none,

    ### Returns:
        (obj):              returns the neural network,
        (fct):              returns the transformation function,

    ### Raises:
        none
    """
    # Create the SSD neural network and the transformation
    net = build_ssd('test')
    net.load_state_dict(torch.load(WEIGHTS, map_location=lambda storage, loc: storage))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
    return net, transform


# -- Public Functions ----------------------------------------------------------

def detect(net, transform, frame):
    """Detects the objects in a frame.

    ### Parameters:
        param1 (obj):       the SDD neural network,
        param2 (fct):       the transformation function,
        param3 (arr):       the input image,

    ### Returns:
        (arr):              returns the labeled image,

    ### Raises:
        none
    """
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    # PyTorch 1.5.1
    # x = Variable(x.unsqueeze(0))
    # y = net(x)
    x = x.unsqueeze(0)
    with torch.no_grad():
        y = net(x)
    detections = y.data

    scale = torch.Tensor([width, height, width, height])
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= THRESHOLD:
            score = detections[0, i, j, 0]
            text = '%s: %.2f' % (labelmap[i - 1], score)
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])),
                          RECT_COLOR, RECT_THICKNESS)

            # add the text with a background
            (text_width, text_height) = cv2.getTextSize(
                text, FONT, fontScale=FONT_SIZE, thickness=FONT_THICKNESS)[0]
            x = int(pt[0])
            y = int(pt[1])
            box_coords = ((x, y), (x + text_width + 2, y - text_height - 8))
            cv2.rectangle(frame, box_coords[0], box_coords[1], RECT_COLOR, cv2.FILLED)
            cv2.putText(frame, text, (int(pt[0] + 2), int(pt[1]) - 6), FONT,
                        FONT_SIZE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            j += 1
    return frame


def process_image(image):
    """Detects objects in an image.

    ### Parameters:
        param1 (arr):       the input image,

    ### Returns:
        (arr):              returns the annotated image,

    ### Raises:
        none
    """
    # Create the SSD neural network and the transformation
    net, transform = _create_net()

    # process the image through the neural network and
    # return the tagged image
    out_image = detect(net.eval(), transform, image)
    return out_image


def process_video(video, output):
    """Detects objects in a video.

    ### Parameters:
        param1 (str):       the input video,
        param2 (str):       the output video,

    ### Returns:
        none

    ### Raises:
        none
    """
    # Create the SSD neural network and the transformation
    net, transform = _create_net()

    # Process the video frame by frame
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(output, fps=fps)
    for i, frame in enumerate(reader):
        frame = detect(net.eval(), transform, frame)
        writer.append_data(frame)
        print(i)
    writer.close()


if __name__ == '__main__':
    filename, type, output = _parse()
    if type != 'image' and type != 'video':
        print('The input must be an image or a video!')
        print('Aborted...')
        exit()

    if type == 'image':
        imageio.imwrite(output, process_image(imageio.imread(filename)))
    else:
        process_video(filename, output)


# -- o ---
