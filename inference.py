import argparse
import os
import warnings
import cv2
import numpy as np

import numpy
import torch
import tqdm
import yaml
from torch.utils import data

from tools import utils
from data.dataset import Dataset

cwd = os.getcwd()
names = "data\yolo_annotations\small.txt"
img_path= "data\images\small"
weight_path = "weights\p2"
conf_thres = 0.05      # 0.001 default 
iou_thres = 0.1        # 0.01 default 

warnings.filterwarnings("ignore")

names = os.path.join(cwd, names)
img_path = os.path.join(cwd, img_path)
weight_path = os.path.join(cwd, weight_path)

@torch.no_grad()
def inference(args, params, weight_path, model=None):

    save_dir = os.path.join(weight_path, 'visualization',  f'conf_t_{conf_thres}_iou_t_{iou_thres}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filenames = []

    if args.test :
        save_dir = os.path.join(weight_path, 'visualization',  f'conf_t_{conf_thres}_iou_t_{iou_thres}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    with open(names) as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append(os.path.join(img_path, filename))

    dataset = Dataset(filenames, args.input_size, params, False)
    loader = data.DataLoader(dataset, 1, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)


    model = torch.load(f'{weight_path}/best.pt', map_location='cuda')['model'].float()
    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))

    video_path = os.path.join(save_dir, 'inference_result.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5

    video_writer = None 

    for samples, targets, shapes, org_img, filename in p_bar:
        samples = samples.cuda()
        targets = targets.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        _, _, height, width = samples.shape  # batch size, channels, height, width

        # Inference
        outputs = model(samples)

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
        outputs = utils.non_max_suppression(outputs, conf_threshold=params['conf_threshold'], iou_threshold=params['iou_threshold'])

        # Metrics
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:]
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if labels.shape[0]:
                    metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                continue

            detections = output.clone()
            utils.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

            # Evaluate
            if labels.shape[0]:
                tbox = labels[:, 1:5].clone()  # target boxes
                tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                utils.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct = numpy.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = utils.box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]
                for j in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                        matches = matches.cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
            metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

            frame = vis_data(
                org_img[0].copy(), 
                tbox,
                labels,
                detections,
                save_dir, 
                filename[0].split('\\')[-1]
            )

            if video_writer is None:
                h, w, _ = frame.shape 
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            
            video_writer.write(frame)
    if video_writer is not None:
        video_writer.release()

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = utils.compute_ap(*metrics)

    # Print results
    print('%10.3g' * 3 % (m_pre, m_rec, mean_ap))

    # Return results
    model.float()  # for training

    print(f'\nresult saved to : {save_dir}\n')

    return map50, mean_ap

def draw_label_with_bg(
    img,
    text,
    x,
    y,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.4,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness=1,
    padding=3
):
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # background rectangle
    cv2.rectangle(
        img,
        (x, y - th - baseline - padding * 2),
        (x + tw + padding * 2, y),
        bg_color,
        -1
    )

    # text
    cv2.putText(
        img,
        text,
        (x + padding, y - baseline - padding),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

def draw_label_with_bg_alpha(
    img, text, x, y,
    bg_color=(0, 0, 0),
    alpha=0.6,
    **kwargs
):
    overlay = img.copy()
    draw_label_with_bg(overlay, text, x, y, bg_color=bg_color, **kwargs)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def vis_data(img, tbox, labels, detections, weight_path, filename, show=True):
    org_img = img.copy()
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2BGR)

    if labels.shape[0] :
        for gt in tbox: 
            x1, y1, x2, y2 = map(int, gt.tolist())
            cv2.rectangle(org_img, (x1, y1), (x2, y2), (255, 0, 0), 1)

            draw_label_with_bg(
                org_img,
                'GT',
                x1,
                max(y1, 15),
                text_color=(255, 255, 255),
                bg_color=(255, 0, 0)
            )
        
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        draw_label_with_bg_alpha(
            img,
            f'small:{conf:.2f}',
            x1,
            max(y1, 15),
            bg_color=(0, 0, 255),
            alpha=0.6
        )
    out = np.concatenate([org_img, img], 1)

    # ===== 실시간 출력 =====
    if show:
        cv2.imshow('GT (left) | Prediction (right)', out)
        key = cv2.waitKey(100) & 0xFF  # 1ms 대기
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt  # 즉시 중단

    save_path = os.path.join(weight_path, filename)
    cv2.imwrite(save_path, out)

    return out


def save_video(img, tbox, labels, detections):
    org_img = img.copy()

    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2BGR)

    # GT
    if labels.shape[0]:
        for gt in tbox:
            x1, y1, x2, y2 = map(int, gt.tolist())
            cv2.rectangle(org_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            draw_label_with_bg(
                org_img,
                'GT',
                x1,
                max(y1, 15),
                text_color=(255, 255, 255),
                bg_color=(255, 0, 0)
            )

    # Detection
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        draw_label_with_bg_alpha(
            img,
            f'defect:{conf:.2f}',
            x1,
            max(y1, 15),
            bg_color=(0, 0, 255),
            alpha=0.6
        )
    out = np.concatenate([org_img, img], axis=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    utils.setup_seed()
    utils.setup_multi_processes()

    with open(r'./tools/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    inference(args, params, weight_path)


if __name__ == "__main__":
    main()
