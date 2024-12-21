import argparse
import os
import time
import cv2
import numpy as np
import pandas as pd
import grabcut as GB

def parse():
    parser = GB.argparse.ArgumentParser()
    parser.add_argument('--eval', type=int, default=1, help='Calculate the metrics')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='Change the rect (x,y,w,h)')
    parser.add_argument('--output_folder', type=str, default='output', help='Folder to save the results')
    return parser.parse_args()


def test_batch():
    args = parse()
    os.makedirs(args.output_folder, exist_ok=True)
    metrics = []

    img_dir = 'data/imgs'
    for img_name in os.listdir(img_dir):
        if img_name.endswith(('.jpg', '.png', '.bmp')):
            input_path = os.path.join(img_dir, img_name)
            base_name = os.path.splitext(img_name)[0]

            if args.use_file_rect:
                rect_path = f"data/bboxes/{base_name}.txt"
                if not os.path.exists(rect_path):
                    print(f"Bounding box file not found for {img_name}. Skipping...")
                    continue
                rect = tuple(map(int, open(rect_path, "r").read().split(' ')))
            else:
                rect = tuple(map(int, args.rect.split(',')))

            img = cv2.imread(input_path)
            if img is None:
                print(f"Failed to read {img_name}. Skipping...")
                continue

            print(f"Running GrabCut on {img_name}")

            start_time = time.time()
            mask, bgGMM, fgGMM = GB.grabcut(img, rect)
            elapsed_time = time.time() - start_time

            mask_binary = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
            img_cut = img * (mask_binary[:, :, np.newaxis])

            cv2.imwrite(os.path.join(args.output_folder, f"{base_name}_img.jpg"), img)
            cv2.imwrite(os.path.join(args.output_folder, f"{base_name}_mask.jpg"), 255 * mask_binary)
            cv2.imwrite(os.path.join(args.output_folder, f"{base_name}_result.jpg"), img_cut)

            if args.eval:
                gt_mask_path = f"data/seg_GT/{base_name}.bmp"
                if os.path.exists(gt_mask_path):
                    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                    gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
                    acc, jac = GB.cal_metric(mask_binary, gt_mask)
                    metrics.append({
                        'Image': img_name,
                        'Accuracy': acc,
                        'Jaccard': jac,
                        'Time': elapsed_time
                    })
                    print(f"{img_name}: Accuracy={acc:.4f}, Jaccard={jac:.4f}, Time={elapsed_time:.2f}s")
                else:
                    print(f"Ground truth mask not found for {img_name}. Skipping metrics...")

    if metrics:
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(args.output_folder, 'metrics.csv'), index=False)
        print(f"Metrics saved to {os.path.join(args.output_folder, 'metrics.csv')}")

def test_blur(img_list):
    args = parse()
    output_folder = 'output_blur'
    os.makedirs(output_folder, exist_ok=True)
    blur_levels = {'no': None, 'low': 5, 'high': 15}
    metrics = []

    for img_name in img_list:
        input_path = os.path.join('data/imgs', img_name)
        base_name = os.path.splitext(img_name)[0]

        if args.use_file_rect:
            rect_path = f"data/bboxes/{base_name}.txt"
            if not os.path.exists(rect_path):
                print(f"Bounding box file not found for {img_name}. Skipping...")
                continue
            rect = tuple(map(int, open(rect_path, "r").read().split(' ')))
        else:
            rect = tuple(map(int, args.rect.split(',')))

        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to read {img_name}. Skipping...")
            continue

        for blur_level, ksize in blur_levels.items():
            if ksize:
                blurred_img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            else:
                blurred_img = img

            start_time = time.time()
            mask, bgGMM, fgGMM = GB.grabcut(blurred_img, rect)
            elapsed_time = time.time() - start_time

            mask_binary = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
            img_cut = blurred_img * (mask_binary[:, :, np.newaxis])

            cv2.imwrite(os.path.join(output_folder, f"{base_name}_img_{blur_level}.jpg"), blurred_img)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask_{blur_level}.jpg"), 255 * mask_binary)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_result_{blur_level}.jpg"), img_cut)

            if args.eval:
                gt_mask_path = f"data/seg_GT/{base_name}.bmp"
                if os.path.exists(gt_mask_path):
                    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                    gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
                    acc, jac = GB.cal_metric(mask_binary, gt_mask)
                    metrics.append({
                        'Image': img_name,
                        'Blur_Level': blur_level,
                        'Accuracy': acc,
                        'Jaccard': jac,
                        'Time': elapsed_time
                    })
                    print(f"{img_name} [{blur_level}]: Accuracy={acc:.4f}, Jaccard={jac:.4f}, Time={elapsed_time:.2f}s")
                else:
                    print(f"Ground truth mask not found for {img_name}. Skipping metrics...")

    if metrics:
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_folder, 'metrics_blur.csv'), index=False)
        print(f"Metrics saved to {os.path.join(output_folder, 'metrics_blur.csv')}")


def test_n_components(img_list, n_list):
    args = parse()
    output_folder = 'output_n_components'
    os.makedirs(output_folder, exist_ok=True)
    metrics = []
    img_dir = 'data/imgs'
    for img_name in img_list:
        input_path = os.path.join(img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]

        # Use bounding box from file or argument
        if args.use_file_rect:
            rect_path = f"data/bboxes/{base_name}.txt"
            if not os.path.exists(rect_path):
                print(f"Bounding box file not found for {img_name}. Skipping...")
                continue
            rect = tuple(map(int, open(rect_path, "r").read().split(' ')))
        else:
            rect = tuple(map(int, args.rect.split(',')))

        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to read {img_name}. Skipping...")
            continue

        print(f"Running GrabCut on {img_name}")

        # Run GrabCut algorithm
        for n in n_list:
            start_time = time.time()
            mask, bgGMM, fgGMM = GB.grabcut(img, rect, n)
            elapsed_time = time.time() - start_time

            # Save the mask and segmented image
            mask_binary = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
            img_cut = img * (mask_binary[:, :, np.newaxis])

            # Save images
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_img_n{n}.jpg"), img)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask_n{n}.jpg"), 255 * mask_binary)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_result_n{n}.jpg"), img_cut)

            # Calculate and save metrics
            if args.eval:
                gt_mask_path = f"data/seg_GT/{base_name}.bmp"
                if os.path.exists(gt_mask_path):
                    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                    gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
                    acc, jac = GB.cal_metric(mask_binary, gt_mask)
                    metrics.append({
                        'Image': img_name,
                        'n_components': n,
                        'Accuracy': acc,
                        'Jaccard': jac,
                        'Time': elapsed_time
                    })
                    print(f"{img_name}: Accuracy={acc:.4f}, Jaccard={jac:.4f}, Time={elapsed_time:.2f}s")
                else:
                    print(f"Ground truth mask not found for {img_name}. Skipping metrics...")

        # Save all metrics to a CSV file
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_folder, 'metrics_n_components.csv'), index=False)
        print(f"Metrics saved to {os.path.join(output_folder, 'metrics_n_components.csv')}")


def test_rect(rect_dict):
    args = parse()
    output_folder = 'output_rect'
    os.makedirs(output_folder, exist_ok=True)
    metrics = []

    for img_name, custom_rect in rect_dict.items():
        input_path = os.path.join('data/imgs', img_name)
        base_name = os.path.splitext(img_name)[0]

        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to read {img_name}. Skipping...")
            continue

        # Define the rectangles
        if args.use_file_rect:
            rect_path = f"data/bboxes/{base_name}.txt"
            if not os.path.exists(rect_path):
                print(f"Bounding box file not found for {img_name}. Skipping medium rectangle...")
                default_rect = None
            else:
                default_rect = tuple(map(int, open(rect_path, "r").read().split(' ')))
        else:
            default_rect = tuple(map(int, args.rect.split(',')))

        large_rect = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

        rects = {
            'custom': custom_rect,
            'default': default_rect,
            'large': large_rect
        }

        for rect_size, rect in rects.items():
            if rect is None:
                continue
            print(f"Running GrabCut on {img_name} with rect = {rect}, rect_size = {rect_size}")
            start_time = time.time()
            mask, bgGMM, fgGMM = GB.grabcut(img, rect)
            elapsed_time = time.time() - start_time

            mask_binary = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
            img_cut = img * (mask_binary[:, :, np.newaxis])

            cv2.imwrite(os.path.join(output_folder, f"{base_name}_img_{rect_size}.jpg"), img)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask_{rect_size}.jpg"), 255 * mask_binary)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_result_{rect_size}.jpg"), img_cut)

            if args.eval:
                gt_mask_path = f"data/seg_GT/{base_name}.bmp"
                if os.path.exists(gt_mask_path):
                    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                    gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
                    acc, jac = GB.cal_metric(mask_binary, gt_mask)
                    metrics.append({
                        'Image': img_name,
                        'Rect_Size': rect_size,
                        'Accuracy': acc,
                        'Jaccard': jac,
                        'Time': elapsed_time
                    })
                    print(f"{img_name} [{rect_size}]: Accuracy={acc:.4f}, Jaccard={jac:.4f}, Time={elapsed_time:.2f}s")
                else:
                    print(f"Ground truth mask not found for {img_name}. Skipping metrics...")

    if metrics:
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_folder, 'metrics_rect.csv'), index=False)
        print(f"Metrics saved to {os.path.join(output_folder, 'metrics_rect.csv')}")


if __name__ == '__main__':
    # test_batch()
    test_n_components(['banana1.jpg', 'cross.jpg', 'bush.jpg', 'fullmoon.jpg'], [1, 3, 5])
    # test_blur(['teddy.jpg', 'llama.jpg', 'flower.jpg'])
    # rect_dict = {'sheep.jpg': (100, 100, 400+100, 320+100), 'flower.jpg': (100, 50, 450+100, 380+50)}
    # test_rect(rect_dict)
