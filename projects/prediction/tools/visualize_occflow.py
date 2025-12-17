import argparse
import glob
import gzip
import os
import pickle

import cv2
import numpy as np
import pickle5
import torch
import tqdm
from PIL import Image

from hat.data.datasets.occflow_dataset import WaypointGrids
from hat.visualize.occflow import flow_rgb_image, occupancy_rgb_image


def gload(filename, use_pickle5=False):
    file = gzip.GzipFile(filename, "rb")
    if use_pickle5:
        res = pickle5.load(file)
    else:
        res = pickle.load(file)
    file.close()
    return res


def save_images(path, occs, flows, one_row=False):
    """Save occupancy and flow result as image."""
    for i in range(len(occs)):
        occs[i] = (occs[i][0] * 255).astype(np.uint8).transpose([1, 2, 0])
        H, W, C = occs[i].shape
        occs[i][0] = np.zeros((W, C))
        occs[i][H - 1] = np.zeros((W, C))
        occs[i][:, 0] = np.zeros((H, C))
        occs[i][:, W - 1] = np.zeros((H, C))
    if one_row:
        occ_im = np.hstack(occs)
    else:
        upper_im = np.hstack(occs[:4])
        lower_im = np.hstack(occs[4:])
        occ_im = np.vstack([upper_im, lower_im])

    for i in range(len(flows)):
        flows[i] = (flows[i][0] * 255).astype(np.uint8).transpose([1, 2, 0])
        H, W, C = flows[i].shape
        flows[i][0] = np.zeros((W, C))
        flows[i][H - 1] = np.zeros((W, C))
        flows[i][:, 0] = np.zeros((H, C))
        flows[i][:, W - 1] = np.zeros((H, C))
    if one_row:
        flows_im = np.hstack(flows)
    else:
        upper_im = np.hstack(flows[:4])
        lower_im = np.hstack(flows[4:])
        flows_im = np.vstack([upper_im, lower_im])

    im = np.vstack([occ_im, flows_im])
    cv2.imwrite(path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def save_gif(path, occs, flows):
    """Save occupancy and flow result as gif."""
    frames = []
    for i in range(len(occs)):
        occs[i] = (occs[i][0] * 255).astype(np.uint8).transpose([1, 2, 0])
        flows[i] = (flows[i][0] * 255).astype(np.uint8).transpose([1, 2, 0])
        H, W, C = occs[i].shape
        occs[i][0] = np.zeros((W, C))
        occs[i][H - 1] = np.zeros((W, C))
        occs[i][:, 0] = np.zeros((H, C))
        occs[i][:, W - 1] = np.zeros((H, C))
        flows[i][0] = np.zeros((W, C))
        flows[i][H - 1] = np.zeros((W, C))
        flows[i][:, 0] = np.zeros((H, C))
        flows[i][:, W - 1] = np.zeros((H, C))
        frame_numpy = np.hstack([occs[i], flows[i]])
        frames.append(Image.fromarray(frame_numpy))
    frame_one = frames[0]
    frame_one.save(
        path,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=500,
        loop=0,
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def convert_to_waypoints(predictions, quantized=False, thres=None):
    """Convert prediction dict to waypoints format."""
    occupancy_numpy = sigmoid(predictions["occ_preds"]).transpose([1, 2, 0])
    flow_numpy = predictions["flow_preds"].transpose([1, 2, 0])
    if quantized:
        occupancy_numpy = np.round(occupancy_numpy * 255) / 255
        flow_numpy = np.clip(np.round(flow_numpy), -128, 127)

    H, W, C = occupancy_numpy.shape
    occupancy_numpy = occupancy_numpy.reshape((H, W, 2, 8))
    if thres is not None:
        occupancy_numpy[occupancy_numpy <= thres] = 0
    flow_numpy = flow_numpy.reshape((H, W, 2, 8))
    occupancy_numpy = np.expand_dims(occupancy_numpy, 0)
    flow_numpy = np.expand_dims(flow_numpy, 0)
    occupancy_numpy = torch.from_numpy(occupancy_numpy)
    flow_numpy = torch.from_numpy(flow_numpy)
    waypoint_logits = WaypointGrids()

    # Slice channels into output predictions.
    class_name = "vehicles"
    for k in range(8):
        observed_occupancy = occupancy_numpy[:, :, :, 0:1, k]
        occluded_occupancy = occupancy_numpy[:, :, :, 1:2, k]
        pred_flow = flow_numpy[:, :, :, :, k]
        getattr(waypoint_logits, class_name).observed_occupancy.append(
            observed_occupancy
        )
        getattr(waypoint_logits, class_name).occluded_occupancy.append(
            occluded_occupancy
        )
        getattr(waypoint_logits, class_name).flow.append(pred_flow)
    return waypoint_logits


def visualize(pred_data, name, gt_data=None, gif=False, occluded=False):
    """Visualize prediction results.

    If ground_truth is provided, also draw road map on the result.
    """
    _, H, W = pred_data["occ_preds"].shape

    start = 0.4
    step = -0.02
    end = start + 7 * step
    thres = np.arange(start, end, step)

    pred_waypoints = convert_to_waypoints(
        pred_data, quantized=args.quantize, thres=thres
    )
    occs = []
    flows = []

    if gt_data:
        roadgraph = torch.from_numpy(
            np.expand_dims(gt_data["output_vis_grids.roadgraph"], 0)
        )
        agent_trails = torch.from_numpy(
            np.expand_dims(gt_data["output_vis_grids.agent_trails"], 0)
        )
    else:
        roadgraph = torch.zeros((1, H, W, 1))
        agent_trails = torch.zeros((1, H, W, 1))

    for k in range(8):
        if occluded:
            occupancy_grids = (
                pred_waypoints.get_observed_occluded_veh_occupancy_at_waypoint(
                    k
                )
            )
        else:
            occupancy_grids = (
                pred_waypoints.get_observed_occupancy_at_waypoint(k)
            )
        occupancy_rgb = (
            occupancy_rgb_image(
                agent_grids=occupancy_grids,
                roadgraph_image=roadgraph,
            )
            .numpy()
            .transpose([0, 3, 1, 2])
        )
        occs.append(occupancy_rgb)

        flow_rgb = (
            flow_rgb_image(
                flow=pred_waypoints.vehicles.flow[k],
                roadgraph_image=roadgraph,
                agent_trails=agent_trails,
            )
            .numpy()
            .transpose([0, 3, 1, 2])
        )
        flows.append(flow_rgb)
    if gif:
        save_gif(os.path.join(output_path, f"{name}.gif"), occs, flows)
    else:
        save_images(os.path.join(output_path, f"{name}.jpg"), occs, flows)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dump_path", help="Dumped predictions path", required=True
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Directory to save visualization",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--reference_path",
        default=None,
        help="Only visualize scenarios present in reference path",
    )
    parser.add_argument(
        "-g", "--gt_path", default=None, help="Visualize with gt road info"
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        default=False,
        help="Whether quantize the predictions",
    )
    parser.add_argument(
        "--occluded",
        action="store_true",
        default=False,
        help="Whether to visualize occluded occupancy",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=-1,
        help="Total number of visualizations",
    )
    parser.add_argument(
        "--gif", default=False, action="store_true", help="Generate GIF"
    )
    args = parser.parse_args()

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    dump_path = args.dump_path

    if args.gt_path:
        gt_token_list = os.path.join(
            args.gt_path, "validation_pkl_list_new.txt"
        )
        gt_data_path = []
        with open(gt_token_list, "r") as f:
            for line in f:
                gt_data_path.append(args.gt_path + line.strip())

        count = min(args.count, len(gt_data_path))
        if count == -1:
            count = len(gt_data_path)
        for gt_file in tqdm.tqdm(gt_data_path[:count]):
            gt_data = gload(gt_file)
            scenario_id = gt_data["scenario_id"]
            dump = os.path.join(dump_path, f"{scenario_id}.pkl")
            name = os.path.basename(dump).split(".")[0]
            with open(dump, "rb") as f:
                data = pickle.load(f)
            visualize(
                data, name, gt_data, gif=args.gif, occluded=args.occluded
            )
    else:
        if args.reference_path:
            dumps = glob.glob(args.reference_path + "*.jpg")
            for i in range(len(dumps)):
                file_name = os.path.basename(dumps[i])
                file_name = file_name.split(".")[0] + ".pkl"
                dumps[i] = os.path.join(dump_path, file_name)
        else:
            dumps = glob.glob(dump_path + "*.pkl")

        count = min(args.count, len(dumps))
        if count == -1:
            count = len(dumps)

        for dump in tqdm.tqdm(dumps[:count]):
            name = os.path.basename(dump).split(".")[0]
            with open(dump, "rb") as f:
                data = pickle.load(f)
            visualize(data, name, gif=args.gif, occluded=args.occluded)
