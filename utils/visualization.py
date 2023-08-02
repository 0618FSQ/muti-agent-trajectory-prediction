import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from demo import Model
from dataset import RoadDataset
from torch.utils.data import DataLoader
from argoverse.map_representation.map_api import ArgoverseMap
# from argoverse.visualization.visualize_sequences import viz_sequence

_ZORDER = {"AGENT": 15, "AV": 10, "OTHERS": 5}
from typing import Dict, List, Optional
from collections import defaultdict
import matplotlib.lines as mlines
import scipy.interpolate as interp

class ViszTool:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.train_config = json.load(open("scene_model_v2.json", "r"))
        self.model = Model(
            agent_feature_size=6,
            map_feature_size=6,
            sublayer=3
        )
        self.model.load_state_dict(torch.load(self.train_config['load_path']))
        self.model.cuda()
        tmp = os.listdir(self.train_config['train_data_directory'])
        self.files = [os.path.join(self.train_config['train_data_directory'], file_name) for file_name in tmp]
        self.dataset = RoadDataset(self.files)
        self.map_api = ArgoverseMap()
    
    def visz(self, idx):
        plt.figure(figsize=(10,10), dpi=400)
        self.visz_map(idx)
        self.visz_gt(idx)
        self.visz_pred(idx)
        
        
    def visz_map(self, idx):
        # file_path = self.files[idx]
        # directories, file_name = os.path.splitext(file_path)
        # id = directories.split("_")[-1]
        # df = pd.read_csv(os.path.join(self.csv_path, id + '.csv'))
        # viz_sequence(df, show=True)

        
        data = self.dataset[idx]
        map_loc = data["map_feature"]
        map_mask = data["map_feature_mask"]

        for i, cl in enumerate(map_loc):
            if not map_mask[i].any():
                continue
            cl = cl[map_mask[i]]
            x = cl[:, 0]
            y = cl[:, 1]
            plt.plot(
                x,
                y,
                "-",
                color="grey",
                alpha=0.3,
                linewidth=1,
                zorder=0,
            )
        # plt.savefig("./map.png")
        
    
    def visz_gt(self, idx):
        data = self.dataset[idx]
        hist_trajs = data["target_agent_history_trajectory"][:, :, :2]
        hist_masks = data["target_agent_history_mask"]
        origs = data['target_agent_local_feature']
        
        fut_trajs = data["location"] + origs[:, :2].unsqueeze(1)
        fut_masks = data["y_mask"]
        
        trajs = torch.cat([hist_trajs, fut_trajs], dim=1)
        masks = torch.cat([hist_masks, fut_masks], dim=1)
        
        
        for orig, traj, mask in zip(origs, trajs, masks ):
            if not mask.any():
                continue
            traj = traj[mask]
            x = traj[:, 0]
            y = traj[:, 1]
            
            plt.plot(x,y,"-",color='b',alpha=1,linewidth=1)
            
            if mask[18]:
                plt.plot(
                    orig[0],
                    orig[1],
                    'o',
                    color= '#d33e4c',
                    alpha=1,
                    markersize=1,
                    zorder=15,
                )            

        # plt.savefig('./test.png')
    
    def visz_pred(self, idx):
        pred_trajs = self.infer(self.model, self.dataset[idx]).squeeze(0).cpu()
        masks = self.dataset[idx]["y_mask"]
        for traj, mask in zip(pred_trajs, masks):
            if not mask.any():
                continue
            traj = traj[mask]
            x = traj[:, 0]
            y = traj[:, 1]
            plt.plot(x,y,"-",color='g',alpha=1,linewidth=1)
        plt.savefig(f'./picture/{idx}.png')
        
    def draw_lane_polygons(
        self,
        ax: plt.Axes,
        lane_polygons: np.ndarray,
        color="y",
                ):
        """Draw a lane using polygons.

        Args:
            ax: Matplotlib axes
            lane_polygons: Array of (N,) objects, where each object is a (M,3) array
            color: Tuple of shape (3,) representing the RGB color or a single character 3-tuple, e.g. 'b'
        """
        for i, polygon in enumerate(lane_polygons):
            ax.plot(polygon[:, 0], polygon[:, 1], color=color, alpha=0.3, zorder=1)

    def visualize_centerline(centerline):
        """Visualize the computed centerline.

        Args:
            centerline: Sequence of coordinates forming the centerline
        """
        line_coords = list(zip(*centerline))
        lineX = line_coords[0]
        lineY = line_coords[1]
        plt.plot(lineX, lineY, "--", color="grey", alpha=1, linewidth=1, zorder=0)
        plt.text(lineX[0], lineY[0], "s")
        plt.text(lineX[-1], lineY[-1], "e")
        plt.axis("equal")

    
    def infer(self, model, data, use_cuda=True):
        
        model.eval()
        with torch.no_grad():

            target_agent_history_trajectory=data["target_agent_history_trajectory"].unsqueeze(0)
            target_agent_history_mask=data["target_agent_history_mask"].unsqueeze(0)
            target_agent_history_cross_mask=data["target_agent_history_cross_mask"].unsqueeze(0)

            agents_history_trajectory=data["agents_history_trajectory"].unsqueeze(0)
            agents_history_mask=data["agents_history_mask"].unsqueeze(0)
            agents_history_cross_mask=data["agents_history_cross_mask"].unsqueeze(0)

            t2a_cross_mask=data["t2a_cross_mask"].unsqueeze(0)
            t2m_cross_mask=data["t2m_cross_mask"].unsqueeze(0)
            a2m_cross_mask=data["a2m_cross_mask"].unsqueeze(0)
            m2a_cross_mask=data["m2a_cross_mask"].unsqueeze(0)

            map_feature=data["map_feature"].unsqueeze(0)
            map_feature_mask=data["map_feature_mask"].unsqueeze(0)
            map_feature_cross_mask=data["map_feature_cross_mask"].unsqueeze(0)

            target_agent_local_feature=data["target_agent_local_feature"].unsqueeze(0)
            agent_local_feature=data['agent_local_feature'].unsqueeze(0)
            map_local_feature=data['new_map_local_feature'].unsqueeze(0)

            future_mask= torch.from_numpy(data["future_mask"]).unsqueeze(0)
            future_cross_mask=data["future_cross_mask"].unsqueeze(0)

            f2h_cross_mask=data["f2h_cross_mask"].unsqueeze(0)
            f2a_cross_mask=data["f2a_cross_mask"].unsqueeze(0)
            f2m_cross_mask=data["f2m_cross_mask"].unsqueeze(0)
            
            # y = data["y"].unsqueeze(0)
            # location = data["location"].unsqueeze(0)
            # y_mask = data["y_mask"].unsqueeze(0)
            
            if use_cuda:
                target_agent_history_trajectory=target_agent_history_trajectory.cuda()
                target_agent_history_mask=target_agent_history_mask.cuda()
                target_agent_history_cross_mask=target_agent_history_cross_mask.cuda()

                agents_history_trajectory=agents_history_trajectory.cuda()
                agents_history_mask=agents_history_mask.cuda()
                agents_history_cross_mask=agents_history_cross_mask.cuda()

                t2a_cross_mask=t2a_cross_mask.cuda()
                t2m_cross_mask=t2m_cross_mask.cuda()
                a2m_cross_mask=a2m_cross_mask.cuda()
                m2a_cross_mask=m2a_cross_mask.cuda()

                map_feature=map_feature.cuda()
                map_feature_mask=map_feature_mask.cuda()
                map_feature_cross_mask=map_feature_cross_mask.cuda()

                target_agent_local_feature=target_agent_local_feature.cuda()
                agent_local_feature=agent_local_feature.cuda()
                map_local_feature=map_local_feature.cuda()

                future_mask=future_mask.cuda()
                future_cross_mask=future_cross_mask.cuda()

                f2h_cross_mask=f2h_cross_mask.cuda()
                f2a_cross_mask=f2a_cross_mask.cuda()
                f2m_cross_mask=f2m_cross_mask.cuda()
                
                # rot = rot.cuda()
                # orig = orig.cuda()
                
            out = model(
                map_feature=map_feature,
                map_local_feature=map_local_feature,
                map_feature_mask=map_feature_mask,
                map_feature_cross_mask=map_feature_cross_mask,

                agent_feature=agents_history_trajectory,
                agent_local_feature=agent_local_feature,
                agent_feature_mask=agents_history_mask,
                agent_feature_cross_mask=agents_history_cross_mask,

                target_agent_feature=target_agent_history_trajectory,
                target_agent_local_feature=target_agent_local_feature,
                target_agent_feature_mask=target_agent_history_mask,
                target_agent_feature_cross_mask=target_agent_history_cross_mask,

                a2m_mask=a2m_cross_mask,
                m2a_mask=m2a_cross_mask,
                t2a_mask=t2a_cross_mask,
                t2m_mask=t2m_cross_mask,
                future_mask=future_mask,
                future_cross_mask=future_cross_mask,
                f2h_cross_mask=f2h_cross_mask,
                f2a_cross_mask=f2a_cross_mask,
                f2m_cross_mask=f2m_cross_mask
            )
                
            # pred_traj = out[:, 0, :, :]
            # real_traj = torch.matmul(pred_traj, rot)
            out = torch.add(target_agent_local_feature[:, :, :2].unsqueeze(dim=-2), out)
               
        return out
    


def viz_sequence(
    df: pd.DataFrame,
    lane_centerlines: Optional[List[np.ndarray]] = None,
    show: bool = True,
    smoothen: bool = True,
):

    # Seq data
    city_name = df["CITY_NAME"].values[0]

    if lane_centerlines is None:
        # Get API for Argo Dataset map
        avm = ArgoverseMap()
        seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    plt.figure(0, figsize=(8, 7))

    x_min = min(df["X"])
    x_max = max(df["X"])
    y_min = min(df["Y"])
    y_max = max(df["Y"])

    if lane_centerlines is None:

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        lane_centerlines = []
        # Get lane centerlines which lie within the range of trajectories
        for lane_id, lane_props in seq_lane_props.items():

            lane_cl = lane_props.centerline

            if (
                np.min(lane_cl[:, 0]) < x_max
                and np.min(lane_cl[:, 1]) < y_max
                and np.max(lane_cl[:, 0]) > x_min
                and np.max(lane_cl[:, 1]) > y_min
            ):
                lane_centerlines.append(lane_cl)

    for lane_cl in lane_centerlines:
        plt.plot(
            lane_cl[:, 0],
            lane_cl[:, 1],
            "-",
            color="grey",
            alpha=1,
            linewidth=1,
            zorder=0,
        )
    frames = df.groupby("TRACK_ID")

    plt.xlabel("Map X")
    plt.ylabel("Map Y")

    color_dict = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}
    object_type_tracker: Dict[int, int] = defaultdict(int)

    # Plot all the tracks up till current frame
    for group_name, group_data in frames:
        object_type = group_data["OBJECT_TYPE"].values[0]

        cor_x = group_data["X"].values
        cor_y = group_data["Y"].values

        if smoothen:
            polyline = np.column_stack((cor_x, cor_y))
            num_points = cor_x.shape[0] * 3
            smooth_polyline = interpolate_polyline(polyline, num_points)
            cor_x = smooth_polyline[:, 0]
            cor_y = smooth_polyline[:, 1]

        plt.plot(
            cor_x,
            cor_y,
            "-",
            color=color_dict[object_type],
            label=object_type if not object_type_tracker[object_type] else "",
            alpha=1,
            linewidth=1,
            zorder=_ZORDER[object_type],
        )

        final_x = cor_x[-1]
        final_y = cor_y[-1]

        if object_type == "AGENT":
            marker_type = "o"
            marker_size = 7
        elif object_type == "OTHERS":
            marker_type = "o"
            marker_size = 7
        elif object_type == "AV":
            marker_type = "o"
            marker_size = 7

        plt.plot(
            final_x,
            final_y,
            marker_type,
            color=color_dict[object_type],
            label=object_type if not object_type_tracker[object_type] else "",
            alpha=1,
            markersize=marker_size,
            zorder=_ZORDER[object_type],
        )

        object_type_tracker[object_type] += 1

    red_star = mlines.Line2D([], [], color="red", marker="*", linestyle="None", markersize=7, label="Agent")
    green_circle = mlines.Line2D(
        [],
        [],
        color="green",
        marker="o",
        linestyle="None",
        markersize=7,
        label="Others",
    )
    black_triangle = mlines.Line2D([], [], color="black", marker="^", linestyle="None", markersize=7, label="AV")

    plt.axis("off")
    plt.savefig("./1.png")
    # if show:
    #     plt.show()
        
def interpolate_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if polyline.shape[0] - len(duplicates) < 4:
        return polyline
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))