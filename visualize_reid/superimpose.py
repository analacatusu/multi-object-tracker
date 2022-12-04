import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cv2

DATA_DIR = "/data/develop/export_mkv_k4a/test_system/"

def plot_cam_grid(axes, cameras, frames):
    # display only for every other column
    # m = np.zeros(axes[0, :].size, dtype=bool)
    # m[::2] = 1

    # for ax, col in zip(axes[0, m], cameras):
    #     ax.set_title(col)

    # for ax, row in zip(axes[:, 0], frames):
    #     ax.set_ylabel("Frame: %s" % row, rotation=90, size='large')

    # load and display images
    for i, frame in enumerate(frames):
        for j, cam in enumerate(cameras):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if cam is None:
                ax.axis('off')
                continue

            file_id = str(frame).zfill(10)
            fpath = os.path.join(DATA_DIR, cam, f"{file_id}_color.jpg")
            color = cv2.imread(fpath,
                               cv2.IMREAD_UNCHANGED)
            if color is None:
                raise Exception(f"Invalid filename {fpath}")
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            fpath = os.path.join(DATA_DIR, cam, f"{file_id}_rgbd.tiff")
            mask = cv2.imread(fpath,
                              cv2.IMREAD_GRAYSCALE)

            # Create mask
            # print("{cam_id} {frame_id} any nonzero?: {any}".format(
            #    cam_id=cam, frame_id=file_id, any=np.sum(mask > 0)))

            # Plot images
            print(f"Frame: {frame} - {cam} into [{i}, {j}]")

            ax.set_title(f"{cam} - {frame}")

            ax.imshow(color)
            ax.imshow(ma.masked_where(mask > 0, mask), 'Reds',
                      interpolation='none', alpha=0.8)


if __name__ == "__main__":
    cameras_1 = ['cn01', 'cn02', 'cn03']
    cameras_2 = ['cn04', 'cn05', 'cn06']
    # frames = [500, 700, 900]
    frames = [5, 505, 1005]

    fig, axes = plt.subplots(nrows=len(frames) * 2, ncols=len(cameras_1), figsize=(6, 12))

    plot_cam_grid(axes[range(0, 2 * len(frames), 2), :], cameras_1, frames)
    plot_cam_grid(axes[range(1, 2 * len(frames), 2), :], cameras_2, frames)

    fig.tight_layout()
    plt.margins(0, 0)
    plt.savefig("rgbd_frames.pdf", bbox_inches='tight', pad_inches=0.1)
