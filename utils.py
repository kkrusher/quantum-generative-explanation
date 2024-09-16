import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle


def get_phaseTransition(config):
    if config.dataset == "SPTS":
        with open(
            f"./data/SPTS_PhaseTransition/SPTS_phaseTransitionDataset_q{config.o_qubits}_n{config.n_data}.npy",
            "rb",
        ) as f:
            phaseTransition = pickle.load(f)
    elif config.dataset == "TFIM":
        with open(
            f"./data/TFIM_PhaseTransition/TFIM_phaseTransitionDataset_q{config.o_qubits}_n{config.n_data}.npy",
            "rb",
        ) as f:
            phaseTransition = pickle.load(f)
    else:
        raise ValueError("Invalid dataset")
    return phaseTransition


def sigmoid_transform(J):
    # from [-inf, inf] to [-pi/2, pi]
    return (2 * torch.sigmoid(J) - 1) * torch.pi / 2


def inverse_sigmoid_transform(Y):
    return torch.log((Y / torch.pi + 0.5) / (1 - (Y / torch.pi + 0.5)))


def process_phase_transition_data(phaseTransitionDataset):
    # 提取 states 和 label
    states = phaseTransitionDataset["states"]
    labels = phaseTransitionDataset["label"]

    # 从pennylane.numpy的tensor格式转换为torch的Tensor格式
    J1 = torch.stack([torch.tensor(state[0]) for state in states])
    J2 = torch.stack([torch.tensor(state[1]) for state in states])
    # 使用 sigmoid 函数映射到 [-1, 1]，然后乘以 pi 映射到 [-pi, pi]
    J1_mapped = sigmoid_transform(J1)
    J2_mapped = sigmoid_transform(J2)

    # print(J1_mapped)

    ground_states = torch.stack([torch.tensor(state[2]) for state in states])

    labels = torch.tensor(labels, dtype=torch.long)
    # 将 labels 转换为独热编码
    # labels = F.one_hot(labels, num_classes=2)

    # 拼接特征矩阵 X
    X = torch.cat((J1_mapped.unsqueeze(1), J2_mapped.unsqueeze(1)), dim=1)
    return X, labels, ground_states


def plot_phase_background(model_type):
    if model_type == "TFIM":
        plt.plot([-1, 1], [-1, 1], color="black", linestyle="-", linewidth=2)
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        plt.xticks(np.arange(-1, 2, 1))
        plt.yticks(np.arange(-1, 2, 1))

        # plt.plot([-4, 4], [-4, 4], color="black", linestyle="-", linewidth=2)
        # plt.xlim((-4, 4))
        # plt.ylim((-4, 4))
        # plt.xticks(np.arange(-4, 6, 2))
        # plt.yticks(np.arange(-4, 6, 2))

        # Add text to the plot
        plt.text(-0.5, 0.5, "I", fontsize=24, fontweight="bold", fontfamily="serif")
        plt.text(0.5, -0.5, "II", fontsize=24, fontweight="bold", fontfamily="serif")
    elif model_type == "SPTS":
        plt.plot([-3, 4], [-4, 3], color="black", linestyle="-", linewidth=2)
        plt.plot([-4, 3], [3, -4], color="black", linestyle="-", linewidth=2)
        plt.plot([-2, 2], [1, 1], color="black", linestyle="-", linewidth=2)
        plt.xlim((-4, 4))
        plt.ylim((-4, 4))
        plt.xticks(np.arange(-4, 6, 2))
        plt.yticks(np.arange(-4, 6, 2))
        # Add text to the plot
        plt.text(0, 3, "I", fontsize=24, fontweight="bold", fontfamily="serif")
        plt.text(0, -3, "I", fontsize=24, fontweight="bold", fontfamily="serif")
        plt.text(3, -1, "II", fontsize=24, fontweight="bold", fontfamily="serif")
        plt.text(-3, -1, "III", fontsize=24, fontweight="bold", fontfamily="serif")
        plt.text(0, 0, "IV", fontsize=24, fontweight="bold", fontfamily="serif")


## 绘制轨迹的函数
def plot_phase_diagram_with_trajactory(
    model_type, exp_name, trajectories, init_true_labels, show_plot=False
):
    plt.figure(figsize=(8, 8))
    plot_phase_background(model_type)

    # 设置轨迹颜色
    # colors = ["blue", "red", "green", "purple"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, trajectory in enumerate(trajectories):
        x = [item[0] for item in trajectory]
        y = [item[1] for item in trajectory]
        init_category = init_true_labels[i]
        color = colors[init_category % len(colors)]

        # 绘制轨迹线
        plt.plot(x, y, color=color, linewidth=2, label=f"Trajectory {i+1}")

        # 计算轨迹线上每个线段的长度，并累积
        distances = [
            ((x[j + 1] - x[j]) ** 2 + (y[j + 1] - y[j]) ** 2) ** 0.5
            for j in range(len(x) - 1)
        ]
        cumulative_distances = [sum(distances[: j + 1]) for j in range(len(distances))]

        # 找到累积距离中最接近总长度一半的索引
        total_length = cumulative_distances[-1]
        midpoint_index = min(
            range(len(cumulative_distances)),
            key=lambda j: abs(cumulative_distances[j] - total_length / 2),
        )

        # 在该位置添加箭头
        head_width = 0.1  # 箭头的宽度
        head_length = 0.2  # 箭头的长度
        plt.arrow(
            x[midpoint_index],
            y[midpoint_index],
            x[midpoint_index + 1] - x[midpoint_index],
            y[midpoint_index + 1] - y[midpoint_index],
            color=color,
            length_includes_head=True,
            head_width=head_width,
            head_length=head_length,
        )

        # 标记起点和终点
        plt.scatter(x[0], y[0], color=color, s=30, zorder=5, marker="o")
        plt.scatter(x[-1], y[-1], color=color, s=30, zorder=5, marker="o")

    plt.xlabel("$J_1$", fontsize=16)
    plt.ylabel("$J_2$", fontsize=16)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig(f"results/{exp_name}/plot_phase_diagram_with_trajectory.png")
    if show_plot:
        plt.show()
    plt.close()


def predict_with_train_dataset(trainer):
    trainer.model.eval()
    x_list = []
    predict_list = []
    with torch.no_grad():
        # todo 使用验证集
        # for X_batch, y_batch in trainer.val_loader:
        for X_batch, y_batch in trainer.train_loader:
            output = trainer.model(X_batch)
            y_predict = output.argmax(axis=1)
            original_J = inverse_sigmoid_transform(X_batch)
            x_list.append(original_J.numpy())
            predict_list.append(y_predict.numpy())
    return x_list, predict_list


# 绘制mode预测结果，使用不同的颜色绘制不同的预测分类
def model_predict_for_validation(x_list, predict_list, model_type):
    plt.figure(figsize=(8, 8))
    plot_phase_background(model_type)

    # Flatten the list of batches into a single list of points
    x_points = np.concatenate(x_list, axis=0)
    predictions = np.concatenate(predict_list, axis=0)

    # Assuming there are two features for each point
    x_coords = x_points[:, 0]
    y_coords = x_points[:, 1]

    # Create a scatter plot of the predictions
    # Use different colors for different predicted classes
    class_label = {0: "I", 1: "II", 2: "III", 3: "IV"}
    for class_index in np.unique(predictions):
        class_mask = predictions == class_index
        plt.scatter(
            x_coords[class_mask],
            y_coords[class_mask],
            label=f"Class {class_label[class_index]}",
        )

    plt.legend()
    plt.show()

    plt.close()


# 输入是单个数据，不能是一个batch
def get_direction_to_control_generation(trainer, input, target, category_num):
    trainer.model.eval()
    trainer.optimizer.zero_grad()

    input = torch.unsqueeze(input, 0)
    input.retain_grad()

    # target = torch.unsqueeze(target, 0)
    # 将每一类的样本目标定位到下一个类别
    target = (torch.unsqueeze(target, 0) + 1) % category_num

    output = trainer.model(input)
    loss = trainer.loss_fn(output, target)
    # print(input, target, output)
    loss.backward()
    # print(trainer.model.classification_params.grad)
    return input.grad[0]
