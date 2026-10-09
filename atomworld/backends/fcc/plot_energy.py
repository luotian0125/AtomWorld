import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Tuple

# 设置全局绘图样式（优化美观度）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # 兼容中英文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300  # 默认高清
plt.rcParams['savefig.dpi'] = 300


def plot_kmc_energy(
    file_path: str = "models/output-rl.e",
    save_path: str = "models/energy_curve.png",
    smooth_window_ratio: float = 0.1,  # 平滑窗口为数据长度的10%（自适应）
    max_smooth_window: int = 200  # 最大平滑窗口限制
) -> None:
    """
    绘制KMC能量演化曲线（增强鲁棒性+可视化效果）
    
    Args:
        file_path: 输入的energy数据文件路径（默认：models/output-rl.e）
        save_path: 输出的图片保存路径（默认：models/energy_curve.png）
        smooth_window_ratio: 平滑窗口占数据长度的比例（自适应调整）
        max_smooth_window: 最大平滑窗口大小（避免窗口过大失真）
    """
    # 1. 基础校验：文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到数据文件 {file_path}")
        return

    # 2. 解析.e文件（增强异常处理）
    steps: List[int] = []
    energies: List[float] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # 跳过空行/注释行
                
                parts = line.split()
                # 严格校验数据格式，避免解析错误
                if len(parts) < 4:
                    print(f"⚠️ 警告：第{line_num}行数据格式错误，跳过 → {line}")
                    continue
                
                # 类型转换容错
                try:
                    step = int(parts[0])
                    energy = float(parts[3])
                    steps.append(step)
                    energies.append(energy)
                except (ValueError, IndexError) as e:
                    print(f"⚠️ 警告：第{line_num}行数据转换失败，跳过 → {line} | 错误：{e}")
                    continue
    except Exception as e:
        print(f"❌ 错误：读取文件 {file_path} 失败 → {e}")
        return

    # 3. 数据量校验：避免空数据绘图
    if len(steps) == 0 or len(energies) == 0:
        print(f"❌ 错误：文件 {file_path} 中无有效能量数据")
        return
    if len(steps) < 2:
        print(f"❌ 错误：有效数据点不足（仅{len(steps)}个），无法绘图")
        return

    # 4. 计算核心能量指标（用于标注）
    init_energy = energies[0]
    final_energy = energies[-1]
    energy_drop = init_energy - final_energy
    energy_drop_rate = (energy_drop / init_energy) * 100 if init_energy != 0 else 0.0
    avg_energy = np.mean(energies)

    # 5. 绘图（增强可视化效果）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制原始能量曲线
    ax.plot(
        steps, energies, 
        color='#1f77b4', linewidth=1.2, alpha=0.7, 
        label='Raw Energy (RL-KMC)', zorder=1
    )
    
    # 绘制平滑趋势线（自适应窗口，避免报错）
    window = min(int(len(energies) * smooth_window_ratio), max_smooth_window)
    if window >= 2:  # 窗口至少为2才做平滑
        smoothed = np.convolve(energies, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[window-1:]  # 对齐平滑后的数据长度
        ax.plot(
            smoothed_steps, smoothed, 
            color='#d62728', linewidth=2.0, alpha=0.9, 
            label=f'Smoothed Trend (Window={window})', zorder=2
        )
    
    # 标注关键指标（提升信息密度）
    ax.scatter([steps[0]], [init_energy], color='#2ca02c', s=60, zorder=3, label=f'Initial: {init_energy:.3f} eV')
    ax.scatter([steps[-1]], [final_energy], color='#ff7f0e', s=60, zorder=3, label=f'Final: {final_energy:.3f} eV')
    
    # 添加能量变化文本标注
    text_box = f"""Energy Change:
Drop: {energy_drop:.3f} eV
Drop Rate: {energy_drop_rate:.2f}%
Avg Energy: {avg_energy:.3f} eV"""
    ax.text(
        0.02, 0.98, text_box, 
        transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8)
    )

    # 图表样式优化
    ax.set_title('System Energy Evolution over KMC Steps', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('KMC Step', fontsize=14, labelpad=10)
    ax.set_ylabel('Total Energy (eV)', fontsize=14, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # 自动调整布局，避免标签截断
    plt.tight_layout()

    # 6. 保存图片（确保目录存在）
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    try:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"🎉 成功！能量演化曲线已保存至：{save_path}")
        # 打印核心指标汇总
        print(f"📊 核心能量指标：")
        print(f"   初始能量: {init_energy:.3f} eV")
        print(f"   最终能量: {final_energy:.3f} eV")
        print(f"   能量下降: {energy_drop:.3f} eV ({energy_drop_rate:.2f}%)")
    except Exception as e:
        print(f"❌ 错误：保存图片 {save_path} 失败 → {e}")
    finally:
        plt.close(fig)  # 关闭画布，释放内存


if __name__ == "__main__":
    # 保持原调用方式不变，直接运行即可
    plot_kmc_energy()