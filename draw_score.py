import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter 

# base directory for SIDE project; can be overridden with SIDE_ROOT env
BASE_DIR = os.environ.get('SIDE_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def read_json_file(filename):
    """读取JSON文件"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"文件 {filename} 未找到")
        return None
    except json.JSONDecodeError:
        print(f"文件 {filename} 不是有效的JSON格式")
        return None

def plot_scores(json_files, colors=None, labels=None):
    """
    绘制多个JSON文件的 score 散点图 + 平滑曲线
    """
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple']
    if labels is None:
        labels = [f"data{i+1}" for i in range(len(json_files))]

    plt.figure(figsize=(12, 8))

    for json_file, color, label in zip(json_files, colors, labels):
        data = read_json_file(json_file)
        if data is None:
            continue
        scores = [item['score'] for item in data]
        x = np.arange(1, len(scores) + 1)

        # 1) 散点：半透明
        plt.scatter(x, scores, color=color, alpha=0.2, s=8, marker=None)

        # 2) 平滑曲线：Savitzky-Golay
        # choose odd window size <= len(scores)
        window = min(51, len(scores))
        if window % 2 == 0:
            window = max(3, window - 1)
        poly_order = min(3, window - 1)
        if len(scores) >= window:
            y_smooth = savgol_filter(scores, window, poly_order)
        else:
            y_smooth = scores
        plt.plot(x, y_smooth, color=color, linewidth=3, label=label)

        print(f"文件 {json_file}: 成功读取 {len(scores)} 个数据点")

    plt.xlabel('Query Index', fontsize=24)
    plt.ylabel('Cosine similarity with harmful direction', fontsize=24)
    # plt.title('Cosine similarity of four different queries and unsafe directions', fontsize=14)
    plt.grid(True, alpha=0.3)
    # plt.legend(fontsize=18)          # 只会出现散点对应的入口，曲线不重复
    plt.legend(fontsize=20,
           loc='upper right',      # 锚点放在左侧竖直居中
           framealpha=0.5,
           bbox_to_anchor=(0.98, 0.8),  # 整体向右移 1.02 倍轴宽
           borderaxespad=0.)       # 与轴无间距
    plt.xlim(0, 200)
    plt.tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    out_dir = os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'score_comparison_plot-消融实验-guide.png'), dpi=300)
# 主函数
def main():
    # JSON文件路径
    json_files = [
        os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores', 'safety_scores_results_391.json'),
        os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores', 'unsafe_scores_results_391.json'),
        # 示例（使用项目相对路径）：
        # os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores', 'unsafe+manipulate_scores_results_391.json')
        # os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores', 'jailbreak+manipulate_scores_results_391.json')
        # os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores', 'adaptive_init_scores_results_391.json')
        # os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores', 'reda_scores_results_391.json')
        # os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores', 'autodan_scores_results_51.json')
        # os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores', 'gcg_scores_results_136.json')
        # os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores', 'side_scores_results_391.json')
        os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores', 'jailbreak+manipulate_scores_results_391.json'),
        os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores', 'manipulate+remove_guide_side_scores_results_391.json'),
        os.path.join(BASE_DIR, 'acl_src', 'llm-adaptive', 'dataset', 'side', 'scores', 'remove_guide_side_scores_results_391.json')

    ]
    
    # 自定义颜色和标签（可选）
    # colors = ['#1f77b4', '#ff7f0e', '#E547F3', '#2ca02c', "#551861", "#229AE0", "#d62728"]  # 蓝色、橙色、绿色、红色
    # labels = ['Harmful', 'Benign', 'adaptive_init', 'REDA', 'AutoDAN', 'GCG', 'SIDE', 'side']
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色、红色
    # labels = ['benign', 'harmful', 'harmful+affirmative prefix']

    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', "#E547F3"]  # 蓝色、橙色、绿色、红色
    labels = ['Benign', 'Harmful', 'side', 'side w/o guide', 'side w/o guide+manipulate']

    # 检查文件是否存在
    for file in json_files:
        if not os.path.exists(file):
            print(f"警告: 文件 {file} 不存在")
    
    # 绘制图表
    plot_scores(json_files, colors, labels)

if __name__ == "__main__":
    main()