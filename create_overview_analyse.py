import matplotlib.pyplot as plt
import argparse  # 新增导入
import os

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='生成数据分析综合报告')
parser.add_argument('--size', type=str, default='10G',
                   help='指定不同数据集')
args = parser.parse_args()

font_name = "simhei"
plt.rcParams['font.family']= font_name # 指定字体，实际上相当于修改 matplotlibrc 文件　只不过这样做是暂时的　下次失效
plt.rcParams['axes.unicode_minus']=False # 正确显示负号，防止变成方框

# 创建综合图1
fig_combined1, axes = plt.subplots(3, 3, figsize=(24, 18))
fig_combined1.suptitle('综合数据分析报告1', fontsize=24, y=1.02)

# 将各子图添加到综合图中
for i, img_path in enumerate([
    '01_age_distribution.png',
    '02_category_distribution.png',
    '03_category_sales_distribution.png',
    '04_category_correlation.png',
    '05_country_distribution.png',
    '06_user_activity_comparison.png',
    '07_total_spent_distribution.png',
    '08_gender_activity_heatmap.png',
    '09_age_activity_density.png'
]):
    if args.size == '10G':
        img = plt.imread(f'{os.getcwd()}/result_imgs/{img_path}')
    else:
        img = plt.imread(f'{os.getcwd()}/result_imgs_30G/{img_path}')
    row, col = divmod(i, 3)
    axes[row, col].imshow(img)
    axes[row, col].axis('off')

plt.tight_layout()
if args.size == '10G':
    fig_combined1.savefig(os.path.join(os.getcwd(),"composed_overview_imgs/overview_analysis1.png"), dpi=300, bbox_inches='tight')
else:
    fig_combined1.savefig(os.path.join(os.getcwd(),"composed_overview_imgs/overview_analysis1_30G.png"), dpi=300, bbox_inches='tight')
plt.close(fig_combined1)

# 创建综合图2
fig_combined2, axes = plt.subplots(3, 2, figsize=(20, 18))
fig_combined2.suptitle('综合数据分析报告2', fontsize=24, y=1.02)

# 将各子图添加到综合图中
for i, img_path in enumerate([
    '10_age_category_distribution.png',
    '11_gender_category_distribution.png',
    '12_age_income_distribution.png',
    '13_country_sales_distribution.png',
    '14_country_spending_distribution.png',
    '15_gender_distribution.png'
]):
    if args.size == '10G':
        img = plt.imread(f'{os.getcwd()}/result_imgs/{img_path}')
    else:
        img = plt.imread(f'{os.getcwd()}/result_imgs_30G/{img_path}')
    row, col = divmod(i, 2)
    axes[row, col].imshow(img)
    axes[row, col].axis('off')
    # axes[row, col].set_title(img_path.split('.')[0], fontsize=12)

plt.tight_layout()
if args.size == '10G':
    fig_combined2.savefig(os.path.join(os.getcwd(),'composed_overview_imgs/overview_analysis2.png'), dpi=300, bbox_inches='tight')
else:
    fig_combined2.savefig(os.path.join(os.getcwd(),'composed_overview_imgs/overview_analysis2_30G.png'), dpi=300, bbox_inches='tight')
plt.close(fig_combined2)