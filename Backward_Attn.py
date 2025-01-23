from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

class Llama3AttentionAnalyzer:
    def __init__(self, model_name, device='cuda'):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # 保持模型bfloat16精度
            attn_implementation="eager",
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 模型配置参数
        self.n_layer = self.model.config.num_hidden_layers
        self.n_head = self.model.config.num_attention_heads
        self.num_key_value_heads = self.model.config.num_key_value_heads
        self.head_size = self.model.config.hidden_size // self.n_head
        self.kv_repeat = self.n_head // self.num_key_value_heads

        # 梯度设置
        for layer in self.model.model.layers:
            layer.self_attn.o_proj.weight.requires_grad_(True)

        # 存储容器
        self.forward_attentions = []
        self.reversed_attentions = []
        self.prompt_len = 0
        self.prompt_list_tmp = []

        # 提取模型名称作为文件夹名
        self.model_name = model_name.split('/')[-1]
        # 创建保存图片的文件夹
        self.save_dir = f"attention_plots_{self.model_name}"
        os.makedirs(self.save_dir, exist_ok=True)

    def analyze(self, prompt, target):
        """全流程分析"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        target_id = self.tokenizer.encode(target, add_special_tokens=False)[0]

        params = []
        for layer in self.model.model.layers:
            layer.self_attn.o_proj.weight.requires_grad_(True)
            params.append(layer.self_attn.o_proj.weight)
        optimizer = torch.optim.Adam(params, lr=1e-1)
        
        # 前向传播
        outputs = self.model(**inputs, output_attentions=True)
        
        # 计算损失
        logits = outputs.logits[0, -1]
        loss = -torch.nn.functional.log_softmax(logits, dim=-1)[target_id]
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 存储输入信息
        self._store_input_info(inputs)
        # 可视化结果
        self._visualize_results(outputs)

    def _store_input_info(self, inputs):
        """存储输入token信息"""
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        self.prompt_len = len(tokens)
        # 将特殊字符Ġ替换为空格，使显示更清晰
        self.prompt_list_tmp = [x.replace('Ġ', ' ') for x in tokens]
        # 同时处理换行符
        self.prompt_list_tmp = [x.replace('\n', '\\n') for x in self.prompt_list_tmp]
        # 简化特殊token的显示
        self.prompt_list_tmp = [
            '<bos>' if x.strip() in ['<s>', '<|begin_of_text|>'] else 
            '<eos>' if x.strip() in ['</s>', '<|end_of_text|>'] else 
            x 
            for x in self.prompt_list_tmp
        ]

    def _visualize_results(self, outputs):
        """可视化注意力模式"""
        self.forward_attentions = []
        self.reversed_attentions = []
        
        for layer_idx in range(self.n_layer):
            grad = self.model.model.layers[layer_idx].self_attn.o_proj.weight.grad
            
            layer_forward = []
            layer_reversed = []
            
            for head_idx in range(self.n_head):
                kv_head_idx = head_idx // self.kv_repeat
                
                # 前向注意力矩阵（保持bfloat16精度）
                attn = outputs.attentions[layer_idx][0, head_idx].to(self.device)
                
                # 反向注意力计算
                values = outputs.past_key_values[layer_idx][1][0, kv_head_idx].to(self.device)
                head_grad = grad[:, head_idx*self.head_size:(head_idx+1)*self.head_size]
                
                # 梯度投影计算
                grad_projection = values @ head_grad.T  # [seq_len, hidden_size]
                grad_projection = grad_projection.view(self.prompt_len, self.n_head, self.head_size)
                current_proj = grad_projection[:, head_idx, :]  # [seq_len, head_size]
                
                # 反向注意力计算
                rev_logits = current_proj @ values.transpose(-1, -2)  # [seq_len, seq_len]
                diag_adjust = torch.diag(attn @ rev_logits)
                reversed_attn = attn * (rev_logits - diag_adjust) / np.sqrt(self.head_size)
                
                # 转换为float32后再转numpy
                layer_forward.append(attn.cpu().detach().float().numpy())  # 关键修改
                layer_reversed.append(reversed_attn.cpu().detach().float().numpy())  # 关键修改
            
            self.forward_attentions.append(layer_forward)
            self.reversed_attentions.append(layer_reversed)

        # 绘制热力图
        self._plot_heatmaps(np.array([[np.linalg.norm(h) for h in layer] for layer in self.forward_attentions]).T, 
                          'Forward Attention')
        self._plot_heatmaps(np.array([[np.linalg.norm(h) for h in layer] for layer in self.reversed_attentions]).T, 
                          'Reversed Attention')

    def _plot_heatmaps(self, data, title):
        """绘制层/头热力图"""
        plt.figure(figsize=(16, 8))
        ax = sns.heatmap(data, cmap='viridis' if 'Forward' in title else 'magma',
                        xticklabels=2, yticklabels=4)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Head Index', fontsize=12)
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f'{title.replace(" ", "_")}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def visualize_individual_heads(self, n_attn_heads_to_show=10):
        """可视化单个注意力头的详细模式"""
        reversed_norms_stats = []
        for layer_idx in range(self.n_layer):
            for head_idx in range(self.n_head):
                value = np.linalg.norm(self.reversed_attentions[layer_idx][head_idx])
                reversed_norms_stats.append((layer_idx, head_idx, value))
        
        reversed_norms_stats.sort(key=lambda x: x[2], reverse=True)
        mask = np.triu(np.ones((self.prompt_len, self.prompt_len), dtype=bool), k=1)
        newcmp = LinearSegmentedColormap.from_list("", ['red', 'white', 'white', 'blue'])

        for i, (layer_idx, head_idx, value) in enumerate(reversed_norms_stats):
            if i >= n_attn_heads_to_show:
                break
            self._plot_single_head(layer_idx, head_idx, mask, newcmp)

    def _plot_single_head(self, layer_idx, head_idx, mask, reversed_cmap):
        """绘制单个注意力头的热力图"""
        forward = self.forward_attentions[layer_idx][head_idx]
        reversed_attn = self.reversed_attentions[layer_idx][head_idx]

        # 绘制前向注意力（使用固定范围）
        self._create_heatmap(forward, 'Greens', 
                            f'L{layer_idx}H{head_idx} Forward Attention', 
                            mask, 
                            vmin=0, vmax=1)
        
        # 绘制反向注意力（使用对称色标）
        max_val = np.abs(reversed_attn).max()
        self._create_heatmap(reversed_attn, reversed_cmap, 
                            f'L{layer_idx}H{head_idx} Reversed Attention',
                            mask, 
                            center=0,
                            vmin=-max_val*1, 
                            vmax=max_val*1)

    def _create_heatmap(self, data, cmap, title, mask, **kwargs):
        """通用热力图绘制函数"""
        plt.figure(figsize=(5.5, 5))
        ax = plt.gca()
        
        # 处理自定义参数
        cbar_ticks = kwargs.pop('cbar_ticks', None)
        vmin = kwargs.get('vmin', None)
        vmax = kwargs.get('vmax', None)
        center = kwargs.get('center', None)
        
        # 创建下三角遮罩（将上三角设为True表示遮罩）
        mask = np.zeros_like(data)
        mask[np.triu_indices_from(mask, k=1)] = True
        
        # 绘制热力图
        sns.heatmap(data, cmap=cmap, annot=False, fmt=".4f", 
                   mask=mask, ax=ax, cbar=False, **kwargs)
        
        # 设置坐标轴标签为token
        ax.set_xticks(np.arange(self.prompt_len) + 0.5)
        ax.set_xticklabels(self.prompt_list_tmp, rotation=60, fontsize=12)
        ax.set_yticks(np.arange(self.prompt_len) + 0.5)
        ax.set_yticklabels(self.prompt_list_tmp, rotation=0, fontsize=12)
        
        # 添加colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.12)
        cax.tick_params(labelsize=14)
        plt.colorbar(ax.get_children()[0], cax=cax)
        
        # 智能设置colorbar刻度
        cbar = ax.collections[0].colorbar
        if vmin is not None and vmax is not None:  # 前向注意力模式
            cbar.set_ticks([0, 0.5, 1])
        elif center is not None:  # 反向注意力对称模式
            max_cbar = round(data.max() * 1, 3)
            min_cbar = round(data.min() * 1, 3)
            cbar.set_ticks([min_cbar, 0, max_cbar])
        
        # 添加阶梯状边界线
        for i in range(data.shape[0]):
            # 添加水平线段
            ax.add_patch(Rectangle((i, i), 1, 0, fill=True, color='black'))
            # 添加垂直线段
            ax.add_patch(Rectangle((i+1, i), 0, 1, fill=True, color='black'))
        
        # 显示所有边框
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.save_dir, f'{title.replace(" ", "_")}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()



if __name__ == "__main__":
    analyzer = Llama3AttentionAnalyzer("meta-llama/Llama-3.2-3B-Instruct")
    analyzer.analyze(
        "I like Italy and France. I visited the city of",
        " Florence"
    )
    # 新增可视化调用
    analyzer.visualize_individual_heads(n_attn_heads_to_show=5)
