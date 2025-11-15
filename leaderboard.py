"""
排行榜模块
处理模型对战积分和战绩统计
"""

import json
import os
from collections import defaultdict

class Leaderboard:
    def __init__(self, data_file='leaderboard_data.json'):
        self.data_file = data_file
        self.battle_records = {}  # 存储模型对战记录 {model_pair_key: {'wins': int, 'losses': int}}
        self.model_scores = defaultdict(int)  # 存储模型积分 {model_path: score}
        self.load_data()
    
    def load_data(self):
        """从文件加载排行榜数据"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.battle_records = data.get('battle_records', {})
                    self.model_scores = defaultdict(int, data.get('model_scores', {}))
        except Exception as e:
            print(f"Error loading leaderboard data: {e}")
    
    def save_data(self):
        """保存排行榜数据到文件"""
        try:
            data = {
                'battle_records': self.battle_records,
                'model_scores': dict(self.model_scores)
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving leaderboard data: {e}")
    
    def get_model_pair_key(self, model1_path, model2_path):
        """获取模型对的唯一键（按字母顺序排序保证一致性）"""
        return ' vs '.join(sorted([model1_path, model2_path]))
    
    def update_battle_result(self, model1_path, model2_path, model1_wins):
        """
        更新对战结果
        Args:
            model1_path: 模型1路径
            model2_path: 模型2路径
            model1_wins: 模型1是否获胜
        """
        # 获取模型对的唯一键
        pair_key = self.get_model_pair_key(model1_path, model2_path)

        # 保存之前的战绩
        old_record = None
        if pair_key in self.battle_records:
            old_record = dict(self.battle_records[pair_key])  # 复制一份

        # 初始化记录
        if pair_key not in self.battle_records:
            self.battle_records[pair_key] = {'wins': 0, 'losses': 0}

        # 更新战绩
        if model1_wins:
            # 确定哪个是实际的赢家和输家
            if model1_path <= model2_path:
                self.battle_records[pair_key]['wins'] += 1
            else:
                self.battle_records[pair_key]['losses'] += 1
        else:
            # 确定哪个是实际的赢家和输家
            if model1_path <= model2_path:
                self.battle_records[pair_key]['losses'] += 1
            else:
                self.battle_records[pair_key]['wins'] += 1

        # 每次更新战绩后都重新计算积分（基于累计的战绩）
        # 首先清除之前这对模型的积分
        if old_record:
            self._clear_pair_scores(model1_path, model2_path, old_record)
        # 然后重新计算积分
        self._calculate_and_update_scores(pair_key, model1_path, model2_path)

        # 保存数据
        self.save_data()

    def _clear_pair_scores(self, model1_path, model2_path, old_record):
        """清除一对模型的积分"""
        # 减去之前这对模型的积分

        wins = old_record['wins']
        losses = old_record['losses']

        # 确定模型1和模型2在记录中的位置
        if model1_path <= model2_path:
            model1_wins = wins
            model2_wins = losses
        else:
            model1_wins = losses
            model2_wins = wins

        # 清除之前的积分
        if model1_wins > model2_wins:
            # 模型1胜出
            self.model_scores[model1_path] -= 3
            self.model_scores[model2_path] -= 1
        elif model2_wins > model1_wins:
            # 模型2胜出
            self.model_scores[model1_path] -= 1
            self.model_scores[model2_path] -= 3
        else:
            # 平局
            self.model_scores[model1_path] -= 2
            self.model_scores[model2_path] -= 2

        # 确保积分不会变成负数
        self.model_scores[model1_path] = max(0, self.model_scores[model1_path])
        self.model_scores[model2_path] = max(0, self.model_scores[model2_path])
    
    def _calculate_and_update_scores(self, pair_key, model1_path, model2_path):
        """计算并更新模型积分"""
        record = self.battle_records[pair_key]
        wins = record['wins']
        losses = record['losses']
        
        # 确定模型1和模型2在记录中的位置
        if model1_path <= model2_path:
            model1_wins = wins
            model2_wins = losses
        else:
            model1_wins = losses
            model2_wins = wins
        
        # 计算积分
        if model1_wins > model2_wins:
            # 模型1胜出
            self.model_scores[model1_path] += 3
            self.model_scores[model2_path] += 1
        elif model2_wins > model1_wins:
            # 模型2胜出
            self.model_scores[model1_path] += 1
            self.model_scores[model2_path] += 3
        else:
            # 平局
            self.model_scores[model1_path] += 2
            self.model_scores[model2_path] += 2
    
    def get_model_score(self, model_path):
        """获取模型积分"""
        return self.model_scores.get(model_path, 0)
    
    def get_battle_record(self, model1_path, model2_path):
        """获取模型对战记录"""
        pair_key = self.get_model_pair_key(model1_path, model2_path)
        return self.battle_records.get(pair_key, {'wins': 0, 'losses': 0})
    
    def get_leaderboard(self, top_n=20):
        """获取排行榜（按积分排序）"""
        # 转换为列表并排序
        scores_list = [(model_path, score) for model_path, score in self.model_scores.items()]
        scores_list.sort(key=lambda x: x[1], reverse=True)
        return scores_list[:top_n]
    
    def get_all_battle_records(self):
        """获取所有对战记录"""
        return self.battle_records

# 全局排行榜实例
leaderboard = Leaderboard()