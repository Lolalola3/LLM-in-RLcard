import openai
import seaborn as sns  
import time
import concurrent.futures
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
from loguru import logger
import json
import os
import rlcard
from rlcard import models
from rlcard.utils import print_card
from tqdm import tqdm
import random
from collections import defaultdict

# --- 配置 ---
BASE_URL = "https://api.toiotech.com/v1"
API_KEY = "sk-Pf2sSgJ6HmV77mmk763c4f5353Ec48B89850CcF029Bd83B8"

# 选定的2个模型
SELECTED_MODELS = [
    "doubao-1-5-thinking-pro-250415",
    "gpt-4.1-2025-04-14",
    "deepseek-r1-250120",
    "deepseek-v3-250324",
    "gemini-2.5-flash-preview-04-17",
    "o4-mini",
]

# 模型配置
THINKING_MODELS = {
    "doubao-1-5-thinking-pro-250415": {
        "model_name": "doubao-1-5-thinking-pro-250415",
        "thinking_chain": True,
        "assigned_role": None  # 将在后面随机分配
    },
    "gemini-2.5-flash-preview-04-17": {
        "model_name": "gemini-2.5-flash-preview-04-17",
        "thinking_chain": True,
        "assigned_role": None
    },
    "gpt-4.1-2025-04-14": {
        "model_name": "gpt-4.1-2025-04-14",
        "thinking_chain": True,
        "assigned_role": None
    },
    "deepseek-v3-250324": {
        "model_name": "deepseek-v3-250324",
        "thinking_chain": False
    },
    "o4-mini": {
        "model_name": "o4-mini",
        "thinking_chain": False  
    },
    "deepseek-r1-250120": {
        "model_name": "deepseek-r1-250120",
        "requires_thinking": True
    },
}

# 游戏配置
GAMES = {
    "limit-holdem": {
        "name": "Limit Texas Hold'em",
        "rlcard_env": "limit-holdem",
        "action_mapping": {
            "call": 0, "raise": 1, "fold": 2, "check": 3,
            0: "call", 1: "raise", 2: "fold", 3: "check"
        },
        "state_features": ["hand", "public_cards", "pot", "all_chips", "current_round"],
    }
}

# 系统角色配置 
SYSTEM_ROLES = {
    "aggressive": {
        "name": "激进型",
        "description": "倾向于主动进攻和冒险决策"
    },
    "conservative": {
        "name": "保守型",
        "description": "倾向于谨慎决策和风险规避"
    },
    "exploitative": {
        "name": "剥削型",
        "description": "分析对手行为模式并针对性利用对手弱点"
    }
}

# 实验参数
MAX_WORKERS = 3
TIMEOUT = 120
N_ROUNDS = 20
NUM_PLAYERS = 6

# --- 工具函数 ---
def get_error_message(e):
    if hasattr(e, 'body') and isinstance(e.body, dict):
        error_info = e.body.get('error', {})
        if isinstance(error_info, dict) and 'message' in error_info:
            return error_info['message']
        return str(e.body)
    elif hasattr(e, 'message'):
        return e.message
    return str(e)

def parse_llm_response(response, game_key):
    """解析LLM返回的JSON格式响应，增强对非标准JSON的容错处理"""
    def clean_json_string(s):
        """清洗JSON字符串中的非法字符"""
        # 替换常见非法字符
        s = s.replace('\u3000', ' ')  # 中文全角空格
        s = s.replace('\xa0', ' ')    # 不间断空格
        # 移除非法控制字符（保留\t\n\r）
        s = ''.join(char for char in s if ord(char) >= 32 or char in '\n\r\t')
        # 压缩连续空白字符
        s = ' '.join(s.split())
        return s.strip()

    try:
        json_str = None
        
        # 情况1：检查是否被 ```json ``` 包裹
        json_start = response.find('```json')
        if json_start >= 0:
            start_idx = json_start + 7
            end_idx = response.find('```', start_idx)
            if end_idx > start_idx:
                json_str = response[start_idx:end_idx].strip()
        
        # 情况2：检查是否被 ``` 包裹（无json标记）
        if json_str is None:
            json_start = response.find('```')
            if json_start >= 0:
                start_idx = json_start + 3
                end_idx = response.find('```', start_idx)
                if end_idx > start_idx:
                    candidate = response[start_idx:end_idx].strip()
                    if candidate.startswith('{') and candidate.endswith('}'):
                        json_str = candidate
        
        # 情况3：检查直接以 { } 包裹
        if json_str is None:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx+1].strip()
        
        # 最终尝试解析（增加清洗步骤）
        if json_str:
            json_str = clean_json_string(json_str)
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as e:
                # 尝试修复常见的JSON格式问题
                json_str = json_str.replace('\n', '\\n').replace('\t', '\\t')
                parsed = json.loads(json_str)
        else:
            response = clean_json_string(response)
            parsed = json.loads(response)
        
        # 验证action是否有效
        if isinstance(parsed.get('action'), (str, int)):
            action = str(parsed['action']).lower()
            action_map = GAMES[game_key]["action_mapping"]
            
            if action.isdigit():
                action_id = int(action)
                if action_id in action_map:
                    return action_id, parsed.get('reasoning', '')
            else:
                for action_id, action_name in action_map.items():
                    if isinstance(action_id, int) and action == action_name:
                        return action_id, parsed.get('reasoning', '')
        
        return 0, "无法解析LLM响应中的有效动作，使用默认动作(call)"
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败. 错误位置: {e.pos}\n原始响应片段: {response[max(0,e.pos-50):e.pos+50]}")
        return 0, "响应格式不符合JSON规范"
    
    except Exception as e:
        logger.error(f"解析LLM响应失败: {str(e)}\n完整响应: {response[:500]}...")
        return 0, f"解析错误: {str(e)}"
    
def decode_limit_holdem_state(state):
    raw_obs=state['raw_obs']
    new_state = {
        "hand": raw_obs.get('hand', []),
        "public_cards": raw_obs.get('public_cards', []),
        "all_chips": raw_obs.get('all_chips', []),
        "pot": sum(raw_obs.get('all_chips', [])),
        "current_round": _determine_round(raw_obs.get('public_cards', [])),
        "legal_actions": raw_obs.get('legal_actions', []),
        "my_chips": raw_obs.get('my_chips', 0),
        "raise_nums": raw_obs.get('raise_nums', []),
        "action_record": state.get('action_record', [])
    }
    
    return new_state

def _determine_round(public_cards):
    num_public = len(public_cards)
    return {
        0: "pre-flop", 3: "flop", 4: "turn", 5: "river"
    }.get(num_public, "unknown")


# --- 核心类 ---
class LLMAgent:
    def __init__(self, model_type, game_key, system_role, position):
        self.model_type = model_type
        self.game_key = game_key
        self.system_role = system_role
        self.position = position
        self.SB_position = None
        self.relative_position= None  
        self.config = THINKING_MODELS[model_type]
        self.game_config = GAMES[game_key]
        self.use_raw = True
        self.history = []
        self.opponent_history = []
        self.client = openai.OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            timeout=TIMEOUT
        )
    
    def step(self, state):
        return self.get_action(state)
    
    def eval_step(self, state):

        action = self.get_action(state)
        action = self.game_config["action_mapping"].get(action, 3)
        legal_action = state['raw_legal_actions']
        probs = [1.0 if a == action else 0.0 for a in legal_action]
        return action, probs
    
    def get_action(self, state):
        decoded_state = decode_limit_holdem_state(state)
        legal_action = decoded_state['legal_actions']
        legal_action_ids = [self.game_config["action_mapping"][a] for a in legal_action]
        
        prompt = self._build_prompt(decoded_state)
        start_time = time.time()
        
        try:
            response = self._call_model(prompt)
            action, reasoning = parse_llm_response(response, self.game_key)
            
            if action not in legal_action_ids:
                logger.warning(f"非法动作 {action}，从合法动作中选择第一个")
                action = legal_action_ids[0]
                
            self.history.append({
                "state": decoded_state,
                "action": action,
                "reasoning": reasoning
            })
            
            duration = time.time() - start_time
            logger.debug(f"{self.model_type} 响应成功 (耗时: {duration:.2f}s)")
            return action
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = get_error_message(e)
            logger.error(f"{self.model_type} 调用失败: {error_msg} (耗时: {duration:.2f}s)")
            return legal_action_ids[0]
    
    def _build_prompt(self, state):
        position_map = {
            0: "小盲位(SB)",
            1: "大盲位(BB)",
            2: "枪口位(UTG)",
            3: "中位(MP)",
            4: "关煞位(CO)",
            5: "按钮位(BTN)"
        }
        
        if self.SB_position is None:
            if state['all_chips'][self.position] == 1:  # 当前玩家是小盲
                self.SB_position = self.position
                self.relative_position = 0  # 自己是小盲位，相对位置为0
            elif state['all_chips'][self.position] == 2:  # 当前玩家是大盲
                self.SB_position = (self.position - 1) % 6
                self.relative_position = 1  # 大盲相对小盲的位置差为1
            elif state['all_chips'][self.position] == 0:
                # 通过all_chips数组找到小盲位(值为1的位置)
                try:
                    self.SB_position = state['all_chips'].index(1)
                except ValueError:
                    # 异常处理：如果没有找到小盲位(理论上不应该发生)
                    self.SB_position = (self.position - 1) % 6  # 默认假设前一位是小盲
                    logger.warning(f"无法确定小盲位，使用默认值: {self.SB_position}")
                
                # 计算当前玩家相对于小盲的位置
                self.relative_position = (self.position - self.SB_position) % 6
        
        prompt = f"""# 游戏背景
你正在参与{self.game_config['name']}游戏，扮演{SYSTEM_ROLES[self.system_role]['description']}角色。

## 游戏规则概述
{self._get_game_rules()}

## 当前游戏状态
{self._format_game_state(state)}

## 可用动作
{self._format_legal_actions(state['legal_actions'])}

## 对手历史动作
{self._format_action_records(state['action_record'])}

## 策略指引
- 分析对手历史行为模式并针对性利用对手弱点
- 考虑位置优劣势: 你的位置是: {position_map.get(self.relative_position, "未知")}
"""

        prompt += """
## 思考要求
请使用思维链方式逐步分析当前局势，包括：
1. 手牌强度和潜在提升空间
2. 对手可能的范围和策略
3. 底池赔率和期望值计算
4. 位置优势和下注轮次
5. 最终决策及理由

## 响应格式要求{
    "reasoning": "你的详细思考过程", 
    "action": "你的选择(0=call, 1=raise, 2=fold, 3=check)",
    "confidence": 0.0-1.0
}"""
        return prompt
    
    def _get_game_rules(self):
        return """
- 限注德州扑克是基于德扑的游戏
- 游戏分为4个下注轮次: 翻牌前(pre-flop), 翻牌(flop), 转牌(turn), 河牌(river)
- 特殊限制：每个下注轮次所有人最多只能加注4次，加注额度固定
- 翻牌前（Pre-flop）和翻牌圈（Flop）每次加注必须等于 1个大盲注（Big Blind）
- 转牌（Turn）和河牌（River）每次加注必须等于 2个大盲注（Big Blind）
- 动作约束：有人下注后必须call/raise/fold，无人下注时可用check代替call
"""
    
    def _format_game_state(self, state):
        formatted = f"""- 当前轮次: {state['current_round']}
- 手牌: {', '.join(state['hand'])}
- 公共牌: {', '.join(state['public_cards']) if state['public_cards'] else '无'}
- 底池总额: {state['pot']}
- 各玩家下注: {state['all_chips']}（下标从0开始）,你的绝对位置编号为 {self.position}，小盲绝对位置编号为 {self.SB_position}
- 所有玩家各个轮次的已加注次数: {state['raise_nums']}
"""
        
        if self.opponent_history:
            formatted += "- 对手最近动作: " + ", ".join(self.opponent_history[-3:]) + "\n"
        
        return formatted
    
    def _format_action_records(self, records):
        if not records:
            return "暂无历史动作记录"
        
        formatted = []
        for player_id, action in records:
            formatted.append(f"玩家{player_id}: {action}")
        return "\n".join(formatted)
    
    def _format_legal_actions(self, legal_actions):
        return "\n".join([f"- {action}" for action in legal_actions])
    
    def _call_model(self, prompt):
        create_kwargs = {
            "model": self.config["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        
        if "gemini" in self.model_type:
            create_kwargs["reasoning_effort"] = "high"
        elif "claude" in self.model_type or "doubao" in self.model_type:
            create_kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": 50}}
        
        response = self.client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content.strip()

# --- 实验逻辑 ---
def run_multi_agent_game(model_agents, n_games=N_ROUNDS):
    """运行多智能体游戏"""
    env = rlcard.make('limit-holdem', config={
        'seed': random.randint(0, 100000), 
        'game_num_players': len(model_agents)
    })
    
    # 运行实验
    results = []
    for _ in tqdm(range(n_games), desc="游戏进度", unit="game"):
        env.set_agents(model_agents)
        trajectories, payoffs = env.run(is_training=False)
        results.append(payoffs)
    
    return np.array(results)  # 返回所有玩家的收益矩阵


def run_experiment_round(model_assignments):
    """运行一轮实验"""
    # 创建模型agents
    model_agents = []
    for i, (model_type, role) in enumerate(model_assignments):
        agent = LLMAgent(model_type, "limit-holdem", system_role=role, position=i)
        model_agents.append(agent)
    
    # 运行游戏
    results = run_multi_agent_game(model_agents)
    
    # 计算指标
    metrics = calculate_metrics(results, model_assignments)
    return metrics

def calculate_metrics(results, model_assignments):
    """计算标准博弈论评估指标"""
    num_games = results.shape[0]
    num_players = results.shape[1]
    
    # 初始化数据结构
    model_stats = defaultdict(lambda: {
        'total_payoff': 0,
        'count': 0,
        'max_payoff': -np.inf,  # 用于计算regret
        'role_stats': defaultdict(lambda: {'payoff': 0, 'count': 0, 'max_payoff': -np.inf})
    })
    
    # 首先计算每个模型的最大可能收益（用于regret计算）
    max_possible = np.max(results, axis=1)  # 每局游戏中的最高得分
    
    # 收集数据
    for game_idx in range(num_games):
        current_max = max_possible[game_idx]
        for player_idx in range(num_players):
            model_type, role = model_assignments[player_idx]
            payoff = results[game_idx, player_idx]
            
            # 更新模型统计
            model_stats[model_type]['total_payoff'] += payoff
            model_stats[model_type]['count'] += 1
            model_stats[model_type]['max_payoff'] = max(model_stats[model_type]['max_payoff'], current_max)
            
            # 更新角色统计
            role_stats = model_stats[model_type]['role_stats'][role]
            role_stats['payoff'] += payoff
            role_stats['count'] += 1
            role_stats['max_payoff'] = max(role_stats['max_payoff'], current_max)
    
    # 计算标准指标
    metrics = {}
    for model_type, stats in model_stats.items():
        avg_payoff = stats['total_payoff'] / stats['count']
        total_max_payoff = stats['max_payoff'] * stats['count']  # 理论最大总收益
        
        # 1. 计算Regret（遗憾值）
        regret = (total_max_payoff - stats['total_payoff']) / stats['count'] if stats['count'] > 0 else 0
        
        # 2. 计算Exploitability（可剥削性）
        # 需要计算当其他玩家使用最佳响应时的收益损失
        # 这里简化计算：使用对最优策略的偏差作为可剥削性
        exploitability = max(0, (total_max_payoff - stats['total_payoff']) / (total_max_payoff + 1e-10))
        
        # 3. 计算Social Welfare（社会福祉）
        social_welfare = stats['total_payoff'] / (stats['count'] * num_players)  # 平均每人每局贡献
        
        # 按角色细分的指标
        role_metrics = {}
        for role, role_stats in stats['role_stats'].items():
            role_avg = role_stats['payoff'] / role_stats['count'] if role_stats['count'] > 0 else 0
            role_max = role_stats['max_payoff'] * role_stats['count']
            role_regret = (role_max - role_stats['payoff']) / role_stats['count'] if role_stats['count'] > 0 else 0
            
            role_metrics[role] = {
                'avg_payoff': role_avg,
                'regret': role_regret,
                'games_played': role_stats['count'],
                'win_rate': np.sum(results[:, [i for i, (m, r) in enumerate(model_assignments) 
                                             if m == model_type and r == role]] > 0) / max(1, role_stats['count'])
            }
        
        metrics[model_type] = {
            'avg_payoff': avg_payoff,
            'regret': regret,
            'exploitability': exploitability,
            'social_welfare': social_welfare,
            'role_performance': role_metrics,
            'total_games': stats['count']
        }
    
    return metrics

def run_full_experiment():
    """运行完整实验(3轮)，每轮6个模型(2激进+2保守+2剥削)，三轮角色轮换"""
    all_metrics = []
    
    # 6个模型列表
    all_models = list(THINKING_MODELS.keys())  
    
    # 定义三轮的角色轮换计划 (model_index : [role1, role2, role3])
    role_rotation_plan = {
        0: ["aggressive", "conservative", "exploitative"],  # 模型0的三轮角色
        1: ["conservative", "exploitative", "aggressive"],
        2: ["exploitative", "aggressive", "conservative"],
        3: ["aggressive", "conservative", "exploitative"],
        4: ["conservative", "exploitative", "aggressive"],
        5: ["exploitative", "aggressive", "conservative"]
    }
    
    for round_idx in range(3):
        logger.info(f"=== 开始第 {round_idx+1} 轮实验 ===")
        
        # 为当前轮次分配角色
        model_assignments = []
        for model_idx in range(6):
            model_type = all_models[model_idx]
            role = role_rotation_plan[model_idx][round_idx]
            model_assignments.append((model_type, role))
        
        # 打印当前分配
        logger.info("当前轮次模型角色分配:")
        for i, (model_type, role) in enumerate(model_assignments):
            logger.info(f"玩家{i}: {model_type} 作为 {SYSTEM_ROLES[role]['name']}")
        
        # 运行实验
        round_metrics = run_experiment_round(model_assignments)
        all_metrics.append(round_metrics)
        
        # 保存中间结果
        os.makedirs("results", exist_ok=True)
        with open(f"results/round_{round_idx}_metrics.json", "w") as f:
            json.dump(round_metrics, f, indent=2)
    
    return all_metrics


def analyze_and_visualize(all_metrics):
    """Analyze and visualize results (supporting new metrics)"""
    os.makedirs("results", exist_ok=True)
    
    # Aggregate data
    metric_names = ['avg_payoff', 'regret', 'exploitability', 'social_welfare']
    model_comparison = {metric: defaultdict(list) for metric in metric_names}
    role_comparison = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for round_metrics in all_metrics:
        for model_type, metrics in round_metrics.items():
            for metric in metric_names:
                model_comparison[metric][model_type].append(metrics[metric])
            
            for role, perf in metrics['role_performance'].items():
                role_comparison[model_type][role]['avg_payoff'].append(perf['avg_payoff'])
                role_comparison[model_type][role]['win_rate'].append(perf['win_rate'])
    
    # 1. Model comparison plot (multi-subplot)
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Model Performance Comparison")
    
    for idx, metric in enumerate(metric_names):
        ax = axs[idx//2, idx%2]
        for model_type, values in model_comparison[metric].items():
            ax.plot(range(1, 4), values, label=model_type, marker='o')
        
        ax.set_title(metric.upper())
        ax.set_xlabel("Round")
        ax.set_ylabel(metric)
        ax.set_xticks([1, 2, 3])
        ax.legend()
        ax.grid()
    
    plt.tight_layout()
    plt.savefig("results/model_metrics_comparison.png")
    plt.close()
    
    # 2. Role performance heatmap
    for model_type in role_comparison.keys():
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        roles = list(SYSTEM_ROLES.keys())
        metrics = ['avg_payoff', 'win_rate']
        data = np.zeros((len(roles), len(metrics)))
        
        for i, role in enumerate(roles):
            for j, metric in enumerate(metrics):
                data[i, j] = np.mean(role_comparison[model_type][role][metric])
        
        # Draw heatmap
        sns.heatmap(data, 
                   annot=True, 
                   fmt=".2f",
                   xticklabels=metrics,
                   yticklabels=[SYSTEM_ROLES[r]['name'] for r in roles],
                   cmap="YlGnBu")
        
        plt.title(f"{model_type} Role Performance Heatmap")
        plt.savefig(f"results/heatmap_{model_type}.png")
        plt.close()
    
    # 3. Regret trend plot
    plt.figure(figsize=(12, 6))
    for model_type, values in model_comparison['regret'].items():
        plt.plot(range(1, 4), values, label=model_type, marker='o')
    
    plt.title("Regret Trend Across Models")
    plt.xlabel("Round")
    plt.ylabel("Regret Value")
    plt.xticks([1, 2, 3])
    plt.legend()
    plt.grid()
    plt.savefig("results/regret_trend.png")
    plt.close()
    
    # 4. Save enhanced final results
    enhanced_results = {
        'model_metrics': model_comparison,
        'role_performance': role_comparison,
        'summary_stats': {
            'best_model': max(
                [(m, np.mean(vals)) for m, vals in model_comparison['social_welfare'].items()],
                key=lambda x: x[1]
            )[0],
            'most_consistent_model': min(
                [(m, np.std(vals)) for m, vals in model_comparison['regret'].items()],
                key=lambda x: x[1]
            )[0],
            'best_performing_role': max(
                [(role, np.mean(role_comparison[model][role]['avg_payoff'])) 
                 for model in role_comparison.keys() 
                 for role in role_comparison[model].keys()],
                key=lambda x: x[1]
            )[0]
        }
    }
    
    with open("results/enhanced_final_results.json", "w") as f:
        json.dump(enhanced_results, f, indent=2)

# --- 主程序 ---
if __name__ == "__main__":
    # 初始化日志
    logger.add("experiment.log", rotation="10 MB")
    
    # 运行实验
    logger.info("开始实验...")
    all_metrics = run_full_experiment()
    
    # 分析和可视化
    analyze_and_visualize(all_metrics)
    
    logger.info("实验完成！")
    logger.info("结果已保存到 results/ 目录")