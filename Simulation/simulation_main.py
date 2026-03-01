import os
import json
import getpass
import math
import random
import time
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
try:
    import MeCab
    import ipadic
except ImportError:
    MeCab = None
    ipadic = None

from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 0. 初期設定と定数
# ==========================================
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OPENAI_API_KEY: ")
client = OpenAI(
    timeout=900.0,  
    max_retries=3    
)

STOP_WORDS = [
    'の', 'に', 'は', 'を', 'が', 'と', 'て', 'で', 'です', 'ます', 
    'した', 'いる', 'ある', 'ような', 'ように', '見え', '見える', 
    '形', '様子', '全体', '部分', '図形', 'タングラム', '思う', '感じ',
    '私', 'あなた', 'それ', 'これ', 'あれ', 'こと', 'もの'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 1. ユーティリティ関数定義
# ==========================================
def call_api_with_retry(func, max_retries=5, initial_wait=1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_msg = str(e)
            if any(x in error_msg for x in ["Rate limit", "429", "500", "503", "502"]):
                if attempt == max_retries - 1: raise e
                wait_time = (initial_wait * (2 ** attempt)) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise e

def _sanitize_str(val: Any) -> Optional[str]:
    if val is None: return None
    if isinstance(val, (dict, list)): return str(val) 
    return str(val)

class suppress_output:
    def __init__(self, suppress=True): pass
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_val, exc_tb): pass

# ==========================================
# 2. データマネージャー
# ==========================================
class DataManager:
    _instance = None
    _model = None       
    
    def __init__(self, csv_path="data.csv"):
        print(f"📥 Loading DataManager from {csv_path}...")
        self.csv_path = csv_path
        self.df = pd.DataFrame()
        self.tangram_centroids = {}
        self.all_labels = []
        self.context_text = "" 
        
        # SBERTモデルのロード
        if DataManager._model is None:
            if SentenceTransformer is None:
                print("⚠️ sentence-transformers not found.")
                DataManager._model = None
            else:
                try:
                    print("⏳ Loading SBERT model...")
                    DataManager._model = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens', device="cpu",model_kwargs={"low_cpu_mem_usage": False})
                    print("✅ SBERT Loaded successfully.")
                except Exception as e:
                    print(f"⚠️ SBERT Load Failed: {e}")
                    DataManager._model = None
        
        self.model = DataManager._model
        
        # MeCabの初期化
        self.tagger = None
        if MeCab and ipadic:
            try:
                self.tagger = MeCab.Tagger(ipadic.MECAB_ARGS + " -Ochasen")
                print("✅ MeCab Tagger initialized.")
            except Exception as e:
                print(f"⚠️ MeCab Init Failed: \n{e}")
        else:
            print("⚠️ MeCab or ipadic library not found. Using raw text for similarity.")

        self._load_and_process()

    def _remove_stopwords(self, text):
        if not isinstance(text, str): return ""
        for word in STOP_WORDS:
            text = text.replace(word, " ")
        return " ".join(text.split())

    def extract_features_from_text(self, text: str) -> str:
        if not self.tagger or not text:
            return self._remove_stopwords(text)
        
        try:
            node = self.tagger.parseToNode(text)
            keywords = []
            while node:
                features = node.feature.split(",")
                pos = features[0]
                word = node.surface
                if pos in ["名詞", "形容詞", "動詞"]:
                    if word not in STOP_WORDS:
                        keywords.append(word)
                node = node.next
            return " ".join(keywords)
        except Exception:
            return self._remove_stopwords(text)

    def _load_and_process(self):
        if not os.path.exists(self.csv_path):
            print(f"⚠️ {self.csv_path} not found. Creating dummy data.")
            dummy_data = {
                'label': ['A1']*5 + ['A2']*5 + ['B1']*5 + ['B2']*5 + ['B3']*5 + ['C1']*5,
                'text': [f'特徴_{i}' for i in range(30)],
                'exp': ['Holistic']*15 + ['Analytic']*15
            }
            self.df = pd.DataFrame(dummy_data)
        else:
            self.df = pd.read_csv(self.csv_path)

        self.df['processed_text'] = self.df['text'].apply(self._remove_stopwords)
        self.all_labels = sorted(self.df['label'].unique().tolist())

        context_lines = []
        for label in self.all_labels:
            features = self.df[self.df['label'] == label]['text'].tolist()
            feat_str = " / ".join(features)
            context_lines.append(f"【ID: {label}】\n特徴: {feat_str}\n")
        self.context_text = "\n".join(context_lines)

        if self.model:
            print("🧮 Calculating Centroids...")
            for label in self.all_labels:
                texts = self.df[self.df['label'] == label]['processed_text'].tolist()
                if texts:
                    vectors = self.model.encode(texts)
                    centroid = np.mean(vectors, axis=0)
                    self.tangram_centroids[label] = centroid
        else:
            for label in self.all_labels:
                self.tangram_centroids[label] = np.random.rand(768)

    def get_most_distinct_target(self, candidate_labels: List[str]) -> str:
        if not candidate_labels: return None
        if len(candidate_labels) == 1: return candidate_labels[0]
        if self.model is None: return random.choice(candidate_labels)

        target_vectors = np.array([self.tangram_centroids[l] for l in candidate_labels])
        sim_matrix = cosine_similarity(target_vectors)
        avg_similarities = np.mean(sim_matrix, axis=1)
        min_sim_idx = np.argmin(avg_similarities)
        return candidate_labels[min_sim_idx]
    
    def encode_text(self, text: str):
        if self.model and text:
            filtered_text = self.extract_features_from_text(text)
            if not filtered_text.strip(): 
                filtered_text = text
            return self.model.encode([filtered_text])[0]
        return np.zeros(768)

# ==========================================
# 3. エージェントクラス定義
# ==========================================

#リーダ
class LeaderAgent:
    def __init__(self, name="Leader", max_char_count=50, csv_path="data.csv"):
        self.name = name
        self.max_char_count = max_char_count
        self.data_manager = DataManager(csv_path)
        self.log_buffer: List[str] = []
        self.current_target_id = None
        self.unnamed_candidates = list(self.data_manager.all_labels)
        self.named_map = {} 
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.expression_counts = {"Holistic": 0, "Analytic": 0, "Mixed": 0}

    def log(self, msg):
        self.log_buffer.append(f"[{self.name}] {msg}")

    def select_next_target(self):
        if not self.unnamed_candidates:
            return None
        best_target = self.data_manager.get_most_distinct_target(self.unnamed_candidates)
        self.current_target_id = best_target
        self.log(f"🎯 Target Selected: {self.current_target_id}")
        return self.current_target_id

    def generate_utterance(self, full_history: str, is_naming_phase: bool = False) -> Dict:
        if not self.current_target_id:
            return {"utterance": "終了です。", "thought_process": "完了", "strategy": "None"}

        target_features = self.data_manager.df[
            self.data_manager.df['label'] == self.current_target_id
        ]['text'].tolist()
        features_snippet = "\n".join(target_features)

        if is_naming_phase:
            task_instruction = f"""
【現在の状況】
相手との共通認識が形成されました。**命名フェーズ**に移行します。
ターゲット図形【{self.current_target_id}】にふさわしい、短く覚えやすい**「名前」**を一つ提案してください。
ただし，これまでの対話履歴からわかる今までに命名した名前とは被らないようにしてください．
また、発話内容にタングラムIDは含めないでください．
発話内容は最大でもひらがなにした時に{self.max_char_count}文字以内に収めてください．
収めながら命名提案ができない場合は「『〜』で」のように最低限の文字数で，発話内容を出力してください．
ただし，発話として出力する際は，漢字やカタカナにすべき語句は必ず変換してから出力してください．
音声対話として自然な発話を出力してください．
自然対話を想定するため箇条書きや括弧書きなど，自然対話では使用されない表現は使用しないでください．

**抽出指示:**
自分の生成した発話(utterance)の中に含まれる以下の表現を**抽出してリスト形式で**JSONに出力してください。
- **Analyticな表現**: 部分的・幾何学的な特徴
- **Holisticな表現**: 全体的・抽象的な印象

【出力フォーマット (JSON)】
{{
  "thought_process": "...",
  "strategy": "Naming_Proposal",
  "analytic_expressions": [],
  "holistic_expressions": ["ウサギ"],
  "utterance": "それでは、この図形を『ウサギ』と呼びませんか？",
  "proposed_name": "ウサギ"
}}
"""
        else:
            remaining_count = len(self.unnamed_candidates)
            remaining_ids_str = ", ".join(self.unnamed_candidates)
            
            task_instruction = f"""
【目標設定: {self.max_char_count}文字以内の説明】
ターゲット図形を，指定した文字数で可能な限り詳細に説明してください．
また，相手の見ている図形は自分の見ている図形と回転角度が異なる可能性があることを考慮してください．
**指示:**
発話内容は**ひらがなにした時に{self.max_char_count}文字ほど**になるようなターゲット図形の説明にしてください。
誤差はプラスマイナス3文字までです．
ただし，発話として出力する際は，漢字やカタカナにすべき語句は必ず変換してから出力してください．
音声対話として自然な発話を出力してください．
ただし，自然対話を想定するため箇条書きや括弧書きなど，自然対話では使用されない表現は発話内容(utterance)では使用しないでください．
また、発話内容にタングラムIDは含めないでください．

また，対話履歴を見て，今のタングラムの説明で過去にすでに伝えていた特徴や要素は発話には含めないようにしてください．
基本的には今の対話から1つ前の命名の対話までに伝えていた特徴や要素はすでに伝えている情報になります．

**抽出指示:**
自分の生成した発話(utterance)の中に含まれる以下の表現を**抽出してリスト形式で**JSONに出力してください。該当なしの場合は空リスト `[]` にしてください。
直前の相手（Follower）の発話からは抽出しないでください．
- **Analyticな表現**: 部分的・幾何学的な特徴のフレーズ（例："三角形がある", "右側が尖っている"）
- **Holisticな表現**: 全体的・抽象的な印象のフレーズ（例："走っている人のようだ", "不安定な感じ"）

**現在の状況分析:**
- 未命名の候補: [{remaining_ids_str}] (計{remaining_count}個)

【出力フォーマット (JSON)】
{{
  "thought_process": "...",
  "strategy": "Description",
  "analytic_expressions": ["..."],
  "holistic_expressions": ["..."],
  "utterance": "走っている人のようなタングラムはありますか？",
  "proposed_name": null
}}
"""

        prompt = f"""
あなたはタングラムゲームの「出題者（Leader）」です。
ターゲット図形【{self.current_target_id}】について話しています。

【ターゲットの特徴データ】
{features_snippet}

【全タングラムの特徴リスト（参考用）】
{self.data_manager.context_text}

{task_instruction}

【ここまでの全対話履歴】
{full_history}
"""
        res = call_api_with_retry(lambda: client.chat.completions.create(
            model="gpt-5", 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            reasoning_effort="low"
        ))
        
        self.total_prompt_tokens += res.usage.prompt_tokens
        self.total_completion_tokens += res.usage.completion_tokens
        
        try:
            data = json.loads(res.choices[0].message.content)
        except:
            data = {"utterance": "Error", "thought_process": "Error", "proposed_name": None}

        
        analytic_list = data.get("analytic_expressions", [])
        holistic_list = data.get("holistic_expressions", [])
        
        if not isinstance(analytic_list, list): analytic_list = []
        if not isinstance(holistic_list, list): holistic_list = []

        has_analytic = len(analytic_list) > 0
        has_holistic = len(holistic_list) > 0
        final_type = "None"

        
        self.expression_counts["Analytic"] += len(analytic_list)
        self.expression_counts["Holistic"] += len(holistic_list)

        if has_analytic and has_holistic:
            final_type = "Mixed"
        elif has_analytic:
            final_type = "Analytic"
        elif has_holistic:
            final_type = "Holistic"
        
        data["proposed_name"] = _sanitize_str(data.get("proposed_name"))

        self.log(f"🧠 Thought: {data.get('thought_process')}")
        self.log(f"📐 Strategy: {data.get('strategy')} | Type: {final_type}")
        if analytic_list:
            self.log(f"   [Analytic]: {analytic_list}")
        if holistic_list:
            self.log(f"   [Holistic]: {holistic_list}")
        if data.get("proposed_name"):
             self.log(f"💡 Proposing Name: {data.get('proposed_name')}")

        return data

    
    def handle_revoke(self, revoked_name):
        target_name = _sanitize_str(revoked_name)
        id_to_remove = None
        
        for tid, name in self.named_map.items():
            if name == target_name:
                id_to_remove = tid
                break
        
        if id_to_remove:
            del self.named_map[id_to_remove]
            if id_to_remove not in self.unnamed_candidates:
                self.unnamed_candidates.append(id_to_remove)
            self.log(f"🔄 Revoked Name: '{target_name}' (ID: {id_to_remove}). Returned to candidates.")
            return True
        else:
            self.log(f"⚠️ Revoke Failed: Name '{target_name}' not found in named_map.")
            return False

    def mark_current_target_done(self, agreed_name):
        if self.current_target_id:
            self.named_map[self.current_target_id] = agreed_name 
            if self.current_target_id in self.unnamed_candidates:
                self.unnamed_candidates.remove(self.current_target_id)
            self.log(f"✅ Naming Completed: ID={self.current_target_id}, Name={agreed_name}")
            self.current_target_id = None

#フォロワー
class FollowerAgent:
    def __init__(self, name="Follower", max_char_count=50, csv_path="dataA.csv"):
        self.name = name
        self.max_char_count = max_char_count 
        self.data_manager = DataManager(csv_path)
        self.log_buffer: List[str] = []
        
        self.pn_probs = {label: 0.0 for label in self.data_manager.all_labels}
        self.pa_probs = {} 
        self.named_history = [] 
        self.named_map = {}
        
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.expression_counts = {"Holistic": 0, "Analytic": 0, "Mixed": 0}

    def log(self, msg):
        self.log_buffer.append(f"[{self.name}] {msg}")

    def respond(self, leader_utterance: str, full_history: str, proposed_name_by_leader: str = None) -> Dict:
        all_ids = self.data_manager.all_labels
        named_ids = self.named_history
        unnamed_ids = [lid for lid in all_ids if lid not in named_ids]

        unnamed_str = ", ".join(unnamed_ids)
        named_str = ", ".join(named_ids)

        if proposed_name_by_leader:
            task_instruction = f"""
【タスク: 命名の合意】
Leaderから名前「{proposed_name_by_leader}」が提案されました。
合意する場合 `accepted_name` に名前を出力、拒否する場合 null。
基本的に提案された命名の拒否は行わないでください．
発話内容(utterance)にタングラムIDは絶対に含めないでください．
発話内容は**最大でもひらがなにした時に{self.max_char_count}文字以内**に収めてください．
ただし，発話として出力する際は，漢字やカタカナにすべき語句は必ず変換してから出力してください．
音声対話として自然な発話を出力してください．
自然対話を想定するため箇条書きや括弧書きなど，自然対話では使用されない表現は使用しないでください．
**抽出指示:**
⚠️**重要:** 抽出対象は、**あなたが今回生成する `utterance` の文言のみ**です。
直前のLeaderの発言内容を含めないでください。
生成した発話(utterance)の中に含まれる以下の表現を**抽出してリスト形式で**JSONに出力してください。
- **Analyticな表現**: 部分的・幾何学的な特徴
- **Holisticな表現**: 全体的・抽象的な印象

【出力フォーマット (JSON)】
{{
  "pn_probabilities": {{...}}, 
  "pa_probabilities": {{...}}, 
  "accepted_name": "{proposed_name_by_leader}",
  "analytic_expressions": [],
  "holistic_expressions": [],
  "thought_process": "...",
  "utterance": "..."
}}
"""
        else:
            task_instruction = f"""
【タスク: 推論】
相手の発話内容から、相手がどのタングラムの説明をしている可能性があるのか推測してください。
また，相手の見ている図形は自分の見ている図形と回転角度が異なる可能性があることを考慮してください．

**1. PN (未命名タングラムの確率) の計算**
未命名の候補リスト [{unnamed_str}] の中で、相手の説明がどれに当てはまるか確率を推定してください。
- 合計が 1.0 になるようにしてください。
- **重要:** 相手の説明に合致する候補が**ない**と判断した場合は、確率を一律低く設定し、思考プロセスに「合致なし」と記述してください。

**2. PA (命名済みタングラムへの当てはまり) の計算**
命名済みリスト [{named_str}] の各タングラムについて、「現在の相手の説明がどれくらい当てはまってしまっているか」を確率(0.0~1.0)で推定してください。
- **相手の説明が、ある命名済みタングラムの特徴と酷似している場合、そのIDの確率を高くしてください**。
- 全く当てはまらない場合は 0.0 に近づけてください。

発話内容(utterance)にタングラムIDは絶対に含めないでください．
また発話内容は，直前に相手の提示してきた説明に当てはまるタングラムがあったかどうかと，当てはまると判断した根拠の特徴を必ず含めるようにしてください．
相手の説明が部分的・幾何学的な特徴のフレーズならば当てはまると判断した根拠の特徴もできるならAnalyticな表現から，
相手の説明が全体的・抽象的な印象のフレーズならば当てはまると判断した根拠の特徴もできるならHolisticな表現から選択してください．
発話内容は**最大でもひらがなにした時に{self.max_char_count}文字以内**に収めてください．
ただし，当てはまると判断した根拠の特徴が入らない場合は，+5文字程度なら許容します．当てはまると判断した根拠の特徴を含めることを優先してください．
ただし，発話として出力する際は，漢字やカタカナにすべき語句は必ず変換してから出力してください．
音声対話として自然な発話を出力してください．
ただし，自然対話を想定するため箇条書きや括弧書きなど，自然対話では使用されない表現は発話内容(utterance)では使用しないでください．
決して新しい特徴や新しい要素の提案は行わないでください．

**拒否の指示:**
もし直前の相手の説明に合致するタングラムが候補の中に**一つも無い**と判断した場合は、`utterance` には **「ありません」** とだけ出力してください。余計な言葉は含めないでください。

**抽出指示:**
⚠️**重要:** 抽出対象は、**あなたが今回生成する `utterance` の文言のみ**です。
直前のLeaderの発言内容（"{leader_utterance}"）から抽出しないでください。
生成した発話(utterance)の中に含まれる以下の表現を**抽出してリスト形式で**JSONに出力してください。該当なしの場合は空リスト `[]` にしてください。
- **Analyticな表現**: 部分的・幾何学的な特徴のフレーズ
- **Holisticな表現**: 全体的・抽象的な印象のフレーズ
また，thought_processにはなぜそのように確率を変動させたのか丁寧に記述してください。

【出力フォーマット (JSON)】
{{
  "pn_probabilities": {{"A2": 0.7, "B1": 0.3, ...}}, 
  "pa_probabilities": {{"A1": 0.8, ...}}, // ★重要: 説明に合致してしまっている確率(高い=撤回候補)
  "accepted_name": null,
  "analytic_expressions": [],
  "holistic_expressions": [],
  "thought_process": "...",
  "utterance": "..."
}}
"""

        prompt = f"""
あなたはタングラムゲームの「回答者（Follower）」です。

【全タングラムの特徴リスト】
{self.data_manager.context_text}

【現在の状況】
- 未命名候補: [{unnamed_str}]
- 命名済み（除外対象）: [{named_str}]

【ここまでの全対話履歴】
{full_history}

【出題者の最新の発言】
"{leader_utterance}"

{task_instruction}
"""
        res = call_api_with_retry(lambda: client.chat.completions.create(
            model="gpt-5", 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            reasoning_effort="low"
        ))
        
        self.total_prompt_tokens += res.usage.prompt_tokens
        self.total_completion_tokens += res.usage.completion_tokens
        
        try:
            data = json.loads(res.choices[0].message.content)
        except:
            data = {
                "pn_probabilities": {}, "pa_probabilities": {}, 
                "accepted_name": None, "utterance": "..."
            }

        analytic_list = data.get("analytic_expressions", [])
        holistic_list = data.get("holistic_expressions", [])
        
        if not isinstance(analytic_list, list): analytic_list = []
        if not isinstance(holistic_list, list): holistic_list = []

        has_analytic = len(analytic_list) > 0
        has_holistic = len(holistic_list) > 0
        final_type = "None"

        self.expression_counts["Analytic"] += len(analytic_list)
        self.expression_counts["Holistic"] += len(holistic_list)

        if has_analytic and has_holistic:
            final_type = "Mixed"
        elif has_analytic:
            final_type = "Analytic"
        elif has_holistic:
            final_type = "Holistic"

        self.pn_probs = data.get("pn_probabilities", {})
        total_pn = sum(float(v) for v in self.pn_probs.values())
        if total_pn > 0:
            self.pn_probs = {k: float(v)/total_pn for k, v in self.pn_probs.items()}
        
        raw_pa = data.get("pa_probabilities", {})
        self.pa_probs = {}
        for named_id in self.named_history:
            if named_id in raw_pa:
                self.pa_probs[named_id] = float(raw_pa[named_id])
            else:
                self.pa_probs[named_id] = 0.0
        
        data["revoke_request"] = None 
        data["accepted_name"] = _sanitize_str(data.get("accepted_name"))
        
        self.log(f"🤔 Thought: {data.get('thought_process')}")
        self.log(f"📐 Strategy: Type:{final_type}")
        if analytic_list:
            self.log(f"   [Analytic]: {analytic_list}")
        if holistic_list:
            self.log(f"   [Holistic]: {holistic_list}")
        
        sorted_pn = sorted(self.pn_probs.items(), key=lambda x: float(x[1]), reverse=True)
        pn_str = ", ".join([f"{k}:{v:.2f}" for k, v in sorted_pn])
        self.log(f"📊 PN (Unnamed): {{{pn_str}}}")
        
        if self.pa_probs:
            sorted_pa = sorted(self.pa_probs.items(), key=lambda x: float(x[1]), reverse=True)
            pa_str = ", ".join([f"{k}:{v:.2f}" for k, v in sorted_pa])
            self.log(f"🛡️ PA (Named - Matching?): {{{pa_str}}}")

        if data.get("accepted_name"):
            self.log(f"🤝 Accepted Name: {data.get('accepted_name')}")

        return data

    def update_named_status(self, target_id, name):
        if target_id not in self.named_history:
            self.named_history.append(target_id)
            self.named_map[target_id] = name 
            self.pa_probs[target_id] = 0.0

    def handle_revoke_accepted(self, revoked_id):
        if revoked_id in self.named_history:
            self.named_history.remove(revoked_id)
            if revoked_id in self.named_map:
                del self.named_map[revoked_id] 
            if revoked_id in self.pa_probs:
                del self.pa_probs[revoked_id]

# ==========================================
# 4. ゲームマスター (進行管理)
# ==========================================
class GameMaster:
    def __init__(self, session_id, max_turns=30, max_char_count=50, leader_data="data.csv", follower_data="dataA.csv"):
        self.session_id = session_id
        self.max_turns = max_turns
        self.max_char_count = max_char_count 
        
        self.leader = LeaderAgent(name="Leader", max_char_count=max_char_count, csv_path=leader_data)
        self.follower = FollowerAgent(name="Follower", max_char_count=max_char_count, csv_path=follower_data)
        self.data_manager = self.leader.data_manager 
        
        self.chronological_log = []
        self.conversation_log = [] 
        self.turn_count = 0
        self.is_naming_phase = False
        self.no_match_counter = 0

    def log_system(self, msg):
        self.chronological_log.append(f"[System] {msg}")

    def _capture_logs(self):
        if self.leader.log_buffer:
            self.chronological_log.extend(self.leader.log_buffer)
            self.leader.log_buffer = []
        if self.follower.log_buffer:
            self.chronological_log.extend(self.follower.log_buffer)
            self.follower.log_buffer = []

    def _get_cost_summary(self):
        l_in = self.leader.total_prompt_tokens
        l_out = self.leader.total_completion_tokens
        f_in = self.follower.total_prompt_tokens
        f_out = self.follower.total_completion_tokens
        total = l_in + l_out + f_in + f_out
        l_counts = self.leader.expression_counts
        f_counts = self.follower.expression_counts 
        
        summary = (
            f"\n📊 Expression Counts:\n"
            f"   [Leader]\n"
            f"     - Holistic: {l_counts['Holistic']}\n"
            f"     - Analytic: {l_counts['Analytic']}\n"
            f"     - Mixed   : {l_counts['Mixed']}\n"
            f"   [Follower]\n"
            f"     - Holistic: {f_counts['Holistic']}\n"
            f"     - Analytic: {f_counts['Analytic']}\n"
            f"     - Mixed   : {f_counts['Mixed']}\n\n"
            f"💰 Token Usage Summary (Session {self.session_id}):\n"
            f"   [Leader]   In: {l_in}, Out: {l_out} (Total: {l_in + l_out})\n"
            f"   [Follower] In: {f_in}, Out: {f_out} (Total: {f_in + f_out})\n"
            f"   --------------------------------------------------\n"
            f"   [TOTAL]    {total} tokens\n"
        )
        return summary

    def save_logs(self):
        file_base = f"s{self.session_id}_Char{self.max_char_count}"
        cost_summary = self._get_cost_summary()
        
        txt_path = os.path.join(SAVE_DIR, f"log_{file_base}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"=== Session {self.session_id} (MaxChar={self.max_char_count}) ===\n")
            f.write(f"Date: {datetime.now().isoformat()}\n\n")
            for line in self.chronological_log:
                f.write(line + "\n")
            f.write("\n" + "="*40 + "\n")
            f.write(cost_summary)
            f.write("\n" + "="*40 + "\n")
            f.write("📊 Final Naming Results:\n")
            f.write(f"[Leader]   {json.dumps(self.leader.named_map, ensure_ascii=False)}\n")
            f.write(f"[Follower] {json.dumps(self.follower.named_map, ensure_ascii=False)}\n")
                
        json_path = os.path.join(SAVE_DIR, f"data_{file_base}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.conversation_log, f, indent=2, ensure_ascii=False)
        
        dialogue_path = os.path.join(SAVE_DIR, f"dialogue_{file_base}.txt")
        with open(dialogue_path, "w", encoding="utf-8") as f:
            for entry in self.conversation_log:
                speaker = entry.get("speaker")
                text = entry.get("text")
                if speaker and text and text != "...":
                    f.write(f"{speaker}: {text}\n")
            
        return txt_path

    def check_similarity(self, text1, text2, threshold=0.5):
        if not text1 or not text2: return False
        v1 = self.data_manager.encode_text(text1)
        v2 = self.data_manager.encode_text(text2)
        sim = cosine_similarity([v1], [v2])[0][0]
        self.log_system(f"📏 Similarity: {sim:.3f}")
        return sim >= threshold

    def run_simulation(self):
        self.log_system(f"🚀 Started Session {self.session_id}. Max Char = {self.max_char_count}")
        
        full_history_text = ""
        self.is_naming_phase = False 
        
        consecutive_no_match = 0
        last_valid_follower_probs = {} 

        while self.turn_count < self.max_turns:
            self.turn_count += 1
            self.log_system(f"--- Turn {self.turn_count} ---")

            if not self.leader.current_target_id:
                prev_target = self.leader.current_target_id
                tid = self.leader.select_next_target()
                if tid != prev_target:
                    consecutive_no_match = 0 
                if not tid:
                    self.log_system("🎉 All Tangrams Identified!")
                    break
                self.is_naming_phase = False 
            
            # 1. リーダ発話
            leader_data = self.leader.generate_utterance(
                full_history_text, 
                is_naming_phase=self.is_naming_phase
            )
            leader_utt = leader_data.get("utterance", "...")
            proposed_name = leader_data.get("proposed_name")
            
            self._capture_logs()
            self.conversation_log.append({
                "turn": self.turn_count, "speaker": "Leader", 
                "text": leader_utt,
                "analytic_expressions": leader_data.get("analytic_expressions", []),
                "holistic_expressions": leader_data.get("holistic_expressions", []),
                "target_id": self.leader.current_target_id
            })
            self.log_system(f"🗣️ Leader: {leader_utt}")
            full_history_text += f"Leader: {leader_utt}\n"

            # 2. フォロワー応答
            follower_data = self.follower.respond(leader_utt, full_history_text, proposed_name_by_leader=proposed_name)
            follower_utt = follower_data.get("utterance", "...")
            
            current_probs = follower_data.get("pn_probabilities", {})
            valid_ids = self.leader.data_manager.all_labels 
            valid_probs = {}
            for k, v in current_probs.items():
                if k in valid_ids:
                    try: valid_probs[k] = float(v)
                    except: pass
            
            if valid_probs:
                last_valid_follower_probs = valid_probs
            
            max_pn = max(valid_probs.values()) if valid_probs else 0.0
            
            # 合致なし判定
            is_rejected = False
            if "ありません" in follower_utt:
                is_rejected = True
                self.log_system(f"⚠️ Follower says 'Arimasen'.")
            elif max_pn < 0.2 and valid_probs: 
                is_rejected = True
                self.log_system(f"⚠️ Low probability detected (Max PN: {max_pn:.2f}).")

            if is_rejected and not self.is_naming_phase:
                consecutive_no_match += 1
                self.log_system(f"⚠️ No match count: {consecutive_no_match}/2")
            else:
                consecutive_no_match = 0

            # 撤回実行判定（2回連続不一致 & 命名履歴あり）
            revoke_req = None
            revoke_req_name = None 
            
            if consecutive_no_match >= 2 and self.follower.named_history:
                # 1. 撤回対象のIDを決定する
                pa_probs = follower_data.get("pa_probabilities", {})
                valid_pa = {k: float(v) for k, v in pa_probs.items() if k in self.follower.named_history}
                
                revoke_target_id = None
                
                if valid_pa:
                    # PA最大のものを撤回対象に
                    revoke_target_id = max(valid_pa, key=valid_pa.get)
                else:
                    revoke_target_id = self.follower.named_history[-1]
                
                # 2. IDから名前を取得
                revoke_name = self.follower.named_map.get(revoke_target_id)
                
                # 3. 対話生成とリクエスト設定 (名前が特定できた場合のみ)
                if revoke_name:
                    # フォロワーの発言 (強制挿入)
                    f_msg = f"あなたの説明だと命名済みの『{revoke_name}』が当てはまります。間違えていたかもしれないのでやり直しませんか？"
                    self.log_system(f"🗣️ Follower (Auto-Revoke): {f_msg}")
                    self.conversation_log.append({"turn": self.turn_count, "speaker": "Follower", "text": f_msg})
                    full_history_text += f"Follower: {f_msg}\n"
                    
                    # リーダの発言 (強制挿入)
                    l_msg = f"わかりました。『{revoke_name}』と命名したタングラムの命名を撤回します。"
                    self.log_system(f"🗣️ Leader (Auto-Revoke): {l_msg}")
                    self.conversation_log.append({"turn": self.turn_count, "speaker": "Leader", "text": l_msg})
                    full_history_text += f"Leader: {l_msg}\n"
                    
                    # フォロワーの発言 (強制挿入)
                    f_msg_2 = "わかりました"
                    self.log_system(f"🗣️ Follower (Auto-Revoke): {f_msg_2}")
                    self.conversation_log.append({"turn": self.turn_count, "speaker": "Follower", "text": f_msg_2})
                    full_history_text += f"Follower: {f_msg_2}\n"
                    
                    revoke_req = revoke_target_id
                    revoke_req_name = revoke_name
                    
                    consecutive_no_match = 0 # カウンタをリセット

            accepted_name = follower_data.get("accepted_name")
            
            self._capture_logs()
            if not revoke_req:
                self.conversation_log.append({
                    "turn": self.turn_count, "speaker": "Follower", 
                    "text": follower_utt,
                    "analytic_expressions": follower_data.get("analytic_expressions", []),
                    "holistic_expressions": follower_data.get("holistic_expressions", []),
                    "accepted_name": accepted_name,
                    "revoke_request": revoke_req
                })
                self.log_system(f"🗣️ Follower: {follower_utt}")
                full_history_text += f"Follower: {follower_utt}\n"

            # 撤回処理
            if revoke_req:
                success = self.leader.handle_revoke(revoke_req_name)
                if success:
                    f_id_to_remove = None
                    for tid, name in self.follower.named_map.items():
                        if name == revoke_req_name:
                            f_id_to_remove = tid
                            break
                    
                    if f_id_to_remove:
                        self.follower.handle_revoke_accepted(f_id_to_remove)
                    
                    self.log_system(f"🔄 Revoke Accepted for '{revoke_req_name}'. Resetting target.")
                    self.leader.current_target_id = None 
                    self.is_naming_phase = False
                    continue

            # ----------------------------------------------------
            # 5. 命名合意 (すれ違い許容)
            # ----------------------------------------------------
            if proposed_name and accepted_name:
                self.log_system(f"✅ Naming Agreement Reached: {accepted_name}")
                
                leader_target_id = self.leader.current_target_id
                self.leader.mark_current_target_done(accepted_name)
                
                if last_valid_follower_probs:
                    follower_believed_id = max(last_valid_follower_probs, key=last_valid_follower_probs.get)
                else:
                    self.log_system(f"⚠️ No valid probability history. Fallback to Leader ID.")
                    follower_believed_id = leader_target_id
                
                self.follower.update_named_status(follower_believed_id, accepted_name)
                
                if leader_target_id != follower_believed_id:
                    self.log_system(f"⚠️ Misunderstanding! Leader named {leader_target_id}, but Follower named {follower_believed_id}.")
                
                self.leader.current_target_id = None
                self.is_naming_phase = False 
                continue

            if not self.is_naming_phase:
                if is_rejected:
                    self.log_system(f"⏳ Follower rejected. Skipping similarity check.")
                else:
                    is_similar = self.check_similarity(leader_utt, follower_utt, threshold=0.5)
                    if is_similar:
                        self.log_system(f"✨ High Similarity Detected. Switching to NAMING PHASE.")
                        self.is_naming_phase = True
            
            elif not self.is_naming_phase:
                 self.log_system(f"⏳ Continue Explanation.")

        self.log_system("🏁 Simulation Finished.")
        print(self._get_cost_summary())
        path = self.save_logs()
        return {
            "session_id": self.session_id,
            "log_file": path,
            "turns": self.turn_count,
            "status": "success"
        }

# ==========================================
# 5. 並列実行ラッパー
# ==========================================
def run_single_session_wrapper(args):
    session_id, max_turns, silent, max_char, l_path, f_path = args
    with suppress_output(suppress=silent):
        try:
            gm = GameMaster(session_id, max_turns, max_char_count=max_char, 
                            leader_data=l_path, follower_data=f_path)
            return gm.run_simulation()
        except Exception as e:
            return {"session_id": session_id, "status": "error", "error": str(e)}

def run_mixed_experiments(config_list: List[Dict], num_experiments_per_config=3, max_workers=3, max_turns=30):
    print(f"⚡ Starting Leader-Follower Experiments (Parallel={max_workers})...")
    
    DataManager("dataR.csv") 

    all_tasks = []
    global_session_id = 0

    for config in config_list:
        max_char = config.get("max_char_count", 50)
        l_path = config.get("leader_data", "data.csv")
        f_path = config.get("follower_data", "dataA.csv")
        
        for _ in range(num_experiments_per_config):
            task_args = (global_session_id, max_turns, True, max_char, l_path, f_path)
            all_tasks.append(task_args)
            global_session_id += 1

    total_tasks = len(all_tasks)
    print(f"📋 Total Tasks: {total_tasks}")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_session_wrapper, t) for t in all_tasks]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="Progress"):
            res = future.result()
            results.append(res)
            
            if res["status"] == "success":
                tqdm.write(f"  ✅ ID:{res['session_id']} | Turns:{res['turns']}")
            else:
                tqdm.write(f"  ❌ ID:{res['session_id']} | Error: {res.get('error')}")

    print("\n📊 Done.")
    return results

if __name__ == "__main__":
    experiment_configs = [
        {
            "max_char_count": 30, #1発話あたりの文字数
            "leader_data": "dataLAB.csv",#リーダに渡すタングラムの特徴のテキストデータ
            "follower_data": "dataFAB.csv" #フォロワーに渡すタングラムの特徴のテキストデータ
        },
    ]

    final_results = run_mixed_experiments(
        config_list=experiment_configs,
        num_experiments_per_config=2,#条件ごとの実行回数
        max_workers=10,#最大並列実行数                
        max_turns=50#最大ターン数
    )