import json
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

class MemoryRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        初始化检索器，加载 Embedding 模型。
        """
        print(f"[Retriever] Loading model: {model_name} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.memory_embeddings = None
        self.raw_memories = [] # 存储原始 JSON 数据

    def _serialize_memory_to_text(self, chunk):
        """
        核心函数：将复杂的 Memory JSON 转换为一段富含语义的文本字符串。
        这是为了让 Embedding 模型能“读懂”结构化数据。
        """
        # 1. 基础信息
        text = f"Time segment: {chunk.get('t_start', 0)}s to {chunk.get('t_end', 0)}s. "
        text += f"Summary: {chunk.get('summary', '')} "

        # 2. 状态更新 (State Update) - 非常重要
        # 将 {"holding": "cup"} 转换为 "holding is cup"
        if 'state_update' in chunk and chunk['state_update']:
            states = [f"{k} is {v}" for k, v in chunk['state_update'].items()]
            text += f"State Context: {', '.join(states)}. "

        # 3. 事件 (Events) - 将动作和对象关联
        if 'events' in chunk and chunk['events']:
            events_desc = []
            for e in chunk['events']:
                # 拼接 verb + obj + detail
                desc = f"{e.get('verb', '')} {e.get('obj', '')}"
                if e.get('detail'):
                    desc += f" ({e.get('detail')})"
                events_desc.append(desc)
            text += f"Actions: {', '.join(events_desc)}. "

        # 4. 实体 (Entities/Objects)
        if 'entities' in chunk and chunk['entities']:
            text += f"Objects involved: {', '.join(chunk['entities'])}."

        return text

    def build_index(self, all_memory_chunks):
        """
        输入所有的 Memory 块，构建向量索引。
        Args:
            all_memory_chunks (list): 包含所有 chunk 数据的列表，格式同 memory_sample.json
        """
        print(f"[Retriever] Building index for {len(all_memory_chunks)} chunks...")
        self.raw_memories = all_memory_chunks
        
        # 1. 将每个 Chunk 转换为文本
        corpus = [self._serialize_memory_to_text(c) for c in all_memory_chunks]
        
        # 2. 批量计算向量 (Embeddings)
        self.memory_embeddings = self.model.encode(
            corpus, 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        print("[Retriever] Index built successfully.")

    def _format_for_evaluation(self, raw_chunk):
        """
        数据清洗层：将 memory_sample.json 的格式 转换为 evaluate_video_qa 需要的格式
        """
        return {
            "chunk_id": raw_chunk.get("chunk_id"),
            # 转换时间格式: 330, 360 -> "330-360"
            "time_range": f"{raw_chunk.get('t_start')}-{raw_chunk.get('t_end')}",
            "summary": raw_chunk.get("summary"),
            # 字段重命名: entities -> objects (适配 evaluation 脚本)
            "objects": raw_chunk.get("entities", []), 
            # 保留其他有用信息以备不时之需
            "state_update": raw_chunk.get("state_update"),
            "events": raw_chunk.get("events")
        }

    def retrieve_for_questions(self, question_bank, top_k=3):
        """
        主功能函数：为题库中的每一道题匹配最相关的 memory。
        
        Args:
            question_bank (list): 题目列表，包含 'id' 和 'question'
            top_k (int): 每题找几个片段
            
        Returns:
            dict: 格式完全符合 evaluate_video_qa 的输入要求
                  { "question_id_1": [mem1, mem2, mem3], ... }
        """
        if self.memory_embeddings is None:
            raise ValueError("请先调用 build_index()！")

        retrieved_map = {}
        
        print(f"[Retriever] Processing {len(question_bank)} questions...")
        
        for q_item in tqdm(question_bank):
            q_id = str(q_item['id']) # 保证 Key 是字符串
            question_text = q_item['question']
            
            # 1. 将问题转换为向量
            q_embedding = self.model.encode(question_text, convert_to_tensor=True)
            
            # 2. 计算相似度 (Cosine Similarity)
            cos_scores = util.cos_sim(q_embedding, self.memory_embeddings)[0]
            
            # 3. 获取 Top-K 索引
            # 加上 min 是防止 memory 总数少于 top_k 报错
            k_actual = min(top_k, len(self.raw_memories))
            top_results = torch.topk(cos_scores, k=k_actual)
            top_indices = top_results.indices.tolist()
            
            # 4. 提取原始 Memory 并格式化
            selected_chunks = []
            for idx in top_indices:
                raw_mem = self.raw_memories[idx]
                formatted_mem = self._format_for_evaluation(raw_mem)
                selected_chunks.append(formatted_mem)
                
            retrieved_map[q_id] = selected_chunks
            
        return retrieved_map

# =========================================================
#  如何与你的其他代码整合 (示例)
# =========================================================
if __name__ == "__main__":
    # 1. 假设这是你的 memory (从 memory_sample.json 读取)
    # 这里直接复制了你提供的 sample 内容作为列表的一个元素
    all_memories = [
        {
            "video_id": "v001",
            "chunk_id": 12,
            "t_start": 330,
            "t_end": 360,
            "summary": "A person walks to the table and picks up a red cup.",
            "entities": ["person", "red cup", "table"],
            "events": [
                {"verb": "walk_to", "obj": "table", "detail": ""},
                {"verb": "pick_up", "obj": "red cup", "detail": "from table"}
            ],
            "state_update": {"holding": "red cup", "red cup location": "in_hand"},
            "evidence_frames": ["path/to/img.jpg"]
        },
        # 假设还有一个无关的片段用于测试
        {
            "video_id": "v001",
            "chunk_id": 99,
            "t_start": 0,
            "t_end": 30,
            "summary": "The screen is black.",
            "entities": [],
            "state_update": {}
        }
    ]

    # 2. 假设这是你的题库 (与 evaluation 脚本格式一致)
    questions = [
        {
            "id": 101,
            "question": "What did the person pick up from the table?",
            "options": {"A": "Cup", "B": "Apple", "C": "Pen", "D": "Phone"},
            "ground_truth": "A"
        }
    ]

    # --- 流程开始 ---
    
    # Step A: 初始化
    retriever = MemoryRetriever()
    
    # Step B: 构建索引
    retriever.build_index(all_memories)
    
    # Step C: 执行检索
    retrieved_map = retriever.retrieve_for_questions(questions, top_k=1)
    
    # Step D: 输出结果 (可直接传给 evaluate_video_qa)
    print("\n=== 检索结果 (Formatted for Evaluation) ===")
    print(json.dumps(retrieved_map, indent=2))