import json
import re
from typing import List, Dict, Any, Callable, Optional

# =============================================================================
# 1. 核心评估逻辑 (保持你原有的逻辑，稍作封装)
# =============================================================================
def evaluate_video_qa(question_bank: List[Dict], retrieved_memories_map: Dict, model_inference_fn: Callable) -> Dict:
    """
    批量评估模式：遍历题库，计算准确率。
    输入:
        question_bank: 题库
        retrieved_memories_map：所用到的memory
        model_inference_fn：模型调用函数
    输出：
        一个json文件，包含每题的预测结果和总准确率
    Args详情:
        ===============================================================================
        question_bank (list): 题库列表，结构如下
        ===============================================================================
            [
                {
                    "id": 1,
                    "question": "What color is the cup the person picked up?",
                    "options": {
                    "A": "Red",
                    "B": "Blue",
                    "C": "Green",
                    "D": "Yellow"
                    },
                    "ground_truth": "A"
                },
                {
                    "id": 2,
                    "question": "Did the robot enter the kitchen?",
                    "options": {
                    "A": "Yes",
                    "B": "No",
                    "C": "Unknown",
                    "D": "Only looked inside"
                    },
                    "ground_truth": "B"
                }
            ]
            
        ===============================================================================
        retrieved_memories_map (dict): 检索到的Memory，结构如下
        ===============================================================================
            {
                "1": [  # question id
                    {
                        "chunk_id": 10,
                        "summary": "The person picked up a red cup from the table.",
                        "objects": ["red cup", "table"],
                        "time_range": "00:05-00:10"
                    },
                    {
                        "chunk_id": 15,
                        "summary": "The person walked to the kitchen area.",
                        "objects": ["person", "kitchen"],
                        "time_range": "00:15-00:20"
                    },
                    {
                        "chunk_id": 22,
                        "summary": "The person placed the cup on the counter.",
                        "objects": ["cup", "counter"],
                        "time_range": "00:25-00:30"
                    }
                ],
                "2": [
                    {
                        "chunk_id": 30,
                        "summary": "The robot approached the kitchen door.",
                        "objects": ["robot", "kitchen door"],
                        "time_range": "01:00-01:05"
                    },
                    {
                        "chunk_id": 35,
                        "summary": "The robot entered the living room instead.",
                        "objects": ["robot", "living room"],
                        "time_range": "01:10-01:15"
                    },
                    {
                        "chunk_id": 40,
                        "summary": "The robot did not go into the kitchen.",
                        "objects": ["robot", "kitchen"],
                        "time_range": "01:20-01:25"
                    }
                ]
        model_inference_fn (function): 一个回调函数，接受 prompt 字符串，返回模型生成的文本。

    Returns:
        dict: 包含总准确率、正确数量以及详细的每题评估结果。
    """
    results = []
    correct_count = 0
    total_questions = len(question_bank)

    print(f"Starting BATCH EVALUATION for {total_questions} questions...")

    for q_item in question_bank:
        q_id = str(q_item['id'])
        question_text = q_item['question']
        options = q_item['options']
        ground_truth = q_item['ground_truth']
        
        # 获取 Memory
        memories = retrieved_memories_map.get(q_id, [])
        
        # 构建 Context 字符串
        context_str = ""
        for i, mem in enumerate(memories):
            context_str += f"[Segment {i+1} | Time: {mem.get('time_range', 'Unknown')}]\n"
            context_str += f"Summary: {mem.get('summary', '')}\n"
            context_str += f"Key Objects: {', '.join(mem.get('objects', []))}\n\n"

        # 构建 Prompt (针对选择题)
        prompt = f"""
        You are an intelligent video assistant.
        
        === Video Memories ===
        {context_str}
        
        === Question ===
        {question_text}
        
        === Options ===
        A. {options['A']}
        B. {options['B']}
        C. {options['C']}
        D. {options['D']}
        
        Based on the video memories provided, choose the correct option.
        Your output must contain ONLY the option letter (A, B, C, or D).
        
        Answer:
        """

        # 调用模型
        raw_output = model_inference_fn(prompt)
        
        # 提取答案
        pred_match = re.search(r'\b([A-D])\b', raw_output.strip().upper())
        prediction = pred_match.group(1) if pred_match else "Unknown"

        # 统计
        is_correct = (prediction == ground_truth)
        if is_correct:
            correct_count += 1

        results.append({
            "id": q_item['id'],
            "question": question_text,
            "agent_prediction": prediction,
            "is_correct": is_correct,
            "raw_model_output": raw_output
        })
        
        print(f"Q{q_id}: Pred={prediction}, Truth={ground_truth} -> {'✅' if is_correct else '❌'}")

    accuracy = correct_count / total_questions if total_questions > 0 else 0
    return {
        "mode": "evaluation",
        "total_questions": total_questions,
        "accuracy": f"{accuracy:.2%}",
        "details": results
    }

# =============================================================================
# 2. 交互式问答逻辑 (新写的模式)
# =============================================================================
def _interactive_qa(context: str, question: str, model_inference_fn: Callable) -> str:
    """
    交互模式：针对单个问题，进行开放式回答。
    """
    # 构建 Prompt (针对开放式问答，不同于选择题)
    prompt = f"""
    You are a helpful video analysis assistant. You are provided with a summary of a video's content.
    
    === Video Context ===
    {context}
    
    === User Question ===
    {question}
    
    Instructions:
    1. Answer the question specifically based on the provided video context.
    2. If the answer is not in the context, say "I cannot find that information in the video summary."
    3. Keep the answer concise and helpful.
    
    Answer:
    """
    
    # 调用模型
    response = model_inference_fn(prompt)
    return response.strip()

# =============================================================================
# 3. 外层统一入口函数 (The Wrapper)
# =============================================================================
def run_qa_system(
    mode: str, 
    model_inference_fn: Callable, 
    **kwargs
) -> Any:
    """
    统一 QA 系统入口。
    
    Args:
        mode (str): "interactive" 或 "evaluation"
        model_inference_fn (func): 模型调用回调函数
        **kwargs: 根据模式传递不同的参数
            - mode="interactive": 需要 'context' (str), 'question' (str)
            - mode="evaluation": 需要 'question_bank' (list), 'retrieved_memories_map' (dict)
            
    Returns:
        String (回答) 或 Dict (评测报告)
    """
    
    if mode == "interactive":
        # 检查参数
        if "context" not in kwargs or "question" not in kwargs:
            raise ValueError("Interactive mode requires 'context' and 'question' in kwargs.")
            
        print("\n[System] Running Interactive QA Mode...")
        return _interactive_qa(
            context=kwargs["context"],
            question=kwargs["question"],
            model_inference_fn=model_inference_fn
        )

    elif mode == "evaluation":
        # 检查参数
        if "question_bank" not in kwargs or "retrieved_memories_map" not in kwargs:
            raise ValueError("Evaluation mode requires 'question_bank' and 'retrieved_memories_map'.")
            
        print("\n[System] Running Batch Evaluation Mode...")
        return evaluate_video_qa(
            question_bank=kwargs["question_bank"],
            retrieved_memories_map=kwargs["retrieved_memories_map"],
            model_inference_fn=model_inference_fn
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'interactive' or 'evaluation'.")