import json
import re
from typing import List, Dict, Any, Callable, Optional

# =============================================================================
# 1. 核心评估逻辑
# =============================================================================
"""
    批量评估模式：遍历题库，计算准确率。
    输入:
        question_bank: 题库
        context: 视频的完整 Summary 文本 (str)
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
        context (str): 视频的完整 Summary 文本 (str)
        model_inference_fn (function): 一个回调函数，接受 prompt 字符串，返回模型生成的文本。

    Returns:
        dict: 一个包含每题的预测结果和总准确率的字典。
    """
def evaluate_video_qa(question_bank: List[Dict], context: str, model_inference_fn: Callable) -> Dict:
    """
    批量评估模式：遍历题库，基于给定的视频 Summary Context 计算准确率。
    最后额外增加一道“总结题”。
    """
    results = []
    correct_count = 0
    total_questions = len(question_bank)

    print(f"Starting BATCH EVALUATION for {total_questions} questions...")

    # 1. 遍历客观题 (选择题)
    for q_item in question_bank:
        q_id = str(q_item['id'])
        question_text = q_item['question']
        options = q_item['options']
        ground_truth = q_item['ground_truth']
        
        prompt = f"""
        You are an intelligent video assistant.
        
        === Video Summary ===
        {context}
        
        === Question ===
        {question_text}
        
        === Options ===
        A. {options['A']}
        B. {options['B']}
        C. {options['C']}
        D. {options['D']}
        
        Based on the video summary provided, choose the correct option.
        Your output must contain ONLY the option letter (A, B, C, or D).
        
        Answer:
        """

        raw_output = model_inference_fn(prompt)
        
        pred_match = re.search(r'\b([A-D])\b', raw_output.strip().upper())
        prediction = pred_match.group(1) if pred_match else "Unknown"

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
    
    # 2. [新增] 额外生成一段视频总结
    print("Generating final summary evaluation...")
    summary_prompt = f"""
    You are a helpful assistant. 
    Based ONLY on the following video context, provide a concise summary of the video's main events.
    
    === Video Context ===
    {context}
    
    === Task ===
    Summarize the video content in 2-3 sentences.
    """
    final_summary = model_inference_fn(summary_prompt).strip()
    
    return {
        "mode": "evaluation",
        "total_questions": total_questions,
        "accuracy_str": f"{accuracy:.2%}",
        "accuracy_val": accuracy,
        "details": results,
        "generated_summary": final_summary  # <--- 将总结结果返回
    }

# =============================================================================
# 2. 交互式问答逻辑
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
    if mode == "interactive":
        if "context" not in kwargs or "question" not in kwargs:
            raise ValueError("Interactive mode requires 'context' and 'question' in kwargs.")
            
        print("\n[System] Running Interactive QA Mode...")
        return _interactive_qa(
            context=kwargs["context"],
            question=kwargs["question"],
            model_inference_fn=model_inference_fn
        )

    elif mode == "evaluation":
        # 修改：检查 context 而不是 retrieved_memories_map
        if "question_bank" not in kwargs or "context" not in kwargs:
            raise ValueError("Evaluation mode requires 'question_bank' and 'context'.")
            
        print("\n[System] Running Batch Evaluation Mode...")
        return evaluate_video_qa(
            question_bank=kwargs["question_bank"],
            context=kwargs["context"],  # 传入 String
            model_inference_fn=model_inference_fn
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'interactive' or 'evaluation'.")