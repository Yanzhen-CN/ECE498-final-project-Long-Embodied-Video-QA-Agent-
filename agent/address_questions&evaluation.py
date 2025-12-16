import json
import re

def evaluate_video_qa(question_bank, retrieved_memories_map, model_inference_fn):
    """
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

    print(f"Starting evaluation for {total_questions} questions...")

    for q_item in question_bank:
        q_id = str(q_item['id']) # 确保 ID 是字符串以匹配字典 Key
        question_text = q_item['question']
        options = q_item['options']
        ground_truth = q_item['ground_truth']
        
        # 1. 获取该题对应的 3 个 Memory Chunk
        # 如果找不到对应的 memory，给一个空的列表，防止报错
        memories = retrieved_memories_map.get(q_id, [])
        
        # 2. 构建 Prompt
        # 将 Memory 转换成字符串文本
        context_str = ""
        for i, mem in enumerate(memories):
            context_str += f"[Segment {i+1} | Time: {mem.get('time_range', 'Unknown')}]\n"
            context_str += f"Summary: {mem.get('summary', '')}\n"
            context_str += f"Key Objects: {', '.join(mem.get('objects', []))}\n\n"

        prompt = f"""
        You are an intelligent video assistant. I will provide you with retrieved memory summaries from a long video and a multiple-choice question.
        
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
        Your output must contain ONLY the option letter (A, B, C, or D). Do not explain.
        
        Answer:
        """

        # 3. 调用模型 (使用传入的函数)
        raw_output = model_inference_fn(prompt)
        
        # 4. 清洗输出，提取 A/B/C/D
        # 即使模型输出了 "The answer is A", 我们也只需要 "A"
        pred_match = re.search(r'\b([A-D])\b', raw_output.strip().upper())
        prediction = pred_match.group(1) if pred_match else "Unknown"

        # 5. 对比答案
        is_correct = (prediction == ground_truth)
        if is_correct:
            correct_count += 1

        # 6. 记录单题结果
        results.append({
            "id": q_item['id'],
            "question": question_text,
            "options": options,
            "ground_truth": ground_truth,
            "agent_prediction": prediction,
            "is_correct": is_correct,
            "raw_model_output": raw_output, # 保留原始输出方便debug
            "used_memories": [m['chunk_id'] for m in memories] # 记录用了哪些chunk
        })
        
        print(f"Q{q_id}: Pred={prediction}, Truth={ground_truth} -> {'✅' if is_correct else '❌'}")

    # 7. 计算总统计
    accuracy = correct_count / total_questions if total_questions > 0 else 0
    
    final_report = {
        "total_questions": total_questions,
        "correct_count": correct_count,
        "accuracy": f"{accuracy:.2%}",
        "accuracy_score": accuracy,
        "details": results
    }
    
    return final_report

# ==========================================
# 使用示例 (假设你已经有数据了)
# ==========================================

# 1. 模拟你的模型调用函数
# 在实际使用中，这里替换成 agent.model.chat(tokenizer, ..., question=prompt)
def my_model_caller(prompt):
    # 这里为了演示，我随机返回一个答案，实际你要调用 InternVL
    # return internvl_model.generate(prompt)
    return "The answer is A" 

# 2. 模拟加载数据 (对应你的输入 JSON)
sample_questions = [
    {
        "id": 1, 
        "question": "What color is the cup?", 
        "options": {"A": "Red", "B": "Blue", "C": "Green", "D": "Black"}, 
        "ground_truth": "A"
    },
    {
        "id": 2, 
        "question": "Where is the cat?", 
        "options": {"A": "Kitchen", "B": "Bedroom", "C": "Garden", "D": "Roof"}, 
        "ground_truth": "B"
    }
]

sample_memories = {
    "1": [{"chunk_id": 10, "summary": "He holds a red cup.", "objects": ["red cup"]}],
    "2": [{"chunk_id": 20, "summary": "Cat sleeps on bed.", "objects": ["cat", "bed"]}]
}

# 3. 运行评估
report = evaluate_video_qa(sample_questions, sample_memories, my_model_caller)

# 4. 输出最终 JSON 文件
output_filename = "D:/HW/senior1/498/final_project/ECE498-final-project-Long-Embodied-Video-QA-Agent-/agent/results/final_evaluation_result.json"
with open(output_filename, "w", encoding='utf-8') as f:
    json.dump(report, f, indent=4, ensure_ascii=False)

print(f"\n评估完成！结果已保存至 {output_filename}")