import json


def calculate_recall_at_k(student_answer_path: str, dataset_path: str) -> None:
    """Calcule le Recall@k réel basé sur la détection des fichiers sources."""
    # similaire au calcul moulinette recall
    try:
        with open(student_answer_path, "r", encoding="utf-8") as f:
            student_data = json.load(f)
            student_results = (
                student_data.get("search_results")
                or student_data.get("generation_results")
                or []
            )
    except (FileNotFoundError,  json.JSONDecodeError, OSError) as e:
        print(f"Error loading student results: {e}")
        return

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            ground_truth_data = json.load(f)
            gt_questions = ground_truth_data.get("rag_questions", [])
    except (FileNotFoundError,  json.JSONDecodeError, OSError) as e:
        print(f"Error loading ground truth dataset: {e}")
        return

    gt_map = {q["question_id"]: q for q in gt_questions if "question_id" in q}
    total_questions = len(student_results)
    questions_with_sources = 0
    recalls = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}

    for s_res in student_results:
        q_id = s_res.get("question_id")
        if q_id not in gt_map or "sources" not in gt_map[q_id]:
            continue

        questions_with_sources += 1
        gt_sources = gt_map[q_id]["sources"]
        retrieved = (s_res.get("retrieved_sources") or
                     s_res.get("sources") or [])

        for target_k in recalls.keys():
            found_count = 0
            truncated_retrieved = retrieved[:target_k]

            for gt_src in gt_sources:
                gt_path: str = gt_src.get("file_path", "").lower()
                source_found = False

                for s_src in truncated_retrieved:
                    s_path: str = s_src.get("file_path", "").lower()

                    if gt_path != "" and (
                        s_path in gt_path
                        or gt_path in s_path
                        or s_path.endswith(gt_path)
                        or gt_path.endswith(s_path)
                    ):
                        source_found = True
                        break

                if source_found:
                    found_count += 1

            if len(gt_sources) > 0:
                recalls[target_k] += (found_count / len(gt_sources))

    print(f"Student data is valid: {True if total_questions > 0 else False}")
    print(f"Total number of questions: {total_questions}")
    print(f"Total number of questions with sources: {questions_with_sources}")
    print(f"Total number of questions with student sources: {total_questions}")
    print("\nEvaluation Results")
    print(f"Questions evaluated: {questions_with_sources}")

    for target_k in sorted(recalls.keys()):
        score = recalls[target_k] / max(1, questions_with_sources)
        print(f"Recall@{target_k}: {min(1.0, score):.3f}")
