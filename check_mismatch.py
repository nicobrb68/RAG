import json

def debug_mismatch(student_file, ground_truth_file):
    with open(student_file, 'r') as f1, open(ground_truth_file, 'r') as f2:
        student = json.load(f1)["search_results"]
        truth = json.load(f2)["rag_questions"]

    # On prend la première question pour comparer
    s_q = student[0]
    t_q = truth[0]
    
    print(f"Question ID: {s_q['question_id']}")
    print(f"Student Path: {s_q['retrieved_sources'][0]['file_path']}")
    print(f"Truth Path: {t_q['sources'][0]['file_path']}")
    
    # Vérification stricte
    if s_q['retrieved_sources'][0]['file_path'] != t_q['sources'][0]['file_path']:
        print("ALERTE: Les chemins ne sont pas identiques !")

if __name__ == "__main__":
    debug_mismatch("data/output/search_results/generated_search_results.json", 
                   "datasets_public/public/AnsweredQuestions/dataset_docs_public.json")