from src.app.evaluator.eval import MinasFloraEvaluation
import os 


def get_eval_metrics(model_path, eval_path, output_save):
    evaluator = MinasFloraEvaluation(model_path, eval_path)
    return evaluator._save_overall_report(output_save)



if __name__ == "__main__":
    model_path = 'src/models/minas_flora_classifier_model'
    eval_path = 'data'
    output_save = 'src/reports/overall_report.json'

    get_eval_metrics(model_path, eval_path, output_save) 