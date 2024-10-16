import spacy
from spacy.tokens import DocBin
from spacy import displacy

def visualize_ner(ner_input, ner_output):
    nlp = spacy.blank("zh")
    # store annotations for rendering
    docs = []
    for text, annotations in zip(ner_input, ner_output):
        doc = nlp(text)
        ents = []
        # required to convert annotations dict of lists into annotation dict
        for annotation in annotations['output']:
            span = doc.char_span(annotation['start'], annotation['end'], label=annotation['type'])
            ents.append(span)
        doc.ents = ents
        docs.append(doc)

    displacy.render(docs, style="ent")


def filter_qa_pairs(dataset):
    questions = [qa["Question"] for qa in dataset]
    ner_questions = [ner_pipeline(q)["output"] for q in questions]
    answers = [qa[qa["Answer"]] for qa in dataset]
    ner_answers = [ner_pipeline(a)["output"] for a in answers]
        
    # find better way to do this
    dataset = dataset.add_column("q_ner", ner_questions)
    dataset = dataset.add_column("a_ner", ner_answers)    
    
    return [qa for qa in dataset if qa["q_ner"] and qa["a_ner"]]



if __name__ == "__main__":
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.models import Model
    from datasets import load_dataset
    
    chinese_tasks = ['ancient_chinese', 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 'chinese_teacher_qualification', 'elementary_chinese', 'ethnology', 'marxist_theory', 'modern_chinese', 'security_study', 'traditional_chinese_medicine']
    
    cmmlu = load_dataset('/scratch/ssd004/scratch/cambish/cmmlu-v1.0.1/cmmlu.py', chinese_tasks, split='test')

