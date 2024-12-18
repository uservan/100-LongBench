import re
import string

import jieba
from fuzzywuzzy import fuzz
import difflib

import json
from typing import List
from collections import Counter
from rouge import Rouge
from .model_utils import OpenAIModel

# prompts inspired by https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG
judge_prompt = """Please act as an impartial judge and evaluate the quality of the provided answer which attempts to answer the provided question based on a provided context.
Although you are not given the context, you will be given a set of correct answers that achieves full scores on all metrics, and you need to assess the provided answers using the correct answers.

Below is your grading rubric:

Fluency:
- Score 0 (incoherent): Incoherent sentences
- Score 1 (coherent, but repetitive answer): Coherent, fluent, grammatically correct but repetitive answers.
- Score 2 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers.

Correctness:
- Score 0 (Incorrect): The answer does not agree with the provided correct answers at all.
- Score 1 (partly correct): Partly agree with one of the provided correct answers (for example, the question asks for a date and a person; the answer gets the date right but the person wrong).
- Score 2 (correct but not fully relevant): Fully agrees with one of the provided correct answers but mentions other completely irrelevant information. Note that extra details provided in the answer, even if not mentioned in the correct answers, should NOT be seen as irrelevant as long as they are relevant to the question to a reasonable extend.
- Score 3 (correct and relevant): Fully agrees with one of the provided correct answers and only provides information relevant to the question. Note that if the answer is longer than the correct answer, as long as everything in the answer is relevant to the question, it should still be given score 3. For example, if the correct answer is "the North Pole" and the answer is "They are headed for the North Pole", it should still be given a score of 3.

Now, read the following question, answer, and correct answers. First think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"fluency": 0, "correctness": 1}}.

Question: {question}
Correct answers: {correct_answers}
Answer: {parsed_output}
"""

# prompts inspired by https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG
fluency_prompt="""Please act as an impartial judge and evaluate the fluency of the provided text. The text should be coherent, non-repetitive, fluent, and grammatically correct.

Below is your grading rubric:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish. Note that even if the answer is coherent, if it is repetitive or incomplete, it should be given a score of 0.
  - Examples:
    - Incomplete: "Summary:"
    - Incoherent: "Summary: The plaintiff the the the the able the the the the the the the the the the able the the the the the Ã�\n"
    - Repetitive: "Summary: The U.S. government brought a criminal case against four defendants. Summary: The U.S. government brought a criminal case against four defendants. Summary: The U.S. government brought a criminal case against four defendants. Summary: The U.S. government brought a criminal case against four defendants."

- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers. If the text is coherent, non-repetitive, and fluent, but the last sentence is truncated, it should still be given a score of 1.
  - Examples:
    - "This case is about an apprenticeship test that had a disparate impact on Black apprenticeship applicants. The Equal Employment Opportunity Commission (EEOC) filed this lawsuit on December 27, 2004, in U.S. District Court for the Southern District of Ohio."
    - "The plaintiffs sought declaratory and injunctive relief, as well as attorneys' fees and costs, under the Americans with Disabilities Act, the Rehabilitation Act of 1973, the Social Security Act, and the Nursing Home Reform Act. The case was certified as a class action on behalf of all Medicaid-eligible adults with disabilities in Cook County, Illinois, who are being, or may in the future be, unnecessarily confined to nursing facilities and with appropriate supports and services may be able to live in a community setting. The defendants denied the allegations and argued that the plaintiffs' claims were not typical of the class and that the class definition was too broad. The case is ongoing, with discovery and expert testimony scheduled for the fall of"

Now, read the provided text, and evaluate the fluency using the rubric. Then output your score in the following json format: {{"fluency": 1}}.

Text: "{text}"
"""

precision_prompt="""Please act as an impartial judge and evaluate the quality of the provided summary of a civil lawsuit. The summary is based on a set of legal documents, and it should contain a short description of the background, the parties involved, and the outcomes of the case.

Below is your grading rubric:
Precision:
- Evaluate the provided summary by deciding if each sentence in the provided summary is supported by the information provided in the expert summary. A sentence is considered supported if its major facts align with the information in the expert summary. A sentence is still considered supported even if some of its minor details, such as dates, entity names, or the names of laws and previous court cases, are not explicitly mentioned in the expert summary. A sentence is not supported if its major facts are not mentioned or contradicted in the expert summary.
- Score: the number of sentences in the provided summary that are supported by the expert summary.
- Examples: use the following examples to guide your evaluation.

Example 1:

Expert summary: "This lawsuit, brought in the the U.S. District Court for the Central District of California, was filed on June 3, 2020. The plaintiffs were represented by attorneys from the ACLU of Southern California. This lawsuit followed nation-wide protests that occurred in response to the killing of George Floyd by a police officer in Minneapolis. While most protests were peaceful, some ended in violence, property destruction, rioting, and looting. Many cities, including Los Angeles and San Bernardino, issued curfews in an attempt to quell these riots. This action challenged these curfews as violations of free speech and assembly, free movement, due process, and challenged the San Bernardino curfew as a violation of the establishment clause (the San Bernardino curfew included a provision that exempted attendants of religious meetings from the curfew.) The plaintiffs sought injunctive and declaratory relief that would void the curfew and prohibit the cities from enforcing them. The following day, June 4th, 2020, the case was assigned to District Judge Philip S. Gutierre and to Magistrate Judge Pedro V. Castillo. Judge Gutierrez informed the parties that he was part of a mandatory alternative dispute resolution (ADR) program and asked the parties to try to form an agreement before going to trial. On July 7, 2020, the plaintiffs voluntarily dismissed the complaint, citing that fact that the city had rescinded the curfews already and not attempted to reinstate them. The case is now closed."

Provided summary: "In June 2020, Black Lives Matter - Los Angeles and several individuals filed a lawsuit in the U.S. District Court for the Central District of California against Los Angeles Mayor Eric Garcetti, other city officials, and the City of San Bernardino, challenging the constitutionality of curfew orders imposed during protests against police violence. The plaintiffs, represented by the ACLU of Southern California, argued that the curfews violated their First Amendment rights to free speech and assembly, as well as their freedom of movement, by suppressing political protests and other activities. The lawsuit also claimed that the curfews were not narrowly tailored to address any emergency and lacked sufficient notice. However, the plaintiffs voluntarily dismissed the case in July 2020 after the defendants lifted the curfew orders and did not reinstate them in the following weeks."

Reasoning: The first sentence in the provided summary is well supported by the expert summary even though some entity names are not explicitly mentioned. The second sentence is also well supported by the expert summary, as it mentions the ACLU of Southern California and the First Amendment rights. The third sentence is not supported by the expert summary, as it does not mention the lack of narrow tailoring or sufficient notice. The fourth sentence is well supported by the expert summary, as it mentions the voluntary dismissal of the case in July 2020. Therefore, the precision score is 3.

Output: {{"precision": 3, "sentence_count": 4}}


Example 2:

Expert summary: "On August 22, 2007, individuals with disabilities filed a lawsuit under the Americans with Disabilities Act (ADA), the Social Security Act, the Rehabilitation Act, and the Nursing Care Reform Act, against various Illinois state officials in the United States District Court for the Northern District of Illinois.  Plaintiffs, represented by private and public interest counsel, asked the court for declaratory and injunctive relief, claiming that they were institutionalized in a nursing facility even though they were capable of living in a more community-integrated setting with appropriate services.  Plaintiffs claimed that Defendants conditioned receipt of long-term care on remaining in an institutionalized setting, even though it would be less expensive for Plaintiffs to receive appropriate care in the community. The Court (Judge Joan H. Lefkow) certified a class as: \"all Medicaid-eligible adults with disabilities in Cook County, Illinois, who are being, or may in the future be, unnecessarily confined to nursing facilities and who, with appropriate supports and services, may be able to live in a community setting.\"  71 Fed.R.Serv.3d 1089. At a status hearing on January 7, 2011, the parties advised Magistrate Judge Maria Valdez that they could conclude settlement discussions without further assistance from the court. On Aug. 29, 2011, the parties jointly moved for the court to approve the consent decree they had agreed upon.  The court held a fairness hearing on Dec. 20, 2011, and ultimately accepted the decree. The consent decree established benchmarks for moving specific numbers of class members out of nursing facilities and into community-based settings. Over the course of the first two-and-a-half years, the decree compelled the state to move 1,100 class members into the community. It also required the state to provide up to $10 million in housing assistance to support the first group of transitioned adults. The decree also compelled the state to develop services needed to adequately support class members who choose to live in the community. It established a monitor to ensure compliance with the decree, and granted $1.2 million in attorneys' fees. The court approved an updated plan following the parties' cross-motion to enter into a cost-neutral plan and supplement and amend the December 2011 consent decree on November 16, 2016. The plan included the transition of class members into community-based settings, and continued evaluations and service plans for the class members. The court retained jurisdiction to oversee the full implementation of the plan. The court approved an updated plan on April 5, 2018. Monitoring by the court appointed monitor (Gail P. Hutchings) is ongoing as of May 20, 2020."

Provided: "Summary: Five Medicaid-eligible individuals with disabilities, Lenil Colbert, Constance Gray, Ernest Reeves, Kenya Lyles, and Dwight Scott, filed a class action lawsuit in the United States District Court for the Northern District of Illinois against Illinois state officials, including Governor Rod R. Blagojevich, Secretary of the Illinois Department of Human Services Carol L. Adams, Director of the Illinois Department of Healthcare and Family Services Barry S. Maram, and Director of the Illinois Department of Public Health Eric E. Whitaker. The plaintiffs alleged that the defendants' policies and practices effectively compel people with disabilities to enter nursing facilities in order to receive long-term care and assistance, forcing them to forego liberty, privacy, independence, and the opportunity to live in the communities of their choice. The plaintiffs sought declaratory and injunctive relief, as well as attorneys' fees and costs, under the Americans with Disabilities Act, the Rehabilitation Act of 1973, the Social Security Act, and the Nursing Home Reform Act. The case was certified as a class action on behalf of all Medicaid-eligible adults with disabilities in Cook County, Illinois, who are being, or may in the future be, unnecessarily confined to nursing facilities and with appropriate supports and services may be able to live in a community setting. The defendants denied the allegations and argued that the plaintiffs' claims were not typical of the class and that the class definition was too broad. The case is ongoing, with discovery and expert testimony scheduled for the fall of"

Reasoning: The first sentence is supported as the expert summary states that "individuals with disabilities filed a lawsuit... against various Illinois state officials", even though some minor details (the name of the people) are not mentioned. The second sentence is not supported as the expert summary does not discuss how the plaintiffs alleged that the defendants' policies forced them to forego their rights. The third sentence is mostly supported as the expert summary mentions that the plaintiffs sought declaratory and injunctive relief, but it does not mention the attorneys' fees and costs, which are minor details. The fourth sentence is supported as the expert summary mentions the class action certification by the court. The fifth sentence is not supported as the expert summary does not mention the defendants' denial of the allegations. The sixth sentence is not supported as the expert summary states that the case was settled through the consent decree, while the provided summary states that the case is ongoing. Therefore, the precision score is 3.

Output: {{"precision": 2, "sentence_count": 6}}

Now, read the provided summary and expert summary, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"precision": 2, "sentence_count": 6}}.

Expert summary: "{expert_summary}"

Provided summary: "{summary}"
"""


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def acc_score(prediction, ground_truth, **kwargs):
    match = re.search(r'\d+', prediction)
    if match: number = match.group()
    else: number=''
    if str(number) == str(ground_truth): return 1
    else: return 0

def kv_retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
    uuids = re.findall(pattern, prediction)
    last_uuid = uuids[-1] if uuids else None
    if str(last_uuid) == str(ground_truth): return 1
    else: return 0

def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)

def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if em_match_list != 0:
        if ground_truth in em_match_list:
            score = (1.0 / len(em_match_list))
        else:
            score = 0.0
    else:
        best_match = None
        highest_similarity = 0
        for string in all_classes:
            similarity = difflib.SequenceMatcher(None, string, prediction).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = string
        score = float(best_match == ground_truth)
    return score
    
def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)

def parse_qa_json(text):
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if len(matches) > 0:
        try:
            r = json.loads(matches[-1])
        except:
            return None
        return r
    return None

def qa_model_score(prediction, ground_truth, **kwargs):
    qa, model= kwargs["qa"], kwargs["model"]
    p = judge_prompt.format(question=qa, correct_answers=ground_truth, parsed_output=prediction)

    o = model.generate(prompt=p)
    s = 0
    if o is not None:
        scores = parse_qa_json(o["output"])
        if scores is not None and "correctness" in scores and "fluency" in scores:
            s = scores
        else:
            print("Warning! Couldn't get a score")
            print(f"GPT-4 output: {o['output']}")
        if scores is not None:
            s = scores["fluency"] * scores["correctness"]
    return s / 6

def parse_sum_json(text):
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if len(matches) > 0:
        try:
            json.loads(matches[-1])
        except:
            matches = re.findall(r"(?:```json)(.+)(?:```)", text, re.DOTALL)
            if len(matches) <=0: return None
        return json.loads(matches[-1])
    return None

def sum_model_score(prediction, ground_truth, **kwargs):
    model = kwargs["model"]
    fp = fluency_prompt.format(text=prediction.strip())
    pp = precision_prompt.format(expert_summary=ground_truth, summary=prediction.strip())
    def get_score(prompt, tries=2):
        o = None
        for _ in range(tries):
            o = model.generate(prompt=prompt)
            if o is not None and o["output"] is not None:
                ret = parse_sum_json(o["output"])
                if ret is not None:
                    return ret, o
        return None, o
    s = 0
    f, fo = get_score(fp)
    if f is None: return s
    p, po = get_score(pp)
    if p is None: return s
    if f is not None and p is not None:
        prec = p["precision"] / p["sentence_count"] if p["sentence_count"] > 0 else 0
        fluency = f["fluency"]
        s = fluency * prec
    return s