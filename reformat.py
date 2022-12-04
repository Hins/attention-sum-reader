import sys
import json
import random
import jieba

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit()

    with open(sys.argv[2], 'w') as wf:
        with open(sys.argv[1], 'r') as f:
            json_obj = json.load(f)
            if json_obj.get("data") is None or not hasattr(json_obj.get("data"), '__iter__'):
                sys.exit()
            for element in json_obj.get("data"):
                if element.get("paragraphs") is None or not hasattr(element.get("paragraphs"), '__iter__'):
                    continue
                for para in element.get("paragraphs"):
                    if para.get("context") is None:
                        continue
                    if para.get("qas") is None or not hasattr(para.get("qas"), '__iter__'):
                        continue
                    wf.write(para.get("context") + "\n")
                    context_words_list = [word for word in jieba.cut(para.get("context"))]
                    for qas in para.get("qas"):
                        question = ""
                        if qas.get("question") is None:
                            continue
                        if qas.get("answers") is None or not hasattr(qas.get("answers"), '__iter__'):
                            continue
                        question += qas.get("question") + "XXXXX\t"
                        for ans in qas.get("answers"):
                            random_list = random.sample(context_words_list, 5)
                            random_list.append(ans["text"])
                            question += ans["text"] + "\t" + "|".join(random_list)
                            break
                        wf.write(question + "\n")