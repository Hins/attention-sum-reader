from __future__ import division
import time
import logging
import json
import argparse
import sys
import os
from rouge import Rouge

def read_json_file(json_file_path):
    with open(json_file_path) as f:
        return json.load(f)

def json_to_dict(json_obj):
    if json_obj.get("data") is None or not hasattr(json_obj.get("data"), '__iter__'):
        return {}
    rtn_dict = {}
    for element in json_obj.get("data"):
        if element.get("paragraphs") is None or not hasattr(element.get("paragraphs"), '__iter__'):
            continue
        for para in element.get("paragraphs"):
            if para.get("qas") is None or not hasattr(para.get("qas"), '__iter__'):
                continue
            for qas in para.get("qas"):
                question = ""
                if qas.get("question") is None or qas.get("answers") is None or len(qas.get("answers")) == 0:
                    continue
                if qas.get("answers")[0].get("text") is None:
                    continue
                rtn_dict[qas.get("question")] = qas.get("answers")[0].get("text")
    return rtn_dict

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', default='/input', help='Location of WIDERFACE root directory')
    parser.add_argument('--output', default='/result', help='Location of WIDERFACE root directory')
    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()
    input_path = args.input
    output_path = args.output

    with open(input_path + "/TagRecord/input.json", 'r') as f:
        predict_json = json.load(f)
    predict_dict = json_to_dict(predict_json)

    with open(input_path + "/Answer/input.json", 'r') as f:
        true_json = json.load(f)
    true_dict = json_to_dict(true_json)

    if len(predict_dict) != len(true_dict):
        sys.exit()

    predict_list = []
    true_list = []
    for k,v in predict_dict.items():
        if k in true_dict:
            predict_list.append(v)
            true_list.append(true_dict[k])

    if len(predict_list) != len(true_list):
        sys.exit()

    r = Rouge()
    score = r.get_scores(predict_list, true_list)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path + "/score.json", 'w') as f:
        json.dump(json.dumps(score), f)