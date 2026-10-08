#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ДЗ 1. Индексация и булев поиск — шаблон решения.

Запуск:
    ./solution.py --build_index --index_dir=index PATH-TO-DATA
    ./solution.py --submission_file=submission.csv --index_dir=index PATH-TO-DATA
"""

import argparse
from timeit import default_timer as timer
import pickle
from nltk import tokenize


def preprocess(text):
    """Токенизация и нормализация. НЕ МЕНЯЙТЕ эту функцию:
    ей же размечен эталон на Kaggle."""
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return [token.lower() for token in tokens]



def build_index(data_dir, index_dir):
    # TODO:
    # - прочитать документы из data_dir/vkmarco-docs.tsv
    #   (без заголовка; поля через таб: DocumentId, URL, Title, Body);
    # - разбить Title и Body на термины функцией preprocess() (URL не индексируем);
    # - построить обратный индекс и сохранить его в index_dir.
    def encode_varbyte_to_buffer(number, buffer):
        while True:
            byte = number & 0x7F
            number >>= 7
            
            if number == 0:
                buffer.append(byte | 0x80)
                break
            else:
                buffer.append(byte)
    inverted_index = {}
    with open(f'{data_dir}/vkmarco-docs.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            row = line.strip().split('\t')
            doc_id, url, title, body = row
            doc_id = int(doc_id[1:])
            for word in set(preprocess(title)) | set(preprocess(body)):
                if word not in inverted_index:
                    buffer = bytearray()
                    encode_varbyte_to_buffer(doc_id, buffer)
                    inverted_index[word] = (doc_id, buffer)
                else:
                    last_id, buffer = inverted_index[word]            
                    gap = doc_id - last_id
                    encode_varbyte_to_buffer(gap, buffer)
                    inverted_index[word] = (doc_id, buffer)
    for word in inverted_index:
        inverted_index[word] = bytes(inverted_index[word][1])

    with open(f'{index_dir}/index.pkl', 'wb') as f:
        pickle.dump(inverted_index, f)

def make_submission(data_dir, index_dir, submission_file):
    # TODO:
    # - загрузить индекс из index_dir;
    # - прочитать запросы из data_dir/vkmarco-doceval-queries.tsv
    #   (без заголовка; поля через таб: QueryId, QueryText) и разбить их функцией preprocess();
    # - для каждого запроса найти документы, в которых есть ВСЕ термины запроса (И-запрос);
    # - для каждой строки data_dir/objects.csv (ObjectId,QueryId,DocumentId) поставить
    #   Label = 1, если документ нашёлся по запросу, иначе 0;
    # - записать submission_file в формате CSV с заголовком ObjectId,Label.
    def decode_varbyte_to_docs(stream):
        docs = []
        current_id = 0
        shift = 0
        n = 0
        for byte in stream:
            if byte < 128:
                n |= (byte << shift)
                shift += 7
            else:
                n |= ((byte & 0x7F) << shift)
                current_id += n
                docs.append(current_id)
                n = 0
                shift = 0
        return docs

    with open(f'{index_dir}/index.pkl', 'rb') as f:
        inverted_index = pickle.load(f)
        
    with open(f'{data_dir}/vkmarco-doceval-queries.tsv', 'r', encoding='utf-8') as f:
        queries = {}
        for line in f:
            row = line.strip().split('\t')
            query_id, query_text = row
            queries[query_id] = set(preprocess(query_text))
    with open(f'{data_dir}/objects.csv', 'r', encoding='utf-8') as f:
        with open(submission_file, 'w', encoding='utf-8') as out_f:
            out_f.write('ObjectId,Label\n')
            next(f) 
            for line in f:
                row = line.strip().split(',')
                object_id, query_id, document_id = row
                document_id = int(document_id[1:])
                query_terms = queries[query_id]
                if all(term in inverted_index and document_id in decode_varbyte_to_docs(inverted_index[term]) for term in query_terms):
                    label = 1
                else:
                    label = 0
                out_f.write(f'{object_id},{label}\n')



def main():
    parser = argparse.ArgumentParser(description='Boolean retrieval homework solution')
    parser.add_argument('--submission_file', help='куда записать сабмишн для Kaggle')
    parser.add_argument('--build_index', action='store_true', help='режим построения индекса')
    parser.add_argument('--index_dir', required=True, help='папка с индексом')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()
    if args.build_index:
        build_index(args.data_dir, args.index_dir)
    else:
        if not args.submission_file:
            parser.error('в режиме генерации сабмишна нужен --submission_file')
        make_submission(args.data_dir, args.index_dir, args.submission_file)
    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
