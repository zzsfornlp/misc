#

# parse bib file

import sys
import bibtexparser
# from mspx.utils import zopen, default_json_serializer

def main(input_path, output_path):
    with open(input_path) as fd:
        bib_database = bibtexparser.load(fd)
    outputs = []
    for entry in bib_database.entries:
        # doc_id = entry['ID'].replace(':', '__')
        doc_id = entry['ID']
        one = {'doc_id': doc_id, 'text': entry['abstract'],
               'info': {k: entry[k] for k in ['title', 'keywords']}}
        outputs.append(one)
    if output_path:
        import json
        # default_json_serializer.save_iter(outputs, output_path)
        with open(output_path, 'w') as fd:
            for one in outputs:
                fd.write(json.dumps(one) + '\n')

# python3 -m mspx.tasks.others.mat.misc0.parse_bib IN OUT
if __name__ == '__main__':
    main(*sys.argv[1:])
