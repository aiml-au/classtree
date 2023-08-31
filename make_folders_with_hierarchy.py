import pandas as pd
from pathlib import Path
import json
import collections 

if __name__ == "__main__": 

    # mode = 'train' #  ['train', 'test', 'new']  | 'train' , 'test': evaluation, 'new': unseen/unlabeled      
    
    # model_name = 'amazon_product_reviews' 
    # data_path = "data/amazon_product_reviews/train_40k.csv"
    # data_path = "data/amazon_product_reviews/val_10k.csv"
    # data_path = "data/amazon_product_reviews/unlabeled_150k.csv"
    # text_col = 'Text'
    # label1_col = 'Cat1'
    # label2_col = 'Cat2'
    # label3_col = 'Cat3'

    # model_name = 'dbpedia'      
    # # data_path = "data/dbpedia/DBPEDIA_train.csv"
    # # data_path = "data/dbpedia/DBPEDIA_val.csv"
    # data_path = "data/dbpedia/DBPEDIA_test.csv"
    # text_col = 'text'
    # label1_col = 'l1'
    # label2_col = 'l2'
    # label3_col = 'l3'

    models = ['amazon_product_reviews', 'dbpedia'] 

    # in the order of train, val, test
    data_paths = {'amazon_product_reviews': ['train_40k.csv', 'val_10k.csv', 'unlabeled_150k.csv'],
                  'dbpedia': ['DBPEDIA_train.csv', 'DBPEDIA_val.csv', 'DBPEDIA_test.csv']}

    for model in models:
        print('Model to train: ', model)

        # See heading names from csv, and change it appropriately.
        if model == 'amazon_product_reviews':
            text_col = 'Text'
            label1_col = 'Cat1'
            label2_col = 'Cat2'
            label3_col = 'Cat3'
        else:
            text_col = 'text'
            label1_col = 'l1'
            label2_col = 'l2'
            label3_col = 'l3'

        for i, mode in enumerate(['train', 'test', 'new']): # ['train', 'test', 'new']
            print('mode: ', mode)

            data_path = f'data/{model}/{data_paths[model][i]}'

            parent_dir = f'{mode}_data/{model}'
            Path(parent_dir).mkdir(parents=True, exist_ok=True)


            columns = pd.read_csv(data_path, nrows=0).columns.tolist()
            print('Columns:', columns)

            df = pd.read_csv(data_path, sep=',')
            # df = pd.read_csv(data_path, sep=',', usecols=[textfile_name_col, text_col, label1_col, label2_col, label3_col] )
            print('# instance:', len(df))
            print('example:\n', df[:3])
            
            index_count = collections.defaultdict(int) # TODO: solve repeated category occuring in different hierarchy.
            naive_level_dict = {} # to save for evaluation dataset

            seen = {}

            if mode == 'test' and os.path.exists('data/naive_level_dict.json'):                        
                with open('data/naive_level_dict.json') as f:
                    naive_level_dict = json.load(f)


            def train_label_name(parent, comparing_label):
                label_in_context  = f'{parent}/{comparing_label}' # naive way to distinguish 'L1-a/L2-b' and 'L1-b/L2-b'

                if comparing_label not in seen: # at first, index its parent
                    seen[comparing_label] = [parent]
                    index_count[comparing_label] = 0
                    return comparing_label

                elif parent not in seen[comparing_label]: # if this label is seen but has different parent from the seen one's parent
                    seen[comparing_label].append(parent)
                    index_count[comparing_label] += 1
                    naive_level_dict[label_in_context] = index_count[comparing_label] # assign incremental number for every this elif case

                    return f'{comparing_label}-{naive_level_dict[label_in_context]}'
                
                else: #if comparing_label in seen and parent in seen[comparing_label]:        
                    if label_in_context in naive_level_dict: # keep giving this number for this label with different parent from the very first seen one's parent
                        return f'{comparing_label}-{naive_level_dict[label_in_context]}'
                    else:
                        return comparing_label

            # For val_data, should use saved naive_level_dict.json file to be aligned with train instances for solving: 'L1-a/L2-b' and 'L1-b/L2-b'
            def eval_label_name(parent, comparing_label):
                label_in_context = f'{parent}/{comparing_label}'
                if label_in_context in naive_level_dict:
                    return f'{comparing_label}({naive_level_dict[label_in_context]})'
                else:
                    return comparing_label
            
            for i, row in df.iterrows():
                if mode in ['train', 'test']:
                    level1_label = row[label1_col].replace(" ", "_").replace("/", "-")
                    level2_label = row[label2_col].replace(" ", "_").replace("/", "-")
                    level3_label = row[label3_col].replace(" ", "_").replace("/", "-")
                    import os.path

                    if mode == 'train':
                        level2_label_ = level2_label
                        if level2_label != 'unknown':
                            level2_label = train_label_name(level1_label, level2_label)
                        if level3_label != 'unknown':    
                            level3_label = train_label_name(level2_label_, level3_label)

                    elif os.path.exists('data/naive_level_dict.json'):     
                        level2_label = eval_label_name(level1_label, level2_label)
                        level3_label = eval_label_name(level2_label, level3_label)

                    level1_path = f'{parent_dir}/{level1_label}'
                    level2_path = f'{level1_path}/{level2_label}'
                    leaf_node_path = f'{level2_path}/{level3_label}'

                    Path(level1_path).mkdir(parents=True, exist_ok=True)

                    if row[label2_col].lower() == 'unknown':
                        leaf_node_path = level1_path
                    elif row[label3_col].lower() == 'unknown':
                        leaf_node_path = level2_path
                        Path(level2_path).mkdir(parents=True, exist_ok=True)
                    else:
                        Path(level2_path).mkdir(parents=True, exist_ok=True)
                        Path(leaf_node_path).mkdir(parents=True, exist_ok=True)    

                else:
                    leaf_node_path = parent_dir

                # file_name = row[textfile_name_col] if textfile_name_col in row else i
                file_name = i # just random name if there's no column like unique_id of id_column is not guaranteed unique 

                with open(f'{leaf_node_path}/{file_name}.txt', 'w') as f:
                    f.write(row[text_col])
    
            if mode == 'train':                
                with open('data/naive_level_dict.json', 'w') as f:
                    json.dump(naive_level_dict, f, indent=4)                    

            print('Finished')
