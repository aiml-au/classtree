import pandas as pd
from pathlib import Path

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

        for i, mode in enumerate(['train', 'test', 'new']):
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
            # ids = []

            for i, row in df.iterrows():
                if mode in ['train', 'test']:
                    level1_path = f'{parent_dir}/{row[label1_col].replace(" ", "_")}'
                    level2_path = f'{level1_path}/{row[label2_col].replace(" ", "_")}'
                    leaf_node_path = f'{level2_path}/{row[label3_col].replace(" ", "_")}' 

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

            print('Finished')
