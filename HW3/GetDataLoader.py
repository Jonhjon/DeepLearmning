from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd



class GetDataLoader():

    def __init__(self, model_name, use_auth_token=None):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token, return_tensors="pt")
        
        original_vocab_size = len(self.tokenizer)
        
        # 設定 pad_token (處理多種情況)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"✅ 使用 eos_token 作為 pad_token")
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                print(f"✅ 使用 unk_token 作為 pad_token")
            else:
                # 如果都沒有,添加新的 pad_token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print(f"⚠️  添加新的 [PAD] token: vocab size {original_vocab_size} -> {len(self.tokenizer)}")
        else:
            print(f"✅ Tokenizer 已有 pad_token: {self.tokenizer.pad_token}")
        
        self.tokenizer.model_max_length = 300  # 設置截斷長度

        pass

    def get_DataLoader(self, data_sum=-1, batch_size=16,train_path='Data\test_data.csv',test_path='Data\test_data.csv',val_path=None):

        # 讀取訓練資料
        df_train = pd.read_csv(train_path, sep=',')
        train_x = df_train['review']
        train_y = df_train['label']
        # 如果沒有提供驗證資料路徑，從訓練資料中分割
        if val_path is None:
            train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y, test_size=0.2, random_state=42
            )
        else:
            # 讀取驗證資料
            df_val = pd.read_csv(val_path, sep=',')
            val_x = df_val['review']
            val_y = df_val['label']

        # 讀取測試資料
        df_test = pd.read_csv(test_path, sep=',')
        test_x = df_test['review']
        test_y = df_test['label']

        # 建立訓練集 DataFrame
        df = pd.DataFrame()
        df['Text'] = train_x.reset_index(drop=True)
        df['labels'] = train_y.reset_index(drop=True)
        train = pd.concat([df['Text'],  df['labels']], axis=1)

        # 建立驗證集 DataFrame
        df = pd.DataFrame()
        df['Text'] = val_x.reset_index(drop=True)
        df['labels'] = val_y.reset_index(drop=True)
        Validation = pd.concat([df['Text'],  df['labels']], axis=1)

        # 建立測試集 DataFrame
        df = pd.DataFrame()
        df['Text'] = test_x.reset_index(drop=True)
        df['labels'] = test_y.reset_index(drop=True)
        Test = pd.concat([df['Text'],  df['labels']], axis=1)


        if data_sum != -1: train = train[:data_sum]
        # if data_sum != -1: Validation = Validation[:data_sum]
        # if data_sum != -1: Test = Test[:data_sum]

        
        train = Dataset.from_pandas(train)
        Validation = Dataset.from_pandas(Validation)
        Test = Dataset.from_pandas(Test)
        datasetDict = DatasetDict({'train': train, 'Validation': Validation, 'Test': Test})

        # 將資料轉成 input_ids
        def tokenize_function(example):
            return self.tokenizer(example["Text"], truncation=True)
        tokenized_datasets = datasetDict.map(tokenize_function, batched=True)


        # 移除不需要的欄位
        print(tokenized_datasets["train"].column_names)
        tokenized_datasets = tokenized_datasets.remove_columns(['Text'])
        tokenized_datasets.set_format("torch")

        print(tokenized_datasets["train"].column_names)


        # 填充資料長度
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        train_dataloader = DataLoader(
            tokenized_datasets["train"], 
            shuffle=False, 
            batch_size=batch_size, 
            collate_fn=data_collator
        )
        val_dataloader = DataLoader(
            tokenized_datasets["Validation"], 
            shuffle=False, 
            batch_size=batch_size, 
            collate_fn=data_collator
        )
        test_dataloader = DataLoader(
            tokenized_datasets["Test"], 
            shuffle=False, 
            batch_size=batch_size, 
            collate_fn=data_collator
           
        )

        return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    use_auth_token = 'your_huggingface_token_here'

    getDataLoader = GetDataLoader(model_name, use_auth_token=use_auth_token)
    train_dataloader, val_dataloader, test_dataloader = getDataLoader.get_DataLoader(data_sum=-1, batch_size=3)


