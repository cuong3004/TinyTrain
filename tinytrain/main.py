import torch
import torchvision
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch 
from mobilevit import mobilevit_xxs

n_epochs = 3
batch_size_train = 128
batch_size_test = 128
learning_rate = 0.0001

random_seed = 42


torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.ImageFolder('../tiny-imagenet-200/train/',
                                transform=torchvision.transforms.Compose([
                                    # torchvision.transforms.Resize(256),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.5,), (0.5,))
                            ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.ImageFolder('../tiny-imagenet-200/train/',
                                transform=torchvision.transforms.Compose([
                                    # torchvision.transforms.Resize(256),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.5,), (0.5,))
                            ])),
    batch_size=batch_size_test, shuffle=False)


# %%

# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# %%

# def soft_cross_entropy(predicts, targets):
#             student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
#             targets_prob = torch.nn.functional.softmax(targets, dim=-1)
#             return (- targets_prob * student_likelihood).mean()

class LitClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()

        import torchvision.models as models
        # self.model = models.mobilenet_v2(pretrained=True)
        # self.model.classifier[1] = nn.Linear(self.model.last_channel, 200)
        self.model = mobilevit_xxs()
        # df_train, df_test = train_test_split(
        #     df, random_state=42, shuffle=True, test_size=0.2
        # )
        # df_train, df_valid = train_test_split(
        #     df_train, random_state=42, shuffle=True, test_size=0.1

        # print(self.teacher_model)

        self.acc = torchmetrics.Accuracy()
        # self.f1 = torchmetrics.F1Score(num_classes=1, multiclass=False)
        # self.pre = torchmetrics.Precision(num_classes=1, multiclass=False)
        # self.rec = torchmetrics.Recall(num_classes=1, multiclass=False)
        

    
    def train_dataloader(self):
        return train_loader
        # dataset = CusttomData(self.df_train, self.tokenizer)
        # return DataLoader(dataset, batch_size=64, num_workers=2, shuffle=True)
    
    def val_dataloader(self):
        return test_loader
        # dataset = CusttomData(self.df_valid, self.tokenizer)
        # return DataLoader(dataset, batch_size=32, num_workers=2)
    
    # def test_dataloader(self):
    #     dataset = CusttomData(self.df_test, self.tokenizer)
    #     return DataLoader(dataset, batch_size=32, num_workers=2)


    def configure_optimizers(self):

        # params = list(self.student_model.parameters()) + list(self.fit_dense.parameters())
        params = list(self.model.parameters())

        optimizer = torch.optim.AdamW(params,
                  lr = 1e-3, # args.learning_rate - default is 5e-5,
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
        # optimizer_fit = AdamW(self.fit_dense.parameters(),
        #           lr = 1e-4, # args.learning_rate - default is 5e-5,
        #           eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
        #         )
        # total_steps = len(self.train_dataloader()) * Epoch

        # # Create the learning rate scheduler.
        # scheduler = get_linear_schedule_with_warmup(optimizer, 
        #                 num_warmup_steps = 0, # Default value in run_glue.py
        #                 num_training_steps = total_steps)
        # return [optimizer], [scheduler]
        # return [optimizer_student, optimizer_fit], []
        
        return optimizer

    def share_batch(self, batch, state):
        image, labels = batch
        
        out_logits = self.model(image) 
        


        loss = F.cross_entropy(out_logits, labels)

        # self.log('train_loss', loss)
        self.log(f"{state}_loss", loss, 
                # on_step=False, on_epoch=True
                )

        acc = self.acc(out_logits, labels)
        # pre = self.pre(student_logits, labels)
        # rec = self.rec(student_logits, labels)
        # f1 = self.f1(student_logits, labels)
        self.log(f'{state}_acc', acc, on_step=False, on_epoch=True)
        # self.log(f'{state}_rec', rec, on_step=False, on_epoch=True)
        # self.log(f'{state}_pre', pre, on_step=False, on_epoch=True)
        # self.log(f'{state}_f1', f1, on_step=False, on_epoch=True)

        # self.log('train_acc', acc, on_step=True, on_epoch=False)
        return loss


    def training_step(self, train_batch, batch_idx):
        loss = self.share_batch(train_batch, "train")
        return loss

    def validation_step(self, val_batch, batch_idx):

        loss = self.share_batch(val_batch, "valid")

    def test_step(self, test_batch, batch_idx):

        loss = self.share_batch(test_batch, "test")
# 
# from fake_review.transformer import TinyBertForSequenceClassification, BertTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
filename = f"model"
checkpoint_callback = ModelCheckpoint(
    filename=filename,
    save_top_k=1,
    verbose=True,
    monitor='valid_loss',
    mode='min',
)
# tokenizer = BertTokenizer("../../bert-base-uncased/vocab.txt")
# %%
model_lit = LitClassification()
# %%
trainer = pl.Trainer(
                    gpus=1, 
                    max_epochs=50,
                    # limit_train_batches=0.2,
                    # reload_dataloaders_every_n_epochs=1,
                    default_root_dir="/content/drive/MyDrive/log_fake_review/log_tiny_train",
                    callbacks=[checkpoint_callback]
                    )
trainer.fit(model_lit)
trainer.test(model_lit)
