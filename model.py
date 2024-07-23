from transformers.modeling_outputs import ModelOutput
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import cosine_similarity
import torch


class PersonaCLRModel(torch.nn.Module):
    def __init__(self, bert_model_name_or_path, 
                 tau=0.05, 
                 latent_size=768,
                 pooling="cls"):
        super().__init__()
        

        self.tau = tau
        self.pooling = pooling # cls or mean
        self.bert = AutoModel.from_pretrained(bert_model_name_or_path)
        config = self.bert.config

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]

    def similarity(self, x, y):
        return cosine_similarity(x, y, dim=-1) / self.tau
    
    def predict(self,
        target_dict,
        reference_dict,
        labels,
        return_z=False,
        **kwargs,
        ):
        
        outputs1 = self.bert(
            **target_dict
        )

        outputs2 = self.bert(
            **reference_dict
        )
        
        if self.pooling == "cls":
            z1 = outputs1[1]
            z2 = outputs2[1]
        elif self.pooling == "mean":
            z1 = self._mean_pooling(outputs1, target_dict["attention_mask"])
            z2 = self._mean_pooling(outputs2, reference_dict["attention_mask"])

        cossim = cosine_similarity(z1, z2, dim=1)
        
        if return_z:
            return cossim, z1, z2
        return cossim


    def forward(
        self,
        target_dict,
        reference_dict,
        labels,
        is_valid,
        **kwargs,
        ):
        
        outputs1 = self.bert(
            **target_dict
        )

        outputs2 = self.bert(
            **reference_dict
        )
        
        if self.pooling == "cls":
            z1 = outputs1[1]
            z2 = outputs2[1]
        elif self.pooling == "mean":
            z1 = self._mean_pooling(outputs1, target_dict["attention_mask"])
            z2 = self._mean_pooling(outputs2, reference_dict["attention_mask"])
        else:
            print("ERROR", self.pooling)

        # constrastive loss
        cos_sim = self.similarity(z1.unsqueeze(1), z2.unsqueeze(0))
        loss_fct = torch.nn.CrossEntropyLoss()
        sim_loss = loss_fct(cos_sim, labels)
        

        return ModelOutput(
            loss=sim_loss,
            logits=cos_sim,
        )
