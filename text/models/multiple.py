import torch.nn.functional as F
import torch
import os


from transformers import FlaubertForSequenceClassification, FlaubertModel
import torchvision.models as models
class Inception(torch.nn.Module):
	def __init__(self, pretrained=True):
		super(Inception, self).__init__()
		num_classes = 14
		self.model = models.inception_v3(pretrained=pretrained, aux_logits=False)
		self.model.fc = torch.nn.Linear(2048, num_classes)

	def forward(self, x):
		output = self.model(x)
		return output


class MLP(torch.nn.Module):
# dataset.num_classes, dataset.num_features, num_layers, hidden
    def __init__(self, num_classes, input_dim, num_mlp_layers, emb_dim, drop_ratio=0, multi_model=False):
        super(MLP, self).__init__()

        self.num_mlp_layers = num_mlp_layers
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.multi_model = multi_model
        self.num_labels = num_classes
        #   self.emb_dim = emb_dim
        # mlp
        module_list = [
            torch.nn.Linear(input_dim, self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_ratio),
        ]

        for i in range(self.num_mlp_layers - 1):
            module_list += [torch.nn.Linear(self.emb_dim, self.emb_dim),
                            torch.nn.BatchNorm1d(self.emb_dim),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(p=self.drop_ratio)]

        # relu is applied in the last layer to ensure positivity
        if not multi_model:
            module_list += [torch.nn.Linear(self.emb_dim,self.num_labels )]

        self.mlp = torch.nn.Sequential(
            *module_list
        )
    def reset_parameters(self):
        for layer in self.mlp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self,x):

        output = self.mlp(x)
        if self.multi_model:
            return output
        else:
            return F.log_softmax(output, dim=-1)



class MultipleModel(torch.nn.Module):

    def __init__(self, num_classes,num_layers,  hidden,  drop_ratio=0):
        super(MultipleModel, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.hidden = hidden
        self.num_labels = num_classes

        self.text_flaubert = FlaubertForSequenceClassification.from_pretrained("flaubert/flaubert_base_uncased", output_hidden_states=True)

        self.jt_specs =  MLP(num_classes = self.num_labels, input_dim=2, num_mlp_layers=self.num_layers, emb_dim=self.hidden,    drop_ratio=self.drop_ratio, multi_model=True)
#3372
        self.last_mlp = MLP(num_classes = self.num_labels, input_dim=1068 , num_mlp_layers=self.num_layers, emb_dim=self.hidden,
                           drop_ratio=self.drop_ratio, multi_model=False)

    def reset_parameters(self):
        self.text_mlp.reset_parameters()
        self.last_mlp.reset_parameters()



    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None,stime=None,media=None,duration=None,labels=None):
        x = self.text_flaubert(input_ids, attention_mask, token_type_ids)
        hidden_states = x[1]
        #https://github.com/huggingface/transformers/issues/1328
        pooled_output = hidden_states[-1]#torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1) #In their paper, BERT gets the best results by concatenating the last four layers
        pooled_output = pooled_output[:, 0, :]
        specs = torch.stack((stime, duration), dim=1)
        x_spects = self.jt_specs(specs)
        input = torch.cat((pooled_output,x_spects), dim=1)
        print(input.shape)
        return self.last_mlp(input)

    def __repr__(self):
        return self.__class__.__name__