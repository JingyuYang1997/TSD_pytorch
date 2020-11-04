import torch
import torch.nn as nn

class AutoRegressiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_confounder):
        super(AutoRegressiveLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_confounder = num_confounder
        self.cell = nn.LSTMCell(self.input_size+1, self.hidden_size)
        self.fc = nn.Sequential(nn.Linear(in_features=self.hidden_size,out_features=self.num_confounder),
                                nn.Tanh())

    def forward(self, inputs, initial_state):
        time_steps = inputs.size(0)
        outputs = []
        # if initial_state:
        h, c, z = initial_state
        for t in range(time_steps):
            combine_input = torch.cat([inputs[t,:,:], z],dim=1)
            h, c = self.cell(combine_input, (h,c))
            z = self.fc(h)
            outputs.append(z)
        # else:
        #     h, c = self.cell(inputs[0, :, :])
        #     outputs.append(h)
        #     for t in range(1, time_steps):
        #         h, c = self.cell(inputs[t,:,:], (h,c))
        #         outputs.append(h)
        return torch.stack(outputs, dim=0), (h,c)

class ComputeLoss(nn.Module):
    def __init__(self):
        super(ComputeLoss, self).__init__()

    def forward(self, target_treatments, treatment_predictions,rnn_input):
        mask = torch.sign(torch.max(torch.abs(rnn_input),dim=-1)[0])
        flat_mask = mask.view(-1,1)
        cross_entropy = -torch.sum(
            (target_treatments*torch.log(torch.clamp(treatment_predictions,1e-10, 1.0))
            +(1-target_treatments)*torch.log(torch.clamp(1-treatment_predictions, 1e-10, 1.0)))*flat_mask,dim=1)
        cross_entropy /= torch.sum(compute_sequence_length(rnn_input))
        loss= torch.sum(cross_entropy)
        a = loss.requires_grad
        return loss

def compute_sequence_length(sequence):
    used = torch.sign(torch.max(torch.abs(sequence), dim=-1)[0])
    length = torch.sum(used, dim=0)
    # length = tf.cast(length, tf.int32)
    return length

