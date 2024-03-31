import torch
import torch.nn as nn
import torch.nn.functional as F


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, routing_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes
        self.routing_iterations = routing_iterations
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.matmul(x.unsqueeze(1), self.weights)
        x = x.view(batch_size, self.num_capsules, self.num_route_nodes, self.out_channels)
        logits = torch.zeros(*x.size()).to(x.device)
        for i in range(self.routing_iterations):
            probs = F.softmax(logits, dim=2)
            outputs = self.squash((probs * x).sum(dim=2, keepdim=True))

            if i != self.routing_iterations - 1:
                delta_logits = (x * outputs).sum(dim=-1)
                logits = logits + delta_logits
        return outputs.squeeze(2)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * input_tensor / torch.sqrt(squared_norm)


class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCapsuleLayer, self).__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in range(num_capsules)
        ])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), -1, u.size(-1))
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * input_tensor / torch.sqrt(squared_norm)



class CapsuleEncoder(nn.Module):
    def __init__(self, image_channels=3, primary_caps_channels=256, primary_caps_num=32, primary_kernel_size=9, primary_stride=2, num_capsules=10, num_route_nodes=32 * 8 * 8, in_channels=8, out_channels=16, routing_iterations=3):
        super(CapsuleEncoder, self).__init__()
        self.conv_layer = nn.Conv2d(image_channels, primary_caps_channels, kernel_size=9, stride=1, padding=0)
        self.primary_capsules = PrimaryCapsuleLayer(primary_caps_num, primary_caps_channels, primary_caps_channels, primary_kernel_size, primary_stride)
        self.digit_capsules = CapsuleLayer(num_capsules, num_route_nodes, in_channels, out_channels, routing_iterations)

    def forward(self, x):
        x = F.relu(self.conv_layer(x))
        x = self.primary_capsules(x)
        x = x.transpose(1, 2)
        x = self.digit_capsules(x)
        return x



class ReshapeCapsuleOutput(nn.Module):
    def __init__(self, target_height, target_width):
        super(ReshapeCapsuleOutput, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, capsule_output):
        batch_size, num_capsules, capsule_dimension = capsule_output.size()

        # Calculate the target number of channels based on desired spatial dimensions
        target_channels = self.target_height * self.target_width

        # Check if the reshaping is possible with the given dimensions
        if num_capsules * capsule_dimension != target_channels:
            raise ValueError("Cannot reshape capsule output to the target size. "
                             "Make sure num_capsules*capsule_dimension matches target_height*target_width.")

        # Reshape to [batch_size, target_channels, 1, 1] then expand to the target size
        reshaped_output = capsule_output.view(batch_size, target_channels, 1, 1)
        reshaped_output = reshaped_output.expand(-1, -1, self.target_height, self.target_width)

        return reshaped_output



if __name__ == '__main__':
    # Example configuration for MNIST
    image_channels = 3
    primary_caps_channels = 256
    primary_caps_num = 32
    primary_kernel_size = 9
    primary_stride = 2
    num_capsules = 10
    num_route_nodes = 32 * 6 * 6  # Adjust based on your primary capsule output
    in_channels = 8
    out_channels = 3