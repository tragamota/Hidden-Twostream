import torch
import torch.nn as nn

class MotionNet(nn.Module):
    def __init__(self):
        super(MotionNet, self).__init__()

        self.conv1 = nn.Conv2d(33, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        self.flow6 = nn.Conv2d(1024, 20, kernel_size=3, stride=1, padding=1)
        self.flow6_upsample = nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1)

        self.deconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.xconv5 = nn.Conv2d(1044, 512, kernel_size=3, stride=1, padding=1)

        self.flow5 = nn.Conv2d(512, 20, kernel_size=3, stride=1, padding=1)
        self.flow5_upsample = nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1)

        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.xconv4 = nn.Conv2d(788, 256, kernel_size=3, stride=1, padding=1)

        self.flow4 = nn.Conv2d(256, 20, kernel_size=3, stride=1, padding=1)
        self.flow4_upsample = nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.xconv3 = nn.Conv2d(404, 128, kernel_size=3, stride=1, padding=1)

        self.flow3 = nn.Conv2d(128, 20, kernel_size=3, stride=1, padding=1)
        self.flow3_upsample = nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.xconv2 = nn.Conv2d(212, 64, kernel_size=3, stride=1, padding=1)

        self.flow2 = nn.Conv2d(64, 20, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):

        out = self.relu(self.conv1(x))
        out = self.relu(self.conv1_1(out))

        out = self.relu(self.conv2(out))
        out_conv2_1 = self.relu(self.conv2_1(out))

        out = self.relu(self.conv3(out_conv2_1))
        out_conv3_1 = self.relu(self.conv3_1(out))

        out = self.relu(self.conv4(out_conv3_1))
        out_conv4_1 = self.relu(self.conv4_1(out))

        out = self.relu(self.conv5(out_conv4_1))
        out_conv5_1 = self.relu(self.conv5_1(out))

        out = self.relu(self.conv6(out_conv5_1))
        out_conv6_1 = self.relu(self.conv6_1(out))

        flow6_out = self.flow6(out_conv6_1)
        flow6_upsampled = self.flow6_upsample(flow6_out)

        out = self.relu(self.deconv5(out_conv6_1))
        out = self.relu(self.xconv5(torch.cat((out, out_conv5_1, flow6_upsampled), dim=1)))

        flow5_out = self.flow5(out)
        flow5_upsampled = self.flow5_upsample(flow5_out)

        out = self.relu(self.deconv4(out))
        out = self.relu(self.xconv4(torch.cat((out, out_conv4_1, flow5_upsampled), dim=1)))

        flow4_out = self.flow4(out)
        flow4_upsampled = self.flow4_upsample(flow4_out)

        out = self.relu(self.deconv3(out))
        out = self.relu(self.xconv3(torch.cat((out, out_conv3_1, flow4_upsampled), dim=1)))

        flow3_out = self.flow3(out)
        flow3_upsampled = self.flow3_upsample(flow3_out)

        out = self.relu(self.deconv2(out))
        out = self.relu(self.xconv2(torch.cat((out, out_conv2_1, flow3_upsampled), dim=1)))

        flow2_out = self.flow2(out)

        return flow2_out, flow3_out, flow4_out, flow5_out, flow6_out


class TinyMotionNet(nn.Module):
    def __init__(self):
        super(TinyMotionNet, self).__init__()

        self.conv1 = nn.Conv2d(33, 64, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)

        self.flow4 = nn.Conv2d(128, 20, kernel_size=3, stride=1, padding=1)
        self.flow4_upsample = nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1)

        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.xconv3 = nn.Conv2d(404, 128, kernel_size=3, stride=1, padding=1)

        self.flow3 = nn.Conv2d(128, 20, kernel_size=3, stride=1, padding=1)
        self.flow3_upsample = nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.xconv2 = nn.Conv2d(212, 64, kernel_size=3, stride=1, padding=1)

        self.flow2 = nn.Conv2d(64, 20, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        conv1_out = self.relu(self.conv1(x))
        conv2_out = self.relu(self.conv2(conv1_out))
        conv3_out = self.relu(self.conv3(conv2_out))
        conv4_out = self.relu(self.conv4(conv3_out))

        flow4_out = self.flow4(conv4_out)
        flow4_up = torch.nn.functional.interpolate(flow4_out, size=conv3_out.shape[-2:], mode='bilinear', align_corners=True)

        deconv3_out = self.relu(self.deconv3(conv4_out))
        xconv3_out = self.relu(self.xconv3(torch.cat((deconv3_out, flow4_up, conv3_out), dim=1)))

        flow3_out = self.flow3(xconv3_out)
        flow3_up = torch.nn.functional.interpolate(flow4_out, size=conv2_out.shape[-2:], mode='bilinear', align_corners=True)

        deconv2_out = self.relu(self.deconv2(xconv3_out))
        xconv2_out = self.relu(self.xconv2(torch.cat((deconv2_out, flow3_up, conv2_out), dim=1)))

        flow2_out = self.flow2(xconv2_out)

        return flow2_out, flow3_out, flow4_out
