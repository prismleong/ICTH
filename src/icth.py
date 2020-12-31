import tensorboardX
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import get_args, get_dataloader, get_model, seed_everything

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class icth(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.test_dataloader = get_dataloader(self.args, train=False)

    def icth(self):
        repeat_times = self.args.repeat_times
        seeds = range(self.args.seed, self.args.seed+repeat_times)
        for seed in seeds:
            seed_everything(seed)

            self.model = get_model(self.args)
            self.optimizer = self.args.optimizer(
                self.model.parameters(), lr=self.args.learning_rate)
            self.train_dataloader = get_dataloader(self.args, train=True)

            self.model = self.model.to(DEVICE)
            for epochs in range(1, self.args.epochs+1):
                self.train_an_epoch()
                with torch.no_grad():
                    self.test()

            self.model = self.model.to('cpu')
            torch.cuda.empty_cache()

    def train_an_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_dataloader)
        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            preds = self.model(imgs)
            loss = F.cross_entropy(preds, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self):
        self.model.eval()
        
        pred_labels = []
        pbar = tqdm(self.test_dataloader)
        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = self.model(imgs)
            loss = F.cross_entropy(preds, labels)
            
            pred_labels.append(preds.argmax(1).cpu())
        pred_labels = torch.cat(pred_labels)


'''

'''

if __name__ == "__main__":
    args = get_args()

    tester = icth(args)
    tester.icth()