import copy
from re import L
import torch
from torch import nn

from tqdm import tqdm
import btorch
from btorch.utils.load_save import save_model, resume

class Module(nn.Module):
    """Base class for all neural network modules.
    Your models should also subclass this class.
    btorch.nn.Module is inhernet from pytorch.nn, hence, all syntax is same as pytorch
    This class provides some highlevel training method like Keras:
        - .fit()
        - .train_net()
        - .train_epoch()
        - .test_epoch()
    You can override these class for your own usage. Just put your training loop here.

    If you decided to use the highlevel training loop. Please set 
        - self._lossfn (default:pytorch Loss Func)[required]
        - self._optimizer (default:pytorch Optimizer)[required]
        - self._lr_scheduler (default:pytorch lr_Scheduler)[optional]
        - self._config (default:dict)[optional]
    You can set them to be a pytorch instance or a dictionary (for advanced uses)
    The default guideline is only for the default highlevel functions.

    All the classmethods in this class can be taken out and use them alone. 
    They are a good starter code for traditional PyTorch training.
    """
    def __init__(self) -> None:
        super(Module, self).__init__()
        self._lossfn = None
        self._optimizer = None
        self._lr_scheduler = None
        self._config = None

    def fit(self, train_loader, eval_loader=None):
        if self._lossfn is None or self._optimizer is None:
            raise ValueError("Print set `self._lossfn` and `self._optimizer`")
        self.train_net(net=self, criterion=self._lossfn, optimizer=self._optimizer,
                   trainloader=train_loader, testloader=eval_loader, lr_scheduler=self._lr_scheduler, config=self._config)

    @classmethod
    def train_net(cls, net, criterion, optimizer, trainloader, testloader=None, lr_scheduler=None, config=None):
        """Standard PyTorch training loop. Override this function when necessary.
        It uses .train_epoch() and .test_epoch()
        
        Args:
            net ([type]): [description]
            criterion ([type]): [description]
            optimizer ([type]): [description]
            trainloader ([type]): [description]
            testloader ([type], optional): [description]. Defaults to None.
            lr_scheduler ([type], optional): [description]. Defaults to None.
            config ([type], optional): [description]. Defaults to None.
        """
        if config is None:
            config = dict()
        start_epoch = config.get("start_epoch", 0)
        max_epoch = config.get("max_epoch", 10)
        device = config.get("device", "cpu")
        save_path = config.get("save", None)
        resume_path = config.get("resume", None)

        if resume_path is not None:
            start_epoch = resume(resume_path, net, optimizer, lr_scheduler)

        train_loss_data = []
        test_loss_data = []

        for epoch in range(start_epoch, max_epoch):
            train_loss = cls.train_epoch(
                net, criterion, trainloader, optimizer, epoch, device)
            train_loss_data.append(train_loss)
            if testloader is not None:
                test_loss = cls.test_epoch(net, criterion, testloader, device)
                test_loss_data.append(test_loss)
            if save_path is not None:
                to_save = dict(train_loss_data=train_loss_data,
                            test_loss_data=test_loss_data)
                if test_loss <= min(test_loss_data, default=999):
                    save_model(net, f"{save_path}_best.pt",
                            to_save, optimizer, lr_scheduler)
                save_model(net, f"{save_path}_latest.pt",
                        to_save, optimizer, lr_scheduler)
                if epoch % 20 == 0:
                    save_model(net, f"{save_path}_{epoch}.pt",
                            to_save, optimizer, lr_scheduler)
            if lr_scheduler is not None:
                lr_scheduler.step()

    @classmethod
    def train_epoch(cls, net, criterion, trainloader, optimizer, epoch_idx, device='cuda', config=None):
        """This is the very basic training function for one epoch. Override this function when necessary
            
        Returns:
            (float): train_loss
        """
        net.train()
        train_loss = 0
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predicted = net(inputs)
            loss = criterion(predicted, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            pbar.set_description(
                f"epoch {epoch_idx+1} iter {batch_idx}: train loss {loss.item():.5f}.")
        return train_loss/(batch_idx+1)

    @classmethod
    def test_epoch(cls, net, criterion, testloader, device='cuda'):
        """This is the very basic evaluating function for one epoch. Override this function when necessary
            
        Returns:
            (float): eval_loss
        """

        net.eval()
        train_loss = 0
        with torch.inference_mode():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                predicted = net(inputs)
                loss = criterion(predicted, targets)
                train_loss += loss.item()
        return train_loss/(batch_idx+1)

    @classmethod
    def overfit_small_batch(cls, net, criterion, loader, optimizer):
        """This is a helper function to check if your model is working by checking if it can overfit a small dataset.
        It uses .train_epoch().
        """
        net_test = copy.deepcopy(net)
        if loader.batch_size > 5:
            loader = btorch.utils.change_batch_size(loader, 5)
        loss_history = []
        for epoch in range(100):
            train_loss = cls.train_epoch(net_test, criterion, loader, optimizer, epoch)
            loss_history.append(train_loss)
        print(loss_history)
        del net_test
        try:
            last_loss = loss_history[-1].item()
            if last_loss < 1e-5:
                print("It looks like your model is working. Please check the loss_history to see whether it is overfitting. Expected to be overfit.")
        except:
            pass
        print("Please check the loss_history to see whether it is overfitting. Expected to be overfit.")
            
    
    # @property
    # def _lossfn(self):
    #     return self.__lossfn

    # @_lossfn.setter
    # def _lossfn(self, __lossfn):
    #     self.__lossfn = __lossfn

    # @property
    # def _optimizer(self):
    #     return self.__optimizer

    # @_optimizer.setter
    # def _optimizer(self, __optimizer):
    #     self.__optimizer = __optimizer

    # @property
    # def _lr_scheduler(self):
    #     return self.__lr_scheduler

    # @_lr_scheduler.setter
    # def _lr_scheduler(self, __lr_scheduler):
    #     self.__lr_scheduler = __lr_scheduler

    # @property
    # def _config(self):
    #     return self.__config

    # @_config.setter
    # def _config(self, __config):
    #     self.__config = __config
